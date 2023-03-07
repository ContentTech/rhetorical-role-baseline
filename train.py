from prettytable import PrettyTable
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from torch.optim import lr_scheduler
from termcolor import colored

import torch
import random
import json
import time
import models
import numpy as np
import os

from eval import eval_model
from utils import tensor_dict_to_gpu, tensor_dict_to_cpu, ResultWriter, get_num_model_parameters, print_model_parameters
from task import Task, Fold
import gc
import copy
# random.seed(1)


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.3, emb_name=''):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name=''):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                #这里加入fgm restore判断是否恢复参数了
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class CosineWarmupScheduler(lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

class SentenceClassificationTrainer:
    '''Trainer for baseline model and also for Sequantial Transfer Learning. '''

    def __init__(self, device, config, task: Task, result_writer:ResultWriter):
        self.device = device
        self.config = config
        self.result_writer = result_writer
        self.cur_result = dict()
        self.cur_result["task"] = task.task_name
        self.cur_result["config"] = config

        self.labels = task.labels
        self.task = task
        print(colored(self.config, 'green'))
        print(colored(self.config['model'], 'green'))
        print(colored(self.labels, 'green'))
        print(colored(self.task.task_name, 'green'))
        print(colored(self.task.dev_metric, 'green'))


    def write_results(self, fold_num, epoch, train_duration, dev_metrics, dev_confusion, test_metrics, test_confusion):
        self.cur_result["fold"] = fold_num
        self.cur_result["epoch"] = epoch
        self.cur_result["train_duration"] = train_duration
        self.cur_result["dev_metrics"] = dev_metrics
        self.cur_result["dev_confusion"] = dev_confusion
        self.cur_result["test_metrics"] = test_metrics
        self.cur_result["test_confusion"] = test_confusion

        self.result_writer.write(json.dumps(self.cur_result))


    def run_training_for_fold(self, fold_num, fold: Fold, initial_model=None, return_best_model=False):

        self.result_writer.log(f'device: {self.device}')

        train_batches, dev_batches, test_batches = fold.train, fold.dev, fold.test

        self.result_writer.log(f"fold: {fold_num}")
        self.result_writer.log(f"train batches: {len(train_batches)}")
        self.result_writer.log(f"dev batches: {len(dev_batches)}")
        self.result_writer.log(f"test batches: {len(test_batches)}")

        # instantiate model per reflection
        if initial_model is None:
            model = getattr(models, self.config["model"])(self.config, [self.task])
        else:
            self.result_writer.log("Loading weights from initial model....")
            model = copy.deepcopy(initial_model)
            # for transfer learning do not transfer the output layer
            model.reinit_output_layer([self.task], self.config)

        self.result_writer.log("Model: " + model.__class__.__name__)
        self.cur_result["model"] = model.__class__.__name__

        model.to(self.device)

        max_train_epochs = self.config["max_epochs"]
        lr = self.config["lr"]
        max_grad_norm = 1.0

        self.result_writer.log(f"Number of model parameters: {get_num_model_parameters(model)}")
        self.result_writer.log(f"Number of model parameters bert: {get_num_model_parameters(model.bert)}")
        # self.result_writer.log(f"Number of model parameters word_lstm: {get_num_model_parameters(model.word_lstm)}")
        # self.result_writer.log(f"Number of model parameters attention_pooling: {get_num_model_parameters(model.attention_pooling)}")
        # self.result_writer.log(f"Number of model parameters sentence_lstm: {get_num_model_parameters(model.sentence_lstm)}")
        # self.result_writer.log(f"Number of model parameters sentence_attention: {get_num_model_parameters(model.multihead_attn)}")
        self.result_writer.log(f"Number of model parameters crf: {get_num_model_parameters(model.crf)}")
        print_model_parameters(model)

        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler()
        # for feature based training use Adam optimizer with lr decay after each epoch (see Jin et al. Paper)
        optimizer = Adam(model.parameters(), lr=lr)
        epoch_scheduler = StepLR(optimizer, step_size=1, gamma=self.config["lr_epoch_decay"])
        # epoch_scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=self.config.get("warmup", 200), max_iters=self.config.get("max_iters", 2000))

        best_dev_result = 0.0
        early_stopping_counter = 0
        epoch = 0
        early_stopping = self.config["early_stopping"]
        best_model = None
        optimizer.zero_grad()

        if self.config.get("fgm", False):
            fgm = FGM(model)

        while epoch < max_train_epochs and early_stopping_counter < early_stopping:
            epoch_start = time.time()

            self.result_writer.log(f'training model for fold {fold_num} in epoch {epoch} ...')

            random.shuffle(train_batches)
            # train model
            model.train()
            for batch_num, batch in enumerate(train_batches):
                # move tensor to gpu
                tensor_dict_to_gpu(batch, self.device)

                # Runs the forward pass with autocasting.
                ctype = self.config.get("dtype", None)
                ctpye = torch.float16 if ctype is not None and ctype == "float16" else torch.float32
                with torch.cuda.amp.autocast(dtype=ctpye):
                    if self.config.get('auxiliary_task', None) is not None:
                        output = model(
                            batch=batch,
                            labels=batch["label_ids"],
                            auxiliary_labels=batch["auxiliary_labels_ids"]
                        )
                    else:
                        output = model(
                            batch=batch,
                            labels=batch["label_ids"]
                        )
                    loss = output["loss"]
                    if self.config.get('use_multi_loss', 0) > 0:
                        loss = loss.sum()
                    else:
                        if 'auxiliary_loss' in output and output['auxiliary_loss'] is not None:
                            auxiliary_loss = output["auxiliary_loss"]
                            mu = self.config.get('mu', 0)
                            # loss = torch.add(loss, torch.mul(auxiliary_loss, mu))
                            loss = torch.add(torch.mul(loss, (1 - mu)), torch.mul(auxiliary_loss, mu))
                        if 'contrastive_loss' in output and output['contrastive_loss'] is not None:
                            lamda = self.config.get('lamda', 0)
                            contrastive_loss = output['contrastive_loss']
                            loss = loss.sum() + lamda * contrastive_loss
                            # loss = torch.add(torch.mul(loss, (1 - lamda)), torch.mul(contrastive_loss, lamda))
                        if "label_contrastive_loss" in output and output['label_contrastive_loss'] is not None:
                            lamda = self.config.get('lamda', 0)
                            label_contrastive_loss = output['label_contrastive_loss']
                            loss = torch.add(torch.mul(loss, (1 - lamda)), torch.mul(label_contrastive_loss, lamda))
                        if "discount_label_loss" in output and output["discount_label_loss"] is not None:
                            lamda = self.config.get('lamda', 0)
                            discount_label_loss = output['discount_label_loss']
                            loss = torch.add(torch.mul(loss, (1 - lamda)), torch.mul(discount_label_loss, lamda))
                        loss = loss.sum()

                accumulation_steps = self.config.get("accumulation_steps", 0)
                if accumulation_steps > 0:
                    loss = loss/accumulation_steps

                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                # Backward passes under autocast are not recommended.
                # Backward ops run in the same dtype autocast chose for corresponding forward ops.
                scaler.scale(loss).backward(retain_graph=True)

                # loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                # optimizer.step()

                if self.config.get("fgm", False):
                    # 对抗训练
                    fgm.attack()  # 在embedding上添加对抗扰动
                    output = model(batch=batch, labels=batch["label_ids"])
                    loss_adv = output["loss"]
                    scaler.scale(loss_adv).backward(retain_graph=True)
                    # loss_adv.backward()  # 反向传播，并在正常的grad基础上，累加对抗训练的梯度
                    fgm.restore()  # 恢复embedding参数
                    # optimizer.step()  # 梯度下降，更新参数
                    # model.zero_grad()

                if accumulation_steps > 0:
                    if (batch_num+1) % accumulation_steps == 0:
                        scaler.unscale_(optimizer) # Unscales the gradients of optimizer's assigned params in-place
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm) # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                        # optimizer's gradients are already unscaled, so scaler.step does not unscale them,
                        # although it still skips optimizer.step() if the gradients contain infs or NaNs.
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update() # Updates the scale for next iteration.
                    optimizer.zero_grad()
                # move batch to cpu again to save gpu memory
                tensor_dict_to_cpu(batch)

                if batch_num % 100 == 0:
                    self.result_writer.log(f"Loss in fold {fold_num}, epoch {epoch}, batch {batch_num}: {loss.item()}")

            train_duration = time.time() - epoch_start

            epoch_scheduler.step()

            # evaluate model
            results={}
            self.result_writer.log(f'evaluating model...')
            dev_metrics, dev_confusion, labels_dict, _ = eval_model(model, dev_batches, self.device, self.task, self.config)
            results['dev_metrics']=dev_metrics
            results['dev_confusion'] = dev_confusion
            results['labels_dict'] = labels_dict
            results['classification_report']=_


            if dev_metrics[self.task.dev_metric] > best_dev_result:
                if return_best_model:
                    best_model = copy.deepcopy(model)
                best_dev_result = dev_metrics[self.task.dev_metric]
                early_stopping_counter = 0
                self.result_writer.log(f"New best dev {self.task.dev_metric} {best_dev_result}!")
                results={}
                test_metrics, test_confusion, labels_dict, _ = eval_model(model, test_batches, self.device, self.task, self.config)
                results['dev_metrics']=dev_metrics
                results['dev_confusion'] = dev_confusion
                results['labels_dict'] = labels_dict
                results['classification_report']=_


                self.write_results(fold_num, epoch, train_duration, dev_metrics, dev_confusion, test_metrics, test_confusion)
                self.result_writer.log(
                    f'*** fold: {fold_num},  epoch: {epoch}, train duration: {train_duration}, dev {self.task.dev_metric}: {dev_metrics[self.task.dev_metric]}, test weighted-F1: {test_metrics["weighted-f1"]}, test macro-F1: {test_metrics["macro-f1"]}, test accuracy: {test_metrics["acc"]}')
            else:
                early_stopping_counter += 1
                self.result_writer.log(f'fold: {fold_num}, epoch: {epoch}, train duration: {train_duration}, dev {self.task.dev_metric}: {dev_metrics[self.task.dev_metric]}')

            epoch += 1
        return best_model



class SentenceClassificationMultitaskTrainer:
    '''Trainer for multitask model.
       Has only small differences to  SentenceClassificationTrainer
        (i.e. no early stopping, two devices to separate models on several gpus)
    '''
    def __init__(self, device, config, tasks, result_writer, device2=None):
        self.device = device
        self.device2 = device2
        self.config = config
        self.result_writer = result_writer
        self.cur_result = dict()
        self.cur_result["tasks"] = [task.task_name for task in tasks]
        self.cur_result["config"] = config

        self.tasks = tasks

    def write_results(self, task, epoch, train_duration, dev_metrics, dev_confusion, test_metrics, test_confusion):
        self.cur_result["task"] = task.task_name
        self.cur_result["epoch"] = epoch
        self.cur_result["train_duration"] = train_duration
        self.cur_result["dev_metrics"] = dev_metrics
        self.cur_result["dev_confusion"] = dev_confusion
        self.cur_result["test_metrics"] = test_metrics
        self.cur_result["test_confusion"] = test_confusion

        self.result_writer.write(json.dumps(self.cur_result))


    def run_training(self, train_batches, dev_batches, test_batches, restart, fold_num, save_models=False, save_best_model_path=None):

        self.result_writer.log(f'device: {self.device}')

        train_batch_count = len(train_batches)
        self.result_writer.log(f"train batches: {train_batch_count}")
        self.result_writer.log(f"dev batches: {len(dev_batches)}")
        self.result_writer.log(f"test batches: {len(test_batches)}")

        # instantiate model per reflection
        model = getattr(models, self.config["model"])(self.config, self.tasks)
        self.result_writer.log("Model: " + model.__class__.__name__)
        self.cur_result["model"] = model.__class__.__name__

        if self.device2 is not None:
            model.to_device(self.device, self.device2)
        else:
            model.to(self.device)

        max_train_epochs = self.config["max_epochs"]
        lr = self.config["lr"]
        max_grad_norm = 1.0

        # for feature based training use Adam optimizer with lr decay after each epoch (see Jin et al. Paper)
        optimizer = Adam(model.parameters(), lr=lr)
        epoch_scheduler = StepLR(optimizer, step_size=1, gamma=self.config["lr_epoch_decay"])

        optimizer.zero_grad()

        best_dev_result = 0.0
        epoch = 0
        while epoch < max_train_epochs:
            epoch_start = time.time()

            self.result_writer.log(f'training model in epoch {epoch} ...')

            random.shuffle(train_batches)
            # train model
            model.train()
            for batch_num, batch in enumerate(train_batches):
                # move tensor to gpu
                tensor_dict_to_gpu(batch, self.device)

                output = model(batch=batch, labels=batch["label_ids"])
                loss = output["loss"]
                loss = loss.sum()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

                # move batch to cpu again to save gpu memory
                tensor_dict_to_cpu(batch)

                if batch_num % 100 == 0:
                    self.result_writer.log(f"Loss in epoch {epoch}, batch {batch_num}: {loss.item()}")

            train_duration = time.time() - epoch_start

            epoch_scheduler.step()

            # evaluate model
            weighted_f1_dev_scores = []
            for task in self.tasks:
                self.result_writer.log(f'evaluating model for task {task.task_name}...')
                dev_metrics, dev_confusion, labels_dict, _ = eval_model(model, dev_batches, self.device, task, self.config)
                test_metrics, test_confusion, labels_dict, _ = eval_model(model, test_batches, self.device, task, self.config)
                self.write_results(task, epoch, train_duration, dev_metrics, dev_confusion, test_metrics, test_confusion)
                self.result_writer.log(
                    f'epoch: {epoch}, train duration: {train_duration}, dev weighted f1: {dev_metrics["weighted-f1"]}, dev {task.dev_metric}: {dev_metrics[task.dev_metric]}, test weighted-F1: {test_metrics["weighted-f1"]}, test micro-F1: {test_metrics["micro-f1"]}. test macro-F1: {test_metrics["macro-f1"]}, test accuracy: {test_metrics["acc"]}')
                weighted_f1_dev_scores.append(test_metrics["weighted-f1"])
            weighted_f1_avg = np.mean(weighted_f1_dev_scores)

            if save_models:
                model_copy = copy.deepcopy(model)
                model_path = os.path.join(save_best_model_path, f'{restart}_{fold_num}_{epoch}_model.pt')
                self.result_writer.log(f"saving model to {model_path}")
                torch.save(model_copy.state_dict(), model_path)

            if weighted_f1_avg > best_dev_result:
                best_dev_result = weighted_f1_avg
                self.result_writer.log(f'*** epoch: {epoch}, mean weighted-F1 dev score: {weighted_f1_avg}')
            else:
                self.result_writer.log(f'epoch: {epoch}, mean weighted-F1 dev score: {weighted_f1_avg}')

            epoch += 1

