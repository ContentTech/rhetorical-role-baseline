import os
import sys
from typing import Any
from transformers import BertTokenizer, BertModel
import torch
from torch import nn
from torch import optim
import numpy as np
import json
import os
import copy
import torch
import logging
from torch.cuda.amp import autocast as ac
# from torch.utils.data import DataLoader, RandomSampler
from transformers import AdamW, get_linear_schedule_with_warmup

from torch.utils.data import DataLoader, Dataset
from data_utils import convert_judgement_data_to_json, yield_data, MyDataset, get_tokenizer, convert_preamble_data_to_json
from model import CRFModel,crf_decode

import logging
logger = logging.getLogger(__name__)


span_id_dir = "../data/label2id/judgement/span_ent2id.json"
span_dict = json.loads(open(span_id_dir).read())

crf_id_dir = "../data/label2id/judgement/crf_ent2id.json"
crf_dict = json.loads(open(crf_id_dir).read())

crf_id_dir_preamble = "../data/label2id/preamble/crf_ent2id.json"
crf_dict_preamble = json.loads(open(crf_id_dir_preamble).read())


span_id_dir_preamble = "../data/label2id/preamble/span_ent2id.json"
span_dict_preamble = json.loads(open(span_id_dir_preamble).read())


def get_preamble_label2name():
    label2name = {}
    for k,v in crf_dict_preamble.items():
        label2name[v] = k 
    return label2name
preamble_label2name = get_preamble_label2name()

def get_label2name():
    label2name = {}
    for k,v in crf_dict.items():
        label2name[v] = k 
    return label2name
label2name = get_label2name()

tokenizer = get_tokenizer()

BERT_DIR = os.path.join("/ossfs/workspace/legal_rr/ptm/checkpoint-48000/")

batch_size = 64
max_sen_length = 512

device = torch.device('cuda')
train_base_dir = '../data/'
dev_base_dir =  '../data/NER_DEV/'

# judgement_train_dir = os.path.join(train_base_dir, 'NER_TRAIN_JUDGEMENT.json')
# judgement_dev_dir = os.path.join(dev_base_dir, 'NER_DEV_JUDGEMENT.json')
# train_examples = convert_judgement_data_to_json(max_sen_length, judgement_train_dir, crf_dict, span_dict, "")

preamble_train_dir = os.path.join(train_base_dir, 'NER_TRAIN_PREAMBLE.json')
preamble_dev_dir = os.path.join(dev_base_dir, 'NER_DEV_PREAMBLE.json')
train_examples = convert_preamble_data_to_json(max_sen_length, preamble_train_dir, crf_dict_preamble, span_dict_preamble, "")
    
train_datas = yield_data(train_examples, batch_size)
train_data_length = len(train_examples)
print ("train example number", train_data_length)

dev_examples = convert_preamble_data_to_json(max_sen_length, preamble_dev_dir, crf_dict_preamble, span_dict_preamble, "")
# dev_examples = convert_judgement_data_to_json(max_sen_length, judgement_dev_dir, crf_dict, span_dict, "")
test_datas = yield_data(dev_examples, batch_size)
test_data_length = len(dev_examples)
print ("dev example number", test_data_length)


def build_optimizer_and_scheduler(model, t_total):
    opt = {"weight_decay":0.01, "other_lr":2e-3,"lr":2e-5,"dropout_prob":0.1,"adam_epsilon":1e-8,"warmup_proportion":0.1}
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        space = name.split('.')
        if space[0] == 'bert_module':
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": opt["weight_decay"], 'lr': opt["lr"]},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt["lr"]},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": opt["weight_decay"], 'lr': opt["other_lr"]},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': opt["other_lr"]},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=opt["lr"], eps=opt["adam_epsilon"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(opt["warmup_proportion"] * t_total), num_training_steps=t_total
    )

    return optimizer, scheduler

    

def train():
    train_epochs = 50 
    train_batch_size = batch_size
    max_grad_norm = 1.0
    num_tags = len(label2name)

    scaler = None

    model  = CRFModel(BERT_DIR, num_tags).to(device)

    use_n_gpus = False
    # if hasattr(model, "module"):
    #     use_n_gpus = True

    t_total = train_data_length * train_epochs

    optimizer, scheduler = build_optimizer_and_scheduler(model, t_total)

    # Train
    logger.info("***** Running training *****")
    logger.info(f"  Num Examples = {train_data_length}")
    logger.info(f"  Num Epochs = {train_epochs}")
    logger.info(f"  Total training batch size = {train_batch_size}")
    logger.info(f"  Total optimization steps = {t_total}")

    global_step = 0

    model.zero_grad()

    fgm, pgd = None, None

    # attack_train_mode = opt.attack_train.lower()
    # if attack_train_mode == 'fgm':
    #     fgm = FGM(model=model)
    # elif attack_train_mode == 'pgd':
    #     pgd = PGD(model=model)

    # pgd_k = 3

    save_steps = t_total // train_epochs
    eval_steps = save_steps

    logger.info(f'Save model in {save_steps} steps; Eval model in {eval_steps} steps')

    log_loss_steps = 20

    avg_loss = 0.

    f1 = 0
    for epoch in range(train_epochs):
        print ("epoch:", epoch)
        for step, batch_data in enumerate(train_datas):
            model.train()

            # for key in batch_data.keys():
            #     batch_data[key] = batch_data[key].to(device)

            # if opt.use_fp16:
            #     with ac():
            #         loss = model(**batch_data)[0]
            # else:
            
            input_ids = batch_data["input_ids"].to(device)
            labels = batch_data["labels"].to(device)
            input_mask = batch_data["input_mask"].to(device)
            token_type_ids = batch_data["token_type_ids"].to(device)
            # print ("batch", len(input_ids))
            

            
            loss = model(input_ids, input_mask, token_type_ids, labels)[0]
            if step % 10==0:
                print ("step, loss", step, loss)

            if use_n_gpus:
                loss = loss.mean()

            # if opt.use_fp16:
            #     scaler.scale(loss).backward()
            # else:
            loss.backward()

            # if fgm is not None:
            #     fgm.attack()

            #     if opt.use_fp16:
            #         with ac():
            #             loss_adv = model(**batch_data)[0]
            #     else:
            #         loss_adv = model(**batch_data)[0]

            #     if use_n_gpus:
            #         loss_adv = loss_adv.mean()

            #     if opt.use_fp16:
            #         scaler.scale(loss_adv).backward()
            #     else:
            #         loss_adv.backward()

            #     fgm.restore()

            # elif pgd is not None:
            #     pgd.backup_grad()

            #     for _t in range(pgd_k):
            #         pgd.attack(is_first_attack=(_t == 0))

            #         if _t != pgd_k - 1:
            #             model.zero_grad()
            #         else:
            #             pgd.restore_grad()

            #         if opt.use_fp16:
            #             with ac():
            #                 loss_adv = model(**batch_data)[0]
            #         else:
            #             loss_adv = model(**batch_data)[0]

            #         if use_n_gpus:
            #             loss_adv = loss_adv.mean()

            #         if opt.use_fp16:
            #             scaler.scale(loss_adv).backward()
            #         else:
            #             loss_adv.backward()

            #     pgd.restore()

            # if opt.use_fp16:
            #     scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # optimizer.step()
            # if opt.use_fp16:
            #     scaler.step(optimizer)
            #     scaler.update()
            # else:
            optimizer.step()

            scheduler.step()
            model.zero_grad()

            global_step += 1

            if global_step % log_loss_steps == 0:
                avg_loss /= log_loss_steps
                logger.info('Step: %d / %d ----> total loss: %.5f' % (global_step, t_total, avg_loss))
                avg_loss = 0.
            else:
                avg_loss += loss.item()
            
            # if global_step % save_steps == 0:
            #     save_model(opt, model, global_step)
        with torch.no_grad():
            true_num = 0
            pred_num = 0
            correct = 0
            for idy, batch in enumerate(test_datas):
                
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                input_mask = batch["input_mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                # token_segs = batch_data["token_segs"]
                # entities = batch_data["entities"]
                # print ("test batch", len(input_ids))
                
                
            
                logits = model(input_ids, input_mask, token_type_ids)
                tokens_out, emissions = logits # tokens, batch, seq

                for i in range(len(input_ids)):
                    y_true = labels[i].tolist() # seq_len
                    y_pred = tokens_out[i]
                    size = len(y_pred)
                    for k in range(size):
                        if y_true[k] >0:
                            true_num += 1 
                        if y_pred[k]>0:
                            pred_num += 1 
                        if y_true[k] == y_pred[k] and y_true[k]>0:
                            correct += 1 
                    if f1 >= 0.85 and idy==0:
                        token_segs = tokenizer.convert_ids_to_tokens(input_ids[i].tolist())[1:size-1]
                        print ("tokens", token_segs)
                        entities = crf_decode(y_true[:size], token_segs, label2name)
                        print ("true entity:", entities)
                        pre_entities =  crf_decode(tokens_out[i], token_segs, label2name)
                        print ("pred entity:", pre_entities)        
                    
                # y_true = torch.masked_select(labels,input_mask.byte())
                # y_pred = tokens_out.view(size=(-1,))
                # print("ytrue", y_true.shape, y_pred.shape)
                # # y_true = torch.multiply(labels, input_mask).

                # print ("label 0", labels[0])
                # print ("label", labels)
                
                # print ("tokens_out 0", tokens_out[0])
                # print ("tokens_out", tokens_out)
            recall=correct/(true_num+1e-8)
            precision=correct/(pred_num+1e-8)
            f1=2*recall*precision/(recall+precision+1e-8)
            print ("recall, precision, f1=", recall, precision, f1 )
            
                

    # clear cuda cache to avoid OOM
    torch.cuda.empty_cache()
    logger.info('Train done')



if __name__=='__main__':
    train()
