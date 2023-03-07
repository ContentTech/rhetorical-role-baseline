import time
import gc
from datetime import datetime
from os import makedirs
import os
import torch
from eval_run import eval_and_save_metrics
from utils import get_device, ResultWriter, log
from task import legaleval_task, pubmed_task
from train import SentenceClassificationTrainer
from models import BertHSLN, BertHSLNWithAuxiliaryTask
from task import Task
from task import LEGALEVAL_TASK, LSP_AUXILIARY_LABELS, BIOE_AUXILIARY_LABELS, BIOE_AUXILIARY_LABELS, LEGALEVAL_LABELS, LEGALEVAL_LABELS_PRES, LEGALEVAL_LABELS_DETAILS, LEGALEVAL_TASK

BASEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# BERT_MODEL = os.path.join(BASEDIR, "bert-base-uncased")
# BERT_MODEL = os.path.join(BASEDIR, "scibert_scivocab_uncased")
# BERT_MODEL = os.path.join(BASEDIR, "zlucia-legalbert")
# BERT_MODEL = os.path.join(BASEDIR, "rhetorical-role-baseline/results/bert_continue_ts_ildc/checkpoint-100000") 
# BERT_MODEL = os.path.join(BASEDIR, "rhetorical-role-baseline/results/zlucia_legalbert_baseline_wo_context_D0/checkpoint-1300") 
# BERT_MODEL = os.path.join(BASEDIR, "rhetorical-role-baseline/results/bert_continue_bz64_baseline_wo_context_D0/checkpoint-900") 
# BERT_MODEL = os.path.join(BASEDIR, "rhetorical-role-baseline/results/bert_continue_bz50/checkpoint-54000")
# BERT_MODEL = os.path.join(BASEDIR, "rhetorical-role-baseline/results/bert_continue_bz64_1121/checkpoint-26000")
# BERT_MODEL = os.path.join(BASEDIR, "rhetorical-role-baseline/results/bert_baseline_wo_context/checkpoint-2000")
# BERT_MODEL = os.path.join(BASEDIR, "rhetorical-role-baseline/results/legalbert_baseline_wo_context/checkpoint-3000")
# BERT_MODEL = os.path.join(BASEDIR, "deberta-v3-base")
# BERT_MODEL = os.path.join(BASEDIR, "rhetorical-role-baseline/results/deberta_continue_only_tasks_data/checkpoint-140000")
# BERT_MODEL = os.path.join(BASEDIR, "rhetorical-role-baseline/results/deberta_continue/checkpoint-1103000")
# BERT_MODEL = os.path.join(BASEDIR, "rhetorical-role-baseline/results/bert_continue_20w/checkpoint-48000")
# BERT_MODEL = os.path.join(BASEDIR, "rhetorical-role-baseline/results/bert_continue_ts_20w/checkpoint-82000")
# BERT_MODEL = os.path.join(BASEDIR, "/mnt/fengyao.hjj/rhetorical-role-baseline/results/bert_continue_ts_40w/checkpoint-131000")
# BERT_MODEL = os.path.join(BASEDIR, "pretrained_models/nlpaueb/legal-bert-base-uncased")
BERT_MODEL = os.path.join(BASEDIR, "rhetorical-role-baseline/results/bert_continue_bz64/checkpoint-48000") 


config = {
    "bert_model": BERT_MODEL,
    "bert_trainable": False,
    "model": BertHSLNWithAuxiliaryTask.__name__,
    "cacheable_tasks": [],

    "dropout": 0.1,
    "word_lstm_hs": 768,
    "att_pooling_dim_ctx": 200,
    "att_pooling_num_ctx": 15,

    "lr": 5e-04,
    "lr_epoch_decay": 0.9,

    # "lr": 1e-3,
    # "warmup": 1000,
    # "max_iters": 20000,

    "batch_size":  1,
    "max_seq_length": 256,
    "max_epochs": 70,
    "early_stopping": 10,
    "accumulation_steps": 2,
    "dtype": "float16",
    # "word_lstm_layer": 2,
    # "output_layer": "simple",
    # "auxiliary_task": 'bioe',
    # "mu": 0.5,
    # "supervised_contrastive_loss": True,
    # "lamda": 0.4,
    # "con_dropout": 0.3,
    # "temperature": 1,
    # "data_augmentation_strategy": "shuffle",
    # "use_multi_loss": 3,
    # "fgm": True,

    # "span_model": False,
    # "decoder": "point_model",
    # "lamda": 0.4,
    # "label_contrastive_loss": True,
    # "num_tags": 14,
    # "discount_label_loss": True,
    # "PGD": True,
    # "PGD_K": 3,
    # "span_task":True,
  	# "span_labels"" True,
    "data_folder": "datasets/legaleval",
    "task_folder_name": ""
    # "labels": ["mask", "RLC", "ANALYSIS", "FAC", "OTHERS"]
}

MAX_DOCS = -1


def c_task(data_folder, task_folder_name, labels, labels_pres):
    def legaleval_task(train_batch_size, max_docs, data_folder=data_folder, task_folder_name=task_folder_name, auxiliary_task=None):
        if auxiliary_task == "lsp":
            auxiliary_labels = LSP_AUXILIARY_LABELS
        elif auxiliary_task == "bioe":
            auxiliary_labels = BIOE_AUXILIARY_LABELS
        else:
            auxiliary_labels = None
        return Task(LEGALEVAL_TASK, labels,
                    train_batch_size, 1, max_docs, short_name="LEVAL", labels_pres=labels_pres,
                    data_folder=data_folder, task_folder_name=task_folder_name, auxiliary_labels=auxiliary_labels)

    def create_task(create_func):
        return create_func(train_batch_size=config["batch_size"], max_docs=MAX_DOCS, auxiliary_task=config.get('auxiliary_task', None))

    task = create_task(legaleval_task)
    return task

data_folder = config.get("data_folder", "datasets/")
task_folder_name = config.get("task_folder_name", None)
labels = config.get("labels", LEGALEVAL_LABELS)
labels_pres = config.get("labels_pres", labels)
task = c_task(data_folder, task_folder_name, labels, labels_pres)

save_best_models = True
device = get_device(0)
timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
run = f"{timestamp}_{task.task_name}_{config['model']}"

# -------------------------------------------

os.makedirs("results/complete_epoch_wise_new", exist_ok=True)
run_results = f'results/{run}'
makedirs(run_results, exist_ok=False)
task.get_folds()
restarts = 1 if task.num_folds == 1 else 1
for restart in range(restarts):
    for fold_num, fold in enumerate(task.get_folds()):
        start = time.time()
        result_writer = ResultWriter(f"{run_results}/{restart}_{fold_num}_results.jsonl")
        result_writer.log(f"Fold {fold_num} of {task.num_folds}")
        result_writer.log(f"Starting training {restart} for fold {fold_num}... ")

        trainer = SentenceClassificationTrainer(device, config, task, result_writer)
        best_model = trainer.run_training_for_fold(fold_num, fold, return_best_model=save_best_models)
        if best_model is not None:
            model_path = os.path.join(run_results, f'{restart}_{fold_num}_model.pt')
            result_writer.log(f"saving best model to {model_path}")
            torch.save(best_model.state_dict(), model_path)

        result_writer.log(f"finished training {restart} for fold {fold_num}: {time.time() - start}")

        # explicitly call garbage collector so that CUDA memory is released
        gc.collect()

log("Training finished.")
log("Calculating metrics...")
eval_and_save_metrics(run_results)
log("Calculating metrics finished")