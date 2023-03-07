import os

import numpy as np
from sklearn.model_selection import KFold, train_test_split

from batch_creator import BatchCreator
from dataset_reader import DocumentsDataset
from utils import log

PUBMED_LABELS = ["DEFAULT", 'mask', "NONE", "PREAMBLE", "FAC", "ISSUE", "ARG_RESPONDENT", "ARG_PETITIONER", "ANALYSIS",
                 "PRE_RELIED", "PRE_NOT_RELIED", "STA", "RLC", "RPC", "RATIO"]
PUBMED_LABELS_PRES = ["DEFAULT", 'mask', "NONE", "PREAMBLE", "FAC", "ISSUE", "ARG_RESPONDENT", "ARG_PETITIONER",
                      "ANALYSIS", "PRE_RELIED", "PRE_NOT_RELIED", "STA", "RLC", "RPC", "RATIO"]


NICTA_LABELS =["NONE", "FAC", "ISSUE", "ARG", "ANALYSIS", "PRE", "STA","RLC", "RPC","RATIO"]
NICTA_LABELS_PRES =["NONE", "FAC", "ISSUE", "ARG", "ANALYSIS", "PRE", "STA","RLC", "RPC","RATIO"]

DRI_LABELS = ["NONE", "FAC", "ISSUE", "ARG", "ANALYSIS", "PRE", "STA","RLC", "RPC","RATIO"]
DRI_LABELS_PRES =["NONE", "FAC", "ISSUE", "ARG", "ANALYSIS", "PRE", "STA","RLC", "RPC","RATIO"]

ART_LABELS = ["NONE", "FAC", "ISSUE", "ARG", "ANALYSIS", "PRE", "STA","RLC", "RPC","RATIO"]
ART_LABELS_PRES = ["NONE", "FAC", "ISSUE", "ARG", "ANALYSIS", "PRE", "STA","RLC", "RPC","RATIO"]

GEN_LABELS = ["mask", "Background", "Problem", "Method", "Result", "Conclusion", "Future Work"]
GEN_LABELS = ["mask", "Background", "Problem", "Contribution", "Method", "Result", "Conclusion", "Future Work"]
GEN_LABELS_PRES = GEN_LABELS


GEN_LABELS = ["PREAMBLE", "NONE", "FAC", "ISSUE", "ARG_RESPONDENT", "ARG_PETITIONER", "ANALYSIS", "PRE_RELIED",
              "PRE_NOT_RELIED", "STA", "RLC", "RPC", "RATIO"]
GEN_LABELS = ["mask", "Background", "Problem", "Contribution", "Method", "Result", "Conclusion", "Future Work"]
GEN_LABELS_PRES = ["mask", "Background", "Problem", "Contribution", "Method", "Result", "Conclusion", "Future Work"]


LEGALEVAL_LABELS = ["mask", "NONE", "PREAMBLE", "FAC", "ISSUE", "ARG_RESPONDENT", "ARG_PETITIONER", "ANALYSIS",
                    "PRE_RELIED", "PRE_NOT_RELIED", "STA", "RLC", "RPC", "RATIO"]
LEGALEVAL_LABELS_PRES = ["mask", "NONE", "PREAMBLE", "FAC", "ISSUE", "ARG_RESPONDENT", "ARG_PETITIONER",
                        "ANALYSIS", "PRE_RELIED", "PRE_NOT_RELIED", "STA", "RLC", "RPC", "RATIO"]
LEGALEVAL_LABELS_DETAILS = ["mask", "None", "Preamble", "Facts", "Issues", "Argument by Petitioner", "Argument by Respondent",
                        "Analysis", "Precedent Relied", "Precedent Not Relied", "Statute", "Ruling by Lower Court", "Ruling by Present Court", "Ratio of the decision"]

# LEGALEVAL_LABELS = ["mask", 'PREAMBLE_B', 'PREAMBLE_I', 'PREAMBLE_E', 'ANALYSIS_B', 'NONE_O', 'FAC_B', 'FAC_I', 'FAC_E', 'RLC_B', 'RLC_I', 'RLC_E', 'ARG_PETITIONER_B', 'ARG_PETITIONER_E', 'ANALYSIS_I', 'ANALYSIS_E', 'STA_B', 'STA_E', 'PRE_RELIED_B', 'PRE_RELIED_I', 'PRE_RELIED_E', 'RPC_B', 'RPC_I', 'RPC_E', 'ISSUE_B', 'RATIO_B', 'RATIO_I', 'RATIO_E', 'STA_I', 'ISSUE_I', 'ISSUE_E', 'ARG_PETITIONER_I', 'ARG_RESPONDENT_B', 'ARG_RESPONDENT_I', 'ARG_RESPONDENT_E', 'PRE_NOT_RELIED_B', 'PRE_NOT_RELIED_I', 'PRE_NOT_RELIED_E']
# LEGALEVAL_LABELS_PRES = ["mask", 'PREAMBLE_B', 'PREAMBLE_I', 'PREAMBLE_E', 'ANALYSIS_B', 'NONE_O', 'FAC_B', 'FAC_I', 'FAC_E', 'RLC_B', 'RLC_I', 'RLC_E', 'ARG_PETITIONER_B', 'ARG_PETITIONER_E', 'ANALYSIS_I', 'ANALYSIS_E', 'STA_B', 'STA_E', 'PRE_RELIED_B', 'PRE_RELIED_I', 'PRE_RELIED_E', 'RPC_B', 'RPC_I', 'RPC_E', 'ISSUE_B', 'RATIO_B', 'RATIO_I', 'RATIO_E', 'STA_I', 'ISSUE_I', 'ISSUE_E', 'ARG_PETITIONER_I', 'ARG_RESPONDENT_B', 'ARG_RESPONDENT_I', 'ARG_RESPONDENT_E', 'PRE_NOT_RELIED_B', 'PRE_NOT_RELIED_I', 'PRE_NOT_RELIED_E']

B20197_LABLES = ["FAC", "RLC", "ARG", "STA", "PRE", "RATIO", "RPC"]
B20197_LABLES_PRES = ["FAC", "RLC", "ARG", "STA", "PRE", "RATIO", "RPC"]
LSP_AUXILIARY_LABELS = ["0", "1", "mask"]
BIOE_AUXILIARY_LABELS = ["B", "I", "O", "E", "mask"]

Y2021M8_LABLES = ["NONE", "FAC", "ISSUE", "ARG_RESPONDENT", "ARG_PETITIONER", "PRE_RELIED", "PRE_NOT_RELIED",
                  "STA", "RLC", "RPC", "RATIO", "DISSENT", "PRE_OVERRULED"]
Y2021M8_LABLES_PRES = ["NONE", "FAC", "ISSUE", "ARG_RESPONDENT", "ARG_PETITIONER", "PRE_RELIED", "PRE_NOT_RELIED",
                  "STA", "RLC", "RPC", "RATIO", "DISSENT", "PRE_OVERRULED"]

DRI_TASK = "DRI"
PUBMED_TASK = "pubmed-20k"
NICTA_TASK = "nicta_piboso"
ART_TASK = "ART"

GEN_DRI_TASK = "DRI_generic"
GEN_PMD_TASK = "PMD_generic"
GEN_NIC_TASK = "NIC_generic"
GEN_ART_TASK = "ART_generic"
LEGALEVAL_TASK = "legaleval"
Y2019B7_TASK = "Y2019B7"
Y2021M8_TASK = "Y2021M8"

def generic_task(task_name, train_batch_size, max_docs):
    return Task(task_name, GEN_LABELS,
                train_batch_size, 1, max_docs, short_name=task_name,
                labels_pres=GEN_LABELS_PRES)


def dri_task(train_batch_size, max_docs):
    # 10-fold cross validation
    return Task(DRI_TASK, DRI_LABELS,
                train_batch_size, 10, max_docs, short_name="DRI",
                labels_pres=DRI_LABELS_PRES)


def art_task(train_batch_size, max_docs):
    # 9-fold cross validation, Accuracy-Metric
    return Task(ART_TASK, ART_LABELS,
                train_batch_size, 9, max_docs,
                dev_metric="acc", short_name="ART", labels_pres=ART_LABELS_PRES)


def art_task_small(train_batch_size, max_docs):
    # 9-fold cross validation, Accuracy-Metric
    return Task(ART_TASK + "_small", ART_LABELS,
                train_batch_size, 9, max_docs,
                dev_metric="acc", portion_training_data=1.0 / 3.0,
                task_folder_name=ART_TASK, short_name="mART", labels_pres=ART_LABELS_PRES)


def pubmed_task(train_batch_size, max_docs, data_folder="datasets/"):
    return Task(PUBMED_TASK, PUBMED_LABELS,
                train_batch_size, 1, max_docs, short_name="PMD", labels_pres=PUBMED_LABELS_PRES,
                data_folder=data_folder)


def pubmed_task_small(train_batch_size, max_docs):
    return Task(PUBMED_TASK + "_small", PUBMED_LABELS,
                train_batch_size, 1, max_docs,
                portion_training_data=1.0 / 20.0, task_folder_name=PUBMED_TASK, short_name="mPMD",
                labels_pres=PUBMED_LABELS_PRES)


def nicta_task(train_batch_size, max_docs):
    return Task(NICTA_TASK, NICTA_LABELS,
                train_batch_size, 1, max_docs, short_name="NIC", labels_pres=NICTA_LABELS_PRES)


def legaleval_task(train_batch_size, max_docs, data_folder="datasets/", auxiliary_task=None):
    if auxiliary_task == "lsp":
        auxiliary_labels = LSP_AUXILIARY_LABELS
    elif auxiliary_task == "bioe":
        auxiliary_labels = BIOE_AUXILIARY_LABELS
    else:
        auxiliary_labels = None
    return Task(LEGALEVAL_TASK, LEGALEVAL_LABELS,
                train_batch_size, 1, max_docs, short_name="LEVAL", labels_pres=PUBMED_LABELS_PRES,
                data_folder=data_folder, auxiliary_labels=auxiliary_labels)


def y2019b7_task(train_batch_size, max_docs, data_folder="datasets/"):
    return Task(Y2019B7_TASK, B20197_LABLES,
                train_batch_size, 1, max_docs, short_name="B20197", labels_pres=PUBMED_LABELS_PRES,
                data_folder=data_folder)

def y2021m8_task(train_batch_size, max_docs, data_folder="datasets/"):
    return Task(Y2021M8_TASK, Y2021M8_LABLES,
                train_batch_size, 1, max_docs, short_name="Y2021M8", labels_pres=PUBMED_LABELS_PRES,
                data_folder=data_folder)


class Fold:
    def __init__(self, train, dev, test):
        self.train = train
        self.dev = dev
        self.test = test


class Task:
    def __init__(self, task_name, labels, train_batch_size, num_fols, max_docs=-1,
                 dev_metric="weighted-f1", portion_training_data=1.0,
                 task_folder_name=None, short_name=None,
                 labels_pres=None,
                 data_folder="datasets/", config=None, auxiliary_labels=None):
        self.labels_pres = labels_pres
        self.short_name = short_name
        self.task_name = task_name
        self.labels = labels
        self.data_dir = os.path.join(data_folder, task_name if task_folder_name is None else task_folder_name)
        self.train_batch_size = train_batch_size
        self.num_folds = num_fols
        self.max_docs = max_docs
        self.folds = None
        self.folds_examples = None
        self.dev_metric = dev_metric
        self.portion_training_data = portion_training_data
        self.config = config
        self.auxiliary_labels = auxiliary_labels

    def get_labels_pres_titled(self):
        '''Labels ordered in presentation-order titled. '''
        return [l.title() for l in self.labels_pres]

    def get_labels_titled(self):
        '''Labels titled. '''
        return [l.title() for l in self.labels]

    def _get_batches(self, examples):
        ds_builder = BatchCreator(
            examples,
            tokenizer=None,
            labels=self.labels,
            batch_sentence_size=self.train_batch_size,
            max_seq_length=None,
            auxiliary_labels=self.auxiliary_labels
        )
        batches = ds_builder.get_batches(task_name=self.task_name)
        return batches

    def _load_full_set(self, file_suffix='scibert'):
        '''Returns one Fold. '''
        log("Loading tokenized data...")
        full_examples = DocumentsDataset(os.path.join(self.data_dir, f"full_{file_suffix}.txt"), max_docs=self.max_docs)
        log("Loading tokenized data finished.")
        return list(full_examples)

    def _load_train_dev_test_examples(self, file_suffix='scibert') -> Fold:

        log("Loading tokenized data...")
        train_examples = DocumentsDataset(os.path.join(self.data_dir, f"train_{file_suffix}.txt"),
                                          max_docs=self.max_docs)
        dev_examples = DocumentsDataset(os.path.join(self.data_dir, f"dev_{file_suffix}.txt"), max_docs=self.max_docs)
        test_examples = DocumentsDataset(os.path.join(self.data_dir, f"test_{file_suffix}.txt"), max_docs=self.max_docs)
        train_examples = self.truncate_train_examples(train_examples)
        return [(train_examples, dev_examples, test_examples)]

    def truncate_train_examples(self, train_examples):
        if self.portion_training_data < 1.0:
            train_examples = list(train_examples)
            new_len = int(len(train_examples) * self.portion_training_data)
            log(f"Truncating training examples with factor {self.portion_training_data} from {len(train_examples)} to {new_len}")
            train_examples = train_examples[0: new_len]
        return train_examples

    def get_all_examples(self, file_suffix='scibert'):
        if self.num_folds == 1:
            train, dev, test = self._load_train_dev_test_examples(file_suffix)[0]
            all_examples = []
            all_examples += list(train)
            print(len(all_examples))
            all_examples += list(dev)
            print(len(all_examples))
            all_examples += list(test)
            print(len(all_examples))
            return all_examples
        else:
            return self._load_full_set(file_suffix)

    def get_folds_examples(self, file_suffix='scibert'):
        if self.folds_examples is not None:
            return self.folds_examples
        self.folds_examples = []
        log(f"Loading data with {self.num_folds} folds...")
        if self.num_folds == 1:
            self.folds_examples = self._load_train_dev_test_examples(file_suffix=file_suffix)
            log(f"Loading folds_examples: {self.folds_examples}")
        else:
            full_examples = np.array(self._load_full_set(file_suffix=file_suffix))
            # assert len(full_examples) % self.num_folds == 0

            self.folds = []
            kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=268)
            for train_index, test_index in kf.split(full_examples):
                train_and_dev = full_examples[train_index]
                test = full_examples[test_index]
                train, dev = train_test_split(train_and_dev, test_size=1.0 / self.num_folds, shuffle=False)

                train = self.truncate_train_examples(train)

                self.folds_examples.append((train, dev, test))

        return self.folds_examples

    def get_folds(self):
        if self.folds is not None:
            return self.folds

        folds_examples = self.get_folds_examples()

        self.folds = []
        log(f"Creating batches for {self.num_folds} folds...")
        for train, dev, test in folds_examples:
            log(f"folds_examples: {len(train)}, {len(dev)}, {len(test)}")
            train_batches = self._get_batches(train)
            dev_batches = self._get_batches(dev)
            test_batches = self._get_batches(test)


            self.folds.append(Fold(train_batches, dev_batches, test_batches))
        log(f"Creating batches finished.")
        return self.folds

    def get_stats_counts(self):
        counts = dict()
        all_examples = self.get_all_examples()
        counts["docs"] = len(all_examples)
        counts["sentences"] = 0
        for d in all_examples:
            for l in d.labels:
                counts["sentences"] += 1
                if l in counts:
                    counts[l] += 1
                else:
                    counts[l] = 1

        return counts

    def get_test_label_counts(self, fold_num):
        fold_num = fold_num % self.num_folds
        _, _, test = self.get_folds_examples()[fold_num]

        counts = [0] * len(self.labels)
        for d in test:
            for l in d.labels:
                label_id = self.labels.index(l)
                counts[label_id] += 1
        return counts
