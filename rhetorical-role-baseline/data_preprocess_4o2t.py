#! /usr/bin/env python
# -*- coding:utf-8 -*-
import json
import os
from transformers import BertTokenizer, DebertaV2Tokenizer
import pandas as pd
from termcolor import colored
BASEDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
LEGALEVAL_LABELS = ["mask", "NONE", "PREAMBLE", "FAC", "ISSUE", "ARG_RESPONDENT", "ARG_PETITIONER", "ANALYSIS",
                 "PRE_RELIED", "PRE_NOT_RELIED", "STA", "RLC", "RPC", "RATIO"]

LABLESY2019B7 = ["Facts", "Ruling by Lower Court", "Argument", "Statute", "Precedent", "Ratio of the decision", "Ruling by Present Court"]
LABLESY2019B7_MAPPING = ["FAC", "RLC", "ARG", "STA", "PRE", "RATIO", "RPC"]

LABLESY2021M8 = ["None", "Fact", "Issue", "ArgumentRespondent", "ArgumentPetitioner", "PrecedentReliedUpon", "PrecedentNotReliedUpon", "Statute", "RulingByLowerCourt", "RulingByPresentCourt", "RatioOfTheDecision", "Dissent", "PrecedentOverruled"]
LABLESY2021M8_MAPPING = ["NONE", "FAC", "ISSUE", "ARG_RESPONDENT", "ARG_PETITIONER", "PRE_RELIED", "PRE_NOT_RELIED", "STA", "RLC", "RPC", "RATIO", "DISSENT", "PRE_OVERRULED"]

REMOVELABELS = ["NONE", "ISSUE"]

MAX_SEQ_LENGTH = 256


def write_in_hsln_format_4_y2019b7(data_files, data_dir, tokenizer, auxiliary_task=None):
    train_final_string = ''
    dev_final_string = ''
    final_string = ''
    # dev_ids = list(range(1, len(data_files), 10))
    dev_ids = []
    for i, file in enumerate(data_files):
        if i in dev_ids:
            final_string = dev_final_string
        else:
            final_string = train_final_string
        file_name = os.path.split(file)[1].split('.')[0]
        previous_label = "None"
        final_string = final_string + '###' + str(file_name) + "\n"

        with open(file, 'r') as f:
            labels = []
            sentences = []
            for line in f:
                sentence_txt, annotation = line.strip().split('\t')[0], line.strip().split('\t')[1]
                sentences.append(sentence_txt)
                labels.append(annotation)
            for ani, (annotation, sentence_txt) in enumerate(zip(labels, sentences)):
                if ani < len(labels) - 1:
                    next_label = labels[ani + 1]
                else:
                    next_label = "mask"
                print(colored(annotation, 'green'), next_label, sentence_txt)
                sentence_txt = sentence_txt.strip().replace("\n", "").replace("\r", "")
                if sentence_txt != "":
                    sent_tokens = tokenizer.encode(sentence_txt, add_special_tokens=True,
                                                   max_length=MAX_SEQ_LENGTH,
                                                   truncation=True)
                    sent_tokens = [str(i) for i in sent_tokens]
                    sent_tokens_txt = " ".join(sent_tokens)
                    if auxiliary_task == 'lsp':
                        if previous_label == annotation:
                            ls = "0"
                        else:
                            ls = "1"
                    if auxiliary_task == 'bioe':
                        if annotation != next_label:
                            ls = "E"
                        if annotation != previous_label:
                            ls = "B"
                        if annotation == previous_label and annotation == next_label:
                            ls = "I"
                        if annotation == "NONE":
                            ls = "O"
                    if LABLESY2019B7_MAPPING[LABLESY2019B7.index(annotation)] in LEGALEVAL_LABELS and LABLESY2019B7_MAPPING[LABLESY2019B7.index(annotation)] not in REMOVELABELS:
                        if auxiliary_task:
                            final_string = final_string + LABLESY2019B7_MAPPING[LABLESY2019B7.index(annotation)] + "\t" + ls + "\t" + sent_tokens_txt + "\n"
                        else:
                            final_string = final_string + LABLESY2019B7_MAPPING[LABLESY2019B7.index(annotation)] + "\t" + sent_tokens_txt + "\n"
                    previous_label = annotation
            final_string = final_string + "\n"
        if i in dev_ids:
            dev_final_string = final_string
        else:
            train_final_string = final_string

    with open(data_dir + '/train_scibert.txt', "w+") as file:
        file.write(train_final_string)
    with open(data_dir + '/dev_scibert.txt', "w+") as file:
        file.write(dev_final_string)
    with open(data_dir + '/test_scibert.txt', "w+") as file:
        file.write(dev_final_string)
    return final_string


def write_in_hsln_format_4_y2021m8(data_dir, tokenizer, auxiliary_task=None):
    names = ['CL', 'IT']
    final_string = ''

    for name in names:
        train_file_name = f'{name}_train.json' if name else 'train.json'
        dev_file_name = f'{name}_dev.json' if name else 'dev.json'
        test_file_name = f'{name}_dev.json' if name else 'test.json'
        print(os.path.join(data_dir, name, train_file_name))
        print(os.path.join(data_dir, name, dev_file_name))
        train_file = json.load(open(os.path.join(data_dir, name, train_file_name), 'r'))
        dev_file = json.load(open(os.path.join(data_dir, name, dev_file_name), 'r'))
        test_file = json.load(open(os.path.join(data_dir, name, test_file_name), 'r'))
        print(len(train_file))
        print(len(dev_file))
        print(len(test_file))
        for file in [train_file, dev_file, test_file]:
            for file_name, v in file.items():
                previous_label = "None"
                final_string = final_string + '###' + str(file_name) + "\n"
                sentences = v['sentences'][2:-2].split("', '") if not isinstance(v['sentences'], list) else v['sentences']
                labels = v['complete']
                assert len(sentences) == len(labels)
                for ani, (annotation, sentence_txt) in enumerate(zip(labels, sentences)):
                    if ani < len(sentences) - 1:
                        next_label = labels[ani+1]
                    else:
                        next_label = "mask"
                    # print(colored(annotation, 'green'), next_label, sentence_txt)
                    sentence_txt = sentence_txt.strip().replace("\n", "").replace("\r", "")
                    if sentence_txt != "":
                        sent_tokens = tokenizer.encode(sentence_txt, add_special_tokens=True,
                                                       max_length=MAX_SEQ_LENGTH,
                                                       truncation=True)
                        sent_tokens = [str(i) for i in sent_tokens]
                        sent_tokens_txt = " ".join(sent_tokens)
                        if auxiliary_task == 'lsp':
                            if previous_label == annotation:
                                ls = "0"
                            else:
                                ls = "1"
                        if auxiliary_task == 'bioe':
                            if annotation != next_label:
                                ls = "E"
                            if annotation != previous_label:
                                ls = "B"
                            if annotation == previous_label and annotation == next_label:
                                ls = "I"
                            if annotation == "None":
                                ls = "O"
                        if annotation not in ['ArgumentRespondent', 'ArgumentPetitioner'] \
                                and LABLESY2021M8_MAPPING[LABLESY2021M8.index(annotation)] in LEGALEVAL_LABELS and LABLESY2021M8_MAPPING[LABLESY2021M8.index(annotation)] not in REMOVELABELS:
                            if auxiliary_task:
                                final_string = final_string + LABLESY2021M8_MAPPING[LABLESY2021M8.index(annotation)] + "\t" + ls + "\t" + sent_tokens_txt + "\n"
                            else:
                                final_string = final_string + LABLESY2021M8_MAPPING[LABLESY2021M8.index(annotation)] + "\t" + sent_tokens_txt + "\n"
                        previous_label = annotation
                final_string = final_string + "\n"
    with open(os.path.join(data_dir, f"train_scibert.txt"), "w+") as file:
        file.write(final_string)

    return final_string


if __name__ == '__main__':
    print('BASEDIR: ', BASEDIR)
    task = 'Y2021M8'
    # BERT_VOCAB = os.path.join(BASEDIR, "zlucia-legalbert")
    # tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)
    # BERT_VOCAB = os.path.join(BASEDIR, "deberta-v3-base")
    BERT_VOCAB = os.path.join(BASEDIR, "rhetorical-role-baseline/results/bert_continue_bz64/checkpoint-48000")
    tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)      

    if task == 'Y2019B7':
        data_dir = f"{BASEDIR}/rhetorical-role-baseline/datasets/Y2019B7/text"
        import os
        import glob
        data_files = sorted(glob.glob(os.path.join(data_dir, '*.txt')))
        print('number of data_files: ', len(data_files))
        write_in_hsln_format_4_y2019b7(data_files, os.path.dirname(data_dir), tokenizer, auxiliary_task='bioe')
        # write_in_hsln_format_4_y2019b7(data_files, os.path.dirname(data_dir), tokenizer)

    if task == 'Y2021M8':
        data_dir = f"{BASEDIR}/rhetorical-role-baseline/datasets/Y2021M8/data_annotated"
        write_in_hsln_format_4_y2021m8(data_dir, tokenizer, auxiliary_task='bioe')
        # write_in_hsln_format_4_y2021m8(data_dir, tokenizer)

