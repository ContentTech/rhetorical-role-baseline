#! /usr/bin/env python
# -*- coding:utf-8 -*-
import json
from termcolor import colored
import pandas as pd
import argparse
from tabulate import tabulate
# 显示所有列
# pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
# pd.set_option('max_colwidth', 100)

LEGALEVAL_LABELS = ["mask", "NONE", "PREAMBLE", "FAC", "ISSUE", "ARG_RESPONDENT", "ARG_PETITIONER",
                        "ANALYSIS", "PRE_RELIED", "PRE_NOT_RELIED", "STA", "RLC", "RPC", "RATIO"]
LEGALEVAL_LABELS_DETAILS = ["mask", "None", "Preamble", "Facts", "Issues", "Argument by Petitioner", "Argument by Respondent",
                        "Analysis", "Precedent Relied", "Precedent Not Relied", "Statute", "Ratio of the decision", "Ruling by Present Court", "Ratio"]


def cmd():
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('-l', '--label', type=str, default=None, help='dev.csv')
    parser.add_argument('-p', '--pred', type=str, default=None, help='prediction.txt')
    args = parser.parse_args()
    print("argparse.args=", args, type(args))
    d = args.__dict__
    for key, value in d.items():
        print('%s = %s' % (key, value))
    return args

import sklearn.metrics as metrics
def calc_metric(df):
    """
    :param df:
    :param threshold
    :return:
    """
    print("-" * 100)
    pred, label = list(df['pred']), list(df['label'])
    macro_f1 = round(metrics.f1_score(y_pred=pred, y_true=label, average="macro"), 4)
    micro_f1 = round(metrics.f1_score(y_pred=pred, y_true=label, average="micro"), 4)
    weighted_f1 = round(metrics.f1_score(y_pred=pred, y_true=label, average="weighted"), 4)
    accuracy_score = round(metrics.accuracy_score(y_pred=pred, y_true=label), 4)
    print("weighted_f1: ", weighted_f1)
    print("accuracy_score: ", accuracy_score)
    print("macro_f1: ", macro_f1)
    print("micro_f1: ", micro_f1)
    print("-"*100)



def confusion(y_true, y_pred):
    from sklearn.metrics import classification_report
    from sklearn.metrics import accuracy_score, average_precision_score, precision_score, f1_score, recall_score
    print('------Macro------')
    print('Macro precision', round(precision_score(y_true, y_pred, average='macro'), 4))
    print('Macro recall', round(recall_score(y_true, y_pred, average='macro'), 4))
    print('Macro f1-score', round(f1_score(y_true, y_pred, average='macro'), 4))
    print('------Micro------')
    print('Micro precision', round(precision_score(y_true, y_pred, average='micro'), 4))
    print('Micro recall', round(recall_score(y_true, y_pred, average='micro'), 4))
    print('Micro f1-score', round(f1_score(y_true, y_pred, average='micro'), 4))
    print('------Weighted------')
    print('Weighted precision', round(precision_score(y_true, y_pred, average='weighted'), 4))
    print('Weighted recall', round(recall_score(y_true, y_pred, average='weighted'), 4))
    print('Weighted f1-score', round(f1_score(y_true, y_pred, average='weighted'), 4))
    t = classification_report(y_true, y_pred)
    print(t)


if __name__ == "__main__":
    """
    python /mnt/fengyao.hjj/rhetorical-role-baseline/experiments/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval_20221208/test_enc_dec_chunk3.csv \
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results/t5_chunk3_3e-5/checkpoint-12400/generated_predictions.txt

    python /mnt/fengyao.hjj/rhetorical-role-baseline/experiments/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval_20221208/test_enc_dec_chunk5.csv \
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results/t5_chunk3_3e-5/checkpoint-12400/generated_predictions.txt

    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results/t5_chunk5_1e-5/generated_predictions.txt
    
    python /mnt/fengyao.hjj/rhetorical-role-baseline/experiments/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/test_enc_dec_chunk5.csv \
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results/t5_chunk5_3e-4/generated_predictions.txt

    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results/t5_chunk3_3e-5/checkpoint-7100/generated_predictions.txt
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results/t5_chunk5_3e-5/generated_predictions.txt

    python /mnt/fengyao.hjj/rhetorical-role-baseline/experiments/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/test_enc_dec_chunk9.csv \
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results/t5_chunk9_3e-5/generated_predictions.txt

    python /mnt/fengyao.hjj/rhetorical-role-baseline/experiments/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/test_enc_dec_chunk9.csv \
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results/t5_chunk9_3e-4/generated_predictions.txt

    python /mnt/fengyao.hjj/rhetorical-role-baseline/experiments/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/test_enc_dec_window1.csv \
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results/t5_window1_3e-5/generated_predictions.txt

    python /mnt/fengyao.hjj/rhetorical-role-baseline/experiments/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/test_enc_dec_window5.csv \
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results/t5_window5_3e-5/checkpoint-6700/generated_predictions.txt
    
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results/t5_window9_3e-5/checkpoint-6000/generated_predictions.txt

    python /mnt/fengyao.hjj/rhetorical-role-baseline/experiments/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/test_enc_dec_window9.csv \
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results/t5_window9_3e-5/checkpoint-6000/generated_predictions.txt


    python /mnt/fengyao.hjj/rhetorical-role-baseline/experiments/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/test_enc_dec_window1.csv \
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results_enc_dec/t5_window1_3e-5/generated_predictions.txt


    python /mnt/fengyao.hjj/rhetorical-role-baseline/experiments/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/test_enc_dec_window1.csv \
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results_enc_dec/t5_window1_3e-5/generated_predictions.txt

    python /mnt/fengyao.hjj/rhetorical-role-baseline/experiments/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/test_enc_dec_window9.csv \
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results_enc_dec/t5_window9_3e-5/generated_predictions.txt

       python /mnt/fengyao.hjj/rhetorical-role-baseline/experiments/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/test_enc_dec_window5.csv \
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results_enc_dec/t5_window5_3e-5/generated_predictions.txt


    python /mnt/fengyao.hjj/rhetorical-role-baseline/experiments/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval_backtranslation/test_enc_dec_window1.csv \
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results_enc_dec/t5_legaleval_backtranslation_window1_3e-5/checkpoint-12200/generated_predictions.txt
    
    python /mnt/fengyao.hjj/rhetorical-role-baseline/experiments/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval_backtranslation/test_enc_dec_window5.csv \
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results_enc_dec/t5_legaleval_backtranslation_window5_3e-5/checkpoint-7900/generated_predictions.txt

    python /mnt/fengyao.hjj/rhetorical-role-baseline/experiments/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval_backtranslation/test_enc_dec_window9.csv \
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results_enc_dec/t5_legaleval_backtranslation_window9_3e-5/checkpoint-4600/generated_predictions.txt



    python /mnt/fengyao.hjj/rhetorical-role-baseline/experiments/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/all/test_enc_dec_window1.csv \
    -p/mnt/fengyao.hjj/rhetorical-role-baseline/results_enc_dec/t5_all_window1_3e-5/generated_predictions.txt


    python /mnt/fengyao.hjj/rhetorical-role-baseline/experiments/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/all/test_enc_dec_window5.csv \
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results_enc_dec/t5_all_window5_3e-5/generated_predictions.txt
    
    python /mnt/fengyao.hjj/rhetorical-role-baseline/rhetorical-role-baseline/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/enc_dec/D4/test_enc_dec_chunk3.csv \
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results/2_result_enc_dec_1221/t5_d4_chunk3/generated_predictions.txt

    python /mnt/fengyao.hjj/rhetorical-role-baseline/rhetorical-role-baseline/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/enc_dec/D4/test_enc_dec_chunk5.csv \
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results/2_result_enc_dec_1221/t5_d4_chunk5/generated_predictions.txt

    python /mnt/fengyao.hjj/rhetorical-role-baseline/rhetorical-role-baseline/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/enc_dec/D4/test_enc_dec_chunk9.csv \
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results/2_result_enc_dec_1221/t5_d4_chunk9_labels_abbr/generated_predictions.txt

    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results/2_result_enc_dec_1221/t5_d4_chunk9/checkpoint-29100/generated_predictions.txt
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results/2_result_enc_dec_1221/t5_d4_chunk9/generated_predictions.txt
    
    python /mnt/fengyao.hjj/rhetorical-role-baseline/rhetorical-role-baseline/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/enc_dec/D4/test_enc_dec_chunk1.csv \
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results/2_result_enc_dec_1221/t5_d4_chunk1/generated_predictions.txt
   
    python /mnt/fengyao.hjj/rhetorical-role-baseline/rhetorical-role-baseline/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/enc_dec/D4/test_enc_dec_chunk13.csv \
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results/2_result_enc_dec_1221/t5_d4_chunk13_accs16/checkpoint-20500/generated_predictions.txt
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results/2_result_enc_dec_1221/t5_d4_chunk13/checkpoint-5500/generated_predictions.txt
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/results/2_result_enc_dec_1221/t5_d4_chunk13_accs16/generated_predictions.txt

    python /mnt/fengyao.hjj/rhetorical-role-baseline/rhetorical-role-baseline/show_result_enc_dec.py \
    -l /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/enc_dec/D3/test_enc_dec_chunk3.csv \
    -p /mnt/fengyao.hjj/rhetorical-role-baseline/2_result_enc_dec_1221/t5_d3_chunk3_3e-5/generated_predictions.txt
    


    """
    args = cmd()
    label_file = args.label
    prediction_file = args.pred
    if label_file and label_file.split('.')[-1] == 'csv':
 
        df = pd.read_csv(label_file, encoding='utf-8')
        # df['label'] = df['labels_abbr']
        # df['pred'] = [LEGALEVAL_LABELS[LEGALEVAL_LABELS_DETAILS.index(l.strip())] for l in open(prediction_file, 'r').readlines()]
        pred = [l.strip() for l in open(prediction_file, 'r').readlines()]
        label = df['label']
        sent_id = df['sent_id']
        doc_id = df['doc_id']
        doc_id_unique = df['doc_id'].unique()
        print('doc_id_unique: ', doc_id_unique)
        preds = []
        labels = []
        if len(pred[0].split()) > 1:
            prev_di = ''
            next_di = doc_id_unique[1]
            j = 0
            for p, l, si, di in zip(pred, label, sent_id, doc_id):
                pr = [' '.join(i.split()[:-1]) for i in p.split(':=')[1:-1]] + [p.split(':=')[-1]]
                la = [' '.join(i.split()[:-1]) for i in l.split(':=')[1:-1]] + [l.split(':=')[-1]]
                # print('pr: ', pr)
                # print('la: ', la)
                assert len(pr) == len(la)
                preds += pr
                labels += la

                # preds.append(pr[-1])
                # labels.append(la[-1])
                # print(pr[-1], '\t', la[-1])
                # if di != prev_di:
                #     preds += pr
                #     labels += la
                #     prev_di = doc_id_unique[j] 
                #     next_di = doc_id_unique[j+1] if j < len(doc_id_unique) - 1 else ''
                #     j += 1
                # elif di != next_di:
                #     preds += pr
                #     labels += la
                # else:
                #     preds.append(pr[int(len(pr)/2)])
                #     labels.append(la[int(len(pr)/2)])
            assert len(preds) == len(labels)
            data = {
                "pred": preds,
                "label": labels
            }
            df = pd.DataFrame(data)
        elif ':=' in pred[0]:
            df['pred'] = [p.split(':=')[1] for p in pred]
            df['label'] = [l.split(':=')[1] for l in list(df['label'])]
            print(len(pred), len(list(df['label'])))
        else:
            df['pred'] = pred
            assert len(pred) == len(label)
        # print(df[['label', 'pred']])
        calc_metric(df)
        confusion(list(df['label']), list(df['pred']))
        print(df.shape[0])
