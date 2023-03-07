#! /usr/bin/env python
# -*- coding:utf-8 -*-
import json
import os
from termcolor import colored
import sys
import argparse
import logging
import pandas as pd

BASEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
LEGALEVAL_LABELS = ["mask", "NONE", "PREAMBLE", "FAC", "ISSUE", "ARG_RESPONDENT", "ARG_PETITIONER", "ANALYSIS",
                 "PRE_RELIED", "PRE_NOT_RELIED", "STA", "RLC", "RPC", "RATIO"]
LABELMAPPING = ["RLC", "ANALYSIS", "FAC"]


def cmd():
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('-f', '--file', type = str, help='input file')
    parser.add_argument('-o', '--output_file', type = str, help='input file')
    args = parser.parse_args()
    print("argparse.args=",args,type(args))
    d = args.__dict__
    for key,value in d.items():
        print('%s = %s'%(key,value))
    return args


if __name__=="__main__":
    """
    python /mnt/fengyao.hjj/rhetorical-role-baseline/rhetorical-role-baseline/data_label_mapping.py \
        -f /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/backtranslation/wmt19-en-de-wmt19-de-en/13labels/train_scibert.txt \
        -o /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/backtranslation/wmt19-en-de-wmt19-de-en/train_scibert.txt
        -f /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/train_scibert.txt \
        -o /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/legaleval/D4/RLC_ANALYSIS_FAC/train_scibert.txt
    """
    args = cmd()
    print(args.file)
    # with open(args.output_file, "w") as fw:
    #     with open(args.file, "r") as f:
    #         for line in f:
    #             new_line = line
    #             if line.split("\t")[0] in LEGALEVAL_LABELS and line.split("\t")[0] not in LABELMAPPING:
    #                 new_line = '\t'.join(["OTHERS"] + line.split("\t")[1:])
    #             fw.write(new_line)                    

    with open(args.output_file, "w") as fw:
        with open(args.file, "r") as f:
            for line in f:
                new_line = line
                if line.split("\t")[0] in ["ISSUE"]:
                    continue
                fw.write(new_line)       