#!/usr/bin/python
# -*- coding:utf-8 -*-
#****************************************************************#
# ScriptName: *.py
# Author: $SHTERM_REAL_USER@alibaba-inc.com
# Create Date: 2022-03-29 10:40
# Modify Author: $SHTERM_REAL_USER@alibaba-inc.com
# Modify Date: 2022-03-29 10:40
# Function:
#***************************************************************#

import sys
import argparse

def cmd():
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('-f', '--file', type=str, help='input file')
    parser.add_argument('-o', '--output', type=str, help='input file')
    # parser.add_argument('-n', '--number', type=int, help='')
    # parser.add_argument('-i', '--input', nargs='+', type=str, help='input file')

    args = parser.parse_args()
    print("argparse.args=",args,type(args))
    d = args.__dict__
    for key,value in d.items():
        print('%s = %s'%(key,value))
    return args


import random
random.seed(4)

if __name__=="__main__":
    """
    python /Users/fengyao/PycharmProjects/rhetorical-role-baseline/datasets/shuffle_tokens.py \
    -f /Users/fengyao/PycharmProjects/rhetorical-role-baseline/datasets/legaleval/train_scibert.txt \
    -o /Users/fengyao/PycharmProjects/rhetorical-role-baseline/datasets/legaleval/train_scibert_.txt 
    
    
    python /Users/fengyao/PycharmProjects/rhetorical-role-baseline/datasets/label_reformate.py \
    -f /Users/fengyao/PycharmProjects/rhetorical-role-baseline/datasets/Y2021M8/data_annotated/train_scibert.txt \
    -o /Users/fengyao/PycharmProjects/rhetorical-role-baseline/datasets/Y2021M8/data_annotated/train_scibert_.txt 
    """
    args = cmd()
    new_labels = []
    with open(args.file, 'r') as f:
        with open(args.output, 'w') as fw:
            for line in f:
                if len(line.strip()) > 0 and len(line.strip().split('\t')) >= 2:
                    text = line.strip().split('\t')[-1].split(' ')[1:-1]
                    random.shuffle(text)
                    new_text = ' '.join(['101'] + text + ['102'])
                    new_text = new_text + '\n'
                    line = '\t'.join(line.strip().split('\t')[0:-1] + [new_text])
                fw.write(line)

