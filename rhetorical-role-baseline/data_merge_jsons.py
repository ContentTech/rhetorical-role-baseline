#!/usr/bin/python
# -*- coding:utf-8 -*-
#****************************************************************#
# ScriptName: ***.py
# Author: $SHTERM_REAL_USER@antgroup.com
# Create Date: yyyy-mm-dd hh:mm
# Modify Author: $SHTERM_REAL_USER@antgroup.com
# Modify Date: yyyy-mm-dd hh:mm
# Function:
#***************************************************************#

import sys
import os
import json
import argparse
from termcolor import colored
from tabulate import tabulate

import pandas as pd
pd.set_option('display.max_columns', None) #显示所有列
pd.set_option('display.max_rows', None) #显示所有行
pd.set_option('max_colwidth', 100) # 设置value的显示长度为100，默认为50

import logging
logging.basicConfig(format='%(asctime)s %(user)-8s %(message)s')
logger = logging.getLogger()
logger.warning('Test: %s', '0', extra={'user': 'fengyao'})

import glob
json_files = glob.glob(os.path.join('.', '*.json'))
print(json_files)


def cmd():
    parser = argparse.ArgumentParser(description='Description')
    parser.add_argument('-d', '--dir', type=str, help='input dir')
    parser.add_argument('-i', '--input', nargs='+', type=str, help='input files')
    parser.add_argument('-o', '--output', type=str, help='output file')
    parser.add_argument('-n', '--number', type=int, help='')        
#	parser.add_argument('integers', metavar='N', type=int, nargs='+', help='an integer for the accumulator')
#	parser.add_argument('--sum', dest='accumulate', action='store_const', const=sum, default=max, help='sum the integers')
    args = parser.parse_args()
    print("argparse.args=",args,type(args))
    d = args.__dict__
    for key,value in d.items():
        print('%s = %s'%(key,value))
    return args


if __name__=="__main__":
    """
    python /mnt/fengyao.hjj/rhetorical-role-baseline/rhetorical-role-baseline/data_merge_jsons.py -d /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/lm/indiankanoon-json -o /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/lm/lm.json
    python /mnt/fengyao.hjj/rhetorical-role-baseline/rhetorical-role-baseline/data_merge_jsons.py -d /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/ILDC/1960_docs -o /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/ILDC/ILDC_multi.json
    """
    args = cmd()
    print(args.dir)
    json_files = glob.glob(os.path.join(args.dir, '*.json'))
    print(json_files, len(json_files))

    all_docs = []
    for input_json in json_files:
        with open(input_json, 'r') as f:
            input_docs = json.load(f)
            all_docs += input_docs
    print(len(all_docs))
    with open(args.output, 'w') as file:
        json.dump(all_docs, file)

