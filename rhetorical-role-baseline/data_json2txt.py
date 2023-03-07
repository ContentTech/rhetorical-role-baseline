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

import logging
logging.basicConfig(format="%(asctime)s %(user)-8s %(message)s")
logger = logging.getLogger()
logger.warning("Test: %s", "0", extra={"user": "fengyao"})

import glob
# json_files = glob.glob(os.path.join(".", "*.json"))

# import jsonlines
# with open("*.jsonl", "r") as f:
#     for line in jsonlines.Reader(f):
#         config = line["config"]
# with open("*.json", "r") as f:
#     doc = json.load(f)
# with open("*.json", "w") as file:
#     json.dump(doc, file)


BASEDIR = os.path.dirname(os.path.realpath(__file__))

def cmd():
    parser = argparse.ArgumentParser(description="Description")
    parser.add_argument("-f", "--input", type=str, help="input dir")
    parser.add_argument("-o", "--output", type=str, help="output file")
    # parser.add_argument("-n", "--number", type=int, help="")          
    # parser.add_argument("integers", metavar="N", type=int, nargs="+", help="an integer for the accumulator")
    # parser.add_argument("--sum", dest="accumulate", action="store_const", const=sum, default=max, help="sum the integers")
    args = parser.parse_args()
    print("argparse.args=",args,type(args))
    d = args.__dict__
    for key,value in d.items():
        print("%s = %s"%(key,value))
    return args


if __name__=="__main__":
    """
    python /mnt/fengyao.hjj/rhetorical-role-baseline/rhetorical-role-baseline/data_json2txt.py \
        -f /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/ILDC/1960_docs/ILDC_multi_1960.json \
        -o /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/ILDC/1960_docs/ILDC_multi_1960.txt
    """
    args = cmd()
    print(args.input)
    with open(args.output, "w") as fw:
        with open(args.input, "r") as f:
            docs = json.load(f)
            print(len(docs))
            for doc in docs:
                for s in doc["annotations"][0]["result"]:
                    text = s["value"]["text"]
                    text = ' '.join(text.strip().replace("\r", "").replace("\n", " ").replace('<\/s>', ' ').split())
                    fw.write(text + "\n")
        
        

