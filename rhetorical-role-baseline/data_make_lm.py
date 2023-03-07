#! /usr/bin/env python
# -*- coding:utf-8 -*-
import sys
import argparse
import glob
import os
import spacy
import json
from lxml import etree
from termcolor import colored


def cmd():
    parser = argparse.ArgumentParser(description='Description')
    # parser.add_argument('-i', '--input', nargs='+', type=str, help='input file')
    parser.add_argument('-i', '--input', type=str, help='')
    parser.add_argument('-o', '--output', type=str, help='')
    args = parser.parse_args()
    print("argparse.args=",args,type(args))
    d = args.__dict__
    for key,value in d.items():
        print('%s = %s'%(key,value))
    return args



def get_lm(data_files, data_dir):
    docs = []
    for i, file in enumerate(data_files):
        file_name = os.path.split(file)[1].split('.')[0]
        doc = {"id": str(file_name), "annotations": [{"result": []}], "data": "", "meta": {"group": "Unknow"}}
        with open(file, 'r') as f:
            labels = []
            sentences = []
            for line in f:
                sentence_txt, annotation = line.strip().split('\t')[0], line.strip().split('\t')[1]
                sentences.append(sentence_txt)
                labels.append(annotation)
            doc["data"] = '\n'.join(sentences)
            for ani, (annotation, sentence_txt) in enumerate(zip(labels, sentences)):
                if len(sentence_txt) > 0 and LABLESY2019B7_MAPPING[LABLESY2019B7.index(annotation)] in LEGALEVAL_LABELS:
                    doc["annotations"][0]["result"].append(
                        {
                            "id": ani,
                            "value":
                            {
                                "start": 0,
                                "end": 0,
                                "text": sentence_txt,
                                "labels": [LABLESY2019B7_MAPPING[LABLESY2019B7.index(annotation)]]
                            }
                        }
                    )
        docs.append(doc)
    with open(data_dir + "/y2019b7.json", "w+") as file:
        json.dump(docs, file)
    return docs


if __name__=="__main__":
    """
    python /mnt/fengyao.hjj/rhetorical-role-baseline/rhetorical-role-baseline/data_make_lm.py \
    -i /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/lm/indiankanoon \
    -o /mnt/fengyao.hjj/rhetorical-role-baseline/datasets/lm/train2.txt
    """
    args = cmd()
    inputs = args.input
    print(inputs)
    file_names = sorted(glob.glob(os.path.join(inputs, '*.html')))
    print(len(file_names))
    nlp = spacy.load("en_core_web_trf")
    with open(args.output, 'w+') as fw:
        for file in file_names:
            docs = []
            print(colored(file, 'green'))
            file_name = os.path.split(file)[1].split('.')[0]
            print("file_name: ", file_name)
            doc = {"id": str(file_name), "annotations": [{"result": []}], "data": "", "meta": {"group": "lm"}}

            with open(file, 'r') as f:
                f = ''.join(f.readlines())
                dom = etree.HTML(f)
                title = dom.xpath('//div[@class="judgments"]/div//text()')
                pre = dom.xpath('//div[@class="judgments"]//pre[@id="pre_1"]//text()')
                p = dom.xpath('//div[@class="judgments"]/p//text()')
                prefixes = ['\n      ', 'Try out our ', 'Premium Member', ' services: ', 'Virtual Legal Assistant', ',  ', 'Query Alert Service', ' and an ad-free experience. ', 'Free for one month', ' and pay only if you like it.', '\n\n    ']
                title = [i for i in title if not any([i == p for p in prefixes])]
                texts = pre + p
                labels = []
                sentences = []
                for text in texts:
                    text = ' '.join(text.strip().replace("\r", "").replace("\n", " ").replace('<\/s>', ' ').split())
                    sents = [str(i) for i in nlp(text).sents]
                    for sentence_txt in sents:
                        if len(sentence_txt.split()) > 1:
                            fw.write(sentence_txt + '\n')
                            sentences.append(sentence_txt)
                            labels.append("DUMMY")

                doc["data"] = '\n'.join(sentences)
                for ani, (annotation, sentence_txt) in enumerate(zip(labels, sentences)):
                    if len(sentence_txt) > 0:
                        doc["annotations"][0]["result"].append(
                            {
                                "id": f"{ani}",
                                "value":
                                {
                                    "start": 0,
                                    "end": 0,
                                    "text": sentence_txt,
                                    "labels": [annotation]
                                }
                            }
                        )
                docs.append(doc)
            with open(os.path.join(os.path.dirname(args.output), f"{file_name}.json"), "w+") as file:
                json.dump(docs, file)
                                    