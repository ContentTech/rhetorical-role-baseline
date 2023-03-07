#! /usr/bin/env python
# -*- coding:utf-8 -*-
import json
import os
import glob
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
REMOVEILDCDOCS = ["ILDC_2018_422.txt"]
import glob
already_down = [os.path.split(f)[1] for f in glob.glob(os.path.join("/mnt/fengyao.hjj/rhetorical-role-baseline/datasets/ILDC/ILDC_jsons_1960", "*.json"))]
print(already_down)
print(len(already_down))
def get_ildc(data_file):
    import spacy
    nlp = spacy.load("en_core_web_trf")
    df = pd.read_csv(data_file)
    num = df.shape[0]
    for item in df.iterrows():
        docs = []
        file_name = f"ILDC_{item[1]['name']}"
        if "ILDC_multi." + file_name + ".json" not in already_down and file_name not in REMOVEILDCDOCS:
            print(f"processing {file_name} ...")
            doc = {"id": str(file_name), "annotations": [{"result": []}], "data": item[1]['text'], "meta": {"group": "ILDC"}}
            sents = [str(i) for i in nlp(item[1]['text']).sents]
            for ani, sentence_txt in enumerate(sents):
                if len(sentence_txt.split()) > 1:
                    doc["annotations"][0]["result"].append(
                        {
                            "id": ani,
                            "value":
                            {
                                "start": 0,
                                "end": 0,
                                "text": sentence_txt,
                                "labels": ["DUMMY"]
                            }
                        }
                    )
            docs.append(doc)
            with open('.'.join(data_file.split('.')[:-1] + [str(file_name), 'json']), "w") as file:
                json.dump(docs, file)
        else:
            print(f"{file_name} already done!")
    return docs

def get_y2019b7(data_files, data_dir):
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


def get_y2021m8(data_dir):
    names = ['CL', 'IT']
    docs = []

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
                doc = {"id": str(file_name), "annotations": [{"result": []}], "data": "", "meta": {"group": name}}
                sentences = v['sentences'][2:-2].split("', '") if not isinstance(v['sentences'], list) else v['sentences']
                labels = v['complete']
                assert len(sentences) == len(labels)
                doc["data"] = '\n'.join(sentences)
                for ani, (annotation, sentence_txt) in enumerate(zip(labels, sentences)):
                    sentence_txt = ' '.join(sentence_txt.strip().replace("\r", "").replace("\n", " ").replace('<\/s>', ' ').split())
                    if sentence_txt != "" and annotation not in ['ArgumentRespondent', 'ArgumentPetitioner'] \
                        and LABLESY2021M8_MAPPING[LABLESY2021M8.index(annotation)] in LEGALEVAL_LABELS:
                        doc["annotations"][0]["result"].append(
                            {
                                "id": ani,
                                "value":
                                {
                                    "start": 0,
                                    "end": 0,
                                    "text": sentence_txt,
                                    "labels": [LABLESY2021M8_MAPPING[LABLESY2021M8.index(annotation)]]
                                }
                            }
                        )
                docs.append(doc)
    with open(os.path.join(data_dir, f"y2021m8.json"), "w+") as file:
        json.dump(docs, file)
    return docs


if __name__ == '__main__':
    print('BASEDIR: ', BASEDIR)
    data_dir = f"{BASEDIR}/rhetorical-role-baseline/datasets/Y2019B7/text"
    # data_files = sorted(glob.glob(os.path.join(data_dir, '*.txt')))
    # print('number of data_files: ', len(data_files))
    # get_y2019b7(data_files, os.path.dirname(data_dir))

    # data_dir = f"{BASEDIR}/rhetorical-role-baseline/datasets/Y2021M8/data_annotated"
    # get_y2021m8(data_dir)

    data_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/ILDC/ILDC_multi/ILDC_multi.csv"
    get_ildc(data_file)

