#! /usr/bin/env python
# -*- coding:utf-8 -*-
import os
import glob
import json
import pandas as pd
from termcolor import colored
#显示所有列
# pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
# pd.set_option('max_colwidth', 20)


BASEDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
LEGALEVAL_LABELS = ["mask", "NONE", "PREAMBLE", "FAC", "ISSUE", "ARG_RESPONDENT", "ARG_PETITIONER",
                        "ANALYSIS", "PRE_RELIED", "PRE_NOT_RELIED", "STA", "RLC", "RPC", "RATIO"]
LEGALEVAL_LABELS_DETAILS = ["mask", "None", "Preamble", "Facts", "Issues", "Argument by Petitioner", "Argument by Respondent",
                        "Analysis", "Precedent Relied", "Precedent Not Relied", "Statute", "Ruling by Lower Court", "Ruling by Present Court", "Ratio of the decision"]

LABLESY2019B7 = ["Facts", "Ruling by Lower Court", "Argument", "Statute", "Precedent", "Ratio of the decision", "Ruling by Present Court"]
LABLESY2019B7_MAPPING = ["FAC", "RLC", "ARG", "STA", "PRE", "RATIO", "RPC"]

LABLESY2021M8 = ["None", "Fact", "Issue", "ArgumentRespondent", "ArgumentPetitioner", "PrecedentReliedUpon", "PrecedentNotReliedUpon", "Statute", "RulingByLowerCourt", "RulingByPresentCourt", "RatioOfTheDecision", "Dissent", "PrecedentOverruled"]
LABLESY2021M8_MAPPING = ["NONE", "FAC", "ISSUE", "ARG_RESPONDENT", "ARG_PETITIONER", "PRE_RELIED", "PRE_NOT_RELIED", "STA", "RLC", "RPC", "RATIO", "DISSENT", "PRE_OVERRULED"]


def window_data(data, output_file, chunk_size=1, sep=':='):
    sentences = data['context']
    labels = data['label']
    labels_abbr = data['labels_abbr']
    doc_ids = data['doc_id']
    sent_ids = data['sent_id']

    max_length = 0
    chunk_sentences = []
    chunk_labels = []
    chunk_labels_abbr = []
    chunk_doc_ids = []
    chunk_sents_id = []

    chunk_sentence = []
    chunk_label = []
    chunk_label_abbr = []
    chunk_sent_id = []

    num_doc = 0

    prev_doc_id = doc_ids[0]
    for (sent, lab, lababb, di, senti) in zip(sentences, labels, labels_abbr, doc_ids, sent_ids):
        if len(chunk_sentence) == chunk_size or (di != prev_doc_id and len(chunk_sentence) > 0):
            chunk_sent = ""
            chunk_lab = ""
            chunk_lababbr = ""
            chunk_sid = ""
            for s, l, la, si in zip(chunk_sentence, chunk_label, chunk_label_abbr, chunk_sent_id):
                chunk_sent = chunk_sent + f"{si}{sep}" + s + " "
                chunk_lab = chunk_lab + f"{si}{sep}" + l + " "
                chunk_lababbr = chunk_lababbr + f"{si}{sep}" + la + " "
                chunk_sid = chunk_sid + f"{si}{sep}" + " "
            chunk_sentences.append(chunk_sent.strip())
            chunk_labels.append(chunk_lab.strip())
            chunk_labels_abbr.append(chunk_lababbr.strip())
            chunk_sents_id.append(chunk_sid.strip())
            chunk_doc_ids.append(f'doc_{num_doc}' + str(prev_doc_id))

            if di != prev_doc_id:
                chunk_sentence = [sent]
                chunk_label = [lab]
                chunk_label_abbr = [lababb]
                chunk_sent_id = [senti]
                num_doc += 1
            else:
                chunk_sentence.pop(0)
                chunk_label.pop(0)
                chunk_label_abbr.pop(0)
                chunk_sent_id.pop(0)
                chunk_sentence.append(sent)
                chunk_label.append(lab)
                chunk_label_abbr.append(lababb)
                chunk_sent_id.append(senti)
            prev_doc_id = di
        else:
            chunk_sentence.append(sent)
            chunk_label.append(lab)
            chunk_label_abbr.append(lababb)
            chunk_sent_id.append(senti)
            prev_doc_id = di

    data = {
        "context": chunk_sentences,
        "label": chunk_labels,
        "labels_abbr": chunk_labels_abbr,
        "doc_id": chunk_doc_ids,
        "sent_id": chunk_sents_id,
    }

    df = pd.DataFrame(data)
    print(df.head(100))
    print('max_length: ', max_length)
    output_file = '.'.join(output_file.split('.')[:-1]) + f'_window{chunk_size}.csv'
    df.to_csv(output_file, index=False)
    return df



def chunk_data(data, output_file, chunk_size=1, sep=':='):
    sentences = data['context']
    labels = data['label']
    labels_abbr = data['labels_abbr']
    doc_ids = data['doc_id']
    sent_ids = data['sent_id']

    max_length = 0
    chunk_sentences = []
    chunk_labels = []
    chunk_labels_abbr = []
    chunk_doc_ids = []
    chunk_sents_id = []

    num = 0
    chunk_sentence = ''
    chunk_label = ''
    chunk_label_abbr = ''
    chunk_sent_id = ''
    prev_doc_id = doc_ids[0]
    num_doc = 0
    for (s, l, la, di, si) in zip(sentences, labels, labels_abbr, doc_ids, sent_ids):
        if num < chunk_size and di == prev_doc_id:
            chunk_sentence = chunk_sentence + f"{si}{sep}" + s + " "
            chunk_label = chunk_label + f"{si}{sep}" + l + " "
            chunk_label_abbr = chunk_label_abbr + f"{si}{sep}" + la + " "
            chunk_sent_id = chunk_sent_id + f"{si}{sep}" + " "
            num += 1
            prev_doc_id = di
        elif num == chunk_size or di != prev_doc_id:
            chunk_sentences.append(chunk_sentence.strip())
            max_length = max(max_length, len(chunk_sentence.split(' ')))
            chunk_labels.append(chunk_label.strip())
            chunk_labels_abbr.append(chunk_label_abbr.strip())
            chunk_sents_id.append(chunk_sent_id.strip())
            if di != prev_doc_id:
                num_doc += 1
            chunk_doc_ids.append(f'doc_{num_doc}' + str(prev_doc_id))
            chunk_sentence = f"{si}{sep}" + s + " "
            chunk_label = f"{si}{sep}" + l + " "
            chunk_label_abbr = f"{si}{sep}" + la + " "
            chunk_sent_id = f"{si}{sep}" + " "
            num = 1
            prev_doc_id = di

    data = {
        "context": chunk_sentences,
        "label": chunk_labels,
        "labels_abbr": chunk_labels_abbr,
        "doc_id": chunk_doc_ids,
        "sent_id": chunk_sents_id,
    }
    df = pd.DataFrame(data)
    # print(df.head(30))
    print('max_length: ', max_length)
    output_file = '.'.join(output_file.split('.')[:-1]) + f'_chunk{chunk_size}.csv'
    df.to_csv(output_file, index=False)
    return df


def write_in_csv_format(input_json, output_file, chunk_size=None, window_size=None):
    json_format = json.load(open(input_json, 'r'))
    sentences = []
    labels = []
    labels_abbr = []
    doc_ids = []
    sent_ids = []
    sent_spans = []
    datas = []
    groups = []
    for file in json_format:
        for sent_id, annotation in enumerate(file['annotations'][0]['result']):
            if annotation['value']['labels'][0].strip() != "":
                doc_ids.append(file['id'])
                sent_spans.append([annotation['value']['start'], annotation['value']['end']])
                sentence_txt = annotation['value']['text']
                sentence_txt = ' '.join(sentence_txt.strip().replace("\r", "").replace("\n", " ").replace('<\/s>', ' ').split())
                sentences.append(sentence_txt)
                labels_abbr.append(annotation['value']['labels'][0])
                labels.append(LEGALEVAL_LABELS_DETAILS[LEGALEVAL_LABELS.index(annotation['value']['labels'][0])])
                sent_ids.append(sent_id)
                datas.append(file['data'])
                groups.append(file['meta']['group'])

    data = {
        "context": sentences,
        "label": labels,
        "labels_abbr": labels_abbr,
        "doc_id": doc_ids,
        "sent_id": sent_ids,
        # "sent_spans": sent_spans,
        # "datas": datas,
        # "groups": groups
    }
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    if chunk_size and chunk_size > 0:
        df = chunk_data(data, output_file, chunk_size)
    elif window_size and window_size > 0:
        df = window_data(data, output_file, window_size)
    else:
        print("please give chunk_size or window_size")
    return df


def write_in_csv_4_y2019b7(data_dir, output_file, chunk_size=None, window_size=None):
    data_files = sorted(glob.glob(os.path.join(data_dir, '*.txt')))
    sentences = []
    labels = []
    labels_abbr = []
    doc_ids = []
    sent_ids = []
    for i, file in enumerate(data_files):
        file_name = os.path.split(file)[1].split('.')[0]
        with open(file, 'r') as f:
            for si, line in enumerate(f):
                sentence_txt, annotation = line.strip().split('\t')[0], line.strip().split('\t')[1]
                if LABLESY2019B7_MAPPING[LABLESY2019B7.index(annotation)] in LEGALEVAL_LABELS:
                    sentences.append(' '.join(sentence_txt.strip().replace("\r", "").replace("\n", " ").replace('<\/s>', ' ').split()))
                    annotation = LABLESY2019B7_MAPPING[LABLESY2019B7.index(annotation)]
                    labels_abbr.append(annotation)
                    labels.append(LEGALEVAL_LABELS_DETAILS[LEGALEVAL_LABELS.index(annotation)])
                    sent_ids.append(si)
                    doc_ids.append(file_name)

    data = {
        "context": sentences,
        "label": labels,
        "labels_abbr": labels_abbr,
        "doc_id": doc_ids,
        "sent_id": sent_ids,
    }
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    if chunk_size and chunk_size > 0:
        df = chunk_data(data, output_file, chunk_size)
    elif window_size and window_size > 0:
        df = window_data(data, output_file, window_size)
    else:
        print("please give chunk_size or window_size")
    return df


def write_in_csv_4_y2021m8(data_dir, output_file, chunk_size=None, window_size=None):
    names = ['CL', 'IT']
    sentences = []
    labels = []
    labels_abbr = []
    doc_ids = []
    sent_ids = []

    for name in names:
        train_file_name = f'{name}_train.json' if name else 'train.json'
        dev_file_name = f'{name}_dev.json' if name else 'dev.json'
        test_file_name = f'{name}_dev.json' if name else 'test.json'
        for json_file_name in [train_file_name, dev_file_name, test_file_name]:
            f = json.load(open(os.path.join(data_dir, name, json_file_name), 'r'))
            for file_name, v in f.items():
                fsentences = v['sentences'][2:-2].split("', '") if not isinstance(v['sentences'], list) else v['sentences']
                flabels = v['complete']
                assert len(fsentences) == len(flabels)
                for ani, (annotation, sentence_txt) in enumerate(zip(flabels, fsentences)):
                    sentence_txt = ' '.join(sentence_txt.strip().replace("\r", "").replace("\n", " ").replace('<\/s>', ' ').split())
                    if sentence_txt != "" and annotation not in ['ArgumentRespondent', 'ArgumentPetitioner'] \
                                and LABLESY2021M8_MAPPING[LABLESY2021M8.index(annotation)] in LEGALEVAL_LABELS:
                        doc_ids.append(file_name)
                        sentences.append(sentence_txt)
                        annotation = LABLESY2021M8_MAPPING[LABLESY2021M8.index(annotation)]
                        labels_abbr.append(annotation)
                        labels.append(LEGALEVAL_LABELS_DETAILS[LEGALEVAL_LABELS.index(annotation)])
                        sent_ids.append(ani)
    data = {
        "context": sentences,
        "label": labels,
        "labels_abbr": labels_abbr,
        "doc_id": doc_ids,
        "sent_id": sent_ids,
    }
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    if chunk_size and chunk_size > 0:
        df = chunk_data(data, output_file, chunk_size)
    elif window_size and window_size > 0:
        df = window_data(data, output_file, window_size)
    else:
        print("please give chunk_size or window_size")
    return df

if __name__ == '__main__':
    chunk_size = None
    window_size = 13

    train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/pubmed-20k/train.json"
    dev_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/pubmed-20k/dev.json"
    write_in_csv_format(dev_file, f"{BASEDIR}/rhetorical-role-baseline/datasets/enc_dec/D4/dev_enc_dec.csv", chunk_size=chunk_size, window_size=window_size)
    write_in_csv_format(dev_file, f"{BASEDIR}/rhetorical-role-baseline/datasets/enc_dec/D4/test_enc_dec.csv", chunk_size=chunk_size, window_size=window_size)
    df1 = write_in_csv_format(train_file, f"{BASEDIR}/rhetorical-role-baseline/datasets/pubmed-20k/train_enc_dec.csv", chunk_size=chunk_size, window_size=window_size)

    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval_wz/legal_res.json"
    # df1 = write_in_csv_format(train_file, f"{BASEDIR}/rhetorical-role-baseline/datasets/all/train_enc_dec.csv", chunk_size=chunk_size, window_size=window_size)

    train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/backtranslation/wmt19-en-de-wmt19-de-en/train.en.json"
    df2 = write_in_csv_format(train_file, f"{BASEDIR}/rhetorical-role-baseline/datasets/backtranslation/wmt19-en-de-wmt19-de-en/train_enc_dec.csv", chunk_size=chunk_size, window_size=window_size)

    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/backtranslation/wmt21-en-de-wmt19-de-en/train.en.wmt19.json"
    # df2_1 = write_in_csv_format(train_file, f"{BASEDIR}/rhetorical-role-baseline/datasets/backtranslation/wmt21-en-de-wmt19-de-en/train_enc_dec.csv", chunk_size=chunk_size, window_size=window_size)

    data_dir = f"{BASEDIR}/rhetorical-role-baseline/datasets/Y2019B7/text"
    output_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/Y2019B7/train_enc_dec.csv"
    df3 = write_in_csv_4_y2019b7(data_dir, output_file, chunk_size=chunk_size, window_size=window_size)

    data_dir = f"{BASEDIR}/rhetorical-role-baseline/datasets/Y2021M8/data_annotated"
    output_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/Y2021M8/data_annotated/train_enc_dec.csv"
    df4 = write_in_csv_4_y2021m8(data_dir, output_file, chunk_size=chunk_size, window_size=window_size)

    dfs = [df1, df2, df3, df4]
    # print('aug: ', df1.shape)
    # print('backtranslation: ', df2.shape)
    # print('backtranslation_1: ', df2_1.shape)
    # print('y2019b7: ', df3.shape)
    # print('y2021m8: ', df4.shape)
    df = pd.concat(dfs)
    print('total: ', df.shape)
    final_file = f"train_enc_dec_chunk{chunk_size}.csv" if chunk_size else f"train_enc_dec_window{window_size}.csv"
    df.to_csv(f"/mnt/fengyao.hjj/rhetorical-role-baseline/datasets/enc_dec/D4/{final_file}", index=False)