import os
from transformers import BertTokenizer, DebertaV2Tokenizer
import pandas as pd
from termcolor import colored
from infer_new import *

BASEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
MAX_SEQ_LENGTH = 256
# BERT_VOCAB = os.path.join(BASEDIR, "bert-base-uncased")
# BERT_VOCAB = os.path.join(BASEDIR, "zlucia-legalbert")
BERT_VOCAB = os.path.join(BASEDIR, "rhetorical-role-baseline/results/bert_continue_bz64/checkpoint-48000")
# BERT_VOCAB = os.path.join(BASEDIR, "deberta-v3-base")


def write_in_csv_format(input_json, output_file):
    json_format = json.load(open(input_json, 'r'))
    sentences = []
    labels = []
    doc_ids = []
    sent_ids = []
    sent_spans = []
    datas = []
    groups = []
    for file in json_format:
        for sent_id, annotation in enumerate(file['annotations'][0]['result']):
            doc_ids.append(file['id'])
            sent_spans.append([annotation['value']['start'], annotation['value']['end']])
            sentence_txt = annotation['value']['text']
            sentence_txt = ' '.join(sentence_txt.strip().replace("\r", "").replace("\n", " ").replace('<\/s>', ' ').split())
            sentences.append(sentence_txt)
            labels.append(annotation['value']['labels'][0])
            sent_ids.append(sent_id)
            datas.append(file['data'])
            groups.append(file['meta']['group'])

    data = {
        "context": sentences,
        "label": labels,
        "doc_id": doc_ids,
        "sent_id": sent_ids,
        "sent_spans": sent_spans,
        "datas": datas,
        "groups": groups
    }
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    return output_file


def write_in_hsln_format(input_dir, output_dir, type='train', auxiliary_task=None):
    tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)
    # tokenizer = DebertaV2Tokenizer.from_pretrained(BERT_VOCAB, do_lower_case=True)
    hsln_format_txt_dirpath = output_dir
    json_format = json.load(open(input_dir))
    final_string = ''
    filename_sent_boundries = {}
    for file in json_format:
        file_name = file['id']
        previous_label = "mask"
        final_string = final_string + '###' + str(file_name) + "\n"
        filename_sent_boundries[file_name] = {"sentence_span": []}
        for ani, annotation in enumerate(file['annotations'][0]['result']):
            if ani < len(file['annotations'][0]['result']) - 1:
                next_label = file['annotations'][0]['result'][ani+1]['value']['labels'][0]
            else:
                next_label = "mask"
            filename_sent_boundries[file_name]['sentence_span'].append(
                [annotation['value']['start'], annotation['value']['end']])

            sentence_txt = annotation['value']['text']
            # sentence_txt = sentence_txt.replace("\r", "")
            sentence_txt = ' '.join(sentence_txt.strip().replace("\r", "").replace("\n", " ").replace('<\/s>', ' ').split())
            if sentence_txt.strip() != "":
                sent_tokens = tokenizer.encode(sentence_txt, add_special_tokens=True, max_length=MAX_SEQ_LENGTH,
                                               truncation=True)
                sent_tokens = [str(i) for i in sent_tokens]
                sent_tokens_txt = " ".join(sent_tokens)
                if len(annotation['value']['labels']) != 1:
                    print(colored(annotation['value']['labels'], 'red'))
                if auxiliary_task == 'lsp':
                    if previous_label == annotation['value']['labels'][0]:
                        ls = "0"
                    else:
                        ls = "1"
                if auxiliary_task == 'bioe':
                    if annotation['value']['labels'][0] != next_label:
                        ls = "E"
                    if annotation['value']['labels'][0] != previous_label:
                        ls = "B"
                    if annotation['value']['labels'][0] == previous_label and annotation['value']['labels'][0] == next_label:
                        ls = "I"
                    if annotation['value']['labels'][0] == "NONE":
                        ls = "O"
                final_string = final_string + annotation['value']['labels'][
                    0] + "\t" + ls + "\t" + sent_tokens_txt + "\n"
                previous_label = annotation['value']['labels'][0]
        final_string = final_string + "\n"

    with open(hsln_format_txt_dirpath + f'/{type}_scibert.txt', "w+") as file:
        file.write(final_string)
    with open(hsln_format_txt_dirpath + f'/{type}_sentece_boundries.json', 'w+') as json_file:
        json.dump(filename_sent_boundries, json_file)
    return filename_sent_boundries


if __name__ == '__main__':

    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/train.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval_wz/legal_res.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/Y2021M8/data_annotated/y2021m8.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/Y2019B7/y2019b7.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/backtranslation/wmt19-en-de-wmt19-de-en/train.en.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/backtranslation/wmt21-en-de-wmt19-de-en/train.en.wmt19.json"

    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/train.ner.all.json"
    # dev_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/pubmed-20k/dev.ner.json"    

    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D3/remove2/train.all.json"
    # dev_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D3/remove2/dev.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/ILDC/272_docs/ILDC_multi.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/ILDC/833_docs/ILDC_multi_833.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/ILDC/833_docs/ILDC_multi_833.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/ILDC/1221_docs/ILDC_multi_1221.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D8/ILDC_multi_pred.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D8.1/ILDC_multi_555_pred.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D8.2/ILDC_multi_833_pred.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D8.3/ILDC_multi_1222_pred.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9/ILDC_multi_272_pred.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.1/ILDC_multi_555_pred.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.2/ILDC_multi_833_pred.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.3/ILDC_multi_1222_pred.json"
    
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.1/all_train_split/0/train.0.json"
    # dev_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.1/all_train_split/0/dev.0.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.1/all_train_split/1/train.1.json"
    # dev_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.1/all_train_split/1/dev.1.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.1/all_train_split/2/train.2.json"
    # dev_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.1/all_train_split/2/dev.2.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.1/all_train_split/3/train.3.json"
    # dev_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.1/all_train_split/3/dev.3.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.1/all_train_split/4/train.4.json"
    # dev_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.1/all_train_split/4/dev.4.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.1/all_train_split/5/train.5.json"
    # dev_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.1/all_train_split/5/dev.5.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.1/all_train_split/6/train.6.json"
    # dev_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.1/all_train_split/6/dev.6.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.1/all_train_split/7/train.7.json"
    # dev_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.1/all_train_split/7/dev.7.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.1/all_train_split/8/train.8.json"
    # dev_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.1/all_train_split/8/dev.8.json"
    # train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.1/all_train_split/9/train.9.json"
    # dev_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D9.1/all_train_split/9/dev.9.json"


    train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D4/train_dev/0/train.0.json"
    dev_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D4/train_dev/0/dev.0.json"
    train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D4/train_dev/1/train.1.json"
    dev_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D4/train_dev/1/dev.1.json"
    train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D4/train_dev/2/train.2.json"
    dev_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D4/train_dev/2/dev.2.json"
    train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D4/train_dev/3/train.3.json"
    dev_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D4/train_dev/3/dev.3.json"
    train_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D4/train_dev/4/train.4.json"
    dev_file = f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/D4/train_dev/4/dev.4.json"


    print('train: ', train_file, len(json.load(open(train_file, 'r'))))
    # print('dev: ', dev_file, len(json.load(open(dev_file, 'r'))))

    # write_in_csv_format(dev_file, f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/dev.csv")
    # write_in_csv_format(dev_file, f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/test.csv")
    # write_in_csv_format(train_file, f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval/train.csv")

    auxiliary_task = 'bioe'
    # write_in_hsln_format(dev_file, output_dir=f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval", type='dev', auxiliary_task=auxiliary_task)
    # write_in_hsln_format(dev_file, output_dir=f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval", type='test', auxiliary_task=auxiliary_task)
    # # write_in_hsln_format(train_file, output_dir=f"{BASEDIR}/rhetorical-role-baseline/datasets/backtranslation/wmt19-en-de-wmt19-de-en", type='train', auxiliary_task=auxiliary_task)
    # # write_in_hsln_format(train_file, output_dir=f"{BASEDIR}/rhetorical-role-baseline/datasets/legaleval_wz", type='train', auxiliary_task=auxiliary_task)
    # write_in_hsln_format(train_file, output_dir=f"{BASEDIR}/rhetorical-role-baseline/datasets/Y2021M8/data_annotated/", type='train', auxiliary_task=auxiliary_task)
    
    write_in_hsln_format(train_file, output_dir=os.path.dirname(train_file), type='train', auxiliary_task=auxiliary_task)
    write_in_hsln_format(dev_file, output_dir=os.path.dirname(train_file), type='dev', auxiliary_task=auxiliary_task)
    write_in_hsln_format(dev_file, output_dir=os.path.dirname(train_file), type='test', auxiliary_task=auxiliary_task)

