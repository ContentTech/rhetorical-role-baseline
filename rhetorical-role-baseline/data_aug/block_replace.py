
import codecs
import copy
import json
import os
import random
import hashlib
import utils
from TreeMix import parsing_stanfordnlp, parse_argument, subtree_exchange_single

LEGALEVAL_LABELS = ["mask", "NONE", "PREAMBLE", "FAC", "ISSUE", "ARG_RESPONDENT", "ARG_PETITIONER", "ANALYSIS",
                 "PRE_RELIED", "PRE_NOT_RELIED", "STA", "RLC", "RPC", "RATIO"]

LABLESY2019B7 = ["Facts", "Ruling by Lower Court", "Argument", "Statute", "Precedent", "Ratio of the decision", "Ruling by Present Court"]
LABLESY2019B7_MAPPING = ["FAC", "RLC", "ARG", "STA", "PRE", "RATIO", "RPC"]

LABLESY2021M8 = ["None", "Fact", "Issue", "ArgumentRespondent", "ArgumentPetitioner", "PrecedentReliedUpon", "PrecedentNotReliedUpon", "Statute", "RulingByLowerCourt", "RulingByPresentCourt", "RatioOfTheDecision", "Dissent", "PrecedentOverruled"]
LABLESY2021M8_MAPPING = ["NONE", "FAC", "ISSUE", "ARG_RESPONDENT", "ARG_PETITIONER", "PRE_RELIED", "PRE_NOT_RELIED", "STA", "RLC", "RPC", "RATIO", "DISSENT", "PRE_OVERRULED"]

file_name = '../../datasets/data_aug/legal_train.json'


argss = parse_argument()

def blocks(content):
    for article in content:
        a_id = article['id']
        items = article['annotations'][0]['result']
        items.sort(key=lambda x:x['value']['start'])

        labels = [item['value']['labels'][0] for item in items]

        combined_labels = []
        for i_label in labels:
            if len(combined_labels) >0 and i_label == combined_labels[-1][-1]:
                combined_labels[-1].append(i_label)
            else:
                combined_labels.append([i_label])

        index = 0
        final_block = []
        for block_label in combined_labels:
            block_item = []
            for i_label in block_label:
                sen_item = items[index]
                block_item.append(sen_item)
                index += 1
            final_block.append(block_item)

        article['annotations'][0]['blocks'] = final_block

    return content


def merge_blocks(content):
    blocks = {"Criminal": {}, "Tax":{}}
    for article in content:
        a_id = article['id']
        a_domain = article['meta']['group']
        a_blocks = article['annotations'][0]['blocks']
        for a_blocks_array in a_blocks:
            a_blocks_type = a_blocks_array[0]['value']['labels'][0]
            if a_blocks_type in blocks[a_domain]:
                blocks[a_domain][a_blocks_type].extend([a_blocks_array])
            else:
                blocks[a_domain][a_blocks_type] = [copy.deepcopy(a_blocks_array)]
    return blocks


def recovery_article(article):
    a_blocks = article['annotations'][0]['blocks']
    sentences = []
    for block in a_blocks:
        sentences.extend(block)
    base_count = 0
    base_text = ''
    for sen in sentences:
        print(sen)
        sen['value']['start'] = base_count
        base_count += len(sen['value']['text'])
        sen['value']['end'] = base_count
        base_text += sen['value']['text']
    article['id'] = utils.md5_decode(base_text)
    article['annotations'][0]['result'] = sentences
    article['data']['text'] = base_text
    return article


def delete_block(content, ratio):
    extend_articles = []
    for article in content:
        a_id = article['id']
        a_domain = article['meta']['group']
        a_blocks = article['annotations'][0]['blocks']
        new_blocks = [item for item in a_blocks if random.random() < ratio]
        if len(new_blocks) == len(a_blocks):
            continue
        new_article = copy.deepcopy(article)
        new_article['annotations'][0]['blocks'] = new_blocks
        new_article['id'] = '121212'
        new_article = recovery_article(new_article)
        extend_articles.append(new_article)

    return extend_articles


def shuffle_inner_block(content, ratio):
    extend_articles = []
    for article in content:
        a_blocks = article['annotations'][0]['blocks']
        new_blocks = []
        for a_block in a_blocks:
            new_block = copy.deepcopy(a_block)
            if random.random() <= ratio:
                random.shuffle(new_block)
            new_blocks.append(new_block)
        new_article = copy.deepcopy(article)
        new_article['annotations'][0]['blocks'] = new_blocks
        new_article['id'] = '121212'
        new_article = recovery_article(new_article)
        extend_articles.append(new_article)

    return extend_articles


def exchange_block(content, val_blocks, ratio):
    extend_articles = []
    for article in content:
        a_id = article['id']
        a_domain = article['meta']['group']
        a_blocks = article['annotations'][0]['blocks']
        new_blocks = []
        change_flag = False
        for a_block in a_blocks:
            a_block_type = a_block[0]['value']['labels'][0]
            if random.random() < ratio:
                new_block = random.choice(val_blocks[a_domain][a_block_type])
                new_blocks.append(new_block)
                change_flag = True
            else:
                new_blocks.append(a_block)
        if not change_flag:
            continue
        new_article = copy.deepcopy(article)
        new_article['annotations'][0]['blocks'] = new_blocks
        new_article = recovery_article(new_article)
        extend_articles.append(new_article)

    return extend_articles


def replace_treemix(content, sen_dict, ratio, min_ratio, max_ratio):
    sen_parse_dict = {}
    for a_domain in sen_dict.keys():
        blocks = sen_dict[a_domain]
        for block_type in blocks.keys():
            for item in blocks[block_type]:
                sen, sen_parse = item[0], item[1]
                sen_parse_dict[sen] = sen_parse
    extend_articles = []
    for article in content:
        a_id = article['id']
        a_domain = article['meta']['group']
        a_blocks = article['annotations'][0]['blocks']
        new_blocks = []
        for a_block in a_blocks:
            a_block_type = a_block[0]['value']['labels'][0]
            new_block = []
            for sen in a_block:
                new_sen = copy.deepcopy(sen)
                if random.random() <= ratio:
                    text = sen['value']['text']
                    text_parsing = sen_parse_dict.get(text, 'None')
                    if text_parsing == 'None' or len(sen_dict[a_domain][a_block_type]) == 0 :
                        continue
                    text_item = random.choice(sen_dict[a_domain][a_block_type])
                    text_b, text_parsing_b = text_item[0], text_item[1]
                    if text_parsing == 'None' or text_parsing_b == 'None':
                        continue
                    res = subtree_exchange_single(argss, text_parsing, 1, text_parsing_b, 2, max_ratio, min_ratio)
                    if res is None:
                        continue
                    new_text, new_label = res
                    print(text)
                    print(new_text)
                    new_sen['text'] = new_text
                new_block.append(sen)
            new_blocks.append(new_block)
        new_article = copy.deepcopy(article)
        new_article['annotations'][0]['blocks'] = new_blocks
        new_article['id'] = '121212'
        new_article = recovery_article(new_article)
        extend_articles.append(new_article)

    return extend_articles


def extract_sen(content):
    sen_dict = {"Criminal": {}, "Tax": {}}
    for article in content[:1]:
        a_id = article['id']
        print(a_id)
        a_domain = article['meta']['group']
        a_blocks = article['annotations'][0]['blocks']
        for a_block in a_blocks:
            a_block_type = a_block[0]['value']['labels'][0]
            for sen in a_block:
                text = sen['value']['text']
                # text_parse = parsing_stanfordnlp(text)
                text_parse = 'parsing_stanfordnlp(text)'
                if text_parse is None:
                    continue
                if a_block_type in sen_dict[a_domain]:
                    sen_dict[a_domain][a_block_type].append([text, text_parse])
                else:
                    sen_dict[a_domain][a_block_type] = [[text, text_parse]]
    return sen_dict


def extract_from_y2019b7():
    yb7_dir = '../../datasets/Y2019B7/text'
    file_names = os.listdir(yb7_dir)

    articles = []
    for file_name in file_names:
        article = {}
        lines = open(os.path.join(yb7_dir, file_name))
        lines = [line.strip() for line in lines]
        sentences = []
        for line in lines:
            sentence_txt, annotation = line.strip().split('\t')[0], line.strip().split('\t')[1]
            new_sen = {"text": sentence_txt, 'label': annotation}
            sentences.append(new_sen)

        article['sentences'] = sentences
        article['blocks'] = get_blocks(sentences)
        articles.append(article)
    return articles


def extract_from_y2021m8():
    ym8_dir = '../../datasets/Y2021M8/text'
    file_names = os.listdir(ym8_dir)

    articles = []
    for file_name in file_names:
        article_details = json.load(open(os.path.join(ym8_dir, file_name), 'r'))
        for key, value in article_details.items():
            article = {}
            texts = value['sentences']
            # texts = json.loads(texts) if type(texts) is str else texts
            texts = texts[2:-2].split("', '") if not isinstance(texts, list) else texts
            labels = value['complete']
            sentences = [{"text": sen, "label": label} for sen, label in zip(texts, labels)]
            article['sentences'] = sentences
            article['blocks'] = get_blocks(sentences)
            articles.append(article)

    return articles

def remove_ym8(articles):
    new_articles = []
    for article in articles:
        sentences = article['sentences']
        val_sentences = []
        for sentence in sentences:
            label = sentence['label']
            if label not in LABLESY2021M8:
                continue
            mapped_label = LABLESY2021M8_MAPPING[LABLESY2021M8.index(label)]
            new_sentence = copy.deepcopy(sentence)
            new_sentence['label'] = mapped_label
            val_sentences.append(new_sentence)
        raw_text = ''.join([sentence['text'] for sentence in val_sentences])
        new_article = {'sentences': val_sentences, 'id': utils.md5_decode(raw_text)}
        new_articles.append(new_article)
    return new_articles


def remove_yb7(articles):
    new_articles = []
    for article in articles:
        sentences = article['sentences']
        val_sentences = []
        for sentence in sentences:
            label = sentence['label']
            if label not in LABLESY2019B7:
                continue
            mapped_label = LABLESY2019B7_MAPPING[LABLESY2019B7.index(label)]
            new_sentence = copy.deepcopy(sentence)
            new_sentence['label'] = mapped_label
            val_sentences.append(new_sentence)
        raw_text = ''.join([sentence['text'] for sentence in val_sentences])
        new_article = {'sentences': val_sentences, 'id': utils.md5_decode(raw_text)}
        new_articles.append(new_article)
    return new_articles


def get_blocks(sentences):
    combined_labels = []
    for sen in sentences:
        sen_text, sen_label = sen['text'], sen['label']
        if len(combined_labels) > 0 and sen_label == combined_labels[-1][-1]:
            combined_labels[-1].append(sen_label)
        else:
            combined_labels.append([sen_label])

    index = 0
    final_block = []
    for block_label in combined_labels:
        block_item = []
        for _ in block_label:
            sen_item = sentences[index]
            block_item.append(sen_item)
            index += 1
        final_block.append(block_item)
    return final_block


def block_replace():
    content = json.load(open(file_name, 'r'))
    content = blocks(content)
    val_blocks = merge_blocks(content)
    # sen_dict = extract_sen(content)
    # sen_dict = json.load(open('../../datasets/data_aug/treemix_base.json', 'r'))
    # print('parse success!')

    # json.dump(sen_dict, open('../../datasets/data_aug/treemix_base.json', 'w'), ensure_ascii=False)
    deleted_articles = delete_block(content, ratio=0.1)
    exchanged_articles = exchange_block(content, val_blocks, ratio=0.4)
    # shuffled_articles = shuffle_inner_block(content, ratio=0.5)

    content.extend(deleted_articles)
    content.extend(exchanged_articles)
    # content.extend(shuffled_articles)

    # treemixed_articles_1 = replace_treemix(content, sen_dict, ratio=0.3, min_ratio=0.1, max_ratio=0.3)
    # treemixed_articles_2 = replace_treemix(content, sen_dict, ratio=0.5, min_ratio=0.1, max_ratio=0.3)
    # treemixed_articles_3 = replace_treemix(content, sen_dict, ratio=0.5, min_ratio=0.1, max_ratio=0.3)
    # content.extend(treemixed_articles_1)
    # content.extend(treemixed_articles_2)
    # content.extend(treemixed_articles_3)
    yb7_articles = extract_from_y2019b7()
    yb7_articles = remove_yb7(yb7_articles)
    ym8_articles = extract_from_y2021m8()
    ym8_articles = remove_ym8(ym8_articles)
    content.extend(yb7_articles)
    print("ADD YB7 DATASETS: {} articles".format(len(yb7_articles)))
    content.extend(ym8_articles)
    print("ADD YM8 DATASETS: {} articles".format(len(ym8_articles)))

    print("TOTAL ARTICLES:{}".format(len(content)))
    json.dump(content, open('../../datasets/data_aug/legal_res.json', 'w'), ensure_ascii=False,indent=2)


if __name__ == '__main__':
    block_replace()