import random
import time

from nltk import Tree
import numpy as np
import os
import argparse
from stanfordcorenlp import StanfordCoreNLP

BASEDIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
# snlp = StanfordCoreNLP(os.path.join(BASEDIR, 'model/stanford_nlp/stanford-corenlp-4.5.1'))

snlp = None

def parsing_stanfordnlp(raw_text):
    st_time = time.time()
    try:
        parsing = snlp.parse(raw_text)
        # print(time.time() - st_time)
        return parsing
    except Exception as e:
        return 'None'

def subtree_exchange_single(args, parsing1, label1, parsing2, label2, lam1, lam2):
    """
    For a pair sentence, exchange subtree and return a label based on subtree length

    Find the candidate subtree, and extract correspoding span, and exchange span

    """
    if args.debug:
        print('check4')
    assert lam1 > lam2
    t1 = Tree.fromstring(parsing1)
    original_sentence = ' '.join(t1.leaves())
    t1_len = len(t1.leaves())
    candidate_subtree1 = list(t1.subtrees(lambda t: lam1 > len(t.leaves()) / t1_len > lam2))
    t2 = Tree.fromstring(parsing2)
    candidate_subtree2 = list(t2.subtrees(lambda t: lam1 > len(t.leaves()) / t1_len > lam2))
    if args.debug:
        print('check5')
    # print('subtree1:',len(candidate_subtree1),'\nsubtree2:',len(candidate_subtree2))
    if len(candidate_subtree1) == 0 or len(candidate_subtree2) == 0:
        return None
    if args.debug:
        print('check6')
    if args.phrase_label:
        if args.debug:
            print('phrase_label')
        tree_labels1 = [tree.label() for tree in candidate_subtree1]
        tree_labels2 = [tree.label() for tree in candidate_subtree2]
        same_labels = list(set(tree_labels1) & set(tree_labels2))
        if not same_labels:
            # print('无相同类型的子树')
            return None
        if args.phrase_length:
            if args.debug:
                print('phrase_lable_length')
            candidate = [(t1, t2) for t1 in candidate_subtree1 for t2 in candidate_subtree2 if
                         len(t1.leaves()) == len(t2.leaves()) and t1.label() == t2.label()]
            candidate1, candidate2 = random.choice(candidate)
        else:
            if args.debug:
                print('phrase_lable')
            select_label = random.choice(same_labels)
            candidate1 = random.choice([t for t in candidate_subtree1 if t.label() == select_label])
            candidate2 = random.choice([t for t in candidate_subtree2 if t.label() == select_label])
    else:
        if args.debug:
            print('no phrase_label')
        if args.phrase_length:
            if args.debug:
                print('phrase_length')
            candidate = [(t1, t2) for t1 in candidate_subtree1 for t2 in candidate_subtree2 if
                         len(t1.leaves()) == len(t2.leaves())]
            candidate1, candidate2 = random.choice(candidate)
        else:
            if args.debug:
                print('normal TreeMix')
            candidate1 = random.choice(candidate_subtree1)
            candidate2 = random.choice(candidate_subtree2)

    exchanged_span = ' '.join(candidate1.leaves())
    exchanged_len = len(candidate1.leaves())
    exchanging_span = ' '.join(candidate2.leaves())
    new_sentence = original_sentence.replace(exchanged_span, exchanging_span)
    # if args.mixup_cross:
    new_label = np.zeros(len(args.label_list))

    exchanging_len = len(candidate2.leaves())
    new_len = t1_len - exchanged_len + exchanging_len

    new_label[int(label2)] += exchanging_len / new_len
    new_label[int(label1)] += (new_len - exchanging_len) / new_len

    # else:
    #     new_label=label1
    if args.showinfo:
        # print('树1 {}'.format(t1))
        # print('树2 {}'.format(t2))
        print('-' * 50)
        print('candidate1:{}'.format([' '.join(x.leaves()) for x in candidate_subtree1]))
        print('candidate2:{}'.format([' '.join(x.leaves()) for x in candidate_subtree2]))
        print('sentence1 ## {}  [{}]\nsentence2 ## {}  [{}]'.format(original_sentence, label1, ' '.join(t2.leaves()),
                                                                    label2))
        print('{}  <=========== {}'.format(exchanged_span, exchanging_span))
        print('new sentence: ## {}'.format(new_sentence))
        print('new label:[{}]'.format(new_label))
    return new_sentence, new_label

def parse_argument():
    parser=argparse.ArgumentParser()
    parser.add_argument('--lam1',type=float,default=0.3)
    parser.add_argument('--lam2',type=float,default=0.1)
    parser.add_argument('--times',default=[2,5],nargs='+',help='augmentation times list')
    parser.add_argument('--min_token',type=int,default=0,help='minimum token numbers of augmentation samples')
    parser.add_argument('--label_name',type=str,default='label')
    parser.add_argument('--phrase_label',action='store_true',help='subtree lable must be same if set')
    parser.add_argument('--phrase_length',action='store_true',help='subtree phrase must be same length if set')
    # parser.add_argument('--data_type',type=str,required=True,help='This is a single classification task or pair sentences classification task')
    parser.add_argument('--seeds',default=[0,1,2,3,4],nargs='+',help='seed list')
    parser.add_argument('--showinfo',action='store_true')
    parser.add_argument('--mixup_cross',action='store_false',help="NO mix across different classes if set")
    parser.add_argument('--low_resource',action='store_true',help="create low source raw and aug datasets if set")
    parser.add_argument('--debug',action='store_true',help="display debug information")
    parser.add_argument('--data',nargs='+',required=False,help='data list')
    parser.add_argument('--proc',type=int,help='processing number for multiprocessing')
    parser.add_argument('--label_list', default=[0,1,2,3,4], help='processing number for multiprocessing')
    args=parser.parse_args()
    return args


def single_sentences_test():
    args = parse_argument()
    sent_a = 'This shows that whereas injury No. 1 was caused by a firearm in the nature of a rifle, injuries 2 and 5 were caused by an ordinary gun.'
    sent_b = "The medical evidence thus falsifies the eye- witnesses' account according to which, the appellant Subhash alone was armed with a double-barrelled gun, the other appellant Shyam Narain being armed with a lathi."
    p_a = parsing_stanfordnlp(sent_a)
    p_b = parsing_stanfordnlp(sent_b)
    print(p_a)
    print(p_b)
    label_1 = 0
    label_2 = 1
    res = subtree_exchange_single(args, p_a, label_1, p_b, label_2, 0.3, 0.1)
    print(res)


if __name__ == '__main__':
    single_sentences_test()