from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm, trange
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import itertools
import argparse
import os
import torch.nn as nn
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from bm25_custom import get_bm25_weights
import nltk
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
import re


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_values(file):
    """
    get label context and response.
    :param file: filel name
    :param get_c_d:
    :return:
    """
    data = open(file, 'r').readlines()
    data = [sent.split('\n')[0].split('\t') for sent in data]
    cr = ["\n".join([sen for sen in a[1:]]) for a in data]
    return cr


def makedocs():
    sample_to_doc = []
    all_docs_str = []
    all_docs_list = []
    doc = []
    corpus_lines = 0
    doc_cnt = 0
    with open("../tf-idf/post_train_ver2.txt", "r", encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
            line = line.strip()
            if line == "":
                if (len(doc) != 0):
                    all_docs_str.append("\n".join(doc))
                    all_docs_list.append(doc)
                doc_cnt = 0
                doc = []
                # remove last added sample because there won't be a subsequent line anymore in the doc
                sample_to_doc.pop()
            else:
                p = re.compile("[^0-9.]")
                line=line.split()
                newline=[]
                for word in line:
                    if "".join(p.findall(word)):
                        newline.append(word)
                line=" ".join(newline)
                # store as one sample
                if doc_cnt >= 10:
                    continue
                sample = {"doc_id": len(all_docs_list),
                          "line": len(doc)}
                sample_to_doc.append(sample)
                if line=="":
                    continue
                doc.append(line.lower())
                doc_cnt += 1
                corpus_lines = corpus_lines + 1

    # if last row in file is not empty
    if all_docs_list[-1] != doc:
        all_docs_list.append(doc)
        all_docs_str.append("\n".join(doc))
        sample_to_doc.pop()

    for doc in all_docs_list:
        if len(doc) == 0:
            print(doc)
    num_docs = len(all_docs_list)
    return all_docs_str, all_docs_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",
                        default=0,
                        type=int,
                        help="The batch size.")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: bert")
    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str,
                        help="bert file location")
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="bert_cache", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--do_lower_case", action='store_true', default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--load", default=False, type=bool)
    parser.add_argument("--save_path",
                        default="../tf-idf/checkpoint/ubuntu.bert.pt",
                        type=str,
                        help="The path to save model.")

    args = parser.parse_args()
    # load bert
    print(args)

    # load the vocab file
    targetlist = get_values('../tf-idf/valid.txt')
    l = 50000
    targetlist = targetlist[:l]
    all_docs_str, all_docs_list = makedocs()

    wholestring = ""

    #tiv = TfidfVectorizer(stop_words="english").fit(all_docs_str)
    #tokenizer = tiv.build_tokenizer()
    #all_docs_numpy = tiv.transform(all_docs_str)
    #all_docs_text_numpy = np.array(all_docs_list)
    # way to find relative docs
    stop = stopwords.words('english') + list(string.punctuation)
    corpus=[]
    strcorpus=[]
    for doc in all_docs_str:
        for sent in doc.split('\n'):
            strcorpus.append(sent)
            corpus.append(list(set([i for i in word_tokenize(sent.lower()) if i not in stop])))
    target = []
    target_part=[]
    for i,doc in enumerate(tqdm(targetlist)):
        if i % 1000==0 and target_part:
            target.append(target_part)
            target_part=[]
        target_part.append(list(set([i for i in word_tokenize(doc.split("\n")[-1].lower()) if i not in stop])))
    if target_part:
        target.append(target_part)

    strcorpus=np.array(strcorpus)
    cnt=0

    for target_part in tqdm(target):
        relevant_str_idx_part = get_bm25_weights(corpus,target_part, n_jobs=-1)

        for relevant_str_idx_single in (relevant_str_idx_part):
            with torch.no_grad():
                top10doc=strcorpus[relevant_str_idx_single]
                prev = ""
                sencount=0
                for i, sen in enumerate(top10doc):
                    if sencount==10:
                        break

                    if  prev != sen and len(sen.split())>2:
                        wholestring += sen + '\n'
                        prev = sen
                        sencount+=1

                if cnt == 440:
                    print("0")
                if prev=="":
                    wholestring+='[empty]\n'
                    print("empty")
                wholestring += '\n'
                cnt+=1

    file = open("bm25_only_dev1_r.txt", "w", encoding='utf-8')
    file.write(wholestring)

    # pickle.dump(dataset, open('ubuntu_data/dataset_1M.pkl', 'wb'))