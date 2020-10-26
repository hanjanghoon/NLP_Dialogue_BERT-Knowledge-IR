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


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}
class Bert_cls(nn.Module):
    def __init__(self,args):
        super(Bert_cls, self).__init__()

        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

        self.bert_config1 = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=1)

        self.bert_tokenizer = BertTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case)
        special_tokens_dict = {'eos_token': '[eos]'}
        num_added_toks = self.bert_tokenizer.add_special_tokens(special_tokens_dict)

        self.bert_model = model_class.from_pretrained(args.model_name_or_path,
                                                   from_tf=bool('.ckpt' in args.model_name_or_path),
                                                   config=self.bert_config1)
        self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))

        self.bert_model = self.bert_model.cuda()
        self.args=args

    def _truncate_seq_pair(self,tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def getcls(self,response,topdoc):
        #if torch.cuda.is_available(): self.cuda()
        bertclslist=[]
        input_ids=[]

        for i,sentence in enumerate(topdoc):

            if(i % self.args.batch_size==0 and input_ids):
                input_ids = torch.tensor(input_ids)
                output = self.bert_model(input_ids.cuda())
                logits = torch.sigmoid(output[0])
                bertclslist.extend(logits)
                input_ids = []


            token_sen=self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize(sentence))
            token_res = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize(response))



            self._truncate_seq_pair(token_sen, token_res, 100 - 3)

            input_id = [self.bert_tokenizer.cls_token_id] + token_sen + [self.bert_tokenizer.sep_token_id] + token_res + [self.bert_tokenizer.sep_token_id]
            padding_length = 100 - len(input_id)
            input_id = input_id + ([0] * padding_length)
            input_ids.append(input_id)

            #x = torch.cat((x, out), 0)

        if(input_ids):
            input_ids=torch.tensor(input_ids)
            output=self.bert_model(input_ids.cuda())
            logits = torch.sigmoid(output[0])
            bertclslist.extend(logits)

        return np.array(bertclslist)

    def load_model(self, path):
        self.load_state_dict(state_dict=torch.load(path))
        if torch.cuda.is_available(): self.cuda()

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
    doc_cnt=0
    with open("post_train_ver2.txt", "r", encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
            line = line.strip()
            if line == "":
                if (len(doc) != 0):
                    all_docs_str.append("\n".join(doc))
                    all_docs_list.append(doc)
                doc_cnt=0
                doc = []
                # remove last added sample because there won't be a subsequent line anymore in the doc
                sample_to_doc.pop()
            else:
                # store as one sample
                if doc_cnt>=5:
                    continue
                sample = {"doc_id": len(all_docs_list),
                          "line": len(doc)}
                sample_to_doc.append(sample)
                doc.append(line.lower())
                doc_cnt+=1
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
                        default="checkpoint/ubuntu.bert.pt",
                        type=str,
                        help="The path to save model.")

    args = parser.parse_args()
    # load bert
    print(args)
    model=Bert_cls(args)

    if (args.load == True):
        print("model load! \n")
        model.load_model(args.save_path)
    # load the vocab file
 #   train_list = get_values('train.txt')
    test_list = get_values('test.txt')
  #  valid_list = get_values('valid.txt')
    targetlist = test_list
    file = open("bert_evi_test0.txt", "w", encoding='utf-8')
    l = 50000
    targetlist = targetlist[:l]
    all_docs_str, all_docs_list = makedocs()

    wholestring = ""

    tiv = TfidfVectorizer(stop_words="english").fit(all_docs_str)
    tokenizer = tiv.build_tokenizer()
    all_docs_numpy = tiv.transform(all_docs_str)
    all_docs_text_numpy = np.array(all_docs_list)
    # way to find relative docs

    for cnt,crset in tqdm(enumerate(targetlist)):

        last = crset.split('\n')[-1]
        #target_cr_numpy = tiv.transform([last])
        target_cr_numpy = tiv.transform([crset])
        score = cosine_similarity(all_docs_numpy, target_cr_numpy).squeeze()
        topk = score.argsort()[::-1]
        top10 = topk[:10]
        top10doc1 = all_docs_text_numpy[top10]
        top10doc = list(itertools.chain.from_iterable(top10doc1))

        #bert로 연관성 찾기.
        with torch.no_grad():
            sentence_score=model.getcls(last,top10doc)
        topk_sentence = sentence_score.argsort()[::-1]
        top10_sentence = topk_sentence[:10]
        prev = ""
        sencount = 1
        for senidx in topk_sentence:
            if sencount == 10:
                break
            if prev != top10doc[senidx]:
                wholestring += top10doc[senidx] + '\n'
                sencount += 1
                prev = top10doc[senidx]
        if cnt==440:
            print("0")
        wholestring += '\n'
        cnt += 1
    file.write(wholestring)

    # pickle.dump(dataset, open('ubuntu_data/dataset_1M.pkl', 'wb'))