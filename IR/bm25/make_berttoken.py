from transformers import BertTokenizer
import pickle
from tqdm import tqdm

def get_values(file, tokenizer=None):
    """
    get label context and response.
    :param file: filel name
    :param get_c_d:
    :return:
    """
    data = open(file, 'r').readlines()
    data = [sent.split('\n')[0].split('\t') for sent in data]
    #data=data[:10000]
    y = [int(a[0]) for a in data]
   # for a in data:
    #    print(a[1:-1])
     #   print(' __EOS__ '.join(a[1:-1]))
    #print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("tt")))
    cr = [ [sen for sen in a[1:]] for a in data]
    #c = [' __EOS__ '.join(a[1:-1]) for a in data]
    cr_list=[]
    cnt=0
    for s in tqdm(cr):
        s_list=[]


        for sen in s[:-1]:
            if len(sen)==0:
                cnt+=1
                continue
            s_list+=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sen+tokenizer.eos_token))
            #print(s_list)
        s_list=s_list+[tokenizer.sep_token_id]
        s_list+=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s[-1]))
        cr_list.append(s_list)
    #c = [[tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(sen)) for sen in s] for s in c]
    print(cnt)
    return y, cr_list

def get_evidence_values(file, tokenizer=None):
    """
    get label context and response.
    :param file: filel name
    :param get_c_d:
    :return:
    """
    data = open(file, 'r').read()
    data=data.split('\n\n')[:-1]
    data = [sent.split('\n') for sent in data]
    #data=data[:10000]
    evi = [ [sen for sen in a] for a in data]
    evi_list=[]
    cnt=0
    listlen=0
    for s in tqdm(evi):
        s_list=[]

        if len(s)==1 and '[empty]' in s:
            evi_list.append(s_list)
            continue

        for sen in s:
            if len(sen)==0:
                print("sibal")
                continue
            s_list+=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sen+tokenizer.eos_token))

           #print("check")
        cnt+=1
        listlen+=len(s_list)
        evi_list.append(s_list)
    print(listlen/cnt)
    #c = [[tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(sen)) for sen in s] for s in c]
    return  evi_list


if __name__ == '__main__':
    #load the vocab file
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)
    special_tokens_dict = {'eos_token': '[eos]', 'additional_special_tokens': ['[soe]','[eoe]']}
    num_added_toks = bert_tokenizer.add_special_tokens(special_tokens_dict)
    '''
    train, test, valid = {}, {}, {}
    train['y'], train['cr'] = get_values('train.txt', tokenizer=bert_tokenizer)
    test['y'], test['cr']= get_values('test.txt',tokenizer=bert_tokenizer)
    valid['y'], valid['cr']= get_values('valid.txt',tokenizer=bert_tokenizer)
    #char_vocab = defaultdict(float)
    dataset = train, valid, test
    pickle.dump(dataset, open('dataset_1M.pkl', 'wb'))
    '''
    #train_evi, valid_evi, test_evi =pickle.load(file=open("bm25_bert30_token1.pkl", 'rb'))

    train_evi = get_evidence_values('bm25_bert30_train6.txt', tokenizer=bert_tokenizer)
    test_evi = get_evidence_values('bm25_bert30_test6.txt', tokenizer=bert_tokenizer)
   # train,_,test= pickle.load(file=open("ubuntu_data/evidence_dataset_1M.pkl", 'rb'))
    valid_evi = get_evidence_values('bm25_bert30_dev6.txt', tokenizer=bert_tokenizer)
    # char_vocab = defaultdict(float)
    dataset = train_evi, valid_evi, test_evi
    pickle.dump(dataset, open('bm25_bert30_token6.pkl', 'wb'))
