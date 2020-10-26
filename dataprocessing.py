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
    for s in cr:
        s_list=[]
        for sen in s[:-1]:
            s_list+=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sen+' tt'))
        s_list=s_list+[tokenizer.sep_token_id]
        s_list+=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s[-1]))
        cr_list.append(s_list)
    #c = [[tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(sen)) for sen in s] for s in c]
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
    evi_len=[]
    for s in tqdm(evi):
        s_list=[]
        s_len=[]
        for sen in s:
            s_len.append(len(s_list))
            s_list+=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sen+' tt'))
        s_len.append(len(s_list))
        evi_list.append(s_list)
        evi_len.append(s_len)
    #c = [[tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(sen)) for sen in s] for s in c]
    return evi_len, evi_list


if __name__ == '__main__':
    #load the vocab file
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)

    train, test, valid = {}, {}, {}
    train['y'], train['cr'] = get_values('ubuntu_data/train.txt', tokenizer=bert_tokenizer)
    test['y'], test['cr']= get_values('ubuntu_data/test.txt',tokenizer=bert_tokenizer)
    valid['y'], valid['cr']= get_values('ubuntu_data/valid.txt',tokenizer=bert_tokenizer)
    #char_vocab = defaultdict(float)
    dataset = train, valid, test
    pickle.dump(dataset, open('ubuntu_data/dataset_1M.pkl', 'wb'))

    train, test, valid = {}, {}, {}
    train['len'], train['evi'] = get_evidence_values('tf-idf-ver1/evidence_train.txt', tokenizer=bert_tokenizer)
    test['len'], test['evi'] = get_evidence_values('tf-idf-ver1/evidence_test.txt', tokenizer=bert_tokenizer)
   # train,_,test= pickle.load(file=open("ubuntu_data/evidence_dataset_1M.pkl", 'rb'))
    valid['len'], valid['evi'] = get_evidence_values('tf-idf-ver1/evidence_dev.txt', tokenizer=bert_tokenizer)
    # char_vocab = defaultdict(float)
    dataset = train, valid, test
    pickle.dump(dataset, open('ubuntu_data/evidence_dataset_1M.pkl', 'wb'))
