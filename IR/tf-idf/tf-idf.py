from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm, trange
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import itertools
import pickle
def get_values(file):
    """
    get label context and response.
    :param file: filel name
    :param get_c_d:
    :return:
    """
    data = open(file, 'r').readlines()
    data = [sent.split('\n')[0].split('\t') for sent in data]
    #data=data[:10000]
   # for a in data:
    #    print(a[1:-1])
     #   print(' __EOS__ '.join(a[1:-1]))
    #print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize("tt")))
    cr = [ "\n".join([sen for sen in a[1:]]) for a in data]
    #c = [' __EOS__ '.join(a[1:-1]) for a in data]

    #c = [[tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(sen)) for sen in s] for s in c]
    return cr



def makedocs():
    sample_to_doc = []
    all_docs_str= []
    all_docs_list=[]
    doc = []
    corpus_lines = 0
    with open("post_train_ver2.txt", "r", encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading Dataset", total=corpus_lines):
            line = line.strip()
            if line == "":
                if (len(doc) != 0):
                    all_docs_str.append("\n".join(doc))
                    all_docs_list.append(doc)
                doc = []
                # remove last added sample because there won't be a subsequent line anymore in the doc
                sample_to_doc.pop()
            else:
                # store as one sample
                sample = {"doc_id": len(all_docs_list),
                          "line": len(doc)}
                sample_to_doc.append(sample)
                doc.append(line.lower())
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
    return all_docs_str,all_docs_list

if __name__ == '__main__':
    #load the vocab file
    train_list = get_values('train.txt')
    test_list = get_values('test.txt')
    valid_list = get_values('valid.txt')
    targetlist=valid_list

    l=100000
    targetlist=targetlist[:l]


    all_docs_str,all_docs_list=makedocs()
    file = open("evidence_dev.txt", "w", encoding='utf-8')
    cnt=0
    wholestring = ""
    tiv = TfidfVectorizer(stop_words="english").fit(all_docs_str)
    tokenizer=tiv.build_tokenizer()
    all_docs_numpy = tiv.transform(all_docs_str)
    all_docs_text_numpy = np.array(all_docs_list)

    #all doc key
    doc_keys = pickle.load(file=open("dockey.pkl", 'rb'))

    # way to find relative docs
    importantword=False
    for crset in tqdm(targetlist):

        #find doc
        #temp = TfidfVectorizer(stop_words="english").fit([crset])

        #print(set(co_command.keys()).intersection(set(temp.vocabulary_.keys())))
        #if set(co_command.keys()).intersection(set(temp.vocabulary_.keys())) is None:
         #   continue



        #print(temp.vocabulary_)
        #tiva = TfidfVectorizer(vocabulary=temp.vocabulary_).fit(all_docs)
        #all_docs_numpy=tiv.transform(all_docs).toarray()
        #step1  관련있는 문서 뽑기.
        if importantword==False:#그냥 docs 부터 tf-idf 방식으로.
            target_cr_numpy=tiv.transform([crset])
            #score=cosine_similarity(target_cr_numpy,all_docs_numpy).squeeze()
            score = cosine_similarity(all_docs_numpy,target_cr_numpy).squeeze()
            topk=score.argsort()[::-1]
            #print(all_docs[topk[0]])
            #print('\n')
            #print(crset)
            #print("Z")
            #print('\n')
            #fine sentence

        relevantcount=0
        if importantword == True:
            sen_token=tokenizer(crset)
            relevant_doc=[]
            relevant_token=[]
            for token in sen_token:
                if token in doc_keys and token not in relevant_token:

                    relevant_doc.append(doc_keys[token])
                    relevant_token.append(token)
                    relevantcount+=1


        top10 = topk[:20]
        top10doc1 = all_docs_text_numpy[top10]




        #step2 여기서 문장 유사도 계산.
        top10doc=list(itertools.chain.from_iterable(top10doc1))

        top10_docs_numpy = tiv.transform(top10doc)

        sentence_score = cosine_similarity(target_cr_numpy, top10_docs_numpy).squeeze()
        topk_sentence = sentence_score.argsort()[::-1]
        top10_sentence = topk_sentence[:10]
        prev=""
        sencount=1
        for senidx in topk_sentence:
            if sencount==10:
                break
            if prev!=top10doc[senidx]:
                wholestring+=top10doc[senidx]+'\n'
                sencount+=1
                prev = top10doc[senidx]

        wholestring+='\n'
        cnt+=1

        #print(top10doclist[topk_sentence[0]])
        #print('\n')
        #print("Z")

    print(cnt)
    file.write(wholestring)


    # pickle.dump(dataset, open('ubuntu_data/dataset_1M.pkl', 'wb'))