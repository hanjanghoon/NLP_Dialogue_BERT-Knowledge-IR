# NLP_Dialogue_BERT-Knowledge-IR

Dialog respose selection 

dataset Ubuntu corpus V1 , External knowledge manual description

외부지식을 직접적으로 사용하기위해 information retrieval (IR) 기법 사용.

TF-IDF , BM-25

Ranking Paragraphs for Improving Answer Recall in Open-Domain Question Answering에서 제안한 Reranker를 바탕으로 reranker 와 IR 알고리즘 융합

추출된 데이터를 학습하기 위해 3가지의 attention 기법을 사용 1. scaled dot 