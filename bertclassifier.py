import time
import argparse
import pickle
import torch
from torch.utils.data import TensorDataset
import os
from transformers import WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer
from NeuralNetwork import NeuralNetwork

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

task_dic = {
    'ubuntu': './dataset/ubuntu_data/',
    'douban': './dataset/DoubanConversaionCorpus/',
    'alime': './dataset/E_commerce/'
}
data_batch_size = {
    "ubuntu": 16,
    "douban": 150,
    "alime": 200
}



## Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--task",
                    default='ubuntu',
                    type=str,
                    help="The dataset used for training and test.")
parser.add_argument("--is_training",
                    default=False,
                    type=bool,
                    help="Training model or evaluating model?")
parser.add_argument("--batch_size",
                    default=0,
                    type=int,
                    help="The batch size.")
parser.add_argument("--learning_rate",
                    default=2e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--l2_reg",
                    default=0.0,
                    type=float,
                    help="The l2 regularization.")
parser.add_argument("--epochs",
                    default=4,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--save_path",
                    default="./checkpoint/",
                    type=str,
                    help="The path to save model.")
parser.add_argument("--score_file_path",
                    default="score_file.txt",
                    type=str,
                    help="The path to save model.")
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
args = parser.parse_args()
args.batch_size = data_batch_size[args.task]
args.save_path += args.task + '.' +"bert.pt"
args.score_file_path = "score.txt"
# load bert


print(args)
print("Task: ", args.task)


def train_model(train,dev,train_evi,dev_evi):
    path = task_dic[args.task]
    model = NeuralNetwork(args=args)
    model.fit(train,dev,train_evi,dev_evi)

def test_model(test,test_evi):
    path = task_dic[args.task]
    model = NeuralNetwork(args=args)
    model.load_model(args.save_path)
    model.evaluate(test,test_evi, is_test=True)



if __name__ == '__main__':
    start = time.time()
    with open('ubuntu_data/dataset_1M.pkl', 'rb') as f:
        train, dev, test = pickle.load(f, encoding='ISO-8859-1')
    with open('ubuntu_data/evidence_bert_dataset1_1_1.pkl', 'rb') as f:
        train_evi, dev_evi, test_evi = pickle.load(f, encoding='ISO-8859-1')

    k = 100000
    l = 50000
    train['cr'] = train['cr'][:k]
    train['y'] = train['y'][:k]

    dev['cr'] = dev['cr'][:l]
    dev['y'] = dev['y'][:l]

    test['cr'] = test['cr'][:l]
    test['y'] = test['y'][:l]

    train_evi=train_evi[:k]
    dev_evi=dev_evi[:l]
    test_evi=test_evi[:l]

    if args.is_training:
        train_model(train,dev,train_evi,dev_evi)
        test_model(test,test_evi)
    #else:
     #   test_model()
        # test_adversarial()
    end = time.time()
    print("use time: ", (end - start) / 60, " min")




