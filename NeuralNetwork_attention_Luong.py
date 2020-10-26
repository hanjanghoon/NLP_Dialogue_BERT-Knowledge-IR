import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from Metrics import Metrics
import logging
from torch.utils.data import TensorDataset,RandomSampler
from transformers import AdamW
from transformers import BertConfig, BertModel, BertTokenizer
from tqdm import tqdm
from torch.utils.data import Dataset
import torch.nn.functional as F

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer)
}

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label,lenidx):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label = label
        self.lenidx=lenidx


class BERTDataset(Dataset):
    def __init__(self,args,train,train_evi,tokenizer):

        # load samples later lazily from disk
        self.train=train
        self.train_evi=train_evi
        self.args=args
        self.bert_tokenizer =tokenizer
    def __len__(self):
        return len(self.train['cr'])
        #return 1000

    def __getitem__(self, item):

        # transform sample to features
        cur_features = convert_examples_to_features(item,self.train,self.train_evi, self.bert_tokenizer)

        cur_tensors = (torch.tensor(cur_features.input_ids),
                       torch.tensor(cur_features.input_mask),
                       torch.tensor(cur_features.segment_ids),
                       torch.tensor(cur_features.label,dtype=torch.float),
                       torch.tensor(cur_features.lenidx)
        )

        return cur_tensors

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
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
            tokens_a.pop(0)
        else:
            tokens_b.pop()


def convert_examples_to_features(item , train, train_evi,bert_tokenizer):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # tokens_a=tokenizer.convert_tokens_to_ids("[CLS]")

    #      for context in X_train_utterances:
    #           for utterance in context:

    #        X_train_utterances,
    # label_map = {label: i for i, label in enumerate(y_train)}
    ex_index=item
    input_ids=train['cr'][item]
    evi_ids=train_evi[item]
    if ex_index % 10000 == 0:
        logger.info("Writing example %d of %d" % (ex_index, len(input_ids)))

    sep=input_ids.index(bert_tokenizer.sep_token_id)
    context=input_ids[:sep]
    response=input_ids[sep+1:]
    _truncate_seq_pair(context, response, 256)

    if len(evi_ids) > 150:
        evi_ids = evi_ids[:150]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    context_len = len(context)
    if (context_len==256) :
        print("error1")

    input_evi_ids = [bert_tokenizer.cls_token_id] + context + [bert_tokenizer.additional_special_tokens_ids[0]] + evi_ids + [bert_tokenizer.additional_special_tokens_ids[1]]\
                    +[bert_tokenizer.sep_token_id] + response  + [bert_tokenizer.sep_token_id]
    evi_len=context_len+1
    context_evi_len = 3 + context_len + len(evi_ids)
    segment_ids = [0] * (context_evi_len+1)  # 컨텍스트 다합친거.
    segment_ids += [1] * (len(input_evi_ids) - context_evi_len-1)  # #이건 리스폰스.
    #처음 1~context_len+1 전까지. contextlen+1에서 context_evi_len까지 context_evi len+1에서 끝 전까지.
    lenidx=[evi_len,context_evi_len ,len(input_evi_ids)-1]

    input_mask = [1] * len(input_evi_ids)

    # Zero-pad up to the sequence length.
    padding_length = 415 - len(input_evi_ids)

    if (padding_length > 0):
        input_ids = input_evi_ids + ([0] * padding_length)
        input_mask = input_mask + ([0] * padding_length)
        segment_ids = segment_ids + ([0] * padding_length)  # 패딩은 0이다.

    #          assert len(input_ids) == 256
    #         assert len(input_mask) == 256
    #        assert len(segment_ids) == 256

    # label_id=y_train[ex_index]
    # label_id = label_map[example.label]

    if ex_index < 1:
        logger.info("*** Example ***")
        logger.info("tokens_idx: %s" % " ".join(
            [str(x) for x in input_ids]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        # logger.info("label: %d " % (utter))

    features =InputFeatures(input_ids=input_ids,
                      input_mask=input_mask,
                      segment_ids=segment_ids,
                      label=train['y'][item],
                      lenidx=lenidx)
    return features

class NeuralNetwork(nn.Module):

    def __init__(self,args):
        super(NeuralNetwork, self).__init__()
        self.args = args
        self.patience = 0
        self.init_clip_max_norm = 5.0
        self.optimizer = None
        self.best_result = [0, 0, 0, 0, 0, 0]
        self.metrics = Metrics(self.args.score_file_path)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

        self.bert_config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            finetuning_task="classification",
            num_labels=1)

        self.bert_tokenizer = BertTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case)
        special_tokens_dict = {'eos_token': '[eos]', 'additional_special_tokens': ['[soe]','[eoe]']}
        num_added_toks = self.bert_tokenizer.add_special_tokens(special_tokens_dict)

        self.bert_model = model_class.from_pretrained(args.model_name_or_path,
                                                      from_tf=bool('.ckpt' in args.model_name_or_path),
                                                      config=self.bert_config)
        self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))

        self.bert_model = self.bert_model.cuda()

        self.attn = nn.Linear(768, 768)
        self.rnn = nn.GRU(
            input_size=768, hidden_size=200,
            num_layers=1, batch_first=True, bidirectional=False
        )
        self.bilinear=nn.Bilinear(768,768,1)
    def forward(self):
        raise NotImplementedError

    def forward_attn(self, x1, x2):
        """
        attention
        x1=T D
        x2=T D -> D
        """
        max_len = x1.size(0)#T D
        x2=x2.mean(dim=0)
        attn = self.attn(x1) # T,D
        attn_energies = attn.mm(x2.unsqueeze(1)) #T,D * D,1 --> T,1
        alpha = F.softmax(attn_energies,dim=0)  # T,1
        alpha=alpha.transpose(0,1) #1,T
        weighted_attn = alpha.mm(x1)  # 1,T * T D= 1 D

        return weighted_attn
    def batch_att_cal(self,bertoutput,lenidx):
        #hid_out,_=self.rnn(bertoutput)
        batchsize=lenidx.shape[0]
        output=torch.zeros(batchsize)
        #context = ho[0:lenidx[:][0]]
        for i in range(batchsize):
            #context_evidence=bertoutput[i,1:lenidx[i][1]]
            context_evidence=torch.cat((bertoutput[i, 1:lenidx[i][0]], bertoutput[i, lenidx[i][0] + 1:lenidx[i][1]]),dim=0)
            response=bertoutput[i,lenidx[i][1]+1:lenidx[i][2]]

            ceattn=self.forward_attn(context_evidence,response)
            rattn=self.forward_attn(response,context_evidence)
            output[i]=self.bilinear(ceattn,rattn)
            #if torch.isnan(output[i])==True:
             #   print("nan")
        return output.cuda()

    def train_step(self, i, data):
        with torch.no_grad():
            batch_ids,batch_mask,batch_seg,batch_y,batch_len = (item.cuda(device=self.device) for item in data)

        self.optimizer.zero_grad()

        output,_ =self.bert_model(batch_ids,batch_mask,batch_seg)
        output=self.batch_att_cal(output,batch_len)

        logits=torch.sigmoid(output)

        loss = self.loss_func(logits, target=batch_y)

        loss.backward()

        self.optimizer.step()
        if i%100==0:
            print('Batch[{}] - loss: {:.6f}  batch_size:{}'.format(i, loss.item(),batch_y.size(0)) )  # , accuracy, corrects
        return loss


    def fit(self, train, dev, train_evi,dev_evi):############################여기가 메인임.

        if torch.cuda.is_available(): self.cuda()

        dataset = BERTDataset(self.args, train,train_evi,self.bert_tokenizer)
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size,  sampler=sampler)

        self.loss_func = nn.BCELoss()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate,  correct_bias=True)#weight_decay=self.args.l2_reg, correct_bias=False)

        for epoch in range(self.args.epochs):
            print("\nEpoch ", epoch+1, "/", self.args.epochs)
            avg_loss = 0

            self.train()
            for i, data in tqdm(enumerate(dataloader)):#원래 배치는 200
               # torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)


                if epoch >= 2 and self.patience >= 3:
                    print("Reload the best model...")
                    self.load_state_dict(torch.load(self.args.save_path))
                    self.adjust_learning_rate()
                    self.patience = 0

                loss = self.train_step(i, data)

                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)

                avg_loss += loss.item()
            cnt = len(train['y']) // self.args.batch_size + 1
            print("Average loss:{:.6f} ".format(avg_loss/cnt))

            self.evaluate(dev,dev_evi)


    def adjust_learning_rate(self, decay_rate=.5):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            self.args.learning_rate = param_group['lr']
        print("Decay learning rate to: ", self.args.learning_rate)


    def evaluate(self, dev,dev_evi,is_test=False):
        y_pred = self.predict(dev,dev_evi)
        with open(self.args.score_file_path, 'w') as output:
            for score, label in zip(y_pred, dev['y']):
                output.write(
                    str(score) + '\t' +
                    str(label) + '\n'
                )

        result = self.metrics.evaluate_all_metrics()
        print("Evaluation Result: \n",
              "MAP:", result[0], "\t",
              "MRR:", result[1], "\t",
              "P@1:", result[2], "\t",
              "R1:",  result[3], "\t",
              "R2:",  result[4], "\t",
              "R5:",  result[5])

        if not is_test and result[3] + result[4] + result[5] > self.best_result[3] + self.best_result[4] + self.best_result[5]:
            print("Best Result: \n",
                  "MAP:", self.best_result[0], "\t",
                  "MRR:", self.best_result[1], "\t",
                  "P@1:", self.best_result[2], "\t",
                  "R1:",  self.best_result[3], "\t",
                  "R2:",  self.best_result[4], "\t",
                  "R5:",  self.best_result[5])
            self.patience = 0
            self.best_result = result
            torch.save(self.state_dict(), self.args.save_path)
            print("save model!!!\n")
        else:
            self.patience += 1


    def predict(self, dev,dev_evi):
        self.eval()
        y_pred = []
  #      for f in features:
   #         print(f.input_ids)
        dataset = BERTDataset(self.args, dev, dev_evi,self.bert_tokenizer)
        dataloader = DataLoader(dataset, batch_size=128)

        for i, data in enumerate(dataloader):
            with torch.no_grad():
                batch_ids, batch_mask, batch_seg, batch_y,batch_len= (item.cuda() for item in data)
            with torch.no_grad():
                output, _ = self.bert_model(batch_ids, batch_mask, batch_seg)
                output = self.batch_att_cal(output, batch_len)
                #for out in torch.isnan(output):
                 #   if out == True:
                  #      print(out)
                logits = torch.sigmoid(output)

            if i % 100==0:
                print('Batch[{}] batch_size:{}'.format(i, batch_ids.size(0)))  # , accuracy, corrects
            y_pred += logits.data.cpu().numpy().tolist()
        return y_pred


    def load_model(self, path):
        self.load_state_dict(state_dict=torch.load(path))
        if torch.cuda.is_available(): self.cuda()

