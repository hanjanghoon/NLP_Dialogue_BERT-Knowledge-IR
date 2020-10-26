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
import torch.nn.init as init

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer)
}

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
logger = logging.getLogger(__name__)

class TransformerBlock(nn.Module):

    def __init__(self, input_size, is_layer_norm=False):
        super(TransformerBlock, self).__init__()
        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        return self.linear2(self.relu(self.linear1(X)))

    def forward(self, Q, K, V, episilon=1e-8):
        '''
        :param Q: (batch_size, max_r_words, embedding_dim)
        :param K: (batch_size, max_u_words, embedding_dim)
        :param V: (batch_size, max_u_words, embedding_dim)
        :return: output: (batch_size, max_r_words, embedding_dim)  same size as Q
        '''
        dk = torch.Tensor([max(1.0, Q.size(-1))]).cuda()

        Q_K = Q.bmm(K.permute(0, 2, 1)) / (torch.sqrt(dk) + episilon)
        Q_K_score = F.softmax(Q_K, dim=-1)  # (batch_size, max_r_words, max_u_words)
        V_att = Q_K_score.bmm(V)

        if self.is_layer_norm:
            X = self.layer_morm(Q + V_att)  # (batch_size, max_r_words, embedding_dim)
            output = self.layer_morm(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X

        return output
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
        self.tokenizer=tokenizer
    def __len__(self):
        return len(self.train['cr'])
        #return 1000

    def __getitem__(self, item):

        # transform sample to features
        cur_features = convert_examples_to_features(item,self.train,self.train_evi, self.tokenizer)

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

    if len(evi_ids) > 250:
        evi_ids = evi_ids[:250]

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

    input_evi_ids = [bert_tokenizer.cls_token_id]+ context +evi_ids + [bert_tokenizer.sep_token_id] + response+[bert_tokenizer.sep_token_id]
    context_evi_len = 1 + context_len + len(evi_ids)
    segment_ids = [0] * (context_evi_len+1)  # 컨텍스트 다합친거.
    segment_ids += [1] * (len(input_evi_ids) - context_evi_len-1)  # #이건 리스폰스.
    #처음 1~context_len+1 전까지. contextlen+1에서 context_evi_len까지 context_evi len+1에서 끝 전까지.
    lenidx=[1+context_len,context_evi_len ,len(input_evi_ids)-1]

    input_mask = [1] * len(input_evi_ids)

    # Zero-pad up to the sequence length.
    padding_length = 512 - len(input_evi_ids)

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

        self.bert_tokenizer = BertTokenizer.from_pretrained( args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case)
        special_tokens_dict = {'eos_token': '[eos]'}
        num_added_toks = self.bert_tokenizer.add_special_tokens(special_tokens_dict)

        self.bert_model = model_class.from_pretrained(args.model_name_or_path,
                                                      from_tf=bool('.ckpt' in args.model_name_or_path),
                                                      config=self.bert_config)
        self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))

        self.bert_model = self.bert_model.cuda()
        '''
        self.attn = nn.Linear(300, 300)
        self.rnn1 = nn.GRU(
            input_size=768, hidden_size=300,
            num_layers=1, batch_first=True, bidirectional=False
        )
        self.bilinear=nn.Bilinear(600,600,1)
        '''
        # multihop
        self.transformer_utt = TransformerBlock(input_size=768)
        self.transformer_eu = TransformerBlock(input_size=768)
        self.transformer_ru = TransformerBlock(input_size=768)

        self.transformer_ett = TransformerBlock(input_size=768)
        self.transformer_ue = TransformerBlock(input_size=768)
        self.transformer_re = TransformerBlock(input_size=768)

        self.transformer_rtt = TransformerBlock(input_size=768)
        self.transformer_ur = TransformerBlock(input_size=768)
        self.transformer_er = TransformerBlock(input_size=768)

        self._projection = nn.Sequential(nn.Linear(4 * 768,200),nn.ReLU())

        self.rnn2=nn.GRU(
            input_size=200, hidden_size=200,
            num_layers=1, batch_first=True, bidirectional=True
        )

        self._classification = nn.Sequential(nn.Dropout(p=0.2),
                                             nn.Linear(2 * 6 *200, 200),
                                             nn.Tanh(),
                                             nn.Dropout(p=0.2),
                                             nn.Linear(200,1))
    def forward(self):
        raise NotImplementedError



    def get_Matching_Map(self, bU_embedding, bE_embedding,bR_embedding,umask,emask,rmask):
        '''
        :param bU_embedding: (batch_size*max_utterances, max_u_words, embedding_dim)
        :param bR_embedding: (batch_size*max_utterances, max_r_words, embedding_dim)
        :return: E: (bsz*max_utterances, max_u_words, max_r_words)
        '''
        #1셀프,2크로스,3셀프-크로스4,셀프-크로스 elementwise product
        Hutt = self.transformer_utt(bU_embedding, bU_embedding, bU_embedding)
        Hue = self.transformer_ue(bU_embedding, bE_embedding, bE_embedding)
        Hur = self.transformer_ur(bU_embedding, bR_embedding, bR_embedding)

        Hett = self.transformer_ett(bE_embedding, bE_embedding, bE_embedding)
        Heu = self.transformer_eu(bE_embedding, bU_embedding, bU_embedding)
        Her = self.transformer_er(bE_embedding, bR_embedding, bR_embedding)

        Hrtt = self.transformer_rtt(bR_embedding, bR_embedding, bR_embedding)
        Hru = self.transformer_ru(bR_embedding, bU_embedding, bU_embedding)
        Hre = self.transformer_re(bR_embedding, bE_embedding, bE_embedding)


        #utterance
        ue_input=torch.cat((Hutt,Hue,Hutt-Hue,Hutt*Hue),dim=-1)
        ur_input=torch.cat((Hutt,Hur,Hutt-Hur,Hutt*Hur),dim=-1)
        #evidence
        eu_input = torch.cat((Hett, Heu, Hett - Heu, Hett * Heu),dim=-1)
        er_input = torch.cat((Hett, Her, Hett - Her, Hett * Her),dim=-1)
        #response
        ru_input = torch.cat((Hrtt, Hru, Hrtt - Hru, Hrtt * Hru),dim=-1)
        re_input = torch.cat((Hrtt, Hre, Hrtt - Hre, Hrtt * Hre),dim=-1)

        ue_input = self._projection(ue_input)
        ur_input = self._projection(ur_input)
        eu_input = self._projection(eu_input)
        er_input = self._projection(er_input)
        ru_input = self._projection(ru_input)
        re_input = self._projection(re_input)

        ue_output,_= self.rnn2(ue_input)
        ur_output,_ = self.rnn2(ur_input)
        eu_output,_ = self.rnn2(eu_input)
        er_output,_ = self.rnn2(er_input)
        ru_output,_ = self.rnn2(ru_input)
        re_output,_ = self.rnn2(re_input)
        '''
        ue_output= ue_output.masked_select(umask)
        ur_output= ur_output.masked_select(umask)
        eu_output= emask
        er_output= emask
        ru_output= rmask
        re_output= rmask
        '''
        maxue,_=ue_output.max(dim=1)
        maxur,_ = ur_output.max(dim=1)
        maxeu,_ = eu_output.max(dim=1)
        maxer,_ = er_output.max(dim=1)
        maxru,_ = ru_output.max(dim=1)
        maxre,_ = re_output.max(dim=1)

        umask=umask.sum(dim=1,keepdim=True)
        emask=emask.sum(dim=1,keepdim=True)
        rmask=rmask.sum(dim=1,keepdim=True)

        meanue = ue_output.sum(dim=1)/umask
        meanur = ur_output.sum(dim=1)/umask
        meaneu = eu_output.sum(dim=1)/emask
        meaner = er_output.sum(dim=1)/emask
        meanru = ru_output.sum(dim=1)/rmask
        meanre = re_output.sum(dim=1)/rmask

        v = torch.cat([maxue+maxur,meanue+meanur ,maxeu+maxer, meaneu+meaner, maxru+maxre, meanru+meanre], dim=1)  # (bsz*max_utterances, channel, max_u_words, max_r_words)

        logits = self._classification(v)
        return logits.squeeze()

    def batch_att_cal(self,bertoutput,lenidx):
        batchsize=lenidx.shape[0]
        output=torch.zeros(batchsize)
        c_arr = torch.zeros((batchsize, 256,768), dtype= torch.float32)
        e_arr = torch.zeros((batchsize, 250,768), dtype=torch.float32)
        r_arr = torch.zeros((batchsize, 150,768), dtype=torch.float32)
        c_mask = torch.zeros((batchsize, 256),dtype=torch.float32)
        e_mask = torch.zeros((batchsize, 250),dtype= torch.float32)
        r_mask = torch.zeros((batchsize, 150),dtype= torch.float32)

        #context = ho[0:lenidx[:][0]]
        for i in range(batchsize):
            c_arr[i, :lenidx[i][0]-1]=bertoutput[i,1:lenidx[i][0]]
            c_mask[i, :lenidx[i][0] - 1] = 1
            e_arr[i, :lenidx[i][1] - lenidx[i][0]] = bertoutput[i, lenidx[i][0]:lenidx[i][1]]
            e_mask[i, :lenidx[i][1] - lenidx[i][0]] = 1
            r_arr[i, :lenidx[i][2] - lenidx[i][1]-1] = bertoutput[i,lenidx[i][1]+1:lenidx[i][2]][:150]
            r_mask[i, :lenidx[i][2] - lenidx[i][1]-1] = 1

        c_arr, e_arr, r_arr = c_arr.cuda(), e_arr.cuda(), r_arr.cuda()
        c_mask, e_mask,r_mask = c_mask.cuda(),e_mask.cuda(), r_mask.cuda()

        logit=self.get_Matching_Map(c_arr,e_arr,r_arr,c_mask,e_mask,r_mask)
        '''
        hc,c=self.rnn(c_arr)
        _,e=self.rnn(e_arr)
        hr, r = self.rnn(r_arr)
        ceattn = self.forward_attn(hc, e,c_mask)
        crattn = self.forward_attn(hc, r,c_mask)
        cattn = torch.cat([ceattn, crattn], dim=2)
        reattn = self.forward_attn(hr, e,r_mask)
        rcattn = self.forward_attn(hr, r,r_mask)
        rattn = torch.cat([reattn, rcattn], dim=2)
        output=self.bilinear(cattn, rattn)
        '''


        return logit

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
                logits = torch.sigmoid(output)

            if i % 100==0:
                print('Batch[{}] batch_size:{}'.format(i, batch_ids.size(0)))  # , accuracy, corrects
            y_pred += logits.data.cpu().numpy().tolist()
        return y_pred


    def load_model(self, path):
        self.load_state_dict(state_dict=torch.load(path))
        if torch.cuda.is_available(): self.cuda()

