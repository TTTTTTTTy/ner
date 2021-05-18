'''
Based on 'https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#bi-lstm-conditional-random-field-discussion'
We modify the code so that it can process multiple sentences at the same time for faster training and predicting
'''
from tokenizers import Tokenizer
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

from transformers import BertTokenizer, BertModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from tagger import Tagger

torch.manual_seed(1)


# Compute log sum exp in a numerically stable way for the forward algorithm


class BERT_CRF(nn.Module):

    def __init__(self, device, tagset_size, bert_model_path='/data/tyc/data/bert/'):
        super(BERT_CRF, self).__init__()
        self.tagset_size = tagset_size + 2  # add [START_TAG]、[STOP_TAG]
        self.device = device
        self.START_TAG = tagset_size
        self.STOP_TAG =  tagset_size + 1

        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path, do_lower_case = True)
        self.bert_model = BertModel.from_pretrained(bert_model_path)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(768, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.  
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))  

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.START_TAG, :] = -10000
        self.transitions.data[:, self.STOP_TAG] = -10000
    
    # def print_args(self):
    #     return f'embedding dim:{self.embedding_dim}\thidden_dim:{self.hidden_dim}\ttarget_size:{self.tagset_size}' \
    #         + f'\tnum_layer:{self.num_layer}\tbidirectional:{self.bidirectional}\tdropout:{self.dropout}'
    
    def log_sum_exp(self, vec): # (B, T, T)
        max_score, _ = torch.max(vec, dim=-1) # [B, T]
        max_score_broadcast = max_score.unsqueeze(-1).expand(vec.size()) # [B, T, T]
        return max_score + \
            torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=-1))  # [B, T]

    def _forward_alg(self, scores, lengths):
        B, L, T = scores.size()
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((B, self.tagset_size), -10000.).to(self.device)
        # START_TAG has all of the score.
        init_alphas[:,self.START_TAG] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas  # 每一行： [log(e^p11 + e^p12 ... + e^p1n), log(e^p21 + e^p22 ... + e^p2n), ,,, log(e^pm1 + e^pm2 ... + e^pmn)
                                   # [当前位置tag为1的路径概率和，  当前位置tag为2的路径概率和， ... , 当前位置tag为m的路径概率和]

        # Iterate through the sentence
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()
            emit_score = scores[:batch_size_t, step].unsqueeze(2).repeat(1, 1, T)   # [batch_size_t, T] -> [batch_size_t, T, T]   [[ x1 x1 x1], [x2, x2, x2], [x3, x3, x3]]
            expaned_forward_var = forward_var[:batch_size_t].unsqueeze(1).repeat(1, T, 1) # [batch_size_t, T] -> [batch_size_t, T, T]    [[ p1 p2 p3], [p1, p2, p3], [p1, p2, p3]]
            next_tag_var = expaned_forward_var + self.transitions + emit_score   # [batch_size_t, T, T] entry(k, i, j) = p_j + t_ji + x_i 
            forward_var[:batch_size_t] = self.log_sum_exp(next_tag_var)
        terminal_var = forward_var + self.transitions[self.STOP_TAG] # [B, T]
        alpha = self.log_sum_exp(terminal_var) # [B]
        return alpha

    def _get_bert_scores(self, sents):
        # convert token to id
        token_ids = [[self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(sent) + [self.tokenizer.sep_token_id] for sent in sents]
        # pad to max length
        for i in range(1, len(token_ids)):
            token_ids[i] += [self.tokenizer.pad_token_id] * (len(token_ids[0]) - len(token_ids[i]))
        # words = self.tokenizer.batch_encode_plus(sents, is_split_into_words=True, padding=True)
        # input_ids, token_type_ids, attention_mask = torch.tensor(words['input_ids']).to(self.device), \
        #     torch.tensor(words['token_type_ids']).to(self.device), torch.tensor(words['attention_mask']).to(self.device)
        final_layer = self.bert_model(torch.tensor(token_ids).to(self.device))[0]
        encoded = final_layer[:, 1:final_layer.size(1)-1 ,:]
        tag_scores = self.hidden2tag(encoded) # [B, L, tag_size]
        return tag_scores

    def _score_sentence(self, scores, lengths, tags):
        # Gives the score of a provided tag sequence
        B, L, T = scores.size()
        gold_scores = torch.zeros(B).to(self.device)
        tags = torch.cat([torch.tensor([self.START_TAG], dtype=torch.long).expand((B, 1)).to(self.device), tags], dim=-1) # [B, L+1]
        for i in range(L):
            batch_size_t = (lengths > i).sum().item()
            t_tags = tags[:batch_size_t, i+1].view(-1, 1)  # [B,1], reshape for gather
            gold_scores[:batch_size_t] += scores[:batch_size_t, i].gather(1, t_tags).view(-1)  # emit_score
            for j in range(batch_size_t):
                gold_scores[j] += self.transitions[tags[j, i + 1], tags[j, i]]     # 有什么函数可以直接进行矩阵操作吗？
        last_tags = tags.gather(1, (lengths-1).view(-1, 1)).view(-1)
        gold_scores += self.transitions[self.STOP_TAG].gather(0, last_tags)
        return gold_scores

    def _viterbi_decode(self, scores, lengths):

        B, L, T = scores.size()   # [B, L, tag_size]
        backpointers = torch.full((B, L, T), self.STOP_TAG).to(self.device)

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((B, self.tagset_size), -10000.).to(self.device)
        init_vvars[:, self.START_TAG] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()
            # next_tag_var[k, i, j]  =  k_th example:  previous[j] + t_ji
            next_tag_var = forward_var[:batch_size_t].unsqueeze(1).repeat(1, T, 1) + self.transitions  # [batch_size_t, T, T] 
            # max_j(previous[j] + t_ji)
            max_scores, best_tag_ids = torch.max(next_tag_var, dim=-1)
            forward_var[:batch_size_t] = max_scores + scores[:batch_size_t, step]
            backpointers[:batch_size_t, step] = best_tag_ids
        path_scores = []
        best_paths = []
        terminal_var = forward_var + self.transitions[self.STOP_TAG]
        _, last_best_ids = torch.max(terminal_var, dim=1)
        backpointers = backpointers.tolist()
        for i in range(B):
            best_tag_id = last_best_ids[i].item()
            best_path = [best_tag_id]
            path_scores.append(terminal_var[i][best_tag_id])
            for j in range(lengths[i]-1 , 0, -1): # from back to front, ignore step 0(start tag)
                best_tag_id = backpointers[i][j][best_tag_id]
                best_path.append(best_tag_id)
            best_path.reverse()
            best_paths.append(best_path)
        return path_scores, best_paths
    
    def neg_log_likelihood(self, sentences, lengths, tags):
        bert_scores = self._get_bert_scores(sentences)
        forward_score = self._forward_alg(bert_scores, lengths)  # log_sum_exp of probabilities of all possibile paths
        gold_score = self._score_sentence(bert_scores, lengths, tags) # log_exp of probabilitt of the correct path
        return (forward_score - gold_score).mean()

    def forward(self, sentences, lengths):  
        # Get the emission scores from the BiLSTM
        bert_scores = self._get_bert_scores(sentences)
        
        # Find the best path, given the features.
        scores, tag_seqs = self._viterbi_decode(bert_scores, lengths)
        return scores, tag_seqs



def toy_test():
    # Make up some training data
    training_data = [(
        "the wall street journal reported today that apple corporation made money",
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia",
        "B I O O O O B".split()
    )]
    
    # get_tag_dictionary
    tag_lst = ["B", 'I', 'O']
    tagger = Tagger(tag_lst)

    # build_model
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model = BERT_CRF(device, len(tag_lst))
    model.to(device)
   
    def batch_prepare_sequence(examples, tagger, device):
        sents = []
        tag_ids_lst, lengths = tagger.convert_batch_tags_to_ids([example[1] for example in examples], padding=True)
        for example in examples:
            sents.append(model.tokenizer.tokenize(example[0]))  # Note: 因为这里的输入分词结果和通过空格分词相同，所以没有做处理，如里一个单词被分为多个wordpiece，需要对tag也进行转换
        return sents, torch.tensor(lengths, dtype=torch.long).to(device), torch.tensor(tag_ids_lst, dtype=torch.long).to(device)

    # build data 
    precheck_sent, precheck_lengths, precheck_tags = batch_prepare_sequence(training_data, tagger, device)

    # Check predictions before training
    print('\n>>> Before Training:\n')
    with torch.no_grad():
        scores, paths = model(precheck_sent, precheck_lengths)
        for i in range(len(training_data)):
            print(f'Example {i}: ')
            print(f'Question: {training_data[i][0]}')
            print(f'ground truth tags: {training_data[i][1]}')
            print(f'predict tags: {tagger.convert_ids_to_tags(paths[i])}')
            print(f'score: {scores[i].item()}')
            print('========================')


    # train model

    print('\n>>> Start Training:\n')
    epoch_num = 200
    optimizer = AdamW(model.parameters(),
                  lr = 1e-5, # args.learning_rate - default is 5e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8
                )
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = epoch_num)

    for epoch in range(epoch_num):

        model.zero_grad()
        # because this is a toy example and dataset size is small, we train them in one batch.
        loss = model.neg_log_likelihood(precheck_sent, precheck_lengths, precheck_tags)
        if (epoch + 1)% 50 == 0:
            print(f'Info: Epoch: {epoch + 1}, Loss: {loss.item()}, lr: {scheduler.get_last_lr()[0]}')
        loss.backward()
        optimizer.step()
        scheduler.step()

    # Check predictions after training
    print('\nAfter Training:\n')
    with torch.no_grad():
        scores, paths = model(precheck_sent, precheck_lengths)
        for i in range(len(training_data)):
            print(f'Example {i}: ')
            print(f'Question: {training_data[i][0]}')
            print(f'ground truth tags: {training_data[i][1]}')
            print(f'predict tags: {tagger.convert_ids_to_tags(paths[i])}')
            print(f'score: {scores[i].item()}')
            print('========================')
        # We got it!

if __name__ == '__main__':
    toy_test()