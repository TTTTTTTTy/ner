'''
Based on 'https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#bi-lstm-conditional-random-field-discussion'
We modify the code so that it can process multiple sentences at the same time for faster training and predicting
'''
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.optim as optim

torch.manual_seed(1)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
PAD_TAG = "<PAD>"

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec): # (B, T, T)
    max_score, _ = torch.max(vec, dim=-1) # [B, T]
    max_score_broadcast = max_score.unsqueeze(-1).expand(vec.size()) # [B, T, T]
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=-1))  # [B, T]

class BiLSTM_CRF(nn.Module):

    def __init__(self, device, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.bilstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=1, batch_first=True, 
                            bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(2 * hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.  
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))  

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    def _forward_alg(self, scores, lengths):
        B, L, T = scores.size()
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((B, self.tagset_size), -10000.).to(self.device)
        # START_TAG has all of the score.
        init_alphas[:,self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas  # 每一行： [log(e^p11 + e^p12 ... + e^p1n), log(e^p21 + e^p22 ... + e^p2n), ,,, log(e^pm1 + e^pm2 ... + e^pmn)
                                   # [当前位置tag为1的路径概率和，  当前位置tag为2的路径概率和， ... , 当前位置tag为m的路径概率和]

        # Iterate through the sentence
        for step in range(L):
            batch_size_t = (lengths > step).sum().item()
            emit_score = scores[:batch_size_t, step].unsqueeze(2).repeat(1, 1, T)   # [batch_size_t, T] -> [batch_size_t, T, T]   [[ x1 x1 x1], [x2, x2, x2], [x3, x3, x3]]
            expaned_forward_var = forward_var[:batch_size_t].unsqueeze(1).repeat(1, T, 1) # [batch_size_t, T] -> [batch_size_t, T, T]    [[ p1 p2 p3], [p1, p2, p3], [p1, p2, p3]]
            next_tag_var = expaned_forward_var + self.transitions + emit_score   # [batch_size_t, T, T] entry(k, i, j) = p_j + t_ji + x_i 
            forward_var[:batch_size_t] = log_sum_exp(next_tag_var)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]] # [B, T]
        alpha = log_sum_exp(terminal_var) # [B]
        return alpha

    def _get_lstm_scores(self, sents, lengths):
        embeds = self.embedding(sents) # [B, L, emb_size]
        packed = pack_padded_sequence(embeds, lengths.cpu(), batch_first=True)
        lstm_out, _ = self.bilstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  # [B, L, hidden_dize * num_direction]
        tag_scores = self.hidden2tag(lstm_out) # [B, L, tag_size]
        return tag_scores

    def _score_sentence(self, scores, lengths, tags):
        # Gives the score of a provided tag sequence
        B, L, T = scores.size()
        gold_scores = torch.zeros(B).to(self.device)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).expand((B, 1)).to(self.device), tags], dim=-1) # [B, L+1]
        for i in range(L):
            batch_size_t = (lengths > i).sum().item()
            t_tags = tags[:batch_size_t, i+1].view(-1, 1)  # [B,1], reshape for gather
            gold_scores[:batch_size_t] += scores[:batch_size_t, i].gather(1, t_tags).view(-1)  # emit_score
            for j in range(batch_size_t):
                gold_scores[j] += self.transitions[tags[j, i + 1], tags[j, i]]     # 有什么函数可以直接进行矩阵操作吗？
        last_tags = tags.gather(1, (lengths-1).view(-1, 1)).view(-1)
        gold_scores += self.transitions[self.tag_to_ix[STOP_TAG]].gather(0, last_tags)
        return gold_scores

    def _viterbi_decode(self, scores, lengths):

        B, L, T = scores.size()   # [B, L, tag_size]
        backpointers = torch.full((B, L, T), self.tag_to_ix[STOP_TAG]).to(self.device)

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((B, self.tagset_size), -10000.).to(self.device)
        init_vvars[:, self.tag_to_ix[START_TAG]] = 0

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
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
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
        lstm_scores = self._get_lstm_scores(sentences, lengths)
        forward_score = self._forward_alg(lstm_scores, lengths)  # log_sum_exp of probabilities of all possibile paths
        gold_score = self._score_sentence(lstm_scores, lengths, tags) # log_exp of probabilitt of the correct path
        return (forward_score - gold_score).mean()

    def forward(self, sentences, lengths):  
        # Get the emission scores from the BiLSTM
        lstm_scores = self._get_lstm_scores(sentences, lengths)
        
        # Find the best path, given the features.
        scores, tag_seqs = self._viterbi_decode(lstm_scores, lengths)
        return scores, tag_seqs


def toy_test():
    # Make up some training data
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia".split(),
        "B I O O O O B".split()
    )]
    
    # get_vocabulary
    word_to_ix = {'[PAD]': 0}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    # get_tag_dictionary
    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, STOP_TAG: 4, PAD_TAG: 5}
    ix_to_tag = {v:k for k, v in tag_to_ix.items()}

    # build_model
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    embedding_dim = 10
    hidden_dim = 5
    model = BiLSTM_CRF(device, len(word_to_ix), tag_to_ix, embedding_dim, hidden_dim)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    def batch_prepare_sequence(examples, word_to_idx, tag_to_idx, device):
        sents = []
        lengths = []
        tags = []
        max_length = len(examples[0][0])
        for example in examples:
            sents.append([word_to_idx[w] for w in example[0]])
            lengths.append(len(sents[-1]))
            while len(sents[-1]) < max_length:
                sents[-1].append(0)
            tags.append([tag_to_idx[tag] for tag in example[1]])
            while len(tags[-1]) < max_length:
                tags[-1].append(5)
        return torch.tensor(sents, dtype=torch.long).to(device), torch.tensor(lengths, dtype=torch.long).to(device), torch.tensor(tags, dtype=torch.long).to(device)

    # build data 
    precheck_sent, precheck_lengths, precheck_tags = batch_prepare_sequence(training_data, word_to_ix, tag_to_ix, device)

    def get_tags_from_ids(tags, ix_to_tag):
        return [ix_to_tag[tag] for tag in tags]

    # Check predictions before training
    print('\n>>> Before Training:\n')
    with torch.no_grad():
        scores, paths = model(precheck_sent, precheck_lengths)
        for i in range(len(training_data)):
            print(f'Example {i}: ')
            print(f'Question: {training_data[i][0]}')
            print(f'ground truth tags: {training_data[i][1]}')
            print(f'predict tags: {get_tags_from_ids(paths[i], ix_to_tag)}')
            print(f'probability: {scores[i].item()}')
            print('========================')


    # train model

    print('\n>>> Start Training:\n')
    epoch_num = 300
    for epoch in range(epoch_num):

        model.zero_grad()
        # because this is a toy example and dataset size is small, we train them in one batch.
        loss = model.neg_log_likelihood(precheck_sent, precheck_lengths, precheck_tags)
        if (epoch + 1)% 100 == 0:
            print(f'Info: Epoch: {epoch + 1}, Loss: {loss.item()}')
        loss.backward()
        optimizer.step()

    # Check predictions after training
    print('\nAfter Training:\n')
    with torch.no_grad():
        scores, paths = model(precheck_sent, precheck_lengths)
        for i in range(len(training_data)):
            print(f'Example {i}: ')
            print(f'Question: {training_data[i][0]}')
            print(f'ground truth tags: {training_data[i][1]}')
            print(f'predict tags: {get_tags_from_ids(paths[i], ix_to_tag)}')
            print(f'probability: {scores[i].item()}')
            print('========================')
        # We got it!

if __name__ == '__main__':
    toy_test()