import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


class DMMN_SDCM(nn.Module):

    def __init__(self, embedding_size, embedding_dimension, embedding_matrix, hidden_size, n_hop):
        super(DMMN_SDCM, self).__init__()

        self.embedding_dimension = embedding_dimension
        self.hidden_size = hidden_size
        self.n_hop = n_hop

        self.embedding_layer = nn.Embedding(embedding_size, embedding_dimension)
        self.embedding_layer.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding_layer.weight.requires_grad = False
        self.activate = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.sen_bilstm = nn.LSTM(input_size=self.embedding_dimension, hidden_size=self.hidden_size, batch_first=True,
                                  bidirectional=True).cuda()
        self.asp_bilstm = nn.LSTM(input_size=self.embedding_dimension, hidden_size=self.hidden_size, batch_first=True,
                                  bidirectional=True).cuda()
        self.intra_asp_bilstm = nn.LSTM(input_size=self.hidden_size * 2, hidden_size=self.hidden_size, batch_first=True,
                                  bidirectional=True).cuda()

        self.sen_att11 = nn.Linear(self.hidden_size, 1).cuda()
        self.sen_att12 = nn.Linear(self.hidden_size * 2, self.hidden_size).cuda()
        self.sen_att21 = nn.Linear(self.hidden_size, 1).cuda()
        self.sen_att22 = nn.Linear(self.hidden_size * 2, self.hidden_size).cuda()
        self.asp_att1 = nn.Linear(self.hidden_size, 1).cuda()
        self.asp_att2 = nn.Linear(self.hidden_size * 2, self.hidden_size).cuda()

        self.var_linear1 = nn.Linear(self.hidden_size * 2, 2).cuda()
        self.var_linear2 = nn.Linear(self.hidden_size * 2, self.hidden_size).cuda()
        self.mean_linear1 = nn.Linear(self.hidden_size * 2, 2).cuda()
        self.mean_linear2 = nn.Linear(self.hidden_size * 2, self.hidden_size).cuda()

        self.attention_list, self.output_linear, self.transform_linear = [], [], []
        for i in range(self.n_hop):
            self.attention_list.append(nn.Linear(self.hidden_size * 5, 1).cuda())
            self.output_linear.append(nn.Linear(self.hidden_size * 2, self.hidden_size).cuda())
            self.transform_linear.append(nn.Linear(self.hidden_size, self.hidden_size).cuda())

        self.aspect_attention = nn.Linear(self.hidden_size * 2, 1).cuda()
        self.predict_linear = nn.Linear(self.hidden_size * 4, 3).cuda()
        self.loss = torch.nn.CrossEntropyLoss()

        self.initialize_weights()

    def soft_cross_entropy(self, input, target):
        """ Cross entropy that accepts soft targets
        Args:
             pred: predictions for neural network
             targets: targets, can be soft
             size_average: if false, sum is returned instead of mean

        Examples::

            input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
            input = torch.autograd.Variable(out, requires_grad=True)

            target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
            target = torch.autograd.Variable(y1)
            loss = cross_entropy(input, target)
            loss.backward()
        """
        logsoftmax = nn.LogSoftmax(dim=1)
        return torch.sum(-target * logsoftmax(input), dim=1)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.0)
                        param.chunk(4)[1].fill_(1)

    def lstm_forward(self, lstm, inputs, seq_lengths):
        sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=True)
        _, desorted_indices = torch.sort(indices, descending=False)
        inputs = inputs[indices]
        packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs, sorted_seq_lengths, batch_first=True)
        lstm.flatten_parameters()
        res, state = lstm(packed_inputs)
        padded_res, _ = nn.utils.rnn.pad_packed_sequence(res, batch_first=True)
        desorted_res = padded_res[desorted_indices]
        return desorted_res

    def forward(self, data, dropout=0.0):
        sentences = torch.tensor(data['sentences']).type(torch.cuda.LongTensor)
        mean = torch.tensor(data['mean']).type(torch.cuda.FloatTensor)
        var = torch.tensor(data['var']).type(torch.cuda.FloatTensor)
        num = torch.tensor(data['num']).type(torch.cuda.LongTensor)
        sentence_lens = torch.tensor(data['sentence_lens']).type(torch.cuda.LongTensor)

        aspects = torch.tensor(data['aspects']).type(torch.cuda.LongTensor)
        aspect_lens = torch.tensor(data['aspect_lens']).type(torch.cuda.LongTensor)
        sentences_locs = torch.tensor(data['sentence_locs']).type(torch.cuda.FloatTensor)
        aspects_locs = torch.tensor(data['aspect_locs']).type(torch.cuda.FloatTensor)
        labels = torch.tensor(data['labels']).type(torch.cuda.LongTensor)

        dropout_layer = nn.Dropout(dropout)

        sentence_inputs = self.embedding_layer(sentences)
        sentence_inputs = dropout_layer(sentence_inputs)
        sentence_outputs = self.lstm_forward(self.sen_bilstm, sentence_inputs, sentence_lens)

        batch_size = sentence_outputs.size()[0]
        max_sentence_len = sentence_outputs.size()[1]
        sentence_mask = torch.ones(batch_size, max_sentence_len).cuda()
        for i in range(batch_size):
            sentence_mask[i, sentence_lens[i]:] = 0

        sentence_outputs_flatten = sentence_outputs.view(-1, self.hidden_size * 2)

        sentence_outputs_weight1 = self.sen_att11(self.activate(self.sen_att12(sentence_outputs_flatten)))
        sentence_outputs_weight1 = sentence_outputs_weight1.view(batch_size, max_sentence_len)
        sentence_outputs_weight1 = sentence_outputs_weight1 - (1 - sentence_mask) * 1e12
        sentence_outputs_weight1 = F.softmax(sentence_outputs_weight1, dim=1).unsqueeze(-1).expand(batch_size,
                                                                                                   max_sentence_len,
                                                                                                   self.hidden_size * 2)
        weighted_sentence_outputs1 = sentence_outputs_weight1 * sentence_outputs
        weighted_sentence_outputs1 = weighted_sentence_outputs1.view(-1, max_sentence_len, self.hidden_size * 2)
        sentence_output1 = torch.sum(weighted_sentence_outputs1, dim=1)

        sentence_outputs_weight2 = self.sen_att21(self.activate(self.sen_att22(sentence_outputs_flatten)))
        sentence_outputs_weight2 = sentence_outputs_weight2.view(batch_size, max_sentence_len)
        sentence_outputs_weight2 = sentence_outputs_weight2 - (1 - sentence_mask) * 1e12
        sentence_outputs_weight2 = F.softmax(sentence_outputs_weight2, dim=1).unsqueeze(-1).expand(batch_size,
                                                                                                   max_sentence_len,
                                                                                                   self.hidden_size * 2)
        weighted_sentence_outputs2 = sentence_outputs_weight2 * sentence_outputs
        weighted_sentence_outputs2 = weighted_sentence_outputs2.view(-1, max_sentence_len, self.hidden_size * 2)
        sentence_output2 = torch.sum(weighted_sentence_outputs2, dim=1)

        pmean_vec = self.activate(self.mean_linear2(sentence_output1))
        pvar_vec = self.activate(self.var_linear2(sentence_output2))

        specific_aspects, specific_aspect_lens, specific_sentences_locs, specific_labels, specific_sentence_outputs = [], [], [], [], []
        total_num = 0
        for i in range(len(aspects)):
            specific_aspects.append(aspects[i, :num[i], :])
            specific_aspect_lens.append(aspect_lens[i, :num[i]])
            specific_sentences_locs.append(sentences_locs[i, :num[i], :, :])
            specific_labels.append(labels[i, :num[i], :])
            aspect_num = num[i].item()
            total_num += aspect_num
            specific_sentence_outputs.append(sentence_outputs[i, :, :].expand(aspect_num, max_sentence_len,
                                                                              self.hidden_size * 2))
        specific_aspects = torch.cat(specific_aspects, dim=0)
        specific_aspect_lens = torch.cat(specific_aspect_lens, dim=0)
        specific_sentences_locs = torch.cat(specific_sentences_locs, dim=0)
        specific_labels = torch.cat(specific_labels, dim=0)
        specific_sentence_outputs = torch.cat(specific_sentence_outputs, dim=0)

        aspect_inputs = self.embedding_layer(specific_aspects)
        aspect_inputs = dropout_layer(aspect_inputs)
        aspect_outputs = self.lstm_forward(self.asp_bilstm, aspect_inputs, specific_aspect_lens)

        max_aspect_len = aspect_outputs.size()[1]
        aspect_mask = torch.ones(total_num, max_aspect_len).cuda()
        for i in range(total_num):
            aspect_mask[i, specific_aspect_lens[i]:] = 0

        aspect_outputs_flatten = aspect_outputs.view(-1, self.hidden_size * 2)
        aspect_outputs_weight = self.asp_att1(self.activate(self.asp_att2(aspect_outputs_flatten)))
        aspect_outputs_weight = aspect_outputs_weight.view(total_num, max_aspect_len)
        aspect_outputs_weight = aspect_outputs_weight - (1 - aspect_mask) * 1e12
        aspect_outputs_weight = F.softmax(aspect_outputs_weight, dim=1).unsqueeze(-1).expand(total_num,
                                                                                             max_aspect_len,
                                                                                             self.hidden_size * 2)
        weighted_aspect_outputs = aspect_outputs_weight * aspect_outputs
        weighted_aspect_outputs = weighted_aspect_outputs.view(-1, max_aspect_len, self.hidden_size * 2)
        aspect_output = torch.sum(weighted_aspect_outputs, dim=1)

        e = torch.zeros([total_num, self.hidden_size]).cuda()
        scores_list = []

        for h in range(self.n_hop):
            sentences_loc = specific_sentences_locs[:, h, :max_sentence_len]
            memory = specific_sentence_outputs * sentences_loc.unsqueeze(-1).expand(total_num, max_sentence_len,
                                                                                    self.hidden_size * 2)
            attention = self.attention_list[h]
            aspect_output_expand = aspect_output.unsqueeze(1).expand(total_num, max_sentence_len,
                                                                     self.hidden_size * 2)
            aspect_output_expand = aspect_output_expand * sentences_loc.unsqueeze(-1).expand(total_num,
                                                                                             max_sentence_len,
                                                                                             self.hidden_size * 2)
            e_expand = e.unsqueeze(1).expand(total_num, max_sentence_len, self.hidden_size)
            e_expand = e_expand * sentences_loc.unsqueeze(-1).expand(total_num, max_sentence_len, self.hidden_size)
            attention_score = attention(torch.cat([memory, aspect_output_expand, e_expand], -1))
            attention_score = attention_score.squeeze(-1) - (1 - sentences_loc) * 1e12
            attention_score = F.softmax(attention_score, dim=1)
            scores_list.append(attention_score)
            i_AL = torch.sum(
                attention_score.unsqueeze(-1).expand(total_num, max_sentence_len, self.hidden_size * 2) * memory,
                dim=1)

            output = self.output_linear[h](i_AL)
            T = self.sigmoid(self.transform_linear[h](e))
            e = output * T + e * (1 - T)

        sen_asp_output, specific_mean_vec, specific_var_vec, specific_glo_vec = [], [], [], []
        cnt = 0
        for i in range(len(num)):
            cur_num = num[i].item()
            cur_e = e[cnt:cnt + cur_num, :]
            sen_asp_output.append(torch.mean(cur_e, dim=0))
            specific_mean_vec.append(pmean_vec[i].unsqueeze(0).expand(cur_num, self.hidden_size))
            specific_var_vec.append(pvar_vec[i].unsqueeze(0).expand(cur_num, self.hidden_size))
            cur_aspect_loc = aspects_locs[i, :cur_num, :cur_num]
            aspect_memory = torch.cat([cur_e.unsqueeze(0).expand(cur_num, cur_num, self.hidden_size), cur_e.unsqueeze(1).expand(cur_num, cur_num, self.hidden_size)], dim=-1)
            weighted_cur_e = aspect_memory * cur_aspect_loc.unsqueeze(
                2).expand(cur_num, cur_num, self.hidden_size * 2)
            # mothod 1: LSTM
            # intra_asp_output, _ = self.intra_asp_bilstm(weighted_cur_e)
            # glo_vec_l = []
            # for j in range(len(intra_asp_output)):
                # glo_vec_l.append(intra_asp_output[j, j, :])
            # glo_vec = torch.cat(glo_vec_l, dim=0).view(cur_num, self.hidden_size * 2)
            # method 2: attention
            glo_vec = torch.sum(
                F.softmax(self.aspect_attention(weighted_cur_e), dim=-1).expand(cur_num, cur_num,
                                                                                self.hidden_size * 2) * weighted_cur_e,
                dim=1)
            specific_glo_vec.append(glo_vec)

            cnt += cur_num
        sen_asp_output = torch.cat(sen_asp_output, dim=0).view(batch_size, self.hidden_size)
        specific_mean_vec = torch.cat(specific_mean_vec, dim=0).view(total_num, self.hidden_size)
        specific_var_vec = torch.cat(specific_var_vec, dim=0).view(total_num, self.hidden_size)
        specific_glo_vec = torch.cat(specific_glo_vec, dim=0).view(total_num, self.hidden_size * 2)

        pmean = self.mean_linear1(torch.cat([pmean_vec, sen_asp_output], dim=1))
        pvar = self.var_linear1(torch.cat([pvar_vec, sen_asp_output], dim=1))
        sentence_cost = self.soft_cross_entropy(pmean, mean) + self.soft_cross_entropy(pvar, var)
        sentence_cost = torch.mean(sentence_cost)

        predict = self.predict_linear(torch.cat([specific_mean_vec, specific_var_vec, specific_glo_vec], dim=1))
        specific_predict_labels = torch.argmax(predict, dim=1)
        specific_labels = torch.argmax(specific_labels, dim=1)
        correct_num = (specific_predict_labels.eq(specific_labels)).sum()
        cost = self.loss(predict, specific_labels)

        return cost, sentence_cost, total_num, correct_num, specific_predict_labels, specific_labels
