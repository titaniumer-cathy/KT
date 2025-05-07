import os

import numpy as np
import torch
from torch import nn

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn import LayerNorm, TransformerEncoder, TransformerEncoderLayer, CrossEntropyLoss

from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def ut_mask(seq_len):
    """ Upper Triangular Mask
    """
    return torch.triu(torch.ones(seq_len,seq_len),diagonal=1).to(dtype=torch.bool).to(device)


class ATDKT(Module):
    '''
        Args:
            question_embeddings: pretrained questions embeddings
            concept_embeddings: pretrained concepts embeddings
            hidden_size: the dimension of the hidden vectors in this model
    '''
    def __init__(self, question_embeddings, concept_embeddings, hidden_size, dropout=0.2, emb_type="qid", num_layers=1, num_attn_heads=6, l1=0.5, l2=0.5, start=50):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_q = question_embeddings.shape[0] + 1
        self.num_c = concept_embeddings.shape[0] + 1
        self.emb_size = question_embeddings.shape[1]
        self.emb_type = emb_type
        self.embedding_size = concept_embeddings.shape[1]

        self.interaction_emb = Embedding(self.num_c * 2, self.embedding_size)

        self.question_emb = Embedding(self.num_q, question_embeddings.shape[1])
        self.concept_emb = Embedding(self.num_c, concept_embeddings.shape[1])
        self.question_emb.weights = F.pad(torch.Tensor(question_embeddings).to(device), (0, 1, 0, 0), mode='constant', value=0)
        self.concept_emb.weights = F.pad(torch.Tensor(concept_embeddings).to(device), (0, 1, 0, 0), mode='constant', value=0)

        self.lstm_layer = LSTM(
            self.embedding_size, self.hidden_size, batch_first=True
        )

        self.dropout_layer = Dropout(dropout)
        self.out_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size//2), nn.ReLU(), nn.Dropout(dropout),
                Linear(self.hidden_size//2, self.num_c))

        self.start = start
        self.l1 = l1
        self.l2 = l2
        if self.emb_type=="trans":
            self.nhead = num_attn_heads
            d_model = self.hidden_size# * 2
            encoder_layer = TransformerEncoderLayer(d_model, nhead=self.nhead)
            encoder_norm = LayerNorm(d_model)
            self.trans = TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)
        else:    
            self.qlstm = LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.qclasifier = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size//2), nn.ReLU(), nn.Dropout(dropout),
                Linear(self.hidden_size//2, self.num_c))
        self.closs = CrossEntropyLoss()


    def forward(self, q, r, c, mask, train=False):
        '''
            Args:
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]
                c:  the concept sequence with the size of [batch_size, n]
            Returns:
                y: the knowledge level about the all questions(KCs)
        '''
        x = c + self.num_c * r
        xemb = self.interaction_emb(x)
        qemb = self.question_emb(q)
        cemb = self.concept_emb(c)
        
        catemb = xemb + qemb + cemb
        if self.emb_type=="trans":
            sm = ut_mask(seq_len = catemb.shape[1])
            qh = self.trans(catemb.transpose(0,1), sm).transpose(0,1)
        else:
            qh, _ = self.qlstm(catemb)
        if train:
            sm = mask.long()
            start = 0
            cpreds = self.qclasifier(qh[:,start:,:])
            flag = sm[:,start:]==1
            y2 = self.closs(cpreds[flag], c[:,start:][flag])
        
        # predict response
        xemb = xemb + qh + cemb + qemb
        h, _ = self.lstm_layer(xemb)

        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y)

        # predict response
        h = self.dropout_layer(h)
        y = self.out_layer(h)
        y = torch.sigmoid(y)

        if train:
            return y, y2
        else:
            return y

    def calculate_loss(self, q, r, c, c_shift, q_shift, r_shift, mask, train=False):
        # calculate y_pred, y_true and loss
        if train:
            y,y2 = self(q.long(), r.long(), c.long(), mask, True)
        else:
            y = self(q.long(), r.long(), c.long(), mask, False)
        y = (y * one_hot(c_shift.long(), self.num_c)).sum(-1)
        y = torch.masked_select(y, mask)
        t = torch.masked_select(r_shift, mask)
        loss1 = binary_cross_entropy(y.double(), t.double())
        if train:
            loss = self.l1*loss1 + self.l2 * y2
            return y, t, loss
        else:
            return y, t, loss1