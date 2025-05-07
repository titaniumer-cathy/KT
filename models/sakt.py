import os

import numpy as np
import torch

from torch.nn import Module, Parameter, Embedding, Sequential, Linear, ReLU, \
    MultiheadAttention, LayerNorm, Dropout
from torch.nn.init import kaiming_normal_
from torch.nn.functional import binary_cross_entropy
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class SAKT(Module):
    '''
        Args:
            question_embeddings: pretrained questions embeddings
            concept_embeddings: pretrained concepts embeddings
            n: length of the sequence
            num_attn_heads: the number of attention heads
            dropout: the dropout rate
    '''
    def __init__(self, question_embeddings, concept_embeddings, n=199, num_attn_heads=6, dropout=0.1):
        super().__init__()
        self.num_q = question_embeddings.shape[0] + 1
        self.num_c = concept_embeddings.shape[0] + 1
        self.n = n
        self.d = question_embeddings.shape[1]
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout

        self.M = Embedding(self.num_q * 2 , self.d)
        self.C = Embedding(self.num_q, self.d)
        self.C.weights = F.pad(torch.Tensor(question_embeddings).to(device), (0, 1, 0, 0), mode='constant', value=0)
        self.P = Parameter(torch.Tensor(self.n, self.d)).to(device)

        kaiming_normal_(self.P)

        self.attn = MultiheadAttention(
            self.d, self.num_attn_heads, dropout=self.dropout
        )
        self.attn_dropout = Dropout(self.dropout)
        self.attn_layer_norm = LayerNorm(self.d)

        self.FFN = Sequential(
            Linear(self.d, self.d),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.d, self.d),
            Dropout(self.dropout),
        )
        self.FFN_layer_norm = LayerNorm(self.d)

        self.pred = Linear(self.d, 1)

    def forward(self, q, r, c, q_shift,c_shift):
        '''
            Args:
                q: the question(KC) sequence with the size of [batch_size, n]
                r: the response sequence with the size of [batch_size, n]
                c: the concept sequence with the size of [batch_size, n]
                q_shift: the question(KC) sequence with the size of [batch_size, n]. It is shift 1 position to the right compared to q
                c_shift: the concept sequence with the size of [batch_size, n]. It is shift 1 position to the right compared to c

            Returns:
                p: the knowledge level about the query
                attn_weights: the attention weights from the multi-head \
                    attention module
        '''
        x = q + self.num_q * r  # (batch, n)
        M = self.M(x).permute(1, 0, 2)
        C = self.C(q_shift).permute(1, 0, 2) # (n, batch, d)

        P = self.P.unsqueeze(1) # (n, 1, d)
        causal_mask = torch.triu(
            torch.ones([C.shape[0], M.shape[0]]), diagonal=1
        ).bool().to(device)
        M = M + P

        S, attn_weights = self.attn(C, M, M, attn_mask=causal_mask)
        S = self.attn_dropout(S) # (n, batch, d)
        S = S.permute(1, 0, 2)# (batch,n,d)
        M = M.permute(1, 0, 2) # (batch,n,d)
        C = C.permute(1, 0, 2) # (batch,n,d)

        S = self.attn_layer_norm(S + C)

        F = self.FFN(S)
        F = self.FFN_layer_norm(F + S)

        p = torch.sigmoid(self.pred(F)).squeeze()

        return p, attn_weights


    def calculate_loss(self, q, r, c, c_shift, q_shift, r_shift, mask, train=False):
        # calculate y_pred, y_true and loss
        p, _ = self(q.long(), r.long(), c.long(), q_shift.long(), c_shift.long())
        p = torch.masked_select(p, mask)
        t = torch.masked_select(r_shift, mask).float()
        loss = binary_cross_entropy(p, t)
        return p, t, loss