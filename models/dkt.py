import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
import torch.nn.functional as F


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class DKT(Module):
    '''
        Args:
            question_embeddings: pretrained questions embeddings
            concept_embeddings: pretrained concepts embeddings
            embedding_size: the dimension of the embedding 
            hidden_size: the dimension of the hidden vectors in this model
    '''
    def __init__(self, question_embeddings, concept_embeddings, embedding_size, hidden_size):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_q = question_embeddings.shape[0] + 1
        self.num_c = concept_embeddings.shape[0] + 1
        self.embedding_size = embedding_size

        self.interaction_emb = Embedding(self.num_c * 2, self.embedding_size)
        self.lstm_layer = LSTM(
            self.embedding_size, self.hidden_size, batch_first=True
        )
        self.out_layer = Linear(self.hidden_size, self.num_c)
        self.dropout_layer = Dropout()

    def forward(self, q, r, c):
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
        h, _ = self.lstm_layer(xemb)
        y = self.out_layer(h)
        y = self.dropout_layer(y)
        y = torch.sigmoid(y)

        return y

    def calculate_loss(self, q, r, c, c_shift, q_shift, r_shift, mask, train=False):
        # calculate y_pred, y_true and loss
        y = self(q.long(), r.long(), c.long())
        y = (y * one_hot(c_shift, self.num_c)).sum(-1)
        y = torch.masked_select(y, mask)
        t = torch.masked_select(r_shift, mask).float()
        loss = binary_cross_entropy(y, t)
        return y, t, loss