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

        self.M = Embedding(self.num_c * 2 , self.d)
        self.C = Embedding(self.num_c, self.d)
        self.C.weights = F.pad(torch.Tensor(concept_embeddings).to(device), (0, 1, 0, 0), mode='constant', value=0)
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
        x = c + self.num_c * r  # (batch, n)
        M = self.M(x).permute(1, 0, 2)
        C = self.C(c_shift).permute(1, 0, 2) # (n, batch, d)

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

    def train_model(
        self, train_loader, val_loader, num_epochs, opt, ckpt_path
    ):
        '''
            Args:
                train_loader: the PyTorch DataLoader instance for training
                val_loader: the PyTorch DataLoader instance for validation
                num_epochs: the number of epochs
                opt: the optimization to train this model
                ckpt_path: the path to save this model's parameters
        '''
        writer = SummaryWriter(log_dir=ckpt_path)
        max_auc = 0
        aucs = []
        loss_means = []
        for epoch in range(1, num_epochs + 1):
            train_loss = []
            all_train_labels = []
            all_train_preds = []

            for data in train_loader:
                q = data["questions"].to(device)
                r = data["responses"].to(device)
                mask = data["selectmasks"].to(device)
                c = data["concepts"].to(device)
                q_shift = data["questions_shift"].to(device)
                r_shift = data["responses_shift"].to(device)
                c_shift = data["concepts_shift"].to(device)
                self.train()

                p, _ = self(q.long(), r.long(), c.long(), q_shift.long(), c_shift.long())
                p = torch.masked_select(p, mask)
                t = torch.masked_select(r_shift, mask).float()

                opt.zero_grad()
                loss = binary_cross_entropy(p, t)
                loss.backward()
                opt.step()

                all_train_labels.append(t.detach().cpu())
                all_train_preds.append(p.detach().cpu())
                train_loss.append(loss.detach().cpu().numpy())

            # Calculate loss, accuracy, roc_auc
            train_labels = torch.cat(all_train_labels)
            train_preds = torch.cat(all_train_preds)

            loss_mean = np.mean(train_loss)
            train_acc = torch.sum(
                (train_preds > 0.5).long() == train_labels.long()
            ).item() / train_labels.shape[0]
            train_auc = metrics.roc_auc_score(
                y_true=train_labels, y_score=train_preds
            )
            
            # TensorBoard logging
            writer.add_scalar("Loss/Train", loss_mean, epoch)
            writer.add_scalar("Accuracy/Train", train_acc, epoch)
            writer.add_scalar("AUC/Train", train_auc, epoch)

            print(
                "Train Epoch: {},   AUC: {},   Loss Mean: {},   ACC: {}"
                .format(epoch, train_auc, loss_mean, train_acc)
            )

            with torch.no_grad():
                y_true = []
                y_score = []
                val_loss = []
                for data in val_loader:
                    q = data["questions"].to(device)
                    r = data["responses"].to(device)
                    mask = data["selectmasks"].to(device)
                    c = data["concepts"].to(device)
                    q_shift = data["questions_shift"].to(device)
                    r_shift = data["responses_shift"].to(device)
                    c_shift = data["concepts_shift"].to(device)
                    self.eval()

                    p, _ = self(q.long(), r.long(), c.long(), q_shift.long(), c_shift.long())
                    p = torch.masked_select(p, mask)
                    t = torch.masked_select(r_shift, mask).float()
                    loss = binary_cross_entropy(p, t)
                    y_true.append(t.detach().cpu())
                    y_score.append(p.detach().cpu())
                    val_loss.append(loss.detach().cpu())
            
                # Calculate loss, accuracy, roc_auc
                val_loss = np.mean(val_loss)
                val_labels = torch.cat(y_true)
                val_preds = torch.cat(y_score)
                val_acc = torch.sum(
                    (val_preds > 0.5).long() == val_labels.long()
                ).item() / val_labels.shape[0]
                val_auc = metrics.roc_auc_score(
                    y_true=val_labels, y_score=val_preds
                )
                # TensorBoard logging
                writer.add_scalar("Loss/Val", val_loss, epoch)
                writer.add_scalar("Accuracy/Val", val_acc, epoch)
                writer.add_scalar("AUC/Val", val_auc, epoch)


                print(
                    "Val Epoch: {},   AUC: {},   Loss Mean: {},    ACC: {}"
                    .format(epoch, val_auc, val_loss, val_acc)
                )

                if val_auc > max_auc:
                    torch.save(
                        self.state_dict(),
                        os.path.join(
                            ckpt_path, "model.ckpt"
                        )
                    )
                    max_auc = val_auc

                aucs.append(val_auc)
                loss_means.append(loss_mean)

        return aucs, loss_means


    def test_model(self, test_loader):
        with torch.no_grad():
            y_true = []
            y_score = []
            test_loss = []
            for data in test_loader:
                q = data["questions"].to(device)
                r = data["responses"].to(device)
                mask = data["selectmasks"].to(device)
                c = data["concepts"].to(device)
                q_shift = data["questions_shift"].to(device)
                r_shift = data["responses_shift"].to(device)
                c_shift = data["concepts_shift"].to(device)
                self.eval()

                p, _ = self(q.long(), r.long(), c.long(), q_shift.long(), c_shift.long())
                p = torch.masked_select(p, mask)
                t = torch.masked_select(r_shift, mask).float()
                loss = binary_cross_entropy(p, t)
                y_true.append(t.detach().cpu())
                y_score.append(p.detach().cpu())
                test_loss.append(loss.detach().cpu())
            
            # Calculate loss, accuracy, roc_auc
            test_loss = np.mean(test_loss)
            test_labels = torch.cat(y_true)
            test_preds = torch.cat(y_score)
            test_acc = torch.sum(
                (test_preds > 0.5).long() == test_labels.long()
            ).item() / test_labels.shape[0]
            test_auc = metrics.roc_auc_score(
                y_true=test_labels, y_score=test_preds
            )
            return test_auc, test_loss, test_acc

