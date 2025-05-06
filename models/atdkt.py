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
    def __init__(self, question_embeddings, concept_embeddings, hidden_size,dropout=0.2, emb_type="qid", num_layers=1, num_attn_heads=6, l1=0.5, l2=0.5, start=50):
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
        if self.emb_type.find("trans") != -1:
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
        if self.emb_type.find("trans") != -1:
            mask = ut_mask(seq_len = catemb.shape[1])
            qh = self.trans(catemb.transpose(0,1), mask).transpose(0,1)
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
        aucs = []
        loss_means = []

        max_auc = 0
        writer = SummaryWriter(log_dir=ckpt_path)

        for epoch in range(1, num_epochs + 1):
            train_loss = []
            all_train_labels = []
            all_train_preds = []
            for data in train_loader:
                q = data["questions"].to(device)
                r = data["responses"].to(device)
                mask = data["selectmasks"].to(device)
                c = data["concepts"].to(device)
                c_shift = data["concepts_shift"].to(device)
                r_shift = data["responses_shift"].to(device)
                self.train()
                y,y2 = self(q.long(), r.long(), c.long(), mask, True)
                y = (y * one_hot(c_shift.long(), self.num_c)).sum(-1)
                y = torch.masked_select(y, mask)
                t = torch.masked_select(r_shift, mask)
                loss1 = binary_cross_entropy(y.double(), t.double())


                loss = self.l1*loss1 + self.l2 * y2
        
                # y = (y * one_hot(c_shift, self.num_c)).sum(-1)
                # y = torch.masked_select(y, mask)
                # t = torch.masked_select(r_shift, mask).float()
                # loss = binary_cross_entropy(y, t)
                opt.zero_grad()
                loss.backward()
                opt.step()
                all_train_labels.append(t.detach().cpu())
                all_train_preds.append(y.detach().cpu())
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
                    r_shift = data["responses_shift"].to(device)
                    c_shift = data["concepts_shift"].to(device)
                    self.eval()
                    y = self(q.long(), r.long(), c.long(), mask)
                    y = (y * one_hot(c_shift.long(), self.num_c)).sum(-1)
                    y = torch.masked_select(y, mask)
                    t = torch.masked_select(r_shift, mask)
                    loss = binary_cross_entropy(y.double(), t.double())
                    y_true.append(t.detach().cpu())
                    y_score.append(y.detach().cpu())
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
                r_shift = data["responses_shift"].to(device)
                c_shift = data["concepts_shift"].to(device)
                self.eval()
                y = self(q.long(), r.long(), c.long(), mask)
                y = (y * one_hot(c_shift.long(), self.num_c)).sum(-1)
                y = torch.masked_select(y, mask)
                t = torch.masked_select(r_shift, mask)
                loss = binary_cross_entropy(y.double(), t.double())
                y_true.append(t.detach().cpu())
                y_score.append(y.detach().cpu())
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
