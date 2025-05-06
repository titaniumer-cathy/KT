import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
from torch.nn.functional import one_hot, binary_cross_entropy
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter
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
                r_shift = data["responses_shift"].to(device)
                c_shift = data["concepts_shift"].to(device)
                self.train()
                y = self(q.long(), r.long(), c.long())
                y = (y * one_hot(c_shift, self.num_c)).sum(-1)
                y = torch.masked_select(y, mask)
                t = torch.masked_select(r_shift, mask).float()
                opt.zero_grad()
                loss = binary_cross_entropy(y, t)
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
                    y = self(q.long(), r.long(), c.long())
                    y = (y * one_hot(c_shift, self.num_c)).sum(-1)
                    y = torch.masked_select(y, mask)
                    t = torch.masked_select(r_shift, mask).float()
                    loss = binary_cross_entropy(y, t)
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
                y = self(q.long(), r.long(), c.long())
                y = (y * one_hot(c_shift, self.num_c)).sum(-1)
                y = torch.masked_select(y, mask)
                t = torch.masked_select(r_shift, mask).float()
                loss = binary_cross_entropy(y, t)
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

