import os
import argparse
import json
import pickle

import torch
import numpy as np
from torch.optim import SGD, Adam
from sklearn import metrics

from data_loaders.XES3G5M import XES3G5MDataModule, XES3G5MDataModuleConfig

from models.dkt import DKT
from models.sakt import SAKT
from models.atdkt import ATDKT
from torch.utils.tensorboard import SummaryWriter


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def train_model(
        model, train_loader, val_loader, num_epochs, opt, ckpt_path
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
            q_shift = data["questions_shift"].to(device)
            r_shift = data["responses_shift"].to(device)
            c_shift = data["concepts_shift"].to(device)
            model.train()
            y, t, loss = model.calculate_loss(q, r, c, c_shift, q_shift, r_shift, mask, True)
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
                q_shift = data["questions_shift"].to(device)
                r_shift = data["responses_shift"].to(device)
                c_shift = data["concepts_shift"].to(device)
                model.eval()
                y, t, loss = model.calculate_loss(q, r, c, c_shift, q_shift, r_shift, mask)
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
                    model.state_dict(),
                    os.path.join(
                        ckpt_path, "model.ckpt"
                    )
                )
                max_auc = val_auc

            aucs.append(val_auc)
            loss_means.append(loss_mean)

    return aucs, loss_means


def test_model(model, test_loader):
    
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
            model.eval()
            y, t, loss = model.calculate_loss(q, r, c, c_shift, q_shift, r_shift, mask)
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


def main(model_name, dataset_name):
    if not os.path.isdir("ckpts"):
        os.mkdir("ckpts")

    ckpt_path = os.path.join("ckpts", model_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, dataset_name)
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)
        model_config = config[model_name]
        train_config = config["train_config"]

    num_epochs = train_config["num_epochs"]
    learning_rate = train_config["learning_rate"]
    optimizer = train_config["optimizer"]  # can be [sgd, adam]
    

    dataset_module = XES3G5MDataModule(config=XES3G5MDataModuleConfig())
    dataset_module.prepare_data()
    dataset_module.setup(stage="fit")

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)
    with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)

    if model_name == "dkt":
        model = DKT(dataset_module.question_embedding(),dataset_module.concept_embedding(), **model_config).to(device)
    elif model_name == "sakt":
        model = SAKT(dataset_module.question_embedding(),dataset_module.concept_embedding(), **model_config).to(device)
    elif model_name == "atdkt":
        model = ATDKT(dataset_module.question_embedding(),dataset_module.concept_embedding(), **model_config).to(device)
    else:
        print("The wrong model name was used...")
        return

    train_loader = dataset_module.train_dataloader()
    val_loader = dataset_module.val_dataloader()


    if optimizer == "sgd":
        opt = SGD(model.parameters(), learning_rate, momentum=0.9)
    elif optimizer == "adam":
        opt = Adam(model.parameters(), learning_rate)

    aucs, loss_means = \
        train_model(
            model,train_loader, val_loader, num_epochs, opt, ckpt_path
        )
    
    with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
        pickle.dump(aucs, f)
    with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
        pickle.dump(loss_means, f)
    
    ### Test Model
    model.load_state_dict(torch.load(os.path.join(ckpt_path, "model.ckpt")))
    dataset_module.setup(stage="test")
    test_loader = dataset_module.test_dataloader()
    auc, loss, acc = test_model(model, test_loader)
    print("Test AUC: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}".format(auc, loss, acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="sakt",
        help="The name of the model to train. \
            The possible models are in [dkt, atdkt, sakt]. \
            The default model is dkt."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="XES3G5M",
        help="The name of the dataset to use in training. \
            The possible datasets are in \
            [XES3G5M]"
    )
    args = parser.parse_args()

    main(args.model_name, args.dataset_name)
