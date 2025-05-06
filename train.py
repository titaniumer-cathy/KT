import os
import argparse
import json
import pickle

import torch

from torch.optim import SGD, Adam

from data_loaders.XES3G5M import XES3G5MDataModule, XES3G5MDataModuleConfig

from models.dkt import DKT
from models.sakt import SAKT
from models.atdkt import ATDKT

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
        model.train_model(
            train_loader, val_loader, num_epochs, opt, ckpt_path
        )
    
    with open(os.path.join(ckpt_path, "aucs.pkl"), "wb") as f:
        pickle.dump(aucs, f)
    with open(os.path.join(ckpt_path, "loss_means.pkl"), "wb") as f:
        pickle.dump(loss_means, f)
    
    ### Test Model
    model.load_state_dict(torch.load(os.path.join(ckpt_path, "model.ckpt")))
    dataset_module.setup(stage="test")
    test_loader = dataset_module.test_dataloader()
    auc, loss, acc = model.test_model(test_loader)
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
