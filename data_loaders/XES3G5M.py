import numpy as np
import pandas as pd
import lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset


class XES3G5MDataModuleConfig:
    """
    Configuration for the data module.
    """

    hf_dataset_ids: dict[str, str] = {
    "sequence": "Atomi/XES3G5M_interaction_sequences",
        "content_metadata": "Atomi/XES3G5M_content_metadata",
        }
    max_seq_length: int = 200
    padding_value: int = -1
    batch_size: int = 64
    val_fold: int = 4


class XES3G5MDataset(Dataset):
    """
    Dataset class for XES3G5M dataset.
    """
    def __init__(self, seq_df: pd.DataFrame, question_embeddings: np.ndarray, concept_embeddings: np.ndarray):
        """Initializes the dataset."""
        self.seq_df = seq_df
        self.question_embeddings = question_embeddings
        self.concept_embeddings = concept_embeddings

    def __len__(self) -> int:
        return len(self.seq_df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Returns a dictionary of input tensors."""
        row = self.seq_df.iloc[idx]
        
        selectmasks = row["selectmasks"][row["selectmasks"] != -1]
        responses = row["responses"][row["responses"] != -1]
        questions = row["questions"][row["questions"] != -1]
        concepts = row["concepts"][row["concepts"] != -1]
        question_embeddings = self.question_embeddings[questions] # (padded_num_questions, emb_dim)
        concept_embeddings = self.concept_embeddings[concepts] # (padded_num_concepts, emb_dim)
        return {
            "questions": torch.LongTensor(questions),
            "concepts": torch.LongTensor(concepts),
            "question_embeddings": torch.Tensor(question_embeddings),
            "concept_embeddings": torch.Tensor(concept_embeddings),
            "selectmasks": torch.Tensor(selectmasks),
            "responses": torch.LongTensor(responses),
        }


class XES3G5MDataModule(pl.LightningDataModule):
    """DataModule class for XES3G5M dataset."""
    def __init__(self, config: XES3G5MDataModuleConfig,) -> None:
        """
        Initializes the data module.
        """
        super().__init__()
        self.hf_dataset_ids = config.hf_dataset_ids
        self.batch_size = config.batch_size
        self.val_fold = config.val_fold
        self.max_seq_length = config.max_seq_length - 1
        self.padding_value = config.padding_value
    
    def question_embedding(self) -> int:
        """
        Returns the length of the questions.
        """
        return self.question_embeddings
    
    def concept_embedding(self) -> int:
        """
        Returns the length of the questions.
        """
        return self.concept_embeddings
    
    def prepare_data(self) -> None:
        """
        Downloads the dataset.
        """
        [load_dataset(hf_dataset_id) for hf_dataset_id in self.hf_dataset_ids.values()]

    def setup(self, stage: str) -> None:
        """
        Loads the dataset.
        """
        datasets = {key: load_dataset(value) for key, value in self.hf_dataset_ids.items()}
        self.datasets = datasets

        seq_df_train_val = datasets["sequence"]["train"].to_pandas()
        val_indices = seq_df_train_val["fold"] == self.val_fold
        self.seq_df_val = seq_df_train_val[val_indices]
        self.seq_df_train = seq_df_train_val[~val_indices]
        self.seq_df_test = datasets["sequence"]["test"].to_pandas()

        question_content_df = datasets["content_metadata"]["question"].to_pandas()
        concept_content_df = datasets["content_metadata"]["concept"].to_pandas()
        self.question_embeddings = np.array([np.array(x) for x in question_content_df["embeddings"].values])
        self.concept_embeddings = np.array([np.array(x) for x in concept_content_df["embeddings"].values])

        if stage == "fit" or stage is None:
            self.train_dataset = XES3G5MDataset(
                seq_df=self.seq_df_train,
                question_embeddings=self.question_embeddings,
                concept_embeddings=self.concept_embeddings,
                )
            self.val_dataset = XES3G5MDataset(
                seq_df=self.seq_df_val,
                question_embeddings=self.question_embeddings,
                concept_embeddings=self.concept_embeddings,
            )
        if stage == "test" or stage is None:
            self.test_dataset = XES3G5MDataset(
                seq_df=self.seq_df_test,
                question_embeddings=self.question_embeddings,
                concept_embeddings=self.concept_embeddings,
                )

    def _collate_fn(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        Collate function for the dataloader.
        """

        # Get list of tensors from the batch
        questions = [x["questions"][:-1] for x in batch]
        questions_shift = [x["questions"][1:] for x in batch]
        concepts = [x["concepts"][:-1] for x in batch]
        concepts_shift = [x["concepts"][1:] for x in batch]
        selectmasks = [x["selectmasks"][1:] for x in batch]
        responses = [x["responses"][:-1] for x in batch]
        responses_shift = [x["responses"][1:] for x in batch]

        # Get the maximum sequence length in this batch
        max_len = max(x.shape[0] for x in questions)
        max_len = min(max_len, self.max_seq_length) # Cap at max_seq_length if needed

        # Pad the sequences if not done already
        for i in range(len(questions)):
            seq_len = questions[i].shape[0]
            if seq_len < max_len:
                questions[i] = torch.nn.functional.pad(questions[i], (0, max_len - seq_len), value=7652)
                questions_shift[i] = torch.nn.functional.pad(questions_shift[i], (0, max_len - seq_len), value=7652)
                concepts[i] = torch.nn.functional.pad(concepts[i], (0, max_len - seq_len), value=1175)
                selectmasks[i] = torch.nn.functional.pad(selectmasks[i], (0, max_len - seq_len), value=self.padding_value)
                concepts_shift[i] = torch.nn.functional.pad(concepts_shift[i], (0, max_len - seq_len), value=1175)
                responses[i] = torch.nn.functional.pad(responses[i], (0, max_len - seq_len), value=self.padding_value)
                responses_shift[i] = torch.nn.functional.pad(responses_shift[i], (0, max_len - seq_len), value=self.padding_value)
            else:
                questions[i] = questions[i][:max_len]
                concepts[i] = concepts[i][:max_len]
                questions_shift[i] = questions_shift[i][:max_len]
                concepts_shift[i] = concepts_shift[i][:max_len]
                selectmasks[i] = selectmasks[i][:max_len]
                responses_shift[i] = responses_shift[i][:max_len]
                responses[i] = responses[i][:max_len]
            selectmasks[i] = (questions[i]!= 7652) * (questions_shift[i]!= 7652)
        # Stack the tensors
        stacked_questions = torch.stack(questions) # (batch_size, max_seq_length)
        stacked_concepts = torch.stack(concepts) # (batch_size, max_seq_length)
        stacked_questions_shift = torch.stack(questions_shift) # (batch_size, max_seq_length)
        stacked_concepts_shift = torch.stack(concepts_shift) # (batch_size, max_seq_length)
        stacked_selectmasks = torch.stack(selectmasks) # (batch_size, max_seq_length)
        stacked_responses = torch.stack(responses) # (batch_size, max_seq_length)
        stacked_responses_shift = torch.stack(responses_shift) # (batch_size, max_seq_length)

        # Replace padding value with 0 for responses
        stacked_responses[stacked_responses == self.padding_value] = 0

        return {
            "questions": stacked_questions,
            "concepts": stacked_concepts,
            "questions_shift": stacked_questions_shift,
            "concepts_shift": stacked_concepts_shift,
            "selectmasks": stacked_selectmasks,
            "responses": stacked_responses,
            "responses_shift": stacked_responses_shift
        }

    def train_dataloader(self) -> DataLoader: # type: ignore[type-arg]
        """
        Returns the training dataloader.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader: # type: ignore[type-arg]
        """
        Returns the validation dataloader.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> DataLoader: # type: ignore[type-arg]
        """
        Returns the test dataloader.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )