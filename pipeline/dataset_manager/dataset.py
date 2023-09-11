import os
import json

import torch
from collections import Counter
import random
from torch.utils.data import Dataset

from common.base import BaseModel

random.seed(2023)


class PipelineDataset(Dataset, BaseModel):
    def __init__(self, config, mode, tokenizer, num_folds: int = None):
        super().__init__()
        self.config = config
        self.num_folds = num_folds
        self.tokenizer = tokenizer
        self.original_data = self.load_dataset(mode=mode)
        self.dynamic_batching = config.dynamic_batching
        if num_folds is None:
            self.data = self.original_data
        else:
            self.data = None
        self.logger.info(f"Load {len(self.data)} examples for {mode} dataset")

    def apply_fold(self, fold_id):
        if fold_id >= self.num_folds:
            raise f"`num_folds` must larger than `fold_id`"
        total_len = len(self.original_data)
        base_indices = total_len // self.num_folds
        fold_eval = self.original_data[fold_id * base_indices:(fold_id + 1) * base_indices]
        self.data = self.original_data[:fold_id * base_indices] + self.original_data[(fold_id + 1) * base_indices:]
        eval_data = PipelineDataset(config=self.config, mode=None, tokenizer=self.tokenizer)
        eval_data.data = fold_eval
        return eval_data

    def load_dataset(self, mode: str = None):
        dataset = []
        if mode is None:
            return dataset
        for ff in os.listdir(self.config.dataset_path):
            if not os.path.isdir(f"{self.config.dataset_path}/{ff}"):
                continue
            for f in os.listdir(f"{self.config.dataset_path}/{ff}"):
                if f.endswith(".json") and f.startswith(mode):
                    dataset += json.load(open(f"{self.config.dataset_path}/{ff}/{f}"))["data"]
        random.shuffle(dataset)
        return dataset

    def __len__(self):
        return len(self.data)

    @staticmethod
    def generate_prompt(sentence):
        raise "This func need implementation"

    @staticmethod
    def generate_label(label_id):
        raise "This func need implementation"

    def generate_training_item(self, item):
        raise "This func need implementation"

    @classmethod
    def get_dummy_example(cls, example):
        raise "This func need implementation"

    def __getitem__(self, idx):
        data_item = self.data[idx]
        return self.generate_training_item(data_item)
