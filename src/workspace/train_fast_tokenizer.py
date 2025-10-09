import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import pathlib
import hydra
import shutil
import inspect
import os
import pickle

from src.workspace.base_workspace import BaseWorkspace
from src.utils.plotting import plot_histogram
from src.model.action.fast_tokenizer import UniversalActionProcessor

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainFastTokenizerWorkspace(BaseWorkspace): 
    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        self.tokenizer_cfg = cfg.tokenizer
        self.dataset = hydra.utils.instantiate(cfg.dataset)
        if cfg.training.normalizer_path is not None:
            normalizer = pickle.load(open(cfg.training.normalizer_path, 'rb'))
            self.dataset.set_normalizer(normalizer)
        else:
            normalizer = self.dataset.get_normalizer()
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        pickle.dump(normalizer, open(normalizer_path, 'wb'))
        
        self.dataloader = DataLoader(self.dataset, collate_fn=self.dataset.get_collator(), **cfg.dataloader)

        if self.cfg.training.save_path is not None:
            self.save_path = pathlib.Path(self.cfg.training.save_path)
        else:
            self.save_path = pathlib.Path(self.output_dir, "tokenizer")
            
    def run(self):
        assert len(self.dataloader) > 0, "No data to calculate tokenizer"
        if self.cfg.training.valid_tokenizer_path is None: 
            action_data = {
                key: [] for key in self.tokenizer_cfg.keys()
            }
            # We need set normalizer for the dataset
            size = 0
            for batch in tqdm(self.dataloader, desc="Loading tokenizer", mininterval=self.cfg.training.tqdm_interval_sec):
                for key in self.tokenizer_cfg.keys():
                    assert key in batch.keys(), f"Key {key} not found in batch"
                    action_data[key].append(batch[key])
                size += 1
                if self.cfg.training.max_corpus_size and size >= self.cfg.training.max_corpus_size:
                    break
            action_data = {k: np.concatenate(v, axis=0) for k, v in action_data.items()}
            self.tokenizer = {
                key: UniversalActionProcessor.fit(
                    action_data[key],
                    scale=self.tokenizer_cfg[key].scale,
                    vocab_size=self.tokenizer_cfg[key].vocab_size,
                    num_workers=self.tokenizer_cfg[key].num_workers,
                ) for key in self.tokenizer_cfg.keys()
            }
            self.save_fast_tokenizer(path=self.save_path)

        self.validate()

    def validate(self):
        valid_tokenizer_path = self.cfg.training.valid_tokenizer_path
        if valid_tokenizer_path is None:
            valid_tokenizer_path = self.save_path
        valid_tokenizer = {
            key: UniversalActionProcessor.from_pretrained(os.path.join(valid_tokenizer_path, key))
            for key in self.tokenizer_cfg.keys()
        }
        loss_list = {
            key: [] for key in self.tokenizer_cfg.keys()
        }
        average_token_length_list = {
            key: [] for key in self.tokenizer_cfg.keys()
        }
        token_length_list = {
            key: [] for key in self.tokenizer_cfg.keys()
        }
        with tqdm(self.dataloader, desc="Validating tokenizer", mininterval=self.cfg.training.tqdm_interval_sec) as tepoch:
            for idx, batch in enumerate(tepoch):
                for key in valid_tokenizer.keys():
                    data = batch[key]
                    batch_tokens = valid_tokenizer[key](data)
                    decoded_actions = valid_tokenizer[key].decode(batch_tokens)

                    loss = np.mean(np.abs(data - decoded_actions))
                    token_length = [len(tokens) for tokens in batch_tokens]
                    average_token_length = np.mean(token_length)
                    token_length_list[key].extend(token_length)
                    tepoch.set_postfix(key=key, loss=loss, average_token_length=average_token_length)
                    loss_list[key].append(loss)
                    average_token_length_list[key].append(average_token_length)
                if self.cfg.training.max_val_steps and idx >= self.cfg.training.max_val_steps:
                    break
        
        for key in valid_tokenizer.keys():
            print(f"Key: {key}")
            print(f"Average L1 loss: {np.mean(loss_list[key])}")
            print(f"Average token length: {np.mean(average_token_length_list[key])}")
            plot_histogram(token_length_list[key], key, self.output_dir)
        
    def save_fast_tokenizer(self, path = None):
        if path is None:
            path = self.save_path
        for key in self.tokenizer_cfg.keys():
            self.tokenizer[key].save_pretrained(os.path.join(path, key))
            source_file_path = inspect.getfile(self.tokenizer[key].__class__)
            target_file_path = pathlib.Path(path).joinpath(key, source_file_path.split("/")[-1])
            shutil.copy(source_file_path, target_file_path)

