import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import pathlib
import hydra
import shutil
import inspect

from src.workspace.base_workspace import BaseWorkspace
from src.model.action.fast_tokenizer import UniversalActionProcessor

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainFastTokenizerWorkspace(BaseWorkspace): 
    def __init__(self, cfg: OmegaConf):
        super().__init__(cfg)

        self.dataset = hydra.utils.instantiate(cfg.dataset)
        self.dataloader = DataLoader(self.dataset, collate_fn=self.dataset.get_collator(), **cfg.dataloader)
            
    def run(self):
        assert len(self.dataloader) > 0, "No data to calculate tokenizer"
        action_data = []
        for batch in tqdm(self.dataloader, desc="Loading tokenizer", mininterval=self.cfg.training.tqdm_interval_sec):
            action_data.append(batch['human_actions'])
        action_data = np.concatenate(action_data, axis=0)
        self.tokenizer = UniversalActionProcessor.fit(
            action_data,
            scale=self.cfg.tokenizer.scale,
            vocab_size=self.cfg.tokenizer.vocab_size,
        )
        
        if self.cfg.training.save_path is not None:
            self.save_path = pathlib.Path(self.cfg.training.save_path)
        else:
            self.save_path = pathlib.Path(self.output_dir, "tokenizer")
        self.save_fast_tokenizer(path=self.save_path)

        self.validate()

    def validate(self):
        valid_tokenizer = UniversalActionProcessor.from_pretrained(self.save_path)
        loss_list = []
        average_token_length_list = []
        with tqdm(self.dataloader, desc="Validating tokenizer", mininterval=self.cfg.training.tqdm_interval_sec) as tepoch:
            for batch in tepoch:
                action_data = batch['human_actions']
                batch_tokens = valid_tokenizer(action_data)
                decoded_actions = valid_tokenizer.decode(batch_tokens)

                loss = np.mean(np.abs(action_data - decoded_actions))
                average_token_length = np.mean([len(tokens) for tokens in batch_tokens])
                tepoch.set_postfix(loss=loss, average_token_length=average_token_length)
                loss_list.append(loss)
                average_token_length_list.append(average_token_length)
        print(f"Average L1 loss: {np.mean(loss_list)}")
        print(f"Average token length: {np.mean(average_token_length_list)}")
        
    def save_fast_tokenizer(self, path = None):
        if path is None:
            path = self.save_path
        self.tokenizer.save_pretrained(path)

        source_file_path = inspect.getfile(self.tokenizer.__class__)
        target_file_path = pathlib.Path(path).joinpath(source_file_path.split("/")[-1])
        shutil.copy(source_file_path, target_file_path)

