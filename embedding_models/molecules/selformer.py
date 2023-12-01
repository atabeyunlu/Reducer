import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"

import pandas as pd
from pandarallel import pandarallel
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch

class SELFORMER(object):

    def __init__(self):
        
        model_name = "HUBioDataLab/SELFormer" # path of the pre-trained model
        config = RobertaConfig.from_pretrained(model_name)
        config.output_hidden_states = True
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name, config=config)

    def get_sequence_embeddings(self, selfies):
  
        token = torch.tensor([self.tokenizer.encode(selfies, add_special_tokens=True, max_length=512, padding=True, truncation=True)])

        output = self.model(token)

        sequence_out = output[0]

        return torch.mean(sequence_out[0], dim=0).tolist()
    
    def get_embeddings(self, df, save_path="/path/to/save", save=False, num_workers=1):
 
        pandarallel.initialize(nb_workers=1,progress_bar=True)
      
        df["sequence_embeddings"] = df.selfies.parallel_apply(self.get_sequence_embeddings)

        if save:
            df.to_csv(save_path, index=False) # save embeddings here

        return df
