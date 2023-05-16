# Based on this notebook: https://www.kaggle.com/code/pastiche/blip-large-training-inference/edit

import os
import gc
import copy
import tempfile
import torch
import hydra
import numpy as np
import pandas as pd
import mlflow
from tqdm import tqdm
from PIL import Image
from torch.optim import lr_scheduler
from transformers import AutoProcessor, AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import BlipForConditionalGeneration
from omegaconf import DictConfig, OmegaConf, open_dict
from helpers.log import log_params_to_mlflow


from sentence_transformers import SentenceTransformer, models

import warnings; warnings.filterwarnings("ignore")
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ['TOKENIZERS_PARALLELISM'] = "False"


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class CustomDataset(Dataset):
    def __init__(self, prompt_list):
        self.prompt_list = prompt_list

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, index):
        prompt = self.prompt_list[index]
        return prompt


CONFIG_DIR = os.getenv('CONFIG_DIR')
CONFIG_NAME = os.getenv('CONFIG_NAME')

assert CONFIG_DIR is not None and CONFIG_DIR != '', 'CONFIG_DIR is not set'
assert CONFIG_NAME is not None and CONFIG_NAME != '', 'CONFIG_NAME is not set'

@hydra.main(config_path=CONFIG_DIR, config_name=CONFIG_NAME, version_base=None)
def main(args : DictConfig):
    print(OmegaConf.to_yaml(args))

    mlflow.set_tracking_uri(args.tracking_uri)
    if args.use_mlflow:
        experiment = mlflow.set_experiment(args.experiment_name)
        mlflow_run = mlflow.start_run(experiment_id=experiment.experiment_id)
        mlflow_run_id = mlflow_run.info.run_id
        with open_dict(args):
            args.run_id = mlflow_run_id
        log_params_to_mlflow(OmegaConf.to_container(args))

    # Set seed
    set_seed(args.seed)

    # Read index file
    index = pd.read_parquet(args.index_path)
    prompt_list = index['caption'].tolist()
    print(f"Number of samples: {len(prompt_list)}")

    # dataset = CustomDataset(prompt_list)
    # Set dataloaders
    # loader = DataLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     num_workers=args.n_workers,
    #     prefetch_factor=args.prefetch_factor,
    #     shuffle=False,
    # )

    # Load model
    model = SentenceTransformer(args.model_name, device=args.device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    print('Device:', args.device)
    # model.to(args.device)

    # Inference
    # out_array = np.zeros((len(dataset), args.embedding_size))
    # for i, batch in enumerate(tqdm(loader)):
    #     embeddings = model.encode(batch)
    #     # embeddings = embeddings.cpu().numpy() # already np in sentence_transformers
    #     out_array[i*args.batch_size:(i+1)*args.batch_size] = embeddings

    out_array = model.encode(
        prompt_list,
        batch_size=args.batch_size,
        show_progress_bar=True,
        device=args.device,
        convert_to_numpy=True,
    )

    # Save embeddings
    part_number = args.index_path.split('.')[0].split('_')[-1]
    os.makedirs(f'/mnt/home/data/laion_vit_and_setntrans/lvs_{part_number}', exist_ok=True)
    np.save(os.path.join(f'/mnt/home/data/laion_vit_and_setntrans/lvs_{part_number}', f'st.npy'), out_array)
    # also move the original from /mnt/home/data/laion/laion2b-en-vit-h-14-embeddings/text_emb/text_emb_{part_number}.npy to the same folder
    os.rename(
        os.path.join(f'/mnt/home/data/laion/laion2b-en-vit-h-14-embeddings/text_emb/text_emb_{part_number}.npy'),
        os.path.join(f'/mnt/home/data/laion_vit_and_setntrans/lvs_{part_number}', f'vit.npy')
    )

    # End run
    _ = gc.collect()
    if args.use_mlflow:
        mlflow.end_run()


if __name__ == "__main__":
    main()
