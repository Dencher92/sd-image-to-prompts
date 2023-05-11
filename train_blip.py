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


import warnings; warnings.filterwarnings("ignore")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TOKENIZERS_PARALLELISM'] = "False"


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class CustomDataset(Dataset):
    def __init__(self, image_list, prompt_list, processor):
        self.image_list = image_list
        self.prompt_list = prompt_list
        self.processor = processor

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        image = Image.open(image_path).convert("RGB")
        prompt = self.prompt_list[index]
        item = self.processor(images=image, text=prompt, padding="max_length", return_tensors="pt")
        return {k:v.squeeze() for k,v in item.items()}


def train_one_epoch(
        model,
        optimizer,
        scheduler,
        dataloader,
        device,
        epoch,
        n_accumulate=1,
        epoch_length=1000
    ):
    model.train()
    dataset_size = 0
    running_loss = 0.0
    bar = tqdm(range(epoch_length), total=epoch_length)
    for step in bar:
        data = next(iter(dataloader))
        input_ids = data['input_ids'].to(device)
        pixel_values = data['pixel_values'].to(device)
        batch_size = input_ids.size(0)
        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
        loss = outputs.loss
        loss = loss / n_accumulate
        loss.backward()
        if (step + 1) % n_accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None: scheduler.step()
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, dataloader, device, epoch, optimizer):
    model.eval()
    dataset_size = 0
    running_loss = 0.0
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        input_ids = data['input_ids'].to(device)
        pixel_values = data['pixel_values'].to(device)
        batch_size = input_ids.size(0)
        outputs = model(input_ids=input_ids, pixel_values=pixel_values, labels=input_ids)
        loss = outputs.loss
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        epoch_loss = running_loss / dataset_size
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    return epoch_loss


import numpy as np
import torch
import os
import copy

import os
import shutil

def run_training(
    model,
    optimizer,
    scheduler,
    num_epochs,
    epoch_length,
    train_loader,
    valid_loader,
    device,
    n_accumulate=1,
    patience=3,
    checkpoints_dir_path="checkpoints",
    max_saved_models=5
):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    prev_epoch_loss = np.inf
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        train_epoch_loss = train_one_epoch(
            model,
            optimizer,
            scheduler,
            dataloader=train_loader,
            device=device,
            epoch=epoch,
            n_accumulate=n_accumulate,
            epoch_length=epoch_length
        )

        val_epoch_loss = valid_one_epoch(
            model,
            valid_loader,
            device=device,
            epoch=epoch,
            optimizer=optimizer
        )

        mlflow.log_metric("train_loss", train_epoch_loss, step=epoch)
        mlflow.log_metric("val_loss", val_epoch_loss, step=epoch)

        if val_epoch_loss <= best_epoch_loss:
            print(f"Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            model_path = os.path.join(checkpoints_dir_path, f"model_{epoch}")
            print(f"Saving model to {model_path}")
            mlflow.pytorch.save_model(model, model_path)

            saved_models = sorted(os.listdir(checkpoints_dir_path), key=lambda x: int(x.split("_")[-1]))
            while len(saved_models) > max_saved_models:
                model_to_remove = saved_models.pop(0)
                shutil.rmtree(os.path.join(checkpoints_dir_path, model_to_remove))

            mlflow.log_metric("val_loss_max", best_epoch_loss, step=epoch)

        if val_epoch_loss > prev_epoch_loss:
            patience_counter += 1
        else:
            patience_counter = 0

        if patience_counter >= patience:
            print(f"Early stopping after {patience} epochs without improvement.")
            break

    print("Best Loss: {:.4f}".format(best_epoch_loss))
    model.load_state_dict(best_model_wts)

    return model


# TODO: make train/test split better, may be by thresholding on sentence-transformer similarity


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
        checkpoints_dir_path = mlflow_run.info.artifact_uri

        if args.is_cluster:
            # костыль, потому что тренька на кластере, а на млфлоу на локалке и до маунта разные пути блять...
            checkpoints_dir_path = checkpoints_dir_path.replace('/mnt/home', '/mnt/home/divashkov')
    else:
        checkpoints_dir_path = os.path.join(args.checkpoints_dir, 'tmp')

    # Set seed
    set_seed(args.seed)

    # Read index file
    index = pd.read_csv(args.index_path)

    # Because in this competition only images with 50 steps of SD are used, we can filter out the rest
    index = index[index['steps'] == 50]

    # Get rid of images with no prompt
    index = index[~index['prompt'].isna()]

    # Sample args.val_size random images and prompt
    val_index = index.sample(args.val_size)
    val_image_list = [os.path.join(args.data_path, x, y) for x, y in zip(val_index['part'], val_index['img_name'])]
    val_prompt_list = list(val_index['prompt'])

    # Sample args.train_size random images and prompts, but prompt should be different from val of first 30 characters
    train_index = index[~index['prompt'].str[:20].isin(val_prompt_list)].sample(args.train_size)
    train_image_list = [os.path.join(args.data_path, x, y) for x, y in zip(train_index['part'], train_index['img_name'])]
    train_prompt_list = list(train_index['prompt'])

    # Load preprocessors and datasets
    processor = AutoProcessor.from_pretrained(args.model_name)
    train_dataset = CustomDataset(train_image_list, train_prompt_list, processor)
    valid_dataset = CustomDataset(val_image_list, val_prompt_list, processor)

    # Set dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        num_workers=args.n_workers,
        shuffle=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        num_workers=args.n_workers,
        shuffle=False,
    )

    # Load model
    device = args.device
    print('Device:', device)

    model = BlipForConditionalGeneration.from_pretrained(args.model_name)
    model.to(device)
    # print number of parameters:
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.t_max, eta_min=args.min_lr)

    # Train
    model = run_training(
        model,
        optimizer,
        scheduler,
        num_epochs=args.epochs,
        epoch_length=args.epoch_length,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        n_accumulate=args.n_accumulate,
        patience=args.patience,
        checkpoints_dir_path=checkpoints_dir_path
    )

    # Release memory
    del train_loader, valid_loader
    _ = gc.collect()

    mlflow.end_run()


if __name__ == "__main__":
    main()
