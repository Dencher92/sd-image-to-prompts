import copy
import os
import gc
import argparse
import shutil
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
import mlflow
from omegaconf import DictConfig, OmegaConf, open_dict
from helpers.log import log_params_to_mlflow
import hydra

class MemmapDataset(Dataset):
    def __init__(self, memmap_file, start_idx, end_idx, shape, dtype='float16'):
        self.data = np.memmap(memmap_file, dtype=dtype, mode='r', shape=shape)
        self.start_idx = start_idx
        self.end_idx = end_idx

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index + self.start_idx].copy())

    def __len__(self):
        return self.end_idx - self.start_idx

class RandomSampler(Sampler):
    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(torch.randint(0, len(self.data_source), (len(self.data_source),)))

    def __len__(self):
        return len(self.data_source)


# TODO: I made the same mistake as the previous time, relu restricts the output to be positive,
# however, the input is not necessarily positive. So, I should probably use a different activation function
# May be that's why encoded embeddings showed better results on knn than decoded back

# class AutoEncoder(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, output_dim),
#             nn.ReLU(),
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(output_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, input_dim),
#             nn.ReLU(),
#         )

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return encoded, decoded


# class AutoEncoderBigger(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 768),
#             nn.ReLU(),
#             nn.Linear(768, 512),
#             nn.ReLU(),
#             nn.Linear(512, output_dim),
#             nn.ReLU(),
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(output_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 768),
#             nn.ReLU(),
#             nn.Linear(768, input_dim),
#             nn.ReLU(),
#         )

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return encoded, decoded


def train_autoencoder(
        train_dataset,
        valid_dataset,
        input_dim,
        output_dim,
        batch_size,
        learning_rate,
        epochs,
        checkpoints_dir,
        epoch_length,
        big=False,
        checkpoint_path=None,
        max_saved_models=5,
        patience=10,
    ):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset), pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder(input_dim, output_dim).to(device) if not big else AutoEncoderBigger(input_dim, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if checkpoint_path is not None:
        print(f"Loading model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))


    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    prev_epoch_loss = np.inf
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        bar = tqdm(range(epoch_length), total=epoch_length, desc=f"Epoch {epoch + 1} - Training")
        for step in bar:
            batch = next(iter(train_loader))
            batch = batch.to(device).float()
            optimizer.zero_grad()
            _, decoded = model(batch)
            loss = criterion(decoded, batch)
            loss.backward()
            optimizer.step()
            bar.set_postfix(loss=loss.item())

        model.eval()
        with torch.no_grad():
            val_epoch_loss = 0.0
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch + 1} - Validation"):
                batch = batch.to(device).float()
                _, decoded = model(batch)
                loss = criterion(decoded, batch)
                val_epoch_loss += loss.item() * batch.size(0)

        val_epoch_loss /= len(valid_dataset)
        print(f"Validation Loss: {val_epoch_loss:.5f}")

        mlflow.log_metric("train_loss", loss.item())
        mlflow.log_metric("val_loss", val_epoch_loss)

        if val_epoch_loss <= best_epoch_loss:
            print(f"Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            model_path = os.path.join(checkpoints_dir, f"model_{epoch}")
            print(f"Saving model to {model_path}")
            mlflow.pytorch.save_model(model, model_path)

            saved_models = sorted(os.listdir(checkpoints_dir), key=lambda x: int(x.split("_")[-1]))
            while len(saved_models) > max_saved_models:
                model_to_remove = saved_models.pop(0)
                shutil.rmtree(os.path.join(checkpoints_dir, model_to_remove))

            mlflow.log_metric("val_loss_best", best_epoch_loss, step=epoch)

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



CONFIG_DIR = os.getenv('CONFIG_DIR')
CONFIG_NAME = os.getenv('CONFIG_NAME')

assert CONFIG_DIR is not None and CONFIG_DIR != '', 'CONFIG_DIR is not set'
assert CONFIG_NAME is not None and CONFIG_NAME != '', 'CONFIG_NAME is not set'

@hydra.main(config_path=CONFIG_DIR, config_name=CONFIG_NAME, version_base=None)
def main(args: DictConfig):

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

    shape = tuple(args.shape)
    

    memmap_data = np.memmap(args.memmap_file, dtype=args.dtype, mode='r', shape=shape)
    total_length = memmap_data.shape[0] // args.input_dim  # total number of data points (each of size input_dim)
    train_length = int(args.train_ratio * total_length)

    train_dataset = MemmapDataset(args.memmap_file, 0, train_length, dtype=args.dtype, shape=shape)
    valid_dataset = MemmapDataset(args.memmap_file, train_length, total_length, dtype=args.dtype, shape=shape)

    train_autoencoder(
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        checkpoints_dir=checkpoints_dir_path,
        epoch_length=args.epoch_length,
        big=args.big,
        checkpoint_path=args.checkpoint_path,
        max_saved_models=5,
        patience=10,
    )

    _ = gc.collect()

    mlflow.end_run()

if __name__ == "__main__":
    # shape is 70697113 1024
    # shape with images is (122289192, 1024)
    # python train.py --memmap_file /path/to/memmap/file.npy --output_dim 256 --input_dim 1024 --batch_size 256 --learning_rate 0.001 --epochs 10 --checkpoint_dir /path/to/checkpoints/dir --train_ratio 0.9
    main()
