
from tqdm import tqdm

import os
import argparse
import numpy as np

from PIL import Image

import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader

import io
import os, sys
import requests
import PIL
from PIL import Image
from PIL import ImageDraw, ImageFont
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF


import yaml

import sys
sys.path.append('./taming-transformers')

from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ


def preprocess(img, target_image_size=512):
    s = min(img.size)
    r = target_image_size / s
    s = (round(r * img.size[1]), round(r * img.size[0]))
    img = TF.resize(img, s, interpolation=PIL.Image.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = torch.unsqueeze(T.ToTensor()(img), 0)

    return img


def preprocess_vqgan(x):
    # idk wtf is this lol
    x = 2.*x - 1.
    return x


class CustomDataset(Dataset):
    def __init__(self, image_list, root_dir):
        self.image_list = image_list
        self.root_dir = root_dir

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        subfolder, image_path, image_number = self.image_list[index]
        img = Image.open(os.path.join(self.root_dir, subfolder, image_path)).convert("RGB")
        img = preprocess(img)
        img = preprocess_vqgan(img)
        return img, image_number


def custom_collate(batch):
    images, image_numbers = zip(*batch)
    images = torch.cat(images, dim=0)
    return images, image_numbers


def save_npy(embeddings, npy_path):
    np.save(npy_path, embeddings)


def get_image_from_memmap(memmap_path, image_number):
    mmap_embeddings = np.memmap(memmap_path, dtype='i2', mode='r+', shape=(14_000_000, 1024))
    return mmap_embeddings[image_number]



def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    if is_gumbel:
        model = GumbelVQ(**config.model.params)
    else:
        model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]  # why cpu here?
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()


def main(args):
    print('Setting up dataloader...')
    torch.set_grad_enabled(False)

    # Set the list of tuples (subfolder, img_path) for all images
    index = pd.read_csv(args.index_path)
    all_images = index[['part_id', 'part', 'img_name', 'image_number']].values

    # Filter the image list based on the specified subfolders
    image_list = []
    for part_number, subfolder, img_path, image_number in tqdm(all_images):
        if int(args.start_part_number) <= part_number <= int(args.end_part_number):
            image_list.append((subfolder, img_path, image_number))

    # Instantiate the dataset and dataloader
    dataset = CustomDataset(image_list, args.data_path)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        num_workers=args.n_workers, pin_memory=True, shuffle=False,
        collate_fn=custom_collate
    )

    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Loading model...')
    print('Device:', device)
    config = load_config(args.model_config_path, display=False)
    model = load_vqgan(config, ckpt_path=args.model_ckpt_path).to(device)

    # Process the images in batches and save the embeddings
    for image_batch in tqdm(dataloader, total=len(dataloader), desc='Vectorizing images'):
        images, image_numbers = image_batch
        images = images.to(device)
        z, _, [_, _, indices] = model.encode(images)
        start_idx = image_numbers[0]

        # indices are batch_size x 1024 flattened vector, so we need to reshape it:
        indices = indices.reshape(-1, 1024)
        assert indices.shape[0] == len(image_numbers)

        indices = indices.cpu().numpy()

        mmap_embeddings = np.memmap(args.memmap_path, dtype='i2', mode='r+', shape=(14_000_000, 1024))
        mmap_embeddings[start_idx:start_idx + len(indices)] = indices


if __name__ == "__main__":
    # last part_id 13998
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the data folder")
    parser.add_argument("--index_path", type=str, help="Path to the index file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--memmap_path", type=str, help="Path to the memmap file")
    parser.add_argument("--n_workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--model_config_path", type=str, help="Path to the model config file")  # "logs/vqgan_imagenet_f16_16384/configs/model.yaml"
    parser.add_argument("--model_ckpt_path", type=str, help="Path to the model checkpoint file")  # "logs/vqgan_imagenet_f16_16384/checkpoints/last.ckpt"
    parser.add_argument("--start_part_number")
    parser.add_argument("--end_part_number")
    args = parser.parse_args()
    main(args)
