import os
import argparse
import numpy as np
import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def load_model(input_dim, output_dim, model_path):
    model = AutoEncoder(input_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def reduce_dimensionality(input_files, output_dir, model):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for input_file in input_files:
        try:
            output_file = os.path.basename(input_file)
            output_path = os.path.join(output_dir, output_file)
            embeddings = np.load(input_file)
            embeddings = torch.from_numpy(embeddings).float()
            with torch.no_grad():
                reduced_embeddings, _ = model(embeddings)
            np.save(output_path, reduced_embeddings.numpy())
        except Exception as e:
            print(f"Error processing file {input_file}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Autoencoder for dimensionality reduction - Inference")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing .npy files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save reduced embeddings")
    parser.add_argument("--start", type=int, default=0, help="Start index for file range")
    parser.add_argument("--end", type=int, default=None, help="End index for file range")
    parser.add_argument("--dim", type=int, default=256, help="Dimension to reduce embeddings to")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")

    args = parser.parse_args()

    input_files = sorted([os.path.join(args.input_dir, fname) for fname in os.listdir(args.input_dir)])
    input_files = input_files[args.start:args.end]

    model = load_model(1024, args.dim, args.model_path)
    reduce_dimensionality(input_files, args.output_dir, model)

if __name__ == "__main__":
    # python inference.py --input_dir /path/to/input/dir --output_dir /path/to/output/dir --start 50 --end 55 --dim 256 --model_path /path/to/model/checkpoint
    main()
