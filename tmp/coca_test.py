import os
import sys
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import open_clip


class CFG:
    device = "cuda"
    seed = 42
    embedding_length = 384
    sentence_model_path = "/kaggle/input/sentence-transformers-222/all-MiniLM-L6-v2"
    model_name = "coca_ViT-L-14"
    model_checkpoint_path = "open-clip-models/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k.bin"


if __name__ == '__main__':
    images = os.listdir('test_submit_dataset/images/')
    imgIds = [i.split('.')[0] for i in images]

    eIds = list(range(CFG.embedding_length))

    imgId_eId = [
        '_'.join(map(str, i)) for i in zip(
            np.repeat(imgIds, CFG.embedding_length),
            np.tile(range(CFG.embedding_length), len(imgIds)))]

    model = open_clip.create_model(CFG.model_name)
    open_clip.load_checkpoint(model, CFG.model_checkpoint_path)

    transform = open_clip.image_transform(
        model.visual.image_size,
        is_train = False,
        mean = getattr(model.visual, 'image_mean', None),
        std = getattr(model.visual, 'image_std', None),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)  # why cuda is not available? probably because for python3.7 there should be different pytorch-with-cuda version installed..

    def make_batches(l, batch_size=16):
        for i in range(0, len(l), batch_size):
            yield l[i:i + batch_size]

    BATCH_SIZE = 16
    NUM_RETURN_SEQUENCES = 3
    NUM_BEAMS = 6
    MIN_LENGTH = 5
    MAX_LENGTH = 75 # why inside it becomes 30? because for beam search it uses seq_len param instead of max_seq_len in coca
    # also for coca with beam search, top_k and top_p are not used as well as temperature etc.
    NUM_BEAM_GROUPS = 3

    images_path = "test_submit_dataset/images"

    prompts = []
    for batch in make_batches(images, BATCH_SIZE):
        images_batch = []
        for i, image_name in enumerate(batch):
            img = Image.open(os.path.join(images_path, image_name)).convert("RGB")
            img = transform(img).unsqueeze(0)
            images_batch.append(img)

        images_batch = torch.cat(images_batch, dim=0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            images_batch = images_batch.to(device)

            # code inside changed:
            # num_beam_hyps_to_keep=3 in beam_scorer, returns sequences, scores
            generated, scores = model.generate(
                images_batch,
                min_seq_len=MIN_LENGTH,
                seq_len=MAX_LENGTH,
                num_beam_hyps_to_keep=NUM_RETURN_SEQUENCES,
                num_beams=NUM_BEAMS,
                num_beam_groups=NUM_BEAM_GROUPS,
            )

        decoded = [open_clip.decode(x) for x in generated]
        for prompt in decoded:
            prompt = (
                prompt
                .split("<end_of_text>")[0]
                .replace("<start_of_text>", "")
                .rstrip(" .,")
            )
            prompts.append(prompt)

    print(prompts)