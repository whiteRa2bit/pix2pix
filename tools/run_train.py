import os
import random

import torch
import numpy as np
from torch.utils.data import DataLoader

from utils import download_data
from dataset import ImageDataset
from model import UnetGenerator, Discriminator
from trainer import Trainer
from config import RANDOM_SEED, DATA_DIR, TRAIN_CONFIG, TRAIN_FRAC

from scheduler import get_gpu_id


def _set_seed(seed=RANDOM_SEED):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(config=TRAIN_CONFIG):
    dataset = ImageDataset(DATA_DIR)
    train_size = int(len(dataset) * TRAIN_FRAC)
    test_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_dataloader = DataLoader(train_dataset, config['train_batch_size'], True)
    val_dataloader = DataLoader(val_dataset, config['val_batch_size'], True)

    generator = UnetGenerator(config)
    discriminator = Discriminator(config)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=config['g_lr'])
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config['d_lr'])
    trainer = Trainer(generator, discriminator, g_optimizer, d_optimizer, \
                  train_dataloader, val_dataloader, config)
    trainer.train()


if __name__ == '__main__':
    _set_seed()
    # download_data()
    main()
