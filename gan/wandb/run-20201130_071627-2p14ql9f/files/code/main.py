import os

import torch
import numpy as np
from torch.utils.data import DataLoader

from dataset import ImageDataset
from model import UnetGenerator
from trainer import Trainer
from config import RANDOM_SEED, TRAIN_DIR, VAL_DIR, TRAIN_CONFIG

def _set_seed(seed=RANDOM_SEED):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    # random.seed(seed)
    np.random.seed(seed)


def main(config=TRAIN_CONFIG):
    train_dataset = ImageDataset(TRAIN_DIR)
    val_dataset = ImageDataset(VAL_DIR)
    train_dataloader = DataLoader(train_dataset, config['train_batch_size'], True)
    val_dataloader = DataLoader(val_dataset, config['val_batch_size'], True)
        
    generator = UnetGenerator(config['in_channels'], config['out_channels'], config['num_downs'])
    optimizer = torch.optim.Adam(generator.parameters(), lr=config['g_lr'])
    trainer = Trainer(generator, optimizer, train_dataloader, val_dataloader, config)
    trainer.train()

if __name__ == '__main__':
    _set_seed()
    main()
