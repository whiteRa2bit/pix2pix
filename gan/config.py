import os

_PROJECT_DIR = '.'
DATA_DIR = os.path.join(_PROJECT_DIR, 'edges2shoes')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
WANDB_PROJECT = 'dl_hse_gan'
CHECKPOINT_DIR = os.path.join(_PROJECT_DIR, 'checkpoints')
RANDOM_SEED = 42
DATA_URL = "https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/edges2shoes.tar.gz"

TRAIN_CONFIG = {
    "train_batch_size": 1,
    "val_batch_size": 64,
    "in_channels": 3,
    "out_channels": 3,
    "num_downs": 7,
    "filters_num": 64,
    "kernel_size": 4,
    "stride": 2,
    "padding": 1,
    "g_lr": 3e-4,
    "epochs_num": 1,
    "log_each": 50,
    "device": "cuda",
    "kernel_size": 4
}
