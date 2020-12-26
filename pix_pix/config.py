import os


_PROJECT_DIR = '/home/pafakanov/data/other/dl/hw3/' 
DATA_DIR = os.path.join(_PROJECT_DIR, 'pokemon_jpg')
WANDB_PROJECT = 'dl_hse_gan'
CHECKPOINT_DIR = os.path.join(_PROJECT_DIR, 'checkpoints')
RANDOM_SEED = 42
TRAIN_FRAC = 0.9

TRAIN_CONFIG = {
    "train_batch_size": 8,
    "val_batch_size": 64,
    "in_channels": 1,
    "out_channels": 3,
    "num_downs": 7,
    "g_lr": 1e-4,
    "d_lr": 1e-4,
    "adv_g_lr": 1e-3,
    "adv_d_lr": 1e-4,
    "l1_weight": 25,
    "g_epochs_num": 2,
    "d_epochs_num": 2,
    "lambda": 10,
    "d_coef": 3,
    "epochs_num": 20,
    "log_each": 25,
    "device": "cuda:1"
}