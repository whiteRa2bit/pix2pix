import os

import tqdm
import torch
import torch.nn.functional as F
import wandb
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from torch.autograd import Variable, grad

from pix_pix.config import TRAIN_CONFIG, WANDB_PROJECT, CHECKPOINT_DIR



class Trainer:
    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, train_dataloader, val_dataloader, config=TRAIN_CONFIG):
        self.generator = generator.to(config['device'])
        self.discriminator = discriminator.to(config['device'])
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.config = config
        self.l1_loss = torch.nn.L1Loss()
        self.adv_loss = F.binary_cross_entropy

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    def _initialize_wandb(self, project_name=WANDB_PROJECT):
        wandb.init(config=self.config, project=project_name)
        wandb.watch(self.generator)
        wandb.watch(self.discriminator)

    def train(self):
        self.generator.train()
        self.discriminator.train()
        self._initialize_wandb()
        
        train_g = True
        train_d = False

        best_val_loss = np.inf
        for epoch in range(self.config['epochs_num']):
            if epoch == self.config["g_epochs_num"]:
                train_g = False
                train_d = True
            elif epoch == self.config["g_epochs_num"] + self.config["d_epochs_num"]:
                train_g = True
                train_d = True

            logger.info(f"Epoch {epoch} started, train generator: {train_g}, train discriminator: {train_d}")
            for i, data in tqdm.tqdm(enumerate(self.train_dataloader)):
                if i == len(self.train_dataloader.dataset) // self.config['train_batch_size']:
                    break

                sketches, real = data
                sketches = sketches.to(self.config["device"])
                real = real.to(self.config["device"])
       
                # Train discriminator
                if train_d:
                    g_sample = self.generator(sketches)
                    d_fake = self.discriminator(g_sample)
                    d_real = self.discriminator(real)
                    gradient_penalty = self._compute_gp(real, g_sample)

                    d_loss = torch.mean(d_fake) - torch.mean(d_real)
                    d_loss_gp = d_loss + gradient_penalty
                    d_loss_gp.backward()
                    self.d_optimizer.step()

                    self._reset_grad()
   
                # Train Generator
                if (i % self.config['d_coef'] == 0 and train_g) or not train_d:
                    g_sample = self.generator(sketches)
                    d_fake = self.discriminator(g_sample)
                    
                    adv_loss = -torch.mean(d_fake)
                    l1_loss = self.l1_loss(g_sample, real)
                    if train_d:
                        for g in self.g_optimizer.param_groups:
                            g['lr'] = self.config["adv_g_lr"]
                        for d in self.d_optimizer.param_groups:
                            d['lr'] = self.config["adv_d_lr"]
                        g_loss = adv_loss + self.config["l1_weight"] * l1_loss
                    else:
                        g_loss = self.config["l1_weight"] * l1_loss
                    g_loss.backward()
                    self.g_optimizer.step()
                    
                    self._reset_grad()


                if i % self.config['log_each'] == 0:
                    try:
                        d_loss_val = d_loss.item()
                    except NameError:
                        d_loss_val = 0
                        gradient_penalty = 0

                    val_metrics = self._compute_metrics(self.val_dataloader)
                    val_loss = val_metrics['loss']
                    val_fig = val_metrics['sample_fig']
                    wandb.log({
                        "G adv loss": adv_loss.item(), \
                        "G l1 loss": l1_loss.item(),
                        "D loss": d_loss_val,
                        "D gradient penalty": gradient_penalty,
                        "Val l1 Loss": val_loss, \
                        "Val sample": val_fig
                    })
                    plt.clf()

                    if val_loss < best_val_loss:
                        self._save_checkpoint(self.generator, "baseline")
                        best_val_loss = val_loss
        logger.info(f"Training finished. Best validation loss: {best_val_loss}")

    def _compute_metrics(self, dataloader):
        self.generator.eval()
        sketches = []
        real = []
        outputs = []

        for data in dataloader:
            batch_sketches, batch_real = data
            batch_sketches = batch_sketches.to(self.config["device"])
            batch_real = batch_real.to(self.config["device"])
            with torch.no_grad():
                batch_outputs = self.generator(batch_sketches)
            sketches.append(batch_sketches)
            real.append(batch_real)
            outputs.append(batch_outputs)

        sketches = torch.cat(sketches)
        real = torch.cat(real)
        outputs = torch.cat(outputs)
        loss = self.l1_loss(outputs, real).item()
        
        idx = np.random.choice(len(real))
        
        sketch_sample = sketches[idx]
        real_sample = real[idx]
        output_sample = outputs[idx]
        
        fig = self._visualize(sketch_sample, real_sample, output_sample)

        return {"loss": loss, "sample_fig": fig}
    
    @staticmethod
    def _visualize(sketch, real, output):
        fig, ax = plt.subplots(1, 3, figsize=(10, 3))
        ax[0].set_title("Sketch")
        ax[0].imshow(sketch.permute(1, 2, 0).squeeze(2).cpu().data, cmap='gray')
        ax[1].set_title("Real")
        ax[1].imshow(real.permute(1, 2, 0).cpu().data)
        ax[2].set_title("Predicted")
        ax[2].imshow(output.permute(1, 2, 0).cpu().data)
        return fig

    @staticmethod
    def _save_checkpoint(model, checkpoint_name):
        checkpoint_dir = os.path.join(CHECKPOINT_DIR, wandb.run.id)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pt")
        torch.save(model.state_dict(), checkpoint_path)
        
    def _reset_grad(self):
        self.generator.zero_grad()
        self.discriminator.zero_grad()

    def _compute_gp(self, X, g_sample):
        alpha = torch.rand((self.config["train_batch_size"], 1,
                            1, 1)).to(self.config['device'])  # TODO: (@whiteRa2bit, 2020-09-25) Fix shape
        x_hat = alpha * X.data + (1 - alpha) * g_sample.data
        x_hat.requires_grad = True
        pred_hat = self.discriminator(x_hat)
        gradients = grad(
            outputs=pred_hat,
            inputs=x_hat,
            grad_outputs=torch.ones(pred_hat.size()).to(self.config['device']),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        gradient_penalty = self.config['lambda'] * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty
