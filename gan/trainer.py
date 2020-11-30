import os

import torch
import numpy as np

from config import TRAIN_CONFIG, WANDB_PROJECT, CHECKPOINT_DIR


class Trainer:
    def __init__(self, model, optimizer, train_dataloader, val_dataloader, config=TRAIN_CONFIG):
        self.model = model.to(config['device'])
        self.optimizer = optimizer
        self.config = config
        self.criterion = torch.nn.L1Loss()

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

    # def _initialize_wandb(self, project_name=WANDB_PROJECT):
    #     wandb.init(config=self.config, project=project_name)
    #     wandb.watch(self.model)

    def train(self):
        self.model.train()
        # self._initialize_wandb()

        best_val_loss = np.inf
        for epoch in range(self.config['epochs_num']):
            print(f"Epoch {epoch} started...")
            for i, data in enumerate(self.train_dataloader):
                sketches, real = data
                sketches = sketches.to(self.config["device"])
                real = real.to(self.config["device"])

                self.optimizer.zero_grad()
                outputs = self.model(sketches)
                loss = self.criterion(outputs, real)

                # compute gradients
                loss.backward()

                # make a step
                self.optimizer.step()

                loss = loss.item()

                if i % self.config['log_each'] == 0:
                    val_metrics = self._compute_metrics(self.val_dataloader)
                    val_loss = val_metrics['loss']
                    val_fig = val_metrics['sample_fig']
                    print(f"Train loss: {round(loss, 3)}, Val loss: {round(val_loss, 3)}")
                    # wandb.log({
                    #     "Train Loss": loss, \
                    #     "Val Loss": val_loss, \
                    #     "Sample": val_fig
                    # })
                    # plt.clf()

                    if val_loss < best_val_loss:
                        self._save_checkpoint(self.model, "baseline")
                        best_val_loss = val_loss
        print(f"Training finished. Best validation loss: {best_val_loss}")

    def _compute_metrics(self, dataloader):
        self.model.eval()
        sketches = []
        real = []
        outputs = []

        for data in dataloader:
            batch_sketches, batch_real = data
            batch_sketches = batch_sketches.to(self.config["device"])
            batch_real = batch_real.to(self.config["device"])
            with torch.no_grad():
                batch_outputs = self.model(batch_sketches)
            sketches.append(batch_sketches)
            real.append(batch_real)
            outputs.append(batch_outputs)

        sketches = torch.cat(sketches)
        real = torch.cat(real)
        outputs = torch.cat(outputs)
        loss = self.criterion(outputs, real).item()

        idx = np.random.choice(len(real))

        sketch_sample = sketches[idx]
        real_sample = real[idx]
        output_sample = outputs[idx]

        # fig = self._visualize(sketch_sample, real_sample, output_sample)
        fig = None

        return {"loss": loss, "sample_fig": fig}

    # @staticmethod
    # def _visualize(sketch, real, output):
    #     fig, ax = plt.subplots(1, 3, figsize=(10, 3))
    #     ax[0].set_title("Sketch")
    #     ax[0].imshow(sketch.permute(1, 2, 0).cpu().data)
    #     ax[1].set_title("Real")
    #     ax[1].imshow(real.permute(1, 2, 0).cpu().data)
    #     ax[2].set_title("Predicted")
    #     ax[2].imshow(output.permute(1, 2, 0).cpu().data)
    #     return fig

    @staticmethod
    def _save_checkpoint(model, checkpoint_name):
        checkpoint_dir = CHECKPOINT_DIR
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.pt")
        torch.save(model.state_dict(), checkpoint_path)
