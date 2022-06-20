import os
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch import nn as nn
from torch.nn import functional as F

from models.scn import SCL, SCLTrainingWrapper


# from models import resnet18_encoder, resnet18_decoder, resnet50_encoder, resnet50_decoder


# from pl_bolts.metrics import mean


class SCN(pl.LightningModule):
    """
    Standard VAE with Gaussian Prior and approx posterior.
    """

    def __init__(
            self,
            # input_height: int,torch.stack([x['test_loss
            backbone: str = 'scn',
            lr: float = 1e-4,
            wd: float = 1e-4,
            **kwargs
    ):
        """
        Args:
            input_height: height of the images
            kl_coeff: coefficient for kl term of the loss
            lr: learning rate for Adam
        """
        self.save_hyperparameters()

        super(SCN, self).__init__()

        feature_extract = True
        num_classes = 1
        use_pretrained = False

        if backbone == "scn":
            """ Scattering Compositional Learner
            """
            model = SCL(
                        image_size = 128,                           # size of image
                        set_size = 5,                               # number of questions + 1 answer
                        conv_channels = [1, 16, 16, 32, 32, 32],    # convolutional channel progression, 1 for greyscale, 3 for rgb
                        conv_output_dim = 80,                       # model dimension, the output dimension of the vision net
                        attr_heads = 10,                            # number of attribute heads
                        attr_net_hidden_dims = [128],               # attribute scatter transform MLP hidden dimension(s)
                        rel_heads = 80,                             # number of relationship heads
                        rel_net_hidden_dims = [64, 23, 5]           # MLP for relationship net
                    )

            self.backbone = SCLTrainingWrapper(model)

    def load_finetune_weights(self, checkpoint):
        print("*" + "load finetune weights ...")
        # CNN.load_fron
        model_temp = SCN.load_from_checkpoint(checkpoint)
        # model.load_finetune_weights(model_temp)
        self.backbone.load_state_dict(model_temp.backbone.state_dict())

    # remove the last layer
    def freeze_pretrained(self):
        for param in self.backbone.parameters()[::-1]:
            param.requires_grad = False

    def init_networks(self):
        # define encoder, decoder, fc_mu and fc_var
        pass

    def forward(self, x):

        ############
        logits = self.backbone(x, x)

        return logits

    def shared_step(self, batch):
        x = batch  # B, 4, H, W

        # creates artificial label
        x_size = x.shape
        perms = torch.stack([torch.randperm(4, device=self.device) for _ in range(x_size[0])], 0)
        y = perms.argmax(1)
        perms = perms + torch.arange(x_size[0], device=self.device)[:, None] * 4
        perms = perms.flatten()
        x = x.reshape([x_size[0] * 4, x_size[2], x_size[3], x_size[4]])  # torch.Size([256, 3, 128, 128])

        x = x[perms]  # torch.Size([256, 3, 128, 128])
        x = x.reshape(x_size)
        x = x[:,:,0,:,:].unsqueeze(2)

        y_hat = self(x)
        return y_hat, y

    def step(self, batch, batch_idx):

        y_hat, y = self.shared_step(batch)
        loss = F.cross_entropy(y_hat, y)
        acc = torch.sum((y == torch.argmax(y_hat, dim=1))).float() / len(y)

        logs = {
            "loss": loss,
            "acc": acc,
        }

        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"metrics/train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"metrics/val_{k}": v for k, v in logs.items()})
        return loss

    def test_step(self, batch, batch_idx):
        # _, logs = self.step(batch, batch_idx)

        y_hat, y = self.shared_step(batch)

        loss = F.cross_entropy(y_hat, y, reduction='none')
        # acc = torch.sum((y == torch.argmax(y_hat, dim=1))).float() / len(y)
        acc = (y == torch.argmax(y_hat, dim=1)) * 1

        logs = {
            "loss": loss,
            "acc": acc,
        }

        results = {f"test_{k}": v for k, v in logs.items()}
        return results

    def test_epoch_end(self, outputs):

        keys = list(outputs[0].keys())
        # results = {k: torch.stack([x[k] for x in outputs]).mean().item() for k in keys}
        results = {k: torch.cat([x[k] for x in outputs]).cpu().numpy() for k in keys}
        self.test_results = results

        # return {'log': results}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # parser.add_argument("--input_height", type=int, default=224, help="input dimensions for reconstruction")

        parser.add_argument("--backbone", type=str, default='scn')
        parser.add_argument("--wd", type=float, default=1e-4)
        parser.add_argument("--lr", type=float, default=1e-4)

        return parser