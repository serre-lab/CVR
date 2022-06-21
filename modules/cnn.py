
import os
from argparse import ArgumentParser


import pytorch_lightning as pl
import torch
from torch import nn as nn
from torch.nn import functional as F

from torchvision import models

from models.vits import vit_small as vit_small_moco

from models.scn import SCL
from models.wren import WReN

class Base(pl.LightningModule):

    def load_finetune_weights(self, checkpoint):
        print("*"*10 + "load finetune weights ...")
        # CNN.load_fron
        model_temp = self.__class__.load_from_checkpoint(checkpoint)
        # model.load_finetune_weights(model_temp)
        self.backbone.load_state_dict(model_temp.backbone.state_dict())

    def load_backbone_weights(self, checkpoint):
        print("*"*10 + "load ckpt weights ...")
        self.backbone.load_state_dict(torch.load(checkpoint)['model'], strict=False)
        

    def freeze_pretrained(self):
        for param in self.backbone.parameters():
            param.requires_grad = False


    def shared_step(self, batch):

        x, task_idx = batch # B, 4, H, W

        # creates artificial label
        x_size = x.shape
        perms = torch.stack([torch.randperm(4, device=self.device) for _ in range(x_size[0])], 0)
        y = perms.argmax(1)
        perms = perms + torch.arange(x_size[0], device=self.device)[:,None]*4
        perms = perms.flatten()

        x = x.reshape([x_size[0]*4, x_size[2], x_size[3], x_size[4]])[perms].reshape([x_size[0], 4, x_size[2], x_size[3], x_size[4]])

        if self.task_embedding:
            y_hat = self(x, task_idx)
        else:
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

        y_hat, y = self.shared_step(batch)
        
        loss = F.cross_entropy(y_hat, y, reduction='none')

        acc = (y == torch.argmax(y_hat, dim=1))*1

        logs = {
            "loss": loss,
            "acc": acc,
        }

        results = {f"test_{k}": v for k, v in logs.items()}
        return results

    def test_epoch_end(self, outputs):
        
        keys = list(outputs[0].keys())

        results = {k: torch.cat([x[k] for x in outputs]).cpu().numpy() for k in keys} 
        self.test_results = results

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)

        

class CNN(Base):

    def __init__(
        self,
        backbone: str ='resnet50',
        lr: float = 1e-4,
        wd: float = 1e-4,
        n_tasks: int = 103,
        mlp_dim: int = 128,
        mlp_hidden_dim: int = 2048,
        task_embedding: int = 0, #64
        ssl_pretrain: bool = False,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            kl_coeff: coefficient for kl term of the loss
            lr: learning rate for Adam
        """
        self.save_hyperparameters()

        super(CNN, self).__init__()

        feature_extract = True
        num_classes = 1
        use_pretrained = False
        self.hidden_size = mlp_dim

        if backbone == "resnet18":
            """ Resnet18
            """
            self.backbone = models.resnet18(pretrained=use_pretrained, progress=False, num_classes=num_classes)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == "resnet50":
            """ Resnet50
            """
            self.backbone = models.resnet50(pretrained=use_pretrained, progress=False, num_classes=num_classes)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()

        elif backbone == "vit_small":
            self.backbone = vit_small_moco(img_size=128, stop_grad_conv1=True)
            self.backbone.head = nn.Identity()
            num_ftrs = self.backbone.embed_dim
                
        if task_embedding>0:
            self.task_embedding = nn.Embedding(n_tasks, task_embedding)
        else:
            self.task_embedding = None
        
        self.mlp = nn.Sequential(nn.Linear(num_ftrs+task_embedding, mlp_hidden_dim), nn.ReLU(), nn.Linear(mlp_hidden_dim, self.hidden_size))

    def init_networks(self):
        # define encoder, decoder, fc_mu and fc_var
        pass

    def forward(self, x, task_idx=None):
        
        x_size = x.shape
        x = x.reshape([x_size[0]*4, x_size[2], x_size[3], x_size[4]])

        x = self.backbone(x)
        
        if task_idx is not None:
            x_task = self.task_embedding(task_idx.repeat_interleave(4))
            x = torch.cat([x, x_task], 1)
        
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=1)
        x = x.reshape([-1, 4, self.hidden_size])
        x = (x[:,:,None,:] * x[:,None,:,:]).sum(3).sum(2)
        x = -x

        return x

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        
        parser.add_argument("--backbone", type=str, default='resnet50')
        parser.add_argument("--wd", type=float, default=1e-4)
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument("--n_tasks", type=int, default=103)
        parser.add_argument("--mlp_dim", type=int, default=128)
        parser.add_argument("--mlp_hidden_dim", type=int, default=2048)
        parser.add_argument("--task_embedding", type=int, default=0)

        return parser

class SCN(Base):

    def __init__(
        self,
        backbone: str ='scl',
        lr: float = 5e-3,
        wd: float = 1e-2,
        n_tasks: int = 103,
        mlp_dim: int = 128,
        mlp_hidden_dim: int = 2048,
        task_embedding: int = 0, #64
        ssl_pretrain: bool = False,
        # task_embedding: 
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

        self.hidden_size = mlp_dim

        if task_embedding>0:
            self.task_embedding = nn.Embedding(n_tasks, task_embedding)
        else:
            self.task_embedding = None
        
        self.backbone = SCL(
            image_size=128,
            set_size=5,
            conv_channels=[3, 16, 16, 32, 32, 32],
            conv_output_dim=80,
            attr_heads=10,
            attr_net_hidden_dims=[128],
            rel_heads=80,
            rel_net_hidden_dims=[64, 23, 5],
            task_emb_size=task_embedding,
        )        


    def forward(self, x, task_idx=None):
        
        
        x_task = self.task_embedding(task_idx) if task_idx is not None else None

        out = self.backbone(x, x_task)

        return out

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--backbone", type=str, default='resnet50')
        parser.add_argument("--wd", type=float, default=5e-3)
        parser.add_argument("--lr", type=float, default=1e-2)

        parser.add_argument("--n_tasks", type=int, default=103)
        parser.add_argument("--task_embedding", type=int, default=0)
        
        return parser


class WREN(Base):

    def __init__(
        self,
        backbone: str ='wren',
        lr: float = 1e-4,
        wd: float = 0,
        n_tasks: int = 103,
        mlp_dim: int = 128,
        mlp_hidden_dim: int = 2048,
        task_embedding: int = 0, #64
        ssl_pretrain: bool = False,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            kl_coeff: coefficient for kl term of the loss
            lr: learning rate for Adam
        """
        self.save_hyperparameters()

        super(WREN, self).__init__()

        feature_extract = True
        num_classes = 1
        use_pretrained = False

        self.hidden_size = mlp_dim

        if task_embedding>0:
            self.task_embedding = nn.Embedding(n_tasks, task_embedding)
        else:
            self.task_embedding = None

        self.backbone = WReN(task_emb_size=task_embedding)

    def forward(self, x, task_idx=None):
        
        x_task = self.task_embedding(task_idx) if task_idx is not None else None

        out = self.backbone(x, x_task)

        return out

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--backbone", type=str, default='wren')
        parser.add_argument("--wd", type=float, default=1e-4)
        parser.add_argument("--lr", type=float, default=1e-4)

        parser.add_argument("--n_tasks", type=int, default=103)
        parser.add_argument("--task_embedding", type=int, default=0)
        
        return parser


class SCNHead(Base):

    def __init__(
        self,
        backbone: str ='resnet50',
        lr: float = 5e-3,
        wd: float = 1e-2,
        n_tasks: int = 103,
        mlp_dim: int = 128,
        mlp_hidden_dim: int = 2048,
        task_embedding: int = 0, #64
        ssl_pretrain: bool = False,
        **kwargs
    ):
        """
        Args:
            input_height: height of the images
            kl_coeff: coefficient for kl term of the loss
            lr: learning rate for Adam
        """
        self.save_hyperparameters()

        super(SCNHead, self).__init__()

        feature_extract = True
        num_classes = 1
        use_pretrained = False
        
        self.hidden_size = mlp_dim

        if backbone == "resnet18":
            """ Resnet18
            """
            self.backbone = models.resnet18(pretrained=use_pretrained, progress=False, num_classes=num_classes)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            
        elif backbone == "resnet50":
            """ Resnet50
            """
            self.backbone = models.resnet50(pretrained=use_pretrained, progress=False, num_classes=num_classes)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()


        if task_embedding>0:
            self.task_embedding = nn.Embedding(n_tasks, task_embedding)
            self.task_embedding_size = task_embedding
        else:
            self.task_embedding = None
            self.task_embedding_size = 0
        
        self.head = SCL(
            image_size=num_ftrs+task_embedding,
            set_size=5,
            conv_channels=[],
            conv_output_dim=mlp_hidden_dim,
            attr_heads=128,
            attr_net_hidden_dims=[256],
            rel_heads=mlp_hidden_dim,
            rel_net_hidden_dims=[64, 23, 5],
            task_emb_size=task_embedding,
        )        

    def load_finetune_weights(self, checkpoint):
        print("*"*10 + "load finetune weights ...")
        model_temp = self.__class__.load_from_checkpoint(checkpoint)
        self.backbone.load_state_dict(model_temp.backbone.state_dict())
        self.head.load_state_dict(model_temp.head.state_dict())

    def load_backbone_weights(self, checkpoint):
        print("*"*10 + "load ckpt weights ...")
        self.backbone.load_state_dict(torch.load(checkpoint)['model'], strict=False)
        
    def forward(self, x, task_idx=None):
        
        x_task = self.task_embedding(task_idx) if task_idx is not None else None

        x_size = x.shape
        x = x.reshape([x_size[0]*4, x_size[2], x_size[3], x_size[4]])
        x = self.backbone(x)
        x = x.reshape([x_size[0], 4, -1])
        out = self.head(x, x_task)

        return out

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument("--backbone", type=str, default='resnet50')
        parser.add_argument("--wd", type=float, default=1e-4)
        parser.add_argument("--lr", type=float, default=1e-3)

        parser.add_argument("--n_tasks", type=int, default=103)
        parser.add_argument("--mlp_hidden_dim", type=int, default=256)
        parser.add_argument("--task_embedding", type=int, default=0)

        # parser.add_argument("--ssl_pretrain", action='store_true')
        
        return parser
