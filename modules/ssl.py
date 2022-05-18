from argparse import ArgumentParser
from typing import Union

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.plugins import DDP2Plugin, DDPPlugin
from torch import nn
from torch.nn import functional as F

from modules import optimizer as optimizer_
from modules import scheduler as scheduler_

from models import vits

import torchvision.models as torchvision_models

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))


class Moco(LightningModule):

    def __init__(
        self,
        # base_encoder: Union[str, torch.nn.Module] = "resnet18",
        arch: str = "resnet50",
        mlp_dim: int = 4096,
        dim: int = 256,
        lr: float = 0.6,
        momentum: float = 0.9,
        weight_decay: float = 1e-6,
        # batch_size: int = 1024,
        # num_workers: int = 8,
        
        moco_dim: int = 128,
        moco_mlp_dim: int = 4096,
        moco_m: int = 0.99,
        moco_m_cos: bool = True,
        moco_t: int = 1.0,
        stop_grad_conv1: bool =False,
        *args,
        **kwargs
    ):

        super().__init__()
        self.save_hyperparameters()


        # self.moco_dim = moco_dim
        # self.moco_mlp_dim = moco_mlp_dim
        self.T = moco_t
        self.moco_m = moco_m
        self._build_projector_and_predictor_mlps(dim, mlp_dim, arch)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)
        
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim, arch):
        
        if 'vit' in arch:
            self.base_encoder = vits.__dict__[arch](img_size=128, stop_grad_conv1=self.hparams.stop_grad_conv1, num_classes=mlp_dim)
            self.momentum_encoder = vits.__dict__[arch](img_size=128, stop_grad_conv1=self.hparams.stop_grad_conv1, num_classes=mlp_dim)

            hidden_dim = self.base_encoder.head.weight.shape[1]
            del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

            # projectors
            self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
            self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

            # predictor
            self.predictor = self._build_mlp(2, dim, mlp_dim, dim)

        else:
            self.base_encoder = torchvision_models.__dict__[arch](zero_init_residual=True, num_classes=mlp_dim)
            self.momentum_encoder = torchvision_models.__dict__[arch](zero_init_residual=True, num_classes=mlp_dim)

            hidden_dim = self.base_encoder.fc.weight.shape[1]
            del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

            # projectors
            self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
            self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

            # predictor
            self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)
    
    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2):
        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))

        with torch.no_grad():  # no gradient
            # print('global step, moco_m', self.trainer.global_step, self.moco_m)
            self._update_momentum_encoder(self.moco_m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)

    def training_step(self, batch, batch_idx):
        (img_1, img_2) = batch
        loss = self(img_1, img_2)
        logs = {"train_loss": loss}
        self.log_dict({f"metrics/train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     (img_1, img_2) = batch
    #     loss = self(img_1, img_2)
    #     log = {"val_loss": loss}
    #     return log

    def configure_optimizers(self):
        lr = self.hparams.lr * self.hparams.batch_size / 256
        lr = self.hparams.lr * self.hparams.batch_size * self.trainer.num_gpus / 256
        print('base_lr', self.hparams.lr)
        print('batch_size', self.hparams.batch_size)
        print('num_gpus', self.trainer.num_gpus)
        print('new_lr', lr)
        # self.batch_size = self.batch_size // 4

        if self.hparams.optimizer == 'lars':
            optimizer = optimizer_.LARS(self.parameters(), lr, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum)
        elif self.hparams.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr, weight_decay=self.hparams.weight_decay)
        
        scheduler = scheduler_.LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs = self.hparams.warmup_epochs * self.n_epoch_steps,
            max_epochs = self.hparams.max_epochs * self.n_epoch_steps,
        )

        sched = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,            
        }

        return [optimizer], [sched]

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        tb_size = self.hparams.batch_size * max(1, self.trainer.gpus)
        self.n_epoch_steps = len(train_loader.dataset) // tb_size

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base', 'resnet50']
        model_names = ['vit_small', 'resnet50']

        parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names, help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
        parser.add_argument('--lr', default=0.6, type=float, metavar='LR', help='initial (base) learning rate', dest='lr')
        parser.add_argument("--momentum", type=float, default=0.9)
        parser.add_argument('--wd', '--weight_decay', default=1e-6, type=float, metavar='W', help='weight decay (default: 1e-6)', dest='weight_decay')

        parser.add_argument('--stop_grad_conv1', action='store_true', help='stop-grad after first conv, or patch embedding')

        parser.add_argument('--moco_dim', default=256, type=int, help='feature dimension (default: 256)')
        parser.add_argument('--moco_mlp_dim', default=4096, type=int, help='hidden dimension in MLPs (default: 4096)')
        parser.add_argument('--moco_m', default=0.99, type=float, help='moco momentum of updating momentum encoder (default: 0.99)')
        parser.add_argument('--moco_m_cos', action='store_true', help='gradually increase moco momentum to 1 with a half-cycle cosine schedule')
        parser.add_argument('--moco_t', default=1.0, type=float, help='softmax temperature (default: 1.0)')

        # parser.add_argument("--flag", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False)

        parser.add_argument('--optimizer', default='lars', type=str, choices=['lars', 'adamw'], help='optimizer used (default: lars)')
        parser.add_argument('--warmup_epochs', default=10, type=int, metavar='N', help='number of warmup epochs')

        return parser

    @staticmethod
    def _use_ddp_or_ddp2(trainer: Trainer) -> bool:
        return isinstance(trainer.training_type_plugin, (DDPPlugin, DDP2Plugin))





# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def mean(res, key):
    # recursive mean for multilevel dicts
    return torch.stack([x[key] if isinstance(x, dict) else mean(x, key) for x in res]).mean()


def accuracy(preds, labels):
    preds = preds.float()
    max_lgt = torch.max(preds, 1)[1]
    num_correct = (max_lgt == labels).sum().item()
    num_correct = torch.tensor(num_correct).float()
    acc = num_correct / len(labels)

    return acc


def precision_at_k(output, target, top_k=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k."""
    with torch.no_grad():
        maxk = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res