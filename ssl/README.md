# MoCo v3 for Self-supervised ResNet and ViT

Code in this folder is adapted from the {original repository}[https://github.com/facebookresearch/moco-v3] of the moco-v3 paper.

## SSL Pretraining

The pretraining scripts are `pretrain_resnet50.sh` and `pretrain_vit_small.sh`.

After the models are trained use `convert_to_deit.py` to place backbone weights in checkpoint files. These checkpoint files can be used to reproduce the SSL-initialization results.

## Citation
```
@Article{chen2021mocov3,
  author  = {Xinlei Chen* and Saining Xie* and Kaiming He},
  title   = {An Empirical Study of Training Self-Supervised Vision Transformers},
  journal = {arXiv preprint arXiv:2104.02057},
  year    = {2021},
}
```
