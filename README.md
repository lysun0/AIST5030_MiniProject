# OFT Finetuning Experiment

This project runs an Orthogonal Finetuning (OFT) experiment using an ImageNet-pretrained ResNet-18 on a downstream task: CIFAR-10 cat vs dog binary classification.

## Method Summary

- Base model: `ResNet-18` pretrained on ImageNet.
- Downstream task: binary classification (`cat` vs `dog`) from CIFAR-10.
- OFT adaptation: a trainable skew-symmetric matrix is transformed into an orthogonal matrix via Cayley transform and inserted before the frozen pretrained classifier.
- Trainable parameters: only OFT matrix (`512 x 512`) while pretrained weights remain frozen.

## Run

```bash
cd OFT
bash setup_env.sh
conda activate oft
CUDA_VISIBLE_DEVICES=0 python run_oft_experiment.py --epochs 20 --batch-size 64 --lr 3e-3
```

## Outputs

Artifacts are written to `results/`:

- `training_log.csv`: epoch-level train/val metrics
- `loss_curve.png`: training and validation loss curves
- `qualitative_before_after.png`: sample predictions before and after OFT
- `metrics.json`: final performance summary
- `oft_resnet18_cifar_cat_dog.pt`: checkpoint with metrics and history