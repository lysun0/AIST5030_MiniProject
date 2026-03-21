#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import ResNet18_Weights, resnet18


PRETRAINED_MODEL_NAME = "ResNet-18"
PRETRAINED_WEIGHTS_NAME = "ImageNet1K_V1"

CAT_IMAGENET_IDXS = [281, 282, 283, 284, 285]
DOG_IMAGENET_IDXS = list(range(151, 269))


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BinaryCIFARWrapper(Dataset):
    def __init__(self, base_dataset: datasets.CIFAR10, indices: List[int]):
        self.base_dataset = base_dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[self.indices[idx]]
        # CIFAR10 label 3 -> cat(0), 5 -> dog(1)
        binary_label = 0 if label == 3 else 1
        return image, binary_label


def split_binary_indices(dataset: datasets.CIFAR10, val_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    cat_dog_indices = [i for i, y in enumerate(dataset.targets) if y in (3, 5)]
    labels = [dataset.targets[i] for i in cat_dog_indices]

    cat_indices = [idx for idx, y in zip(cat_dog_indices, labels) if y == 3]
    dog_indices = [idx for idx, y in zip(cat_dog_indices, labels) if y == 5]

    rng = random.Random(seed)
    rng.shuffle(cat_indices)
    rng.shuffle(dog_indices)

    cat_val = int(len(cat_indices) * val_ratio)
    dog_val = int(len(dog_indices) * val_ratio)

    val_indices = cat_indices[:cat_val] + dog_indices[:dog_val]
    train_indices = cat_indices[cat_val:] + dog_indices[dog_val:]

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


class OFTResNetBinary(nn.Module):
    def __init__(self, eps: float = 1e-4):
        super().__init__()
        # Pretrained backbone being finetuned with OFT: ResNet-18 (ImageNet1K_V1 weights).
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = backbone.fc

        for param in self.parameters():
            param.requires_grad = False

        feature_dim = self.fc.in_features
        self.oft_raw = nn.Parameter(torch.zeros(feature_dim, feature_dim))
        self.eps = eps

        self.register_buffer("cat_idx", torch.tensor(CAT_IMAGENET_IDXS, dtype=torch.long))
        self.register_buffer("dog_idx", torch.tensor(DOG_IMAGENET_IDXS, dtype=torch.long))

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def _cayley(self) -> torch.Tensor:
        # Cayley transform: R = (I - A)^(-1) (I + A), where A is skew-symmetric.
        a = self.oft_raw - self.oft_raw.transpose(0, 1)
        dim = a.shape[0]
        eye = torch.eye(dim, device=a.device, dtype=a.dtype)
        left = eye - a
        right = eye + a
        left = left + self.eps * eye
        r = torch.linalg.solve(left, right)
        return r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._features(x)
        r = self._cayley()
        feat_t = feat @ r

        logits_1000 = self.fc(feat_t)
        cat_logit = torch.logsumexp(logits_1000.index_select(1, self.cat_idx), dim=1)
        dog_logit = torch.logsumexp(logits_1000.index_select(1, self.dog_idx), dim=1)
        logits_2 = torch.stack([cat_logit, dog_logit], dim=1)
        return logits_2


def build_dataloaders(data_dir: str, batch_size: int, num_workers: int, val_ratio: float, seed: int):
    weights = ResNet18_Weights.IMAGENET1K_V1
    base_t = weights.transforms()
    mean = list(base_t.mean)
    std = list(base_t.std)
    normalize = transforms.Normalize(mean=mean, std=std)

    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    base_train = datasets.CIFAR10(root=data_dir, train=True, transform=train_tfms, download=True)
    base_train_eval = datasets.CIFAR10(root=data_dir, train=True, transform=eval_tfms, download=False)
    base_test = datasets.CIFAR10(root=data_dir, train=False, transform=eval_tfms, download=True)

    train_idx, val_idx = split_binary_indices(base_train, val_ratio=val_ratio, seed=seed)
    test_idx = [i for i, y in enumerate(base_test.targets) if y in (3, 5)]

    train_ds = BinaryCIFARWrapper(base_train, train_idx)
    val_ds = BinaryCIFARWrapper(base_train_eval, val_idx)
    test_ds = BinaryCIFARWrapper(base_test, test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> EvalResult:
    model.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_count += bs

    return EvalResult(
        loss=total_loss / max(1, total_count),
        accuracy=accuracy_score(all_labels, all_preds),
        precision=precision_score(all_labels, all_preds, zero_division=0),
        recall=recall_score(all_labels, all_preds, zero_division=0),
        f1=f1_score(all_labels, all_preds, zero_division=0),
    )


def train_oft(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
) -> List[Dict[str, float]]:
    optimizer = torch.optim.AdamW([model.oft_raw], lr=lr, weight_decay=weight_decay)
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            running_loss += loss.item() * bs
            total += bs

        train_loss = running_loss / max(1, total)
        val_result = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_result.loss,
            "val_accuracy": val_result.accuracy,
            "val_f1": val_result.f1,
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{epochs} | train_loss={train_loss:.4f} | "
            f"val_loss={val_result.loss:.4f} | val_acc={val_result.accuracy:.4f} | val_f1={val_result.f1:.4f}"
        )

    return history


def save_history_csv(history: List[Dict[str, float]], path: str) -> None:
    if not history:
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def plot_loss_curve(history: List[Dict[str, float]], path: str) -> None:
    epochs = [int(h["epoch"]) for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, marker="o", label="Train Loss")
    plt.plot(epochs, val_losses, marker="s", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("OFT Finetuning Loss Curves")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def denormalize(img: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    mean_t = torch.tensor(mean, device=img.device).view(3, 1, 1)
    std_t = torch.tensor(std, device=img.device).view(3, 1, 1)
    return img * std_t + mean_t


def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device, max_items: int = 8):
    images_out = []
    labels_out = []
    preds_out = []
    probs_out = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)

            for i in range(images.size(0)):
                images_out.append(images[i].cpu())
                labels_out.append(int(labels[i].cpu().item()))
                preds_out.append(int(preds[i].cpu().item()))
                probs_out.append(float(confs[i].cpu().item()))
                if len(images_out) >= max_items:
                    return images_out, labels_out, preds_out, probs_out

    return images_out, labels_out, preds_out, probs_out


def plot_qualitative(
    images,
    labels,
    before_preds,
    before_probs,
    after_preds,
    after_probs,
    path: str,
):
    class_names = ["cat", "dog"]
    weights = ResNet18_Weights.IMAGENET1K_V1
    base_t = weights.transforms()
    mean = list(base_t.mean)
    std = list(base_t.std)

    n = len(images)
    cols = 4
    rows = int(math.ceil(n / cols))

    plt.figure(figsize=(4 * cols, 3.5 * rows))
    for i in range(n):
        ax = plt.subplot(rows, cols, i + 1)
        img = denormalize(images[i], mean, std).permute(1, 2, 0).numpy()
        img = np.clip(img, 0.0, 1.0)
        ax.imshow(img)
        title = (
            f"GT: {class_names[labels[i]]}\n"
            f"Before: {class_names[before_preds[i]]} ({before_probs[i]:.2f})\n"
            f"After: {class_names[after_preds[i]]} ({after_probs[i]:.2f})"
        )
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.suptitle("Qualitative Predictions: Before vs After OFT", fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def main():
    parser = argparse.ArgumentParser(description="OFT finetuning on CIFAR-10 cat vs dog with pretrained ResNet-18")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    model = OFTResNetBinary().to(device)
    print(f"Pretrained model: {PRETRAINED_MODEL_NAME} ({PRETRAINED_WEIGHTS_NAME})")
    params_info = count_parameters(model)
    print(f"Parameter count: total={params_info['total']}, trainable={params_info['trainable']}")

    before_test = evaluate(model, test_loader, device)
    print(
        "Before OFT | "
        f"loss={before_test.loss:.4f}, acc={before_test.accuracy:.4f}, "
        f"precision={before_test.precision:.4f}, recall={before_test.recall:.4f}, f1={before_test.f1:.4f}"
    )

    sample_images, sample_labels, before_preds, before_probs = collect_predictions(model, test_loader, device, max_items=8)

    history = train_oft(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    after_test = evaluate(model, test_loader, device)
    print(
        "After OFT  | "
        f"loss={after_test.loss:.4f}, acc={after_test.accuracy:.4f}, "
        f"precision={after_test.precision:.4f}, recall={after_test.recall:.4f}, f1={after_test.f1:.4f}"
    )

    _, _, after_preds, after_probs = collect_predictions(model, test_loader, device, max_items=8)

    history_csv_path = os.path.join(args.output_dir, "training_log.csv")
    loss_curve_path = os.path.join(args.output_dir, "loss_curve.png")
    qualitative_path = os.path.join(args.output_dir, "qualitative_before_after.png")
    metrics_json_path = os.path.join(args.output_dir, "metrics.json")
    model_ckpt_path = os.path.join(args.output_dir, "oft_resnet18_cifar_cat_dog.pt")

    save_history_csv(history, history_csv_path)
    plot_loss_curve(history, loss_curve_path)
    plot_qualitative(
        sample_images,
        sample_labels,
        before_preds,
        before_probs,
        after_preds,
        after_probs,
        qualitative_path,
    )

    torch.save(
        {
            "state_dict": model.state_dict(),
            "args": vars(args),
            "params": params_info,
            "before_test": before_test.__dict__,
            "after_test": after_test.__dict__,
            "history": history,
        },
        model_ckpt_path,
    )

    metrics = {
        "task": "CIFAR-10 cat vs dog binary classification",
        "model": f"ImageNet-pretrained {PRETRAINED_MODEL_NAME} ({PRETRAINED_WEIGHTS_NAME}) + OFT (Cayley orthogonal transform)",
        "params": params_info,
        "before_test": before_test.__dict__,
        "after_test": after_test.__dict__,
        "improvement": {
            "accuracy": after_test.accuracy - before_test.accuracy,
            "f1": after_test.f1 - before_test.f1,
            "precision": after_test.precision - before_test.precision,
            "recall": after_test.recall - before_test.recall,
        },
        "artifacts": {
            "training_log_csv": history_csv_path,
            "loss_curve_png": loss_curve_path,
            "qualitative_png": qualitative_path,
            "checkpoint": model_ckpt_path,
        },
    }

    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics to: {metrics_json_path}")
    print("Done.")


if __name__ == "__main__":
    main()
