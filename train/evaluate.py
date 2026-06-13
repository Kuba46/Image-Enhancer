"""
evaluate.py — оценка качества модели.

Метрики:
  - MAE по каждому параметру (brightness, contrast, saturation)
  - Корреляция предсказанного и реального
  - PSNR между результатом предсказания и результатом реальных меток
  - Scatter plot: предсказанное vs реальное

Использование:
    python evaluate.py
    python evaluate.py --n 200
    python evaluate.py --raw data/raw --expert data/expert_c --n 200
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import keras
from train import extract_features, compute_targets


def load_keras_model() -> keras.Model:
    keras_path = Path("model/best_keras.keras")
    if not keras_path.exists():
        raise FileNotFoundError(f"Модель не найдена: {keras_path}\nСначала: python train.py")
    return keras.models.load_model(str(keras_path))


def load_scaler(scaler_path: Path):
    with open(scaler_path) as f:
        params = json.load(f)
    return np.array(params["mean"]), np.array(params["scale"])


def apply_corrections(img: np.ndarray, brightness: float, contrast: float, saturation: float) -> np.ndarray:
    result    = img.copy() + brightness * 0.3
    c_factor  = 1.0 + contrast   * 0.5
    result    = (result - 0.5) * c_factor + 0.5
    lum       = (0.299 * result[..., 0] + 0.587 * result[..., 1] + 0.114 * result[..., 2])[..., np.newaxis]
    result    = lum + (result - lum) * (1.0 + saturation * 0.5)
    return np.clip(result, 0.0, 1.0)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a - b) ** 2)
    return float('inf') if mse == 0 else float(20 * np.log10(1.0 / np.sqrt(mse)))


def evaluate(args):
    raw_dir    = Path(args.raw)
    expert_dir = Path(args.expert)

    model = load_keras_model()
    print(f"Модель загружена: {model.count_params()} параметров")

    scaler_mean, scaler_scale = load_scaler(Path(args.scaler))

    raw_files    = {p.stem: p for p in sorted(raw_dir.glob("*.jpg"))}
    expert_files = {p.stem: p for p in sorted(expert_dir.glob("*.jpg"))}
    common       = sorted(set(raw_files) & set(expert_files))[:args.n]
    print(f"Оцениваем на {len(common)} изображениях...\n")

    pred_list, true_list, psnr_list = [], [], []
    skipped = 0

    for name in common:
        try:
            features        = extract_features(raw_files[name])
            features_scaled = (features - scaler_mean) / scaler_scale
            pred            = model.predict(features_scaled[np.newaxis, :], verbose=0)[0]
            true            = compute_targets(raw_files[name], expert_files[name])

            pred_list.append(pred.tolist())
            true_list.append(true.tolist())

            raw_img   = np.array(Image.open(raw_files[name]).convert("RGB"), dtype=np.float32) / 255.0
            enhanced  = apply_corrections(raw_img, pred[0],  pred[1],  pred[2])
            reference = apply_corrections(raw_img, true[0],  true[1],  true[2])
            psnr_list.append(psnr(enhanced, reference))

        except Exception as e:
            skipped += 1
            print(f"  Пропуск {name}: {e}")

    if not pred_list:
        print("Нет данных для оценки.")
        return

    pred_all = np.array(pred_list)
    true_all = np.array(true_list)

    if skipped:
        print(f"Пропущено: {skipped}\n")

    param_names = ["brightness", "contrast", "saturation"]

    print("Дисперсия предсказаний (если ~0 — модель коллапсировала):")
    for i, name in enumerate(param_names):
        print(f"  {name:<15} std={pred_all[:, i].std():.4f}  "
              f"min={pred_all[:, i].min():.3f}  max={pred_all[:, i].max():.3f}")
    print()

    print("=" * 52)
    print(f"{'Параметр':<15} {'MAE':>8} {'корреляция':>12} {'std ошибки':>12}")
    print("-" * 52)
    for i, name in enumerate(param_names):
        errors = np.abs(pred_all[:, i] - true_all[:, i])
        mae    = errors.mean()
        std    = errors.std()
        if pred_all[:, i].std() < 1e-6 or true_all[:, i].std() < 1e-6:
            corr_str = "   n/a"
        else:
            corr     = np.corrcoef(pred_all[:, i], true_all[:, i])[0, 1]
            corr_str = f"{corr:>12.4f}"
        print(f"{name:<15} {mae:>8.4f} {corr_str} {std:>12.4f}")

    print("=" * 52)
    print(f"Средний PSNR:   {np.mean(psnr_list):.2f} dB")
    print(f"Медиана PSNR:   {np.median(psnr_list):.2f} dB")
    print(f"(> 30 dB — хорошо, > 35 dB — отлично)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Предсказанное vs реальное (n={len(pred_all)})", fontsize=13)
    for i, (ax, name) in enumerate(zip(axes, param_names)):
        ax.scatter(true_all[:, i], pred_all[:, i], alpha=0.4, s=12, color="#4f8ef7")
        ax.plot([-1, 1], [-1, 1], "r--", lw=1)
        ax.set_xlabel(f"Реальное ({name})")
        ax.set_ylabel(f"Предсказание ({name})")
        ax.set_title(name)
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1])
    plt.tight_layout()
    plt.savefig("evaluation_scatter.png", dpi=120)
    print(f"\nГрафик сохранён: evaluation_scatter.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw",    default="data/raw")
    parser.add_argument("--expert", default="data/c")
    parser.add_argument("--scaler", default="model/scaler.json")
    parser.add_argument("--n",      type=int, default=200)
    evaluate(parser.parse_args())