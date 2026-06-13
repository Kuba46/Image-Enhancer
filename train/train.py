"""
MIT-Adobe FiveK — pipeline обучения MLP для предсказания параметров коррекции.

Входные данные:
    data/raw/      — исходные изображения JPG (папка raw из Kaggle датасета)
    data/expert_c/ — эталоны Expert C JPG    (папка c из Kaggle датасета)

Выходные данные:
    model/tfjs/model.json + group1-shard1of1.bin
    model/scaler.json

Порядок запуска:
    1. Скопировать папки raw/ и c/ из датасета в data/raw/ и data/expert_c/
    2. python train.py
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image

import keras
from keras import layers

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


RAW_DIR      = Path("data/raw")
EXPERT_DIRS  = [Path("data/c")]
MODEL_DIR    = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

IMG_PREVIEW_SIZE = (256, 256)
N_HISTOGRAM_BINS = 16
RANDOM_SEED      = 42
EPOCHS           = 100
BATCH_SIZE       = 64


def rgb_to_hsv_array(rgb: np.ndarray):
    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        s = np.where(maxc != 0, (maxc - minc) / maxc, 0.0)
    v = maxc
    return s, v


def extract_features(img_path: Path) -> np.ndarray:
    """30 признаков из гистограммы и статистик пикселей."""
    img = Image.open(img_path).convert("RGB").resize(IMG_PREVIEW_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0

    lum      = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    lum_flat = lum.flatten()

    hist, _ = np.histogram(lum_flat, bins=N_HISTOGRAM_BINS, range=(0.0, 1.0))
    hist = hist / hist.sum()

    mean_lum    = lum_flat.mean()
    std_lum     = lum_flat.std()
    med_lum     = float(np.median(lum_flat))
    dark_frac   = float((lum_flat < 0.25).mean())
    bright_frac = float((lum_flat > 0.75).mean())

    sat, val = rgb_to_hsv_array(arr)
    mean_sat = sat.mean()
    std_sat  = sat.std()
    mean_val = val.mean()

    mean_r = arr[..., 0].mean()
    mean_g = arr[..., 1].mean()
    mean_b = arr[..., 2].mean()

    # Перцентили яркости — различаем тусклое от тёмного
    p5, p25, p75, p95 = np.percentile(lum_flat, [5, 25, 75, 95])

    # Энтропия гистограммы — мера "богатства" деталей
    hist_nz = hist[hist > 0]
    entropy = float(-np.sum(hist_nz * np.log2(hist_nz)))

    return np.concatenate([
        hist,
        [mean_lum, std_lum, med_lum],
        [dark_frac, bright_frac],
        [mean_sat, std_sat, mean_val],
        [mean_r, mean_g, mean_b],
        [mean_r - mean_b, mean_r - mean_g, mean_g - mean_b],
        [p5, p25, p75, p95, entropy],   # +5 признаков = итого 35
    ]).astype(np.float32)


def compute_targets(raw_path: Path, expert_path: Path) -> np.ndarray:
    """
    Вычисляет дельты между исходником и эталоном по трём параметрам.
    Возвращает [brightness_delta, contrast_delta, saturation_delta] ∈ [-1, 1].
    """
    def stats(path):
        img = Image.open(path).convert("RGB").resize(IMG_PREVIEW_SIZE)
        arr = np.array(img, dtype=np.float32) / 255.0
        lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        sat, _ = rgb_to_hsv_array(arr)
        return lum.mean(), lum.std(), sat.mean()

    raw_lum,    raw_contrast,    raw_sat    = stats(raw_path)
    expert_lum, expert_contrast, expert_sat = stats(expert_path)

    brightness = float(np.clip((expert_lum      - raw_lum)      / 0.3,  -1.0, 1.0))
    contrast   = float(np.clip((expert_contrast - raw_contrast) / 0.25, -1.0, 1.0))
    saturation = float(np.clip((expert_sat      - raw_sat)      / 0.3,  -1.0, 1.0))

    return np.array([brightness, contrast, saturation], dtype=np.float32)


def _img_stats(path: Path):
    """Статистики изображения для compute_targets — выносим отдельно для кэша."""
    img = Image.open(path).convert("RGB").resize(IMG_PREVIEW_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    sat, _ = rgb_to_hsv_array(arr)
    return lum.mean(), lum.std(), sat.mean()


def build_dataset():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Папка не найдена: {RAW_DIR}")

    available_experts = [d for d in EXPERT_DIRS if d.exists()]
    if not available_experts:
        raise FileNotFoundError(
            f"Папки экспертов не найдены. Скопируйте из Kaggle датасета:\n"
            f"  a/ → data/a/,  b/ → data/b/,  c/ → data/c/,  d/ → data/d/,  e/ → data/e/"
        )

    raw_files = {p.stem: p for p in sorted(RAW_DIR.glob("*.jpg"))}
    print(f"Исходников (raw): {len(raw_files)}")

    # ── Кэш признаков и статистик raw-изображений ─────────────────────────────
    # raw читается один раз — не 5 раз для каждого эксперта
    print("Кэширование признаков raw-изображений...")
    raw_features_cache = {}
    raw_stats_cache    = {}
    for i, (name, path) in enumerate(raw_files.items()):
        if i % 500 == 0:
            print(f"  {i}/{len(raw_files)}...")
        try:
            raw_features_cache[name] = extract_features(path)
            raw_stats_cache[name]    = _img_stats(path)
        except Exception as e:
            print(f"  Пропуск {name}: {e}")

    print(f"Закэшировано: {len(raw_features_cache)} исходников\n")

    X, y = [], []
    total_pairs = 0

    for expert_dir in available_experts:
        expert_files = {p.stem: p for p in sorted(expert_dir.glob("*.jpg"))}
        common       = sorted(set(raw_features_cache) & set(expert_files))
        total_pairs += len(common)
        print(f"  {expert_dir.name}/: {len(common)} пар", end="", flush=True)

        expert_y = []
        for name in common:
            try:
                # Признаки берём из кэша — не читаем raw повторно
                features = raw_features_cache[name]

                # Статистики expert считаем один раз
                raw_lum, raw_contrast, raw_sat       = raw_stats_cache[name]
                exp_lum, exp_contrast, exp_sat       = _img_stats(expert_files[name])

                brightness = float(np.clip((exp_lum      - raw_lum)      / 0.3,  -1.0, 1.0))
                contrast   = float(np.clip((exp_contrast - raw_contrast) / 0.25, -1.0, 1.0))
                saturation = float(np.clip((exp_sat      - raw_sat)      / 0.3,  -1.0, 1.0))

                X.append(features)
                y.append([brightness, contrast, saturation])
                expert_y.append([brightness, contrast, saturation])
            except Exception as e:
                print(f"\n    Пропуск {name}: {e}", end="")

        ey = np.array(expert_y)
        print(f" ✓  mean=[{ey.mean(axis=0)[0]:+.2f}, {ey.mean(axis=0)[1]:+.2f}, {ey.mean(axis=0)[2]:+.2f}]"
              f"  (brightness, contrast, saturation)")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"\nВсего пар: {len(X)} (из {total_pairs})")
    print(f"Метки — mean: {y.mean(axis=0).round(3)}, std: {y.std(axis=0).round(3)}")
    if y.std(axis=0).max() < 0.05:
        print("⚠️  Маленькая дисперсия меток — проверьте пары изображений")

    return X, y


def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        layers.Dense(32, activation="relu"),
        layers.Dense(3, activation="tanh"),
    ], name="image_enhancer")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def train():
    print("=== Сборка датасета ===")
    X, y = build_dataset()
    print(f"X: {X.shape}, y: {y.shape}")
    print(f"Метки — mean: {y.mean(axis=0).round(3)}, std: {y.std(axis=0).round(3)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    scaler_params = {
        "mean":  scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
    }
    with open(MODEL_DIR / "scaler.json", "w") as f:
        json.dump(scaler_params, f, indent=2)
    print("scaler.json сохранён")

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.15, random_state=RANDOM_SEED
    )

    model = build_model(X_train.shape[1])
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-5
        ),
        keras.callbacks.ModelCheckpoint(
            MODEL_DIR / "best_keras.keras", save_best_only=True
        ),
    ]

    print("\n=== Обучение ===")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    val_loss, val_mae = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nVal Loss (MSE): {val_loss:.4f} | Val MAE: {val_mae:.4f}")

    export_model(model, MODEL_DIR)


def export_model(model: keras.Model, model_dir: Path):
    """
    Экспортирует модель в формат совместимый с worker.js —
    без tensorflowjs, напрямую через numpy.

    Сохраняет model_dir/tfjs/model.json со структурой:
      { layers: [{ weights, bias, activation }] }
    """
    import struct

    tfjs_dir = model_dir / "tfjs"
    tfjs_dir.mkdir(parents=True, exist_ok=True)

    layers_data = []
    all_weights = []   # все float32 веса подряд — для бинарного шарда
    weight_specs = []  # манифест весов для model.json

    for layer in model.layers:
        # Пропускаем всё кроме Dense — BatchNorm не нужен при инференсе
        if not isinstance(layer, keras.layers.Dense):
            continue

        cfg        = layer.get_config()
        activation = cfg.get("activation", "linear")
        weights    = layer.get_weights()   # [kernel, bias]

        kernel, bias = weights[0], weights[1]

        # Имя слоя для манифеста
        lname = layer.name

        # Записываем спецификации весов
        weight_specs.append({
            "name":  f"{lname}/kernel",
            "shape": list(kernel.shape),
            "dtype": "float32",
        })
        weight_specs.append({
            "name":  f"{lname}/bias",
            "shape": list(bias.shape),
            "dtype": "float32",
        })

        all_weights.append(kernel.astype(np.float32).flatten())
        all_weights.append(bias.astype(np.float32).flatten())

        layers_data.append({
            "name":       lname,
            "activation": activation,
            "kernel":     kernel.shape,
            "bias_size":  bias.shape[0],
        })

    # Бинарный шард весов
    shard_path = tfjs_dir / "group1-shard1of1.bin"
    flat = np.concatenate(all_weights).astype(np.float32)
    with open(shard_path, "wb") as f:
        f.write(flat.tobytes())

    # Считаем байтовые смещения для манифеста
    byte_offset = 0
    for spec, arr in zip(weight_specs, all_weights):
        spec["byteOffset"] = byte_offset
        byte_offset += arr.nbytes

    # model.json в формате TF.js LayersModel
    # (совместим с model-loader.js в браузере)
    model_json = {
        "format":         "layers-model",
        "generatedBy":    "custom-exporter",
        "convertedBy":    None,
        "modelTopology": {
            "model_config": {
                "class_name": "Sequential",
                "config": {
                    "name": model.name,
                    "layers": [
                        {
                            "class_name": layer.get_config().get("__class__", "Dense")
                                          if hasattr(layer, "get_config") else "Dense",
                            "config": {
                                "name":       layer.name,
                                "activation": layer.get_config().get("activation", "linear"),
                            }
                        }
                        for layer in model.layers
                        if layer.get_weights()
                    ]
                }
            }
        },
        "weightsManifest": [
            {
                "paths":   ["group1-shard1of1.bin"],
                "weights": weight_specs,
            }
        ],
        # Дополнительный ключ для model-loader.js — плоский список слоёв
        "layersMeta": [
            {
                "name":       d["name"],
                "activation": d["activation"],
                "kernelShape": d["kernel"],
                "biasSize":   d["bias_size"],
            }
            for d in layers_data
        ],
    }

    with open(tfjs_dir / "model.json", "w") as f:
        json.dump(model_json, f, indent=2)

    print(f"Модель экспортирована в {tfjs_dir}/")
    print(f"  model.json:              {(tfjs_dir / 'model.json').stat().st_size / 1024:.1f} KB")
    print(f"  group1-shard1of1.bin:    {shard_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    train()