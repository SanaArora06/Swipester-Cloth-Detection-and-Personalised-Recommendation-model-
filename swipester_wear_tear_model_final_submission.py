# %% [markdown]
# # Swipester Wear, Tear & Stain Detection Model
# ### Final submission notebook/script aligned with the midterm methodology
#
# This cleaned version keeps the same overall logic described in the project deliverables:
#
# 1. data cleaning and dataset organization  
# 2. resizing and normalization for MobileNetV2  
# 3. data augmentation for robustness  
# 4. class-imbalance handling with class weights  
# 5. transfer learning with MobileNetV2  
# 6. phase-1 frozen training, followed by phase-2 fine-tuning  
# 7. validation + test evaluation, confusion matrix, and export to `.keras` and `.tflite`
#
# You can run this file as:
#
# ```bash
# python swipester_wear_tear_model_final_submission.py
# ```
#
# Or convert it into a notebook with the companion generator cell that Codex created.

# %% [markdown]
# ## 1. Imports and configuration
# In Colab, install missing packages first if needed:
#
# ```python
# !pip install -q tensorflow matplotlib seaborn scikit-learn pillow
# ```

# %%
from __future__ import annotations

import json
import os
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image, ImageFile
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 3
INITIAL_EPOCHS = 12
FINE_TUNE_EPOCHS = 8
UNFREEZE_LAST_N = 30
CLASSES = ["good", "stained", "torn"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

PROJECT_DIR = Path.cwd()
RAW_DATASET_DIR = PROJECT_DIR / "raw_dataset"
DATASET_DIR = PROJECT_DIR / "dataset"
DATASET_ZIP = PROJECT_DIR / "dataset.zip"
ARTIFACT_DIR = PROJECT_DIR / "swipester_mobilenetv2_artifacts"
BEST_MODEL_PATH = ARTIFACT_DIR / "best_model.keras"
FINAL_KERAS_PATH = ARTIFACT_DIR / "swipester_wear_tear_model.keras"
FINAL_TFLITE_PATH = ARTIFACT_DIR / "swipester_wear_tear_model.tflite"
LABELS_PATH = ARTIFACT_DIR / "labels.json"
SUMMARY_PATH = ARTIFACT_DIR / "training_summary.json"
TRAINING_PLOT_PATH = ARTIFACT_DIR / "training_curves.png"
VAL_CM_PATH = ARTIFACT_DIR / "confusion_matrix_validation.png"
TEST_CM_PATH = ARTIFACT_DIR / "confusion_matrix_test.png"


@dataclass
class RunConfig:
    seed: int = SEED
    image_size: int = IMG_SIZE
    batch_size: int = BATCH_SIZE
    initial_epochs: int = INITIAL_EPOCHS
    fine_tune_epochs: int = FINE_TUNE_EPOCHS
    unfreeze_last_n: int = UNFREEZE_LAST_N
    classes: list[str] = None

    def __post_init__(self) -> None:
        if self.classes is None:
            self.classes = CLASSES.copy()


CONFIG = RunConfig()


def in_colab() -> bool:
    return "google.colab" in sys.modules


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# %% [markdown]
# ## 2. Dataset utilities
# The code supports two dataset layouts:
#
# 1. **Already split**
# ```text
# dataset/
#   train/good, stained, torn
#   val/good, stained, torn
#   test/good, stained, torn
# ```
#
# 2. **Raw per-class folders**
# ```text
# raw_dataset/
#   good/
#   stained/
#   torn/
# ```
#
# If only `raw_dataset/` exists, the script creates an 80/10/10 split automatically.

# %%
def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([path for path in folder.iterdir() if is_image_file(path)])


def maybe_extract_dataset_zip(zip_path: Path, target_root: Path) -> None:
    if not zip_path.exists() or DATASET_DIR.exists():
        return
    import zipfile

    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(target_root)


def ensure_split_structure(dataset_dir: Path, classes: list[str]) -> None:
    for split in ("train", "val", "test"):
        for class_name in classes:
            (dataset_dir / split / class_name).mkdir(parents=True, exist_ok=True)


def build_dataset_split_from_raw(
    raw_dir: Path,
    dataset_dir: Path,
    classes: list[str],
    seed: int = SEED,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> None:
    print("Creating 80/10/10 train/val/test split from raw_dataset/ ...")
    ensure_split_structure(dataset_dir, classes)
    rng = random.Random(seed)

    for class_name in classes:
        src_dir = raw_dir / class_name
        images = list_images(src_dir)
        if not images:
            raise FileNotFoundError(f"No images found for class '{class_name}' in {src_dir}")

        rng.shuffle(images)
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val

        split_map = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }

        for split, split_images in split_map.items():
            target_dir = dataset_dir / split / class_name
            for image_path in split_images:
                target_path = target_dir / image_path.name
                shutil.copy2(image_path, target_path)

        print(f"{class_name:8s} -> train={n_train:4d}, val={n_val:4d}, test={n_test:4d}")


def dataset_ready(dataset_dir: Path, classes: list[str]) -> bool:
    for split in ("train", "val", "test"):
        for class_name in classes:
            if not list_images(dataset_dir / split / class_name):
                return False
    return True


def maybe_prepare_dataset() -> None:
    maybe_extract_dataset_zip(DATASET_ZIP, PROJECT_DIR)

    if dataset_ready(DATASET_DIR, CLASSES):
        print("Using existing dataset/train|val|test folders.")
        return

    if RAW_DATASET_DIR.exists():
        if DATASET_DIR.exists():
            shutil.rmtree(DATASET_DIR)
        build_dataset_split_from_raw(RAW_DATASET_DIR, DATASET_DIR, CLASSES, seed=SEED)
        return

    raise FileNotFoundError(
        "Dataset not found. Provide either a split dataset/ directory or raw_dataset/ with good, stained, and torn folders."
    )


def count_dataset_images(dataset_dir: Path, classes: list[str]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        summary[split] = {}
        for class_name in classes:
            summary[split][class_name] = len(list_images(dataset_dir / split / class_name))
    return summary


def print_dataset_summary(summary: dict[str, dict[str, int]]) -> None:
    print("\nDataset summary")
    print("=" * 50)
    for split, counts in summary.items():
        split_total = sum(counts.values())
        print(f"{split.upper():5s} total = {split_total}")
        for class_name in CLASSES:
            print(f"  {class_name:8s}: {counts.get(class_name, 0)}")
    print("=" * 50)


# %% [markdown]
# ## 3. Visual sanity check
# Preview a few training examples so you can catch obvious labeling or formatting errors before training.

# %%
def preview_training_images(dataset_dir: Path, classes: list[str], samples_per_class: int = 4) -> None:
    fig, axes = plt.subplots(len(classes), samples_per_class, figsize=(14, 9))
    fig.suptitle("Swipester dataset preview", fontsize=15, fontweight="bold")
    colors = {"good": "#4ade80", "stained": "#fbbf24", "torn": "#f87171"}

    for row, class_name in enumerate(classes):
        images = list_images(dataset_dir / "train" / class_name)[:samples_per_class]
        for col in range(samples_per_class):
            ax = axes[row][col] if len(classes) > 1 else axes[col]
            ax.axis("off")
            if col < len(images):
                img = Image.open(images[col]).convert("RGB")
                ax.imshow(img)
            if col == 0:
                ax.set_ylabel(class_name.upper(), color=colors[class_name], fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## 4. Data preprocessing and augmentation
# This section keeps the same process as the midterm plan:
#
# - resize to **224 × 224**
# - normalize inputs consistently for MobileNetV2
# - augment the training split with rotations, shifts, flips, zoom, and brightness variation
#
# The only refinement is that we use `preprocess_input`, which is better aligned with ImageNet-pretrained MobileNetV2 weights.

# %%
def make_generators(dataset_dir: Path):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        horizontal_flip=True,
        zoom_range=0.10,
        brightness_range=(0.8, 1.2),
        fill_mode="nearest",
    )
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        dataset_dir / "train",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=SEED,
    )
    val_generator = eval_datagen.flow_from_directory(
        dataset_dir / "val",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    test_generator = eval_datagen.flow_from_directory(
        dataset_dir / "test",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    return train_generator, val_generator, test_generator


def compute_training_class_weights(train_generator) -> dict[int, float]:
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_generator.classes),
        y=train_generator.classes,
    )
    class_weights = {class_index: float(weight) for class_index, weight in enumerate(weights)}
    print("\nClass weights")
    for class_name, class_index in train_generator.class_indices.items():
        print(f"  {class_name:8s}: {class_weights[class_index]:.3f}")
    return class_weights


# %% [markdown]
# ## 5. Model definition: MobileNetV2 transfer learning
# The model follows the process described in the deliverables:
#
# - MobileNetV2 base pretrained on ImageNet
# - remove the original classifier head
# - attach a light custom head for `good / stained / torn`
# - train the head first, then fine-tune deeper layers

# %%
def build_model():
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.20)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="swipester_mobilenetv2")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model


def unfreeze_top_layers(base_model: keras.Model, unfreeze_last_n: int = UNFREEZE_LAST_N) -> None:
    base_model.trainable = True
    for layer in base_model.layers[:-unfreeze_last_n]:
        layer.trainable = False

    # Keep BatchNorm layers frozen for more stable fine-tuning.
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False


def make_callbacks() -> list[keras.callbacks.Callback]:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            BEST_MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


# %% [markdown]
# ## 6. Phase-1 training and phase-2 fine-tuning
# Phase 1 trains the custom head while the MobileNetV2 base stays frozen.  
# Phase 2 unfreezes the top MobileNetV2 layers and fine-tunes them with a smaller learning rate.

# %%
def train_model(model, base_model, train_generator, val_generator, class_weights):
    callbacks = make_callbacks()

    print("\nStarting phase 1: frozen-base training")
    history1 = model.fit(
        train_generator,
        epochs=INITIAL_EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    print("\nStarting phase 2: fine-tuning")
    unfreeze_top_layers(base_model, UNFREEZE_LAST_N)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_generator,
        epochs=INITIAL_EPOCHS + FINE_TUNE_EPOCHS,
        initial_epoch=len(history1.history["loss"]),
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    return history1, history2


# %% [markdown]
# ## 7. Evaluation, plots, and confusion matrices
# We report validation and test accuracy, a class-wise precision/recall/F1 report, and confusion matrices.

# %%
def combine_histories(history1, history2) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for key in history1.history:
        merged[key] = history1.history.get(key, []) + history2.history.get(key, [])
    return merged


def plot_training_curves(history1, history2, output_path: Path) -> None:
    history = combine_histories(history1, history2)
    epochs = range(1, len(history["accuracy"]) + 1)
    phase_boundary = len(history1.history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Swipester MobileNetV2 training curves", fontsize=14, fontweight="bold")

    axes[0].plot(epochs, history["accuracy"], label="Train accuracy", linewidth=2)
    axes[0].plot(epochs, history["val_accuracy"], label="Validation accuracy", linewidth=2)
    axes[0].axvline(phase_boundary, linestyle="--", color="gray", alpha=0.7, label="Fine-tuning starts")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history["loss"], label="Train loss", linewidth=2)
    axes[1].plot(epochs, history["val_loss"], label="Validation loss", linewidth=2)
    axes[1].axvline(phase_boundary, linestyle="--", color="gray", alpha=0.7, label="Fine-tuning starts")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.show()


def evaluate_split(model, generator, split_name: str) -> dict:
    loss, accuracy = model.evaluate(generator, verbose=0)
    generator.reset()
    predictions = model.predict(generator, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = generator.classes
    class_names = list(generator.class_indices.keys())

    report = classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    print(f"\n{split_name.upper()} RESULTS")
    print("=" * 50)
    print(f"Loss:     {loss:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(true_classes, predicted_classes, target_names=class_names, zero_division=0))

    return {
        "loss": float(loss),
        "accuracy": float(accuracy),
        "true_classes": true_classes.tolist(),
        "predicted_classes": predicted_classes.tolist(),
        "class_names": class_names,
        "report": report,
    }


def save_confusion_matrix(
    true_classes: list[int],
    predicted_classes: list[int],
    class_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(true_classes, predicted_classes)
    cm_percent = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_percent = np.nan_to_num(cm_percent) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Raw counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Oranges", xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Row-normalized (%)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.show()


# %% [markdown]
# ## 8. Save the model for submission / deployment
# The exported files match the names already used in the earlier Swipester workflow:
#
# - `swipester_wear_tear_model.keras`
# - `swipester_wear_tear_model.tflite`
# - `labels.json`
# - `training_summary.json`

# %%
def export_models(best_model: keras.Model, class_names: list[str], summary_payload: dict) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    best_model.save(FINAL_KERAS_PATH)
    print(f"Saved Keras model -> {FINAL_KERAS_PATH}")

    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    FINAL_TFLITE_PATH.write_bytes(tflite_model)
    print(f"Saved TFLite model -> {FINAL_TFLITE_PATH}")

    LABELS_PATH.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    keras_size = FINAL_KERAS_PATH.stat().st_size / (1024 * 1024)
    tflite_size = FINAL_TFLITE_PATH.stat().st_size / (1024 * 1024)
    print(f"Keras size:  {keras_size:.2f} MB")
    print(f"TFLite size: {tflite_size:.2f} MB")


# %% [markdown]
# ## 9. Single-image inference helper
# This mirrors the seller-upload flow in the Swipester app.

# %%
def predict_clothing_image(model: keras.Model, image_path: str | Path, class_names: list[str]) -> tuple[str, float]:
    image_path = Path(image_path)
    img = keras_image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    probabilities = model.predict(img_array, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(keras_image.load_img(image_path))
    axes[0].axis("off")
    axes[0].set_title("Input image", fontweight="bold")

    colors = {"good": "#4ade80", "stained": "#fbbf24", "torn": "#f87171"}
    bar_colors = [colors[name] for name in class_names]
    bars = axes[1].barh(class_names, probabilities * 100, color=bar_colors)
    axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Confidence (%)")
    axes[1].set_title("Predicted probabilities", fontweight="bold")
    for bar, value in zip(bars, probabilities * 100):
        axes[1].text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center")

    plt.tight_layout()
    plt.show()

    return predicted_class, confidence


# %% [markdown]
# ## 10. Main training pipeline

# %%
def main() -> None:
    set_global_seed(SEED)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    maybe_prepare_dataset()
    summary = count_dataset_images(DATASET_DIR, CLASSES)
    print_dataset_summary(summary)
    preview_training_images(DATASET_DIR, CLASSES, samples_per_class=4)

    train_generator, val_generator, test_generator = make_generators(DATASET_DIR)
    class_weights = compute_training_class_weights(train_generator)

    model, base_model = build_model()
    history1, history2 = train_model(model, base_model, train_generator, val_generator, class_weights)
    plot_training_curves(history1, history2, TRAINING_PLOT_PATH)

    best_model = keras.models.load_model(BEST_MODEL_PATH)

    validation_results = evaluate_split(best_model, val_generator, "validation")
    test_results = evaluate_split(best_model, test_generator, "test")

    save_confusion_matrix(
        validation_results["true_classes"],
        validation_results["predicted_classes"],
        validation_results["class_names"],
        VAL_CM_PATH,
        "Validation confusion matrix",
    )
    save_confusion_matrix(
        test_results["true_classes"],
        test_results["predicted_classes"],
        test_results["class_names"],
        TEST_CM_PATH,
        "Test confusion matrix",
    )

    summary_payload = {
        "config": asdict(CONFIG),
        "dataset_summary": summary,
        "class_weights": class_weights,
        "validation": {
            "loss": validation_results["loss"],
            "accuracy": validation_results["accuracy"],
            "classification_report": validation_results["report"],
        },
        "test": {
            "loss": test_results["loss"],
            "accuracy": test_results["accuracy"],
            "classification_report": test_results["report"],
        },
        "artifacts": {
            "best_model": str(BEST_MODEL_PATH),
            "keras_model": str(FINAL_KERAS_PATH),
            "tflite_model": str(FINAL_TFLITE_PATH),
            "labels": str(LABELS_PATH),
            "training_plot": str(TRAINING_PLOT_PATH),
            "validation_confusion_matrix": str(VAL_CM_PATH),
            "test_confusion_matrix": str(TEST_CM_PATH),
        },
    }

    export_models(best_model, validation_results["class_names"], summary_payload)

    print("\nFinal submission summary")
    print("=" * 60)
    print("Model: MobileNetV2 transfer learning + fine-tuning")
    print(f"Classes: {', '.join(validation_results['class_names'])}")
    print(f"Validation accuracy: {validation_results['accuracy'] * 100:.2f}%")
    print(f"Test accuracy:       {test_results['accuracy'] * 100:.2f}%")
    print(f"Keras export:        {FINAL_KERAS_PATH}")
    print(f"TFLite export:       {FINAL_TFLITE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
