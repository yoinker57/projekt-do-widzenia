import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TimesformerForVideoClassification,
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
)
import decord
from decord import VideoReader, cpu
from PIL import Image
from sklearn.metrics import accuracy_score
import random
import optuna
import sys

# Ensure decord works correctly
decord.bridge.set_bridge("torch")

# Configuration
DATA_DIR = "data"
FILMS_DIR = os.path.join(DATA_DIR, "films")
SPLIT_FILE = "split.csv"
MODEL_CKPT = "facebook/timesformer-base-finetuned-k400"
BATCH_SIZE = 4
RESIZE_TO = 224

# Default DB URL (can be overridden by env var)
# Using 'localhost' assumes running outside docker but connecting to dockerized DB mapped port.
# If running inside docker, use 'optuna-db' hostname.
DB_URL = os.environ.get(
    "OPTUNA_DB_URL", "postgresql://optuna:password@192.168.0.150:5432/optuna_db"
)
STUDY_NAME = "timesformer_optimization"


def parse_intervals(csv_path):
    """
    Parses a CSV file to find labeled intervals.
    Returns a list of dicts: {'label': label, 'start_frame': start, 'end_frame': end}
    """
    try:
        df = pd.read_csv(csv_path, header=None)
    except pd.errors.EmptyDataError:
        return []

    intervals = []
    if len(df) == 0:
        return intervals

    current_label = None
    start_frame = None

    # Assuming column 2 is label
    labels = df.iloc[:, 2].values

    for i, label in enumerate(labels):
        if label != -1:  # Valid label
            if current_label is None:
                current_label = label
                start_frame = i
            elif label != current_label:
                intervals.append(
                    {
                        "label": current_label,
                        "start_frame": start_frame,
                        "end_frame": i - 1,
                    }
                )
                current_label = label
                start_frame = i
        else:
            if current_label is not None:
                intervals.append(
                    {
                        "label": current_label,
                        "start_frame": start_frame,
                        "end_frame": i - 1,
                    }
                )
                current_label = None
                start_frame = None

    if current_label is not None:
        intervals.append(
            {
                "label": current_label,
                "start_frame": start_frame,
                "end_frame": len(labels) - 1,
            }
        )

    return intervals


def prepare_dataset(split_file):
    split_df = pd.read_csv(split_file)

    train_clips = []
    test_clips = []

    all_labels = set()

    print(f"Processing {len(split_df)} files...")

    for _, row in split_df.iterrows():
        file_id = row["id"]
        split_type = row["split"]

        csv_path = os.path.join(DATA_DIR, f"{file_id}.csv")
        video_path = os.path.join(FILMS_DIR, f"{file_id}.mp4")

        if not os.path.exists(csv_path) or not os.path.exists(video_path):
            print(f"Missing file for ID: {file_id}")
            continue

        intervals = parse_intervals(csv_path)

        for interval in intervals:
            clip = {
                "video_path": video_path,
                "start_frame": interval["start_frame"],
                "end_frame": interval["end_frame"],
                "label": interval["label"],
            }

            all_labels.add(interval["label"])

            if split_type == "train":
                train_clips.append(clip)
            else:
                test_clips.append(clip)

    return train_clips, test_clips, sorted(list(all_labels))


# Prepare data globally once (to avoid re-parsing every trial)
print("Loading dataset...")
train_clips_all, test_clips, unique_labels = prepare_dataset(SPLIT_FILE)

# Class Balancing Logic
TARGET_COUNT = 160
random.seed(42)
class_1_clips = [c for c in train_clips_all if c["label"] == 1]
other_clips = [c for c in train_clips_all if c["label"] != 1]

if len(class_1_clips) > TARGET_COUNT:
    print(f"Undersampling Class 1 from {len(class_1_clips)} to {TARGET_COUNT}...")
    class_1_clips = random.sample(class_1_clips, TARGET_COUNT)

train_clips_balanced = other_clips + class_1_clips
random.shuffle(train_clips_balanced)
print(f"Training on {len(train_clips_balanced)} clips (Balanced)")

label2id = {int(label): int(i) for i, label in enumerate(unique_labels)}
id2label = {int(i): int(label) for label, i in label2id.items()}


class ExerciseDataset(Dataset):
    def __init__(self, clips, processor, num_frames=8):
        self.clips = clips
        self.processor = processor
        self.num_frames = num_frames

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        video_path = clip["video_path"]
        start_f = clip["start_frame"]
        end_f = clip["end_frame"]
        label = clip["label"]

        try:
            vr = VideoReader(video_path, ctx=cpu(0))
        except Exception as e:
            print(f"Error reading {video_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        total_frames = len(vr)
        start_f = max(0, min(start_f, total_frames - 1))
        end_f = max(0, min(end_f, total_frames - 1))

        if start_f >= end_f:
            indices = [start_f] * self.num_frames
        else:
            indices = np.linspace(start_f, end_f, self.num_frames).astype(int)

        video = vr.get_batch(indices)
        if hasattr(video, "asnumpy"):
            video = video.asnumpy()
        else:
            video = video.numpy()

        inputs = self.processor(list(video), return_tensors="pt")

        return {
            "pixel_values": inputs.pixel_values.squeeze(),
            "labels": torch.tensor(label2id[label]),
        }


def objective(trial):
    # Hyperparameters to tune
    num_frames = trial.suggest_int("num_frames", 4, 16)

    print(f"\n--- Trial {trial.number}: NUM_FRAMES={num_frames} ---")

    processor = AutoImageProcessor.from_pretrained(MODEL_CKPT)
    model = TimesformerForVideoClassification.from_pretrained(
        MODEL_CKPT,
        num_labels=len(unique_labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    train_ds = ExerciseDataset(train_clips_balanced, processor, num_frames=num_frames)
    test_ds = ExerciseDataset(test_clips, processor, num_frames=num_frames)

    output_dir = f"optuna_results/trial_{trial.number}"

    args = TrainingArguments(
        output_dir=output_dir,
        remove_unused_columns=False,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        num_train_epochs=3,  # Keep epochs low for optimization speed, or increase for better results
        fp16=True if torch.cuda.is_available() else False,
        report_to="none",
        disable_tqdm=True,  # Less noise in logs
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {"accuracy": accuracy_score(labels, predictions)}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    accuracy = metrics["eval_accuracy"]

    return accuracy


if __name__ == "__main__":
    print(f"Connecting to Optuna Storage: {DB_URL}")

    # Retry connection logic could be added, but letting it fail fast is okay for now
    try:
        study = optuna.create_study(
            study_name=STUDY_NAME,
            storage=DB_URL,
            load_if_exists=True,
            direction="maximize",
        )
        print("Study loaded/created.")

        # Run optimization
        # Each worker runs N trials.
        # User can run this script in parallel on multiple machines.
        n_trials = 10
        print(f"Running {n_trials} trials...")
        study.optimize(objective, n_trials=n_trials)

        print("Best params:", study.best_params)
        print("Best value:", study.best_value)

    except Exception as e:
        print(f"Error connecting to DB or running study: {e}")
        print("Ensure the Docker container is running: docker-compose up -d")
        sys.exit(1)
