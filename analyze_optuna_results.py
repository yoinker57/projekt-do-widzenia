import os
import re
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from transformers import (
    TimesformerForVideoClassification,
    AutoImageProcessor,
    Trainer,
    TrainingArguments,
)
import decord
from decord import VideoReader, cpu
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    recall_score,
)
from tqdm import tqdm

# Configuration
OPTUNA_RESULTS_DIR = "optuna_results"
DATA_DIR = "data"
FILMS_DIR = os.path.join(DATA_DIR, "films")
SPLIT_FILE = "split.csv"
BATCH_SIZE = 4

# Decord setup
decord.bridge.set_bridge("torch")


def parse_intervals(csv_path):
    try:
        df = pd.read_csv(csv_path, header=None)
    except Exception:
        return []
    intervals = []
    if len(df) == 0:
        return intervals
    current_label, start_frame = None, None
    labels = df.iloc[:, 2].values
    for i, label in enumerate(labels):
        if label != -1:
            if current_label is None:
                current_label, start_frame = label, i
            elif label != current_label:
                intervals.append(
                    {
                        "label": current_label,
                        "start_frame": start_frame,
                        "end_frame": i - 1,
                    }
                )
                current_label, start_frame = label, i
        else:
            if current_label is not None:
                intervals.append(
                    {
                        "label": current_label,
                        "start_frame": start_frame,
                        "end_frame": i - 1,
                    }
                )
                current_label, start_frame = None, None
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
    test_clips = []
    all_labels = set()
    for _, row in split_df.iterrows():
        file_id, split_type = row["id"], row["split"]
        csv_path = os.path.join(DATA_DIR, f"{file_id}.csv")
        video_path = os.path.join(FILMS_DIR, f"{file_id}.mp4")
        if not os.path.exists(csv_path) or not os.path.exists(video_path):
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
            if split_type != "train":
                test_clips.append(clip)
    return test_clips, sorted(list(all_labels))


class ExerciseDataset(Dataset):
    def __init__(self, clips, processor, num_frames=8):
        self.clips = clips
        self.processor = processor
        self.num_frames = num_frames
        self.label2id = {
            int(label): int(i)
            for i, label in enumerate(sorted(list(set(c["label"] for c in clips))))
        }

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        clip = self.clips[idx]
        try:
            vr = VideoReader(clip["video_path"], ctx=cpu(0))
            total_frames = len(vr)
            start_f = max(0, min(clip["start_frame"], total_frames - 1))
            end_f = max(0, min(clip["end_frame"], total_frames - 1))

            if start_f >= end_f:
                indices = [start_f] * self.num_frames
            else:
                indices = np.linspace(start_f, end_f, self.num_frames).astype(int)

            video = vr.get_batch(indices).numpy()
            inputs = self.processor(list(video), return_tensors="pt")
            return {
                "pixel_values": inputs.pixel_values.squeeze(),
                "labels": torch.tensor(self.label2id[clip["label"]]),
            }
        except Exception as e:
            print(f"Error loading {clip['video_path']}: {e}")
            # Fallback to next item
            return self.__getitem__((idx + 1) % len(self))


def main():
    # 1. Discover models in optuna_results
    if not os.path.exists(OPTUNA_RESULTS_DIR):
        print(f"Directory {OPTUNA_RESULTS_DIR} not found.")
        return

    frames_dirs = []
    for d in os.listdir(OPTUNA_RESULTS_DIR):
        match = re.search(r"frames_(\d+)", d)
        if match:
            num_frames = int(match.group(1))
            frames_dirs.append((num_frames, os.path.join(OPTUNA_RESULTS_DIR, d)))

    # Sort by number of frames
    frames_dirs.sort(key=lambda x: x[0])

    print(f"Found {len(frames_dirs)} models: {[f[0] for f in frames_dirs]} frames.")

    results = []
    test_clips, unique_labels = prepare_dataset(SPLIT_FILE)
    label2id = {int(label): int(i) for i, label in enumerate(unique_labels)}
    id2label = {int(i): int(label) for label, i in label2id.items()}

    processor = AutoImageProcessor.from_pretrained(
        "facebook/timesformer-base-finetuned-k400"
    )

    for num_frames, model_dir in frames_dirs:
        model_path = os.path.join(model_dir, "final_model")
        if not os.path.exists(model_path):
            print(f"Skipping {model_dir}: final_model not found.")
            continue

        print(f"\nProcessing model with {num_frames} frames from {model_path}...")

        try:
            model = TimesformerForVideoClassification.from_pretrained(model_path)
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            continue

        test_ds = ExerciseDataset(test_clips, processor, num_frames=num_frames)
        # Ensure the dataset uses the global label map
        test_ds.label2id = label2id

        args = TrainingArguments(
            output_dir=os.path.join("temp_eval", f"frames_{num_frames}"),
            per_device_eval_batch_size=BATCH_SIZE,
            fp16=True if torch.cuda.is_available() else False,
            report_to="none",
        )

        trainer = Trainer(model=model, args=args)

        print(f"Evaluating...")
        output = trainer.predict(test_ds)
        y_true = output.label_ids
        y_pred = np.argmax(output.predictions, axis=1)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")

        results.append(
            {"num_frames": num_frames, "accuracy": acc, "f1": f1, "recall": recall}
        )

        print(
            f"Results for {num_frames} frames: Acc={acc:.4f}, F1={f1:.4f}, Recall={recall:.4f}"
        )

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[id2label[i] for i in range(len(unique_labels))],
            yticklabels=[id2label[i] for i in range(len(unique_labels))],
        )
        plt.title(f"Confusion Matrix - {num_frames} Frames")
        plt.xlabel("Prediction")
        plt.ylabel("True Label")
        cm_filename = f"matrix_{num_frames}.png"
        plt.savefig(cm_filename)
        plt.close()
        print(f"Saved confusion matrix to {cm_filename}")

    # Plot aggregated metrics
    if results:
        df_res = pd.DataFrame(results)
        print("\nAggregated Results:")
        print(df_res)

        plt.figure(figsize=(10, 6))
        plt.plot(df_res["num_frames"], df_res["accuracy"], marker="o", label="Accuracy")
        plt.plot(df_res["num_frames"], df_res["f1"], marker="s", label="F1-Score")
        plt.plot(df_res["num_frames"], df_res["recall"], marker="^", label="Recall")

        plt.xlabel("Number of Frames")
        plt.ylabel("Score")
        plt.title("Metrics vs Number of Frames")
        plt.legend()
        plt.grid(True)
        plt.savefig("metrics_vs_frames.png")
        plt.close()
        print("Saved metrics plot to metrics_vs_frames.png")
    else:
        print("No results to plot.")


if __name__ == "__main__":
    main()
