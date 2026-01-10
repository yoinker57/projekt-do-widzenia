import os
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
    TrainingArguments
)
import decord
from decord import VideoReader, cpu
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score
import random

# Konfiguracja decord
decord.bridge.set_bridge("torch")

# --- PARAMETRY ---
MODEL_PATH = "optuna_results/frames_5_trial_0/final_model"
DATA_DIR = "data"
FILMS_DIR = os.path.join(DATA_DIR, "films")
SPLIT_FILE = "split.csv"
BATCH_SIZE = 4
NUM_FRAMES = 5  # Upewnij się, że to odpowiada ustawieniom z Trial 0

# --- LOGIKA PRZYGOTOWANIA DANYCH (z Twojego kodu) ---

def parse_intervals(csv_path):
    try:
        df = pd.read_csv(csv_path, header=None)
    except Exception:
        return []
    intervals = []
    if len(df) == 0: return intervals
    current_label, start_frame = None, None
    labels = df.iloc[:, 2].values
    for i, label in enumerate(labels):
        if label != -1:
            if current_label is None:
                current_label, start_frame = label, i
            elif label != current_label:
                intervals.append({"label": current_label, "start_frame": start_frame, "end_frame": i - 1})
                current_label, start_frame = label, i
        else:
            if current_label is not None:
                intervals.append({"label": current_label, "start_frame": start_frame, "end_frame": i - 1})
                current_label, start_frame = None, None
    if current_label is not None:
        intervals.append({"label": current_label, "start_frame": start_frame, "end_frame": len(labels) - 1})
    return intervals

def prepare_dataset(split_file):
    split_df = pd.read_csv(split_file)
    test_clips = []
    all_labels = set()
    for _, row in split_df.iterrows():
        file_id, split_type = row["id"], row["split"]
        csv_path = os.path.join(DATA_DIR, f"{file_id}.csv")
        video_path = os.path.join(FILMS_DIR, f"{file_id}.mp4")
        if not os.path.exists(csv_path) or not os.path.exists(video_path): continue
        intervals = parse_intervals(csv_path)
        for interval in intervals:
            clip = {"video_path": video_path, "start_frame": interval["start_frame"], 
                    "end_frame": interval["end_frame"], "label": interval["label"]}
            all_labels.add(interval["label"])
            if split_type != "train":
                test_clips.append(clip)
    return test_clips, sorted(list(all_labels))

# Pobieramy etykiety i klipy testowe
test_clips, unique_labels = prepare_dataset(SPLIT_FILE)
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
                "labels": torch.tensor(label2id[clip["label"]]),
            }
        except Exception as e:
            print(f"Błąd ładowania {clip['video_path']}: {e}")
            return self.__getitem__((idx + 1) % len(self))

# --- GŁÓWNA FUNKCJA EWALUACJI ---

def evaluate_saved_model():
    print(f"Wczytywanie modelu z: {MODEL_PATH}")
    
    # Wczytujemy procesor z bazy (aby uniknąć błędu braku pliku w folderze checkpointu)
    processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
    model = TimesformerForVideoClassification.from_pretrained(MODEL_PATH)
    
    test_ds = ExerciseDataset(test_clips, processor, num_frames=NUM_FRAMES)
    
    args = TrainingArguments(
        output_dir="./temp_eval",
        per_device_eval_batch_size=BATCH_SIZE,
        fp16=True if torch.cuda.is_available() else False,
        report_to="none"
    )
    
    trainer = Trainer(model=model, args=args)

    print(f"Analiza {len(test_ds)} klipów testowych...")
    output = trainer.predict(test_ds)
    
    y_true = output.label_ids
    y_pred = np.argmax(output.predictions, axis=1)

    # Obliczanie metryk
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    
    print("\n" + "="*40)
    print(f"WYNIKI EWALUACJI:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Recall:   {recall:.4f}")
    print("="*40)

    # Szczegółowy raport klasyfikacji
    target_names = [f"Klasa {id2label[i]}" for i in range(len(unique_labels))]
    print("\nSzczegółowy raport per klasa:")
    print(classification_report(y_true, y_pred, target_names=target_names))

    # Generowanie macierzy pomyłek
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=[id2label[i] for i in range(len(unique_labels))],
        yticklabels=[id2label[i] for i in range(len(unique_labels))]
    )
    plt.title(f'Macierz Pomyłek - {os.path.basename(MODEL_PATH)}')
    plt.xlabel('Predykcja (Model)')
    plt.ylabel('Prawda (Etykieta)')
    
    # Zapis i wyświetlenie
    plt.savefig("macierz_pomylek.png")
    print("\nMacierz pomyłek została zapisana do pliku: macierz_pomylek.png")
    plt.show()

if __name__ == "__main__":
    evaluate_saved_model()