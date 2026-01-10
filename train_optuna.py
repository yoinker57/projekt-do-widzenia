import os
import traceback
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
from optuna.storages import RDBStorage
from datetime import datetime, timedelta
import sys

# Konfiguracja decord
decord.bridge.set_bridge("torch")

# Konfiguracja
DATA_DIR = "data"
FILMS_DIR = os.path.join(DATA_DIR, "films")
SPLIT_FILE = "split.csv"
MODEL_CKPT = "facebook/timesformer-base-finetuned-k400"
BATCH_SIZE = 4
DB_URL = os.environ.get("OPTUNA_DB_URL", "postgresql://optuna:password@51.83.132.188:5432/optuna_db")
STUDY_NAME = "timesformer_optimization"

# --- Logika ładowania danych (bez zmian) ---
def parse_intervals(csv_path):
    try:
        df = pd.read_csv(csv_path, header=None)
    except Exception: return []
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
        elif current_label is not None:
            intervals.append({"label": current_label, "start_frame": start_frame, "end_frame": i - 1})
            current_label, start_frame = None, None
    if current_label is not None:
        intervals.append({"label": current_label, "start_frame": start_frame, "end_frame": len(labels) - 1})
    return intervals

def prepare_dataset(split_file):
    split_df = pd.read_csv(split_file)
    train_clips, test_clips, all_labels = [], [], set()
    for _, row in split_df.iterrows():
        f_id = row["id"]
        csv_p, vid_p = os.path.join(DATA_DIR, f"{f_id}.csv"), os.path.join(FILMS_DIR, f"{f_id}.mp4")
        if not os.path.exists(csv_p) or not os.path.exists(vid_p): continue
        intervals = parse_intervals(csv_p)
        for interval in intervals:
            clip = {"video_path": vid_p, "start_frame": interval["start_frame"], "end_frame": interval["end_frame"], "label": interval["label"]}
            all_labels.add(interval["label"])
            if row["split"] == "train": train_clips.append(clip)
            else: test_clips.append(clip)
    return train_clips, test_clips, sorted(list(all_labels))

print("Ładowanie danych...")
train_clips_all, test_clips, unique_labels = prepare_dataset(SPLIT_FILE)
TARGET_COUNT = 160
random.seed(42)
c1 = [c for c in train_clips_all if c["label"] == 1]
others = [c for c in train_clips_all if c["label"] != 1]
if len(c1) > TARGET_COUNT: c1 = random.sample(c1, TARGET_COUNT)
train_clips_balanced = others + c1
random.shuffle(train_clips_balanced)

label2id = {int(l): i for i, l in enumerate(unique_labels)}
id2label = {i: int(l) for i, l in enumerate(unique_labels)}

class ExerciseDataset(Dataset):
    def __init__(self, clips, processor, num_frames=8):
        self.clips, self.processor, self.num_frames = clips, processor, num_frames
    def __len__(self): return len(self.clips)
    def __getitem__(self, idx):
        clip = self.clips[idx]
        try:
            vr = VideoReader(clip["video_path"], ctx=cpu(0))
            idx_pts = np.linspace(max(0, clip["start_frame"]), min(clip["end_frame"], len(vr)-1), self.num_frames).astype(int)
            video = vr.get_batch(idx_pts).numpy()
            inputs = self.processor(list(video), return_tensors="pt")
            return {"pixel_values": inputs.pixel_values.squeeze(), "labels": torch.tensor(label2id[clip["label"]])}
        except Exception: return self.__getitem__((idx + 1) % len(self))

# --- GŁÓWNA FUNKCJA CELU ---
def objective(trial):
    # 1. Losujemy parametr
    num_frames = trial.suggest_int("num_frames", 4, 16)

    # 2. Blokada duplikacji: Sprawdzamy czy ktoś już to liczy lub policzył
    # Pobieramy wszystkie triale ze studia
    all_trials = trial.study.get_trials(deepcopy=False)
    
    for t in all_trials:
        # Pomijamy samego siebie
        if t.number == trial.number:
            continue
        
        # Sprawdzamy czy inny trial ma te same parametry
        if t.params.get("num_frames") == num_frames:
            # Jeśli zakończony - nie ma sensu powtarzać
            if t.state == optuna.trial.TrialState.COMPLETE:
                print(f"Pominięto num_frames={num_frames} - wynik już jest w bazie (Trial {t.number})")
                raise optuna.exceptions.TrialPruned()
            
            # Jeśli w toku - sprawdzamy czy nie "zawisł" (np. starszy niż 3h)
            if t.state == optuna.trial.TrialState.RUNNING:
                if t.datetime_start and datetime.now() - t.datetime_start < timedelta(hours=3):
                    print(f"Pominięto num_frames={num_frames} - ktoś inny właśnie to liczy (Trial {t.number})")
                    raise optuna.exceptions.TrialPruned()

    print(f"\n>>> Rozpoczynam trening dla NUM_FRAMES={num_frames} (Trial {trial.number})")

    processor = AutoImageProcessor.from_pretrained(MODEL_CKPT)
    model = TimesformerForVideoClassification.from_pretrained(
        MODEL_CKPT, num_labels=len(unique_labels), id2label=id2label, label2id=label2id,
        num_frames=num_frames, ignore_mismatched_sizes=True
    )

    train_ds = ExerciseDataset(train_clips_balanced, processor, num_frames=num_frames)
    test_ds = ExerciseDataset(test_clips, processor, num_frames=num_frames)

    output_dir = f"optuna_results/frames_{num_frames}"
    args = TrainingArguments(
        output_dir=output_dir, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=1, learning_rate=5e-5, per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE, gradient_accumulation_steps=4,
        warmup_ratio=0.1, num_train_epochs=3, fp16=torch.cuda.is_available(),
        logging_steps=10, load_best_model_at_end=True, metric_for_best_model="accuracy",
        report_to="none", disable_tqdm=True
    )

    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=test_ds,
        compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1))}
    )

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final_model"))
    
    accuracy = trainer.evaluate()["eval_accuracy"]
    return accuracy

if __name__ == "__main__":
    storage = RDBStorage(
        url=DB_URL,
        heartbeat_interval=60,
        grace_period=120,
        engine_kwargs={"pool_pre_ping": True, "pool_recycle": 1800}
    )

    try:
        study = optuna.create_study(
            study_name=STUDY_NAME, storage=storage, load_if_exists=True, direction="maximize"
        )
        # Uruchamiamy pętlę - n_trials=20, ale dzięki blokadzie i tak przeliczy tylko unikalne
        study.optimize(objective, n_trials=20) 

    except Exception as e:
        print(f"Błąd: {e}")
        traceback.print_exc()
        sys.exit(1)