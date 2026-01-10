import optuna
from optuna.trial import TrialState

# Dane połączenia
DB_URL = "postgresql://optuna:password@51.83.132.188:5432/optuna_db"
STUDY_NAME = "timesformer_optimization"

def add_completed_trial(num_frames, accuracy):
    study = optuna.load_study(study_name=STUDY_NAME, storage=DB_URL)
    
    # Tworzymy nowy, gotowy rekord w bazie
    study.add_trial(
        optuna.trial.create_trial(
            params={"num_frames": num_frames},
            distributions={"num_frames": optuna.distributions.IntDistribution(4, 16)},
            value=accuracy,
            state=TrialState.COMPLETE,
        )
    )
    print(f"Dodano pomyślnie wynik {accuracy} dla num_frames={num_frames}")

if __name__ == "__main__":
    # WPISZ TUTAJ NUMER TRIALU, KTÓRY ZAWODZIŁ (prawdopodobnie 14)
    NUM_FRAMES = 10
    ACCURACY = 0.9609940232777603
    
    add_completed_trial(NUM_FRAMES, ACCURACY)