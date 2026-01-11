# Projekt z przedmiotu "Widzenie komputerowe"
Autorzy: Jakub Bizan, Kacper Jurek, Jakub Kroczek, Radosław Niżnik, Piotr Wolanin

Wykorzystaliśmy model TimeSformer do predykcji ćwiczeń ze zbioru danych otrzymanych na zajęciach

## Środowisko testowe
Wykorzystaliśmy framework Optuna do trenowania hiperparametru (liczby klatek), wstępnie uruchomiliśmy modele na lokalnych zasobach aby sprawdzić czy konfiguracja i wersje bibliotek są poprawne. Później postawiliśmy bazę i stworzyliśmy klaster obliczeniowy na własnych komputerach (wykorzystywaliśmy do liczenia RTX5070 oraz RTX3070), a później już lokalnie na jednym komputerze testowaliśmy działanie modeli.

## Wyniki obliczeń

Poniżej umieszczone porównanie modeli ze zmienną liczbą klatek, w sumie wytrenowaliśmy i zewaluowaliśmy 13 modeli z różną wartością hiperparametru (liczby klatek). Najlepsze wyniki wyszły dla liczby klatek równej 6, a drugie najlepsze dla liczby klatek równej 14

![](metrics_vs_frames.png)

Macierz pomyłek dla najlepszej liczby klatek (6)

![](matrix_6.png)
