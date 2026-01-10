# Projekt z przedmiotu "Widzenie komputerowe"
Autorzy: Jakub Bizan, Kacper Jurek, Jakub Kroczek, Radosław Niżnik, Piotr Wolanin

Wykorzystaliśmy model TimeSformer do predykcji ćwiczeń ze zbioru danych otrzymanych na zajęciach

## Środowisko testowe
Wykorzystaliśmy framework Optuna do trenowania hiperparametru (liczby klatek), wstępnie uruchomiliśmy modele na lokalnych zasobach. Z powodu zbyt długiego czasu trwania obliczeń, przenieśliśmy trenowanie modeli na klaster obliczeniowy Ares.

## Wyniki obliczeń

Poniżej umieszczone porównanie modeli ze zmienną liczbą klatek, najlepsze wyniki wyszły dla liczby klatek równej 6, a drugie najlepsze dla liczby klatek równej 14

![](metrics_vs_frames.png)

Macierz pomyłek dla najlepszej liczby klatek (6)

![](matrix_6.png)
