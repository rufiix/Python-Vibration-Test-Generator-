# README: Aplikacja do Testów Wibracji

## Opis
Aplikacja **Vibration Test App** to narzędzie do generowania i odtwarzania sygnałów dźwiękowych w celu testowania wibracji. Umożliwia generowanie sygnału sinusoidalnego lub losowego z zadanym widmem mocy (Random Sloped), z możliwością sterowania częstotliwością, głośnością i czasem trwania. Aplikacja pozwala również na odtwarzanie sekwencji presetów z różnymi czasami trwania i poziomami głośności. Interfejs użytkownika jest oparty na bibliotece **NiceGUI**, a przetwarzanie dźwięku wykorzystuje **PyAudio** i **NumPy**.

## Funkcjonalności
- **Tryby sygnału**: 
  - Sygnał sinusoidalny o zadanej częstotliwości.
  - Sygnał losowy (Random Sloped) z widmem mocy interpolowanym na podstawie punktów referencyjnych.
- **Parametry wejściowe**:
  - Częstotliwość (dla sygnału sinusoidalnego, w Hz).
  - Czas trwania (w sekundach, opcjonalnie nieograniczony).
  - Głośność (w zakresie 0–1, regulowana suwakiem i polem tekstowym).
- **Sekwencja presetów**: Odtwarzanie sekwencji sygnałów losowych z predefiniowanymi czasami trwania i poziomami głośności.
- **Metryki w czasie rzeczywistym**:
  - Wyświetlanie czasu odtwarzania.
  - Obliczanie RMS (wartości skutecznej sygnału, w g).
  - Obliczanie błędu PSD (Power Spectral Density) dla sygnału losowego w odniesieniu do zadanego widma.
- **Interfejs graficzny**:
  - Intuicyjne pola wejściowe, suwak głośności, przyciski sterujące.
  - Dynamiczne wyświetlanie metryk i statusu sekwencji.

## Wymagania
- Python 3.8+
- Biblioteki:
  - `nicegui`
  - `pyaudio`
  - `numpy`
  - `scipy`
- System z obsługą dźwięku (np. karta dźwiękowa).

## Instalacja
1. Sklonuj repozytorium lub pobierz plik źródłowy.
2. Zainstaluj wymagane biblioteki:
   ```bash
   pip install nicegui pyaudio numpy scipy
   ```
3. Uruchom aplikację:
   ```bash
   python vibration_test_app.py
   ```

## Użycie
1. Uruchom aplikację. Interfejs będzie dostępny pod adresem `http://localhost:8080`.
2. Wybierz tryb sygnału (sine/random).
3. Wprowadź parametry:
   - Dla sygnału sinusoidalnego: częstotliwość (np. 440 Hz).
   - Dla sygnału losowego: czas trwania (np. 8 s).
   - Głośność (0–1, domyślnie 1.0).
4. Kliknij **Start**, aby rozpocząć odtwarzanie.
5. Użyj przycisku **Stop**, aby zatrzymać, lub **Reset**, aby zresetować czas.
6. Aby odtworzyć sekwencję presetów, kliknij **Play sequence**.
7. Monitoruj metryki (czas, RMS, błąd PSD) w czasie rzeczywistym.

## Struktura kodu
- **Klasa `VibrationTestApp`**:
  - Inicjalizacja parametrów audio i stanu odtwarzania.
  - Budowa interfejsu użytkownika (`build_ui`).
  - Walidacja danych wejściowych (`validate_inputs`).
  - Generowanie sygnału losowego z zadanym widmem (`generate_noise_sloped`).
  - Obsługa odtwarzania dźwięku (`audio_callback`, `start_playback`, `stop_playback`).
  - Aktualizacja metryk w czasie rzeczywistym (`update_time_and_metrics`).
  - Odtwarzanie sekwencji presetów (`play_sequence`).
  - Sprzątanie zasobów (`cleanup`).
- **Parametry audio**:
  - Częstotliwość próbkowania: 22050 Hz.
  - Rozmiar bufora: 1024 ramki.
  - Docelowy RMS sygnału losowego: 6.9 g.
- **Widmo PSD** (dla sygnału losowego):
  - Punkty referencyjne: od 20 Hz (0.01) do 2000 Hz (0.01).
  - Interpolacja logarytmiczna dla ciągłego widma.

## Uwagi
- Sygnał losowy wymaga zdefiniowanego czasu trwania.
- Metryki RMS i PSD są obliczane tylko po upływie 1 sekundy odtwarzania.
- Głośność można regulować w czasie rzeczywistym za pomocą suwaka.
- Aplikacja automatycznie zatrzymuje odtwarzanie po osiągnięciu zadanego czasu (jeśli określony).
- W przypadku błędów wejściowych (np. ujemna częstotliwość), wyświetlane są powiadomienia.

