# Python Vibration Test Signal Generator

## Overview

This project is a desktop application for generating and playing audio signals designed for vibration testing. The user interface is built with the **NiceGUI** framework, while audio processing is handled by **PyAudio** and **NumPy**.

The application supports two main signal generation modes:
1.  **Sine Wave**: Generates a pure sinusoidal signal at a user-specified frequency.
2.  **Random Sloped**: Generates random noise with a power spectral density (PSD) shaped by a set of reference points, simulating specific vibration profiles.

It allows for real-time control over frequency, volume, and duration, and can play a sequence of presets. The application also provides real-time metrics, such as RMS (Root Mean Square) and PSD error, making it a useful tool for engineering and testing environments.

---

## Features

* **Dual Signal Modes**: Generate either a pure sine wave or random noise with a sloped power spectrum.
* **Real-Time Control**: Interactively start, stop, and cancel signal playback.
* **Configurable Parameters**:
    * **Frequency**: Set the frequency for the sine wave signal.
    * **Duration**: Define the test duration in seconds (0 for infinite playback).
    * **Volume**: Adjust the output volume using a slider and text input (0-1 range).
* **Preset Sequencing**: Play a pre-defined sequence of test signals with varying durations, frequencies, and volumes.
* **Live Metrics Display**:
    * **Playback Time**: A running timer shows the elapsed time.
    * **RMS Calculation**: Computes and displays the signal's Root Mean Square value in real-time.
    * **PSD Error**: For random signals, it calculates the percentage error between the generated signal's Power Spectral Density and the target spectrum.
* **User-Friendly GUI**: An intuitive interface built with NiceGUI, featuring clear input fields, controls, and status displays.

---

## Technology Stack

* **Language**: Python 3.8+
* **GUI Framework**: NiceGUI
* **Audio Processing**: PyAudio, NumPy, SciPy

---

## How to Run

### Prerequisites

* Python 3.8 or higher.
* A system with a working sound card.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-name>
    ```

2.  **Install the required libraries:**
    ```bash
    pip install nicegui pyaudio numpy scipy
    ```

3.  **Run the application:**
    ```bash
    python sine_wave_generator.py
    ```

### Usage

1.  Open your web browser and navigate to `http://localhost:8080`.
2.  Select the desired signal mode (`sine` or `random`).
3.  Enter the required parameters (frequency for sine, duration for random, etc.).
4.  Click **Start** to begin playback.
5.  Use the **Stop** and **Cancel** buttons to control the test.
6.  To run a series of tests, click **Play sequence**.

---

## Code Structure

The application is encapsulated within the `VibrationTestApp` class in `sine_wave_generator.py`.

* **`__init__(self)`**: Initializes audio parameters, UI state variables, and calls `build_ui`.
* **`build_ui(self)`**: Constructs the entire NiceGUI user interface, including input fields, buttons, sliders, and labels for metrics.
* **`validate_inputs(self, ...)`**: Parses and validates user input for frequency, duration, and volume.
* **`generate_noise_sloped(self, ...)`**: Creates the random noise signal. It uses logarithmic interpolation between defined PSD points (`[20Hz, 0.01]` to `[2000Hz, 0.01]`) to shape the spectrum of the generated signal.
* **`audio_callback(self, ...)`**: The core PyAudio callback function. It generates audio chunks on-the-fly for the sine wave or serves pre-generated chunks for the random signal.
* **`start_playback(self, ...)`**: Initializes and starts the PyAudio stream after validating inputs and (for random mode) generating the signal data.
* **`stop_playback(self)` / `toggle_playback(self)` / `cancel_playback(self)`**: Handle the playback state (stopping, pausing, resuming, canceling).
* **`play_sequence(self)`**: An asynchronous method that iterates through the `presets` list, configuring and running each test in sequence.
* **`update_time_and_metrics(self)`**: A timer-driven function that updates the elapsed time display and calculates real-time RMS and PSD error.
* **`cleanup(self)`**: A shutdown hook to properly close the PyAudio stream and terminate resources.
