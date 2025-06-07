import pyaudio
import numpy as np
import tkinter as tk
from tkinter import messagebox
import time

class SineWaveApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sine Wave Generator")
        self.root.geometry("400x450")

        # --- Parametry strumienia ---
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.sample_rate = 22050
        self.frames_per_buffer = 1024

        # --- Zmienne sterujące odtwarzaniem ---
        self.playing = False
        self.phase = 0.0
        self.phase_increment = 0.0
        self.start_time = None
        self.duration = None
        self.elapsed_time = 0.0

        # --- GUI ---

        tk.Label(root, text="Frequency (Hz):").grid(row=0, column=0, padx=5, pady=5)
        self.freq_entry = tk.Entry(root)
        self.freq_entry.grid(row=0, column=1, padx=5, pady=5)
        self.freq_entry.insert(0, "440.0")

        tk.Label(root, text="Duration (s, optional):").grid(row=1, column=0, padx=5, pady=5)
        self.duration_entry = tk.Entry(root)
        self.duration_entry.grid(row=1, column=1, padx=5, pady=5)
        self.duration_entry.insert(0, "")

        tk.Label(root, text="Volume:").grid(row=2, column=0, padx=5, pady=5)
        self.volume = 1.0
        self.volume_scale = tk.Scale(
            root, from_=0.0, to=1.0, resolution=0.01,
            orient=tk.HORIZONTAL, command=self.on_volume_change
        )
        self.volume_scale.set(1.0)
        self.volume_scale.grid(row=2, column=1, padx=5, pady=5)

        self.time_label = tk.Label(root, text="Elapsed time: 0.0s")
        self.time_label.grid(row=3, column=0, columnspan=2, pady=10)

        self.start_button = tk.Button(root, text="Start", command=self.start_playback)
        self.start_button.grid(row=4, column=0, padx=5, pady=10)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_playback, state=tk.DISABLED)
        self.stop_button.grid(row=4, column=1, padx=5, pady=10)

        self.reset_button = tk.Button(root, text="Reset", command=self.reset_time, state=tk.DISABLED)
        self.reset_button.grid(row=5, column=0, columnspan=2, pady=5)

        # --- Sekwencja presetów ---

        tk.Label(root, text="Presets (freq Hz, duration s, volume):").grid(row=6, column=0, columnspan=2, pady=10)

        self.presets = [
            {"freq": 440, "duration": 2, "volume": 0.5},
            {"freq": 550, "duration": 1.5, "volume": 0.7},
            {"freq": 660, "duration": 2.5, "volume": 0.3},
            {"freq": 770, "duration": 1, "volume": 1.0},
            {"freq": 880, "duration": 3, "volume": 0.6},
        ]

        self.preset_entries = []
        for i, preset in enumerate(self.presets):
            freq_e = tk.Entry(root, width=6)
            freq_e.grid(row=7+i, column=0, padx=5)
            freq_e.insert(0, str(preset["freq"]))

            dur_e = tk.Entry(root, width=6)
            dur_e.grid(row=7+i, column=1, padx=5)
            dur_e.insert(0, str(preset["duration"]))

            vol_e = tk.Entry(root, width=6)
            vol_e.grid(row=7+i, column=2, padx=5)
            vol_e.insert(0, str(preset["volume"]))

            self.preset_entries.append((freq_e, dur_e, vol_e))

        # Nowy przycisk do odtwarzania sekwencji
        self.play_sequence_button = tk.Button(root, text="Play sequence", command=self.play_sequence)
        self.play_sequence_button.grid(row=12, column=0, columnspan=3, pady=15)

        # Label pokazujący aktualny preset w trakcie sekwencji
        self.current_preset_label = tk.Label(root, text="")
        self.current_preset_label.grid(row=13, column=0, columnspan=3, pady=10)

        # Do sterowania sekwencją
        self.sequence_index = 0
        self.sequence_playing = False

    def on_volume_change(self, val):
        self.volume = float(val)

    def validate_inputs(self):
        try:
            freq = float(self.freq_entry.get())
            if freq <= 0:
                raise ValueError("Frequency must be positive.")

            duration_text = self.duration_entry.get().strip()
            duration = None
            if duration_text:
                duration = float(duration_text)
                if duration <= 0:
                    raise ValueError("Duration must be positive.")

            return freq, duration
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            return None, None

    def audio_callback(self, in_data, frame_count, time_info, status):
        if not self.playing:
            return (None, pyaudio.paComplete)

        t = np.arange(frame_count, dtype=np.float32)
        data_block = (self.volume * np.sin(self.phase + self.phase_increment * t)).astype(np.float32)
        self.phase = (self.phase + self.phase_increment * frame_count) % (2 * np.pi)

        self.elapsed_time = time.time() - self.start_time
        if (self.duration is not None) and (self.elapsed_time >= self.duration):
            self.playing = False
            return (data_block.tobytes(), pyaudio.paComplete)

        return (data_block.tobytes(), pyaudio.paContinue)

    def update_time_label(self):
        if self.playing or self.sequence_playing:
            self.time_label.config(text=f"Elapsed time: {self.elapsed_time:.1f}s")
            self.root.after(50, self.update_time_label)
        else:
            self.time_label.config(text="Elapsed time: 0.0s")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.reset_button.config(state=tk.DISABLED)
            self.play_sequence_button.config(state=tk.NORMAL)
            self.current_preset_label.config(text="")

    def start_playback(self):
        if self.playing:
            return

        freq, duration = self.validate_inputs()
        if freq is None:
            return

        self.duration = duration
        self.phase = 0.0
        self.phase_increment = 2 * np.pi * freq / self.sample_rate
        self.start_time = time.time()
        self.elapsed_time = 0.0
        self.playing = True

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.reset_button.config(state=tk.NORMAL)
        self.play_sequence_button.config(state=tk.DISABLED)
        self.time_label.config(text="Elapsed time: 0.0s")

        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.frames_per_buffer,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()

        self.root.after(50, self.update_time_label)

    def stop_playback(self):
        if not self.playing and not self.sequence_playing:
            return

        self.playing = False
        self.sequence_playing = False

        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def reset_time(self):
        if self.playing:
            self.start_time = time.time()
            self.elapsed_time = 0.0
            self.time_label.config(text="Elapsed time: 0.0s")

    def cleanup(self):
        self.playing = False
        self.sequence_playing = False
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

    # --- Nowa funkcja do odtwarzania sekwencji presetów ---
    def play_sequence(self):
        if self.playing or self.sequence_playing:
            return  # już coś gra

        # Wczytujemy dane z GUI, walidujemy
        presets = []
        for i, (freq_e, dur_e, vol_e) in enumerate(self.preset_entries):
            try:
                f = float(freq_e.get())
                d = float(dur_e.get())
                v = float(vol_e.get())
                if f <= 0 or d <= 0 or not (0 <= v <= 1):
                    raise ValueError
                presets.append({"freq": f, "duration": d, "volume": v})
            except ValueError:
                messagebox.showerror("Invalid preset", f"Błąd w presetcie nr {i+1}")
                return

        self.presets = presets
        self.sequence_index = 0
        self.sequence_playing = True
        self.play_sequence_button.config(state=tk.DISABLED)
        self.current_preset_label.config(text="Starting sequence...")

        self.play_next_in_sequence()

    def play_next_in_sequence(self):
        if self.sequence_index >= len(self.presets):
            # Koniec sekwencji
            self.sequence_playing = False
            self.current_preset_label.config(text="Sequence finished")
            self.play_sequence_button.config(state=tk.NORMAL)
            return

        preset = self.presets[self.sequence_index]
        self.sequence_index += 1

        # Aktualizujemy label
        self.current_preset_label.config(
            text=f"Playing preset {self.sequence_index}: freq={preset['freq']} Hz, duration={preset['duration']}s, volume={preset['volume']}"
        )

        # Ustawiamy parametry do odtwarzania
        self.phase = 0.0
        self.phase_increment = 2 * np.pi * preset["freq"] / self.sample_rate
        self.duration = preset["duration"]
        self.volume = preset["volume"]
        self.volume_scale.set(self.volume)  # aktualizujemy suwak głośności dla info

        self.start_time = time.time()
        self.elapsed_time = 0.0
        self.playing = True

        # Otwieramy strumień
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()

        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=self.frames_per_buffer,
            stream_callback=self.audio_callback
        )
        self.stream.start_stream()

        # Odświeżamy GUI
        self.stop_button.config(state=tk.NORMAL)
        self.reset_button.config(state=tk.NORMAL)
        self.start_button.config(state=tk.DISABLED)

        # Monitorujemy zakończenie danego preset odtwarzania, odpalamy kolejny po duration
        self.root.after(int(self.duration * 1000) + 100, self.check_and_continue_sequence)

        self.root.after(50, self.update_time_label)

    def check_and_continue_sequence(self):
        # Po zakończeniu obecnego preset przechodzimy do następnego, jeśli sequence_playing nadal true
        if not self.playing and self.sequence_playing:
            # Zamykamy obecny stream
            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None

            # Kontynuujemy kolejnym presetem
            self.play_next_in_sequence()

if __name__ == "__main__":
    root = tk.Tk()
    app = SineWaveApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
    root.mainloop()
