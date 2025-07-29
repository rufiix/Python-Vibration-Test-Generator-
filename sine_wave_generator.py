import nicegui
from nicegui import ui
import pyaudio
import numpy as np
import asyncio
import time
from scipy.signal import welch
from collections import deque

class VibrationTestApp:
    def __init__(self):
        # Parametry audio / DSP
        self.sample_rate = 22050
        self.frames_per_buffer = 1024
        self.target_rms = 6.9
        self.p = pyaudio.PyAudio()
        self.stream = None

        # Stan pojedynczego sygnału
        self.playing = False
        self.paused = False
        self.start_time = 0.0
        self.elapsed = 0.0
        self.duration = 0.0
        self.volume = 1.0
        self.phase = 0.0
        self.phase_inc = 0.0
        self.noise_data = None
        self.noise_idx = 0

        # Stan sekwencji
        self.sequence_playing = False
        self.seq_paused = False
        self.seq_current = 0
        self.seq_elapsed = 0.0

        # Bufor do analizy na żywo
        self.record_buffer = deque(maxlen=int(5 * self.sample_rate / self.frames_per_buffer))

        # Ustawienia predefiniowane (presety)
        self.presets = [
            {'duration': 2.0, 'frequency': 200.0, 'volume': 0.5},
            {'duration': 1.5, 'frequency': 440.0, 'volume': 0.7},
            {'duration': 2.5, 'frequency': 800.0, 'volume': 0.3},
            {'duration': 1.0, 'frequency': 1000.0, 'volume': 1.0},
            {'duration': 3.0, 'frequency': 1500.0, 'volume': 0.6},
        ]

        self.build_ui()
        ui.timer(0.2, self.update_time_and_metrics)

    def build_ui(self):
        ui.label('Vibration Test: Sine & Random Sloped')\
          .style('font-size:1.2em; margin-bottom:10px;')

        with ui.row():
            ui.label('Signal mode:')
            self.mode = ui.radio(['sine', 'random'], value='sine')

        with ui.row():
            ui.label('Frequency (Hz)')
            self.freq_input = ui.input(value='440.0').props('type=number step=1')

        with ui.row():
            ui.label('Duration (s)')
            self.duration_input = ui.input(value='8.0').props('type=number step=0.1')

        with ui.row():
            ui.label('Volume (0–1)')
            self.volume_input = ui.input(value='1.0').props('type=number step=0.01')
            ui.slider(min=0, max=1, step=0.01)\
              .bind_value(self.volume_input, 'value')\
              .bind_value(self, 'volume')

        ui.button('Start', on_click=self.start_playback)
        self.stop_resume_btn = ui.button('Stop', on_click=self.toggle_playback)\
                                  .props('color=warning')\
                                  .bind_visibility_from(self, 'playing')
        self.cancel_btn = ui.button('Cancel', on_click=self.cancel_playback)\
                                  .props('color=negative')\
                                  .bind_visibility_from(self, 'playing')
        ui.button('Reset Time', on_click=self.reset_time)\
          .bind_visibility_from(self, 'playing')

        ui.separator()
        self.time_label = ui.label('Time: 0.0 s')
        self.rms_label = ui.label('RMS (g): –')
        self.psd_label = ui.label('PSD err (%): –')
        ui.separator()

        ui.label('Sequence presets:')
        for p in self.presets:
            with ui.row().style('align-items:center; gap:20px;'):
                ui.input(value=str(p['duration']))\
                  .props('type=number step=0.1 placeholder="duration (s)"')\
                  .bind_value(p, 'duration')
                ui.input(value=str(p['frequency']))\
                  .props('type=number step=1 placeholder="frequency (Hz)"')\
                  .bind_value(p, 'frequency')
                vol_in = ui.input(value=str(p['volume']))\
                           .props('type=number step=0.01 placeholder="volume"')\
                           .bind_value(p, 'volume')
                ui.slider(min=0, max=1, step=0.01, value=p['volume'])\
                  .bind_value(p, 'volume')\
                  .bind_value(vol_in, 'value')

        ui.button('Play sequence', on_click=lambda: asyncio.create_task(self.play_sequence()))
        self.seq_stop_resume = ui.button('Stop Sequence', on_click=self.toggle_sequence)\
                                  .props('color=warning')\
                                  .bind_visibility_from(self, 'sequence_playing')
        self.seq_cancel = ui.button('Cancel Sequence', on_click=self.cancel_sequence)\
                                  .props('color=negative')\
                                  .bind_visibility_from(self, 'sequence_playing')
        self.seq_label = ui.label('')

    def validate_inputs(self, sine_mode=True):
        try:
            f = float(self.freq_input.value) if sine_mode else None
            d = float(self.duration_input.value)
            v = float(self.volume)
            if sine_mode and f <= 0:
                raise ValueError('Frequency must be > 0')
            if d < 0:
                raise ValueError('Duration must be >= 0')
            return f, d, v
        except Exception as e:
            ui.notify(str(e), color='negative')
            return None, None, None

    def generate_noise_sloped(self, duration):
        N = int(duration * self.sample_rate)
        if N == 0:
            return np.array([], dtype=np.float32)
        M = 1 << (N - 1).bit_length()
        freqs = np.fft.rfftfreq(M, 1 / self.sample_rate)
        tgt = np.array([
            [20, 0.01], [50, 0.03], [100, 0.05],
            [200, 0.06], [500, 0.04], [1000, 0.02],
            [2000, 0.01]
        ])
        logF, logP = np.log10(tgt[:, 0]), np.log10(tgt[:, 1])
        psd_ref = 10 ** np.interp(
            np.log10(np.clip(freqs, 20, None)),
            logF, logP,
            left=logP[0], right=logP[-1]
        )
        df = self.sample_rate / M
        mag = np.sqrt(psd_ref * df)
        phase = np.exp(1j * 2 * np.pi * np.random.rand(len(freqs)))
        spectrum = mag * phase
        noise = np.fft.irfft(spectrum, n=M)[:N].astype(np.float32)
        noise -= noise.mean()
        rms_cur = np.sqrt(np.mean(noise**2))
        if rms_cur > 0:
            noise /= rms_cur
            noise *= self.target_rms
        return noise

    def audio_callback(self, in_data, frame_count, time_info, status):
        # POPRAWKA: Dodano warunek `self.paused` dla większej stabilności
        if not self.playing or self.paused:
            return np.zeros(frame_count, dtype=np.float32).tobytes(), pyaudio.paComplete

        if self.mode.value == 'sine':
            t = np.arange(frame_count, dtype=np.float32)
            block = (self.volume * np.sin(self.phase + self.phase_inc * t)).astype(np.float32)
            self.phase = (self.phase + self.phase_inc * frame_count) % (2 * np.pi)
        else:
            # POPRAWKA: Poprawiono logikę obsługi ostatniego bloku, aby uniknąć błędów
            remaining_samples = len(self.noise_data) - self.noise_idx
            if remaining_samples >= frame_count:
                s, e = self.noise_idx, self.noise_idx + frame_count
                block = self.volume * self.noise_data[s:e]
                self.noise_idx = e
            else:
                last_chunk = self.volume * self.noise_data[self.noise_idx:]
                block = np.pad(last_chunk, (0, frame_count - len(last_chunk)), 'constant')
                self.noise_idx += len(last_chunk)
                self.playing = False

        self.record_buffer.append(block.copy())
        self.elapsed = time.time() - self.start_time

        if self.duration and self.elapsed >= self.duration:
            self.playing = False

        code = pyaudio.paContinue if self.playing else pyaudio.paComplete
        return block.tobytes(), code

    def start_playback(self, offset: float = 0.0):
        if self.playing:
            return

        sine_mode = (self.mode.value == 'sine')
        f, d, v = self.validate_inputs(sine_mode)
        if d is None:
            return

        # POPRAWKA: Kolejność operacji zmieniona, aby najpierw generować dane
        if sine_mode:
            self.phase_inc = 2 * np.pi * f / self.sample_rate
            self.phase = (2 * np.pi * f * offset) % (2 * np.pi)
            self.noise_data = None
        else:
            self.noise_data = self.generate_noise_sloped(d)
            if not np.any(self.noise_data):
                return
            self.noise_idx = min(int(offset * self.sample_rate), len(self.noise_data))

        # Resetowanie stanu dla pojedynczego sygnału
        self.paused = False
        self.duration = d
        self.volume = v
        self.start_time = time.time() - offset
        self.elapsed = offset
        self.playing = True
        self.record_buffer.clear()
        
        self.stop_resume_btn.set_text('Stop')

        if self.stream:
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

    def stop_playback(self):
        self.playing = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def reset_time(self):
        if self.playing and not self.paused:
            self.start_time = time.time()
            self.elapsed = 0.0
            if self.mode.value == 'sine':
                self.phase = 0.0
            else:
                self.noise_idx = 0


    def toggle_playback(self):
        # POPRAWKA: Uproszczono i naprawiono logikę pauzy/wznowienia
        if not self.playing:
            return

        if not self.paused:
            # Pauza pojedynczego sygnału
            self.paused = True
            # self.elapsed jest już aktualne dzięki audio_callback
            self.stop_playback() # Zatrzymuje i zamyka strumień
            self.stop_resume_btn.set_text('Resume')
        else:
            # Wznowienie pojedynczego sygnału
            self.paused = False
            self.stop_resume_btn.set_text('Stop')
            # Użyj zapisanego czasu `elapsed` jako przesunięcia (`offset`)
            self.start_playback(offset=self.elapsed)

    async def play_sequence(self):
        if self.sequence_playing:
            return
        self.sequence_playing = True
        self.seq_paused = False
        self.seq_current = 0
        self.seq_elapsed = 0.0
        self.seq_stop_resume.set_text('Stop Sequence')

        while self.seq_current < len(self.presets) and self.sequence_playing:
            p = self.presets[self.seq_current]
            self.seq_label.set_text(
                f'Preset {self.seq_current+1}: '
                f'{p["duration"]} s, {p["frequency"]} Hz, vol={p["volume"]}'
            )
            self.mode.value = 'sine'
            self.freq_input.value = str(p['frequency'])
            self.duration_input.value = str(p['duration'])
            self.volume_input.value = str(p['volume'])

            # Rozpocznij lub wznów ten preset od `seq_elapsed`
            self.start_playback(offset=self.seq_elapsed)

            while self.playing and not self.seq_paused and self.sequence_playing:
                await asyncio.sleep(0.05)

            if not self.sequence_playing:
                break # Anulowano

            if self.seq_paused:
                # Pauza: czekaj na wznowienie
                while self.seq_paused and self.sequence_playing:
                    await asyncio.sleep(0.1)
                # Po wznowieniu pętla kontynuuje bez inkrementacji seq_current
                continue

            # Odtwarzanie zakończone normalnie -> następny preset
            self.seq_elapsed = 0.0
            self.seq_current += 1
            await asyncio.sleep(0.1) # Krótka przerwa między presetami

        if self.sequence_playing:
            self.seq_label.set_text('Sequence completed')
        self.sequence_playing = False
        self.seq_paused = False
        self.seq_stop_resume.set_text('Stop Sequence')
        self.stop_playback()

    def toggle_sequence(self):
        # POPRAWKA: Uproszczono logikę. Zmiana flagi `seq_paused` wystarczy.
        if not self.sequence_playing:
            return

        if not self.seq_paused:
            # Pauza sekwencji
            self.seq_elapsed = time.time() - self.start_time
            self.stop_playback() # Zatrzymuje i zamyka strumień
            self.seq_paused = True
            self.seq_stop_resume.set_text('Resume Sequence')
        else:
            # Wznowienie sekwencji - sama zmiana flagi
            self.seq_paused = False
            self.seq_stop_resume.set_text('Stop Sequence')
            # Pętla `play_sequence` zajmie się resztą

    def cancel_playback(self):
        self.stop_playback()
        self.paused = False
        self.elapsed = 0.0
        self.time_label.set_text('Time: 0.0 s')
        self.rms_label.set_text('RMS (g): –')
        self.psd_label.set_text('PSD err (%): –')

    def cancel_sequence(self):
        self.sequence_playing = False # To zakończy pętlę w `play_sequence`
        self.stop_playback()
        self.seq_paused = False
        self.seq_current = 0
        self.seq_elapsed = 0.0
        self.seq_stop_resume.set_text('Stop Sequence')
        self.seq_label.set_text('Sequence cancelled')

    def update_time_and_metrics(self):
        is_active = self.playing and not self.paused
        if is_active:
             self.time_label.set_text(f'Time: {self.elapsed:.1f} s')
             if self.elapsed >= 1.0 and self.record_buffer:
                data = np.concatenate(list(self.record_buffer))
                rms = np.sqrt(np.mean(data**2))
                self.rms_label.set_text(f'RMS (g): {rms:.3f}')

                if self.mode.value == 'random':
                    f, Pxx = welch(data, fs=self.sample_rate, nperseg=1024)
                    tgt = np.array([
                        [20, 0.01], [50, 0.03], [100, 0.05],
                        [200, 0.06], [500, 0.04], [1000, 0.02],
                        [2000, 0.01]
                    ])
                    logF, logP = np.log10(tgt[:, 0]), np.log10(tgt[:, 1])
                    mask = (f >= 20) & (f <= 2000)
                    if np.any(mask):
                        interp = 10 ** np.interp(np.log10(f[mask]), logF, logP)
                        err = np.abs(Pxx[mask] - interp) / interp * 100
                        self.psd_label.set_text(f'PSD err (%): {err.mean():.1f}')
                else:
                    self.psd_label.set_text('PSD err (%): –')

    def cleanup(self):
        self.stop_playback()
        self.p.terminate()

app = VibrationTestApp()
ui.on('shutdown', app.cleanup)
ui.run(title='Vibration Test App', port=8080)
