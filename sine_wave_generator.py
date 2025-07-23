from nicegui import ui
import pyaudio
import numpy as np
import asyncio
import time
from scipy.signal import welch
from collections import deque

class VibrationTestApp:
    def __init__(self):
        # Audio / DSP parameters
        self.sample_rate = 22050
        self.frames_per_buffer = 1024
        self.target_rms = 6.9
        self.p = pyaudio.PyAudio()
        self.stream = None

        # Single‐signal state
        self.playing = False
        self.paused = False
        self.remaining = None
        self.start_time = 0.0
        self.duration = 0.0
        self.volume = 1.0
        self.phase = 0.0
        self.phase_inc = 0.0

        # Sequence state
        self.sequence_playing = False
        self.seq_paused = False
        self.seq_current = 0
        self.seq_pause_time = 0.0

        # Live‐analysis buffer
        self.record_buffer = deque(maxlen=int(5 * self.sample_rate / self.frames_per_buffer))

        # Presets
        self.presets = [
            {'duration': 2.0, 'frequency': 200.0, 'volume': 0.5},
            {'duration': 1.5, 'frequency': 440.0, 'volume': 0.7},
            {'duration': 2.5, 'frequency': 800.0, 'volume': 0.3},
            {'duration': 1.0, 'frequency': 1000.0, 'volume': 1.0},
            {'duration': 3.0, 'frequency': 1500.0, 'volume': 0.6},
        ]

        # Build UI and start timer
        self.build_ui()
        ui.timer(0.2, self.update_time_and_metrics)

    def build_ui(self):
        ui.label('Vibration Test: Sine & Random Sloped')\
          .style('font-size:1.2em; margin-bottom:10px;')

        with ui.row():
            ui.label('Signal mode:')
            self.mode = ui.radio(['sine', 'random'], value='sine')

        with ui.row():
            self.freq_input = ui.input('Frequency (Hz)', value='440.0')\
                                 .props('type=number step=1')
        with ui.row():
            self.duration_input = ui.input('Duration (s)', value='8.0')\
                                     .props('type=number step=0.1')
        with ui.row():
            self.volume_input = ui.input('Volume (0–1)', value='1.0')\
                                  .props('type=number step=0.01')
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
        self.rms_label  = ui.label('RMS (g): –')
        self.psd_label  = ui.label('PSD err (%): –')
        ui.separator()

        ui.label('Sequence presets (duration, frequency, volume):')
        for p in self.presets:
            with ui.row().style('align-items:center; gap:20px; margin-bottom:5px;'):
                ui.input(value=p['duration'])\
                  .props('type=number step=0.1 placeholder="duration (s)"')\
                  .bind_value(p, 'duration')
                ui.input(value=p['frequency'])\
                  .props('type=number step=1 placeholder="frequency (Hz)"')\
                  .bind_value(p, 'frequency')
                vol_in = ui.input(value=p['volume'])\
                            .props('type=number step=0.01 placeholder="volume"')\
                            .bind_value(p, 'volume')
                ui.slider(min=0.0, max=1.0, step=0.01, value=p['volume'])\
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
            if sine_mode:
                fs = self.freq_input.value
                if not fs:
                    raise ValueError('Podaj częstotliwość')
                f = float(fs)
                if f <= 0:
                    raise ValueError('Frequency must be > 0')
            else:
                f = None

            ds = self.duration_input.value
            if ds is None or ds == '' or float(ds) == 0.0:
                d = None
            else:
                d = float(ds)
                if d < 0:
                    raise ValueError('Duration must be >= 0')

            v = float(self.volume)
            return f, d, v
        except Exception as e:
            ui.notify(str(e), color='negative')
            return None, None, None

    def generate_noise_sloped(self, duration):
        N = int(duration * self.sample_rate)
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
            left=10 ** logP[0], right=10 ** logP[-1]
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
        if not self.playing:
            return np.zeros(frame_count, dtype=np.float32).tobytes(), pyaudio.paComplete

        vol = self.volume
        if self.mode.value == 'sine':
            t = np.arange(frame_count, dtype=np.float32)
            block = (vol * np.sin(self.phase + self.phase_inc * t)).astype(np.float32)
            self.phase = (self.phase + self.phase_inc * frame_count) % (2 * np.pi)
        else:
            s = self.noise_idx
            e = s + frame_count
            block = vol * self.noise_data[s:e]
            self.noise_idx = e
            if e >= len(self.noise_data):
                self.playing = False
                block = np.pad(block, (0, e - len(self.noise_data)), 'constant')

        self.record_buffer.append(block.copy())
        self.elapsed = time.time() - self.start_time

        if self.duration is not None and self.elapsed >= self.duration:
            self.playing = False
            return block.tobytes(), pyaudio.paComplete

        return block.tobytes(), pyaudio.paContinue

    def start_playback(self):
        if self.playing:
            return

        sine_mode = (self.mode.value == 'sine')
        f, d, v = self.validate_inputs(sine_mode)
        if sine_mode and f is None:
            return
        if not sine_mode and d is None:
            ui.notify('Podaj czas trwania dla Random Sloped', color='negative')
            return

        # Reset single‐signal pause state
        self.paused = False
        self.remaining = None

        self.duration   = d
        self.volume     = v
        self.start_time = time.time()
        self.elapsed    = 0.0
        self.playing    = True
        self.record_buffer.clear()

        if sine_mode:
            self.phase     = 0.0
            self.phase_inc = 2 * np.pi * f / self.sample_rate
            self.noise_data = None
        else:
            self.noise_data = self.generate_noise_sloped(d)
            self.noise_idx  = 0

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
        if not self.playing:
            return
        self.playing = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def reset_time(self):
        if self.playing:
            self.start_time = time.time()
            self.elapsed    = 0.0

    def toggle_playback(self):
        if not self.paused:
            # Pause single signal
            self.paused    = True
            self.stream.stop_stream()
            self.remaining = max(0.0, self.duration - (time.time() - self.start_time))
            self.stop_resume_btn.set_text('Resume')
        else:
            # Resume single signal
            self.paused = False
            self.stop_resume_btn.set_text('Stop')
            self.start_time = time.time() - (self.duration - self.remaining)
            self.prepare_playback_offset(self.duration - self.remaining)

    def cancel_playback(self):
        # Cancel single‐signal playback completely
        self.paused    = False
        self.remaining = None
        self.playing   = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        # Reset metrics display
        self.elapsed = 0.0
        self.time_label.set_text('Time: 0.0 s')
        self.rms_label.set_text('RMS (g): –')
        self.psd_label.set_text('PSD err (%): –')

    async def play_sequence(self):
        if self.sequence_playing:
            return
        self.sequence_playing = True
        self.seq_paused       = False
        self.seq_current      = 0

        while self.seq_current < len(self.presets) and self.sequence_playing:
            p = self.presets[self.seq_current]
            self.seq_label.set_text(
                f'Preset {self.seq_current+1}: {p["duration"]} s, '
                f'freq={p["frequency"]} Hz, vol={p["volume"]}'
            )
            self.mode.value         = 'sine'
            self.freq_input.value   = p['frequency']
            self.duration_input.value = p['duration']
            self.volume_input.value = p['volume']
            self.start_playback()

            # wait until playback ends or paused/cancelled
            while self.playing and not self.seq_paused and self.sequence_playing:
                await asyncio.sleep(0.05)

            # if paused, wait until resume
            while self.seq_paused and self.sequence_playing:
                await asyncio.sleep(0.1)

            if not self.seq_paused:
                self.seq_current += 1

        if self.sequence_playing:
            self.seq_label.set_text('Sequence completed')
        self.sequence_playing = False
        self.seq_paused       = False

    def toggle_sequence(self):
        if not self.sequence_playing or not self.stream:
            return

        if not self.seq_paused:
            # Pause sequence
            self.seq_pause_time = time.time()
            self.stream.stop_stream()
            self.seq_paused     = True
            self.seq_stop_resume.set_text('Resume Sequence')
        else:
            # Resume sequence
            pause_duration     = time.time() - self.seq_pause_time
            self.start_time   += pause_duration
            self.seq_paused     = False
            self.seq_stop_resume.set_text('Stop Sequence')
            self.stream.start_stream()

    def cancel_sequence(self):
        # 1. Stop any ongoing playback
        self.stop_playback()

        # 2. Reset sequence state
        self.sequence_playing = False
        self.seq_paused       = False
        self.seq_current      = 0
        self.seq_pause_time   = 0.0

        # 3. Restore Stop Sequence button label
        self.seq_stop_resume.set_text('Stop Sequence')

        # 4. Notify
        self.seq_label.set_text('Sequence cancelled')

    def prepare_playback_offset(self, offset):
        """Adjust phase or noise index and restart stream at offset."""
        if self.mode.value == 'sine':
            f = float(self.freq_input.value)
            self.phase     = (2 * np.pi * f * offset) % (2 * np.pi)
            self.phase_inc = 2 * np.pi * f / self.sample_rate
        else:
            idx = int(offset * self.sample_rate)
            self.noise_idx = min(idx, len(self.noise_data))
        self.playing = True
        self.stream.start_stream()

    def update_time_and_metrics(self):
        self.time_label.set_text(f'Time: {self.elapsed:.1f} s')
        if self.playing and self.elapsed >= 1.0 and self.record_buffer:
            data = np.concatenate(list(self.record_buffer))
            rms  = np.sqrt(np.mean(data**2))
            self.rms_label.set_text(f'RMS (g): {rms:.3f}')
            f, Pxx = welch(data, fs=self.sample_rate, nperseg=1024)
            tgt = np.array([
                [20, 0.01], [50, 0.03], [100, 0.05],
                [200,0.06], [500,0.04], [1000,0.02],
                [2000,0.01]
            ])
            logF, logP = np.log10(tgt[:,0]), np.log10(tgt[:,1])
            mask       = (f >= 20) & (f <= 2000)
            interp     = 10 ** np.interp(np.log10(f[mask]), logF, logP)
            err        = np.abs(Pxx[mask] - interp) / interp * 100
            self.psd_label.set_text(f'PSD err (%): {np.mean(err):.1f}')

    def cleanup(self):
        self.playing = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

# Run the application
app = VibrationTestApp()
ui.on('shutdown', app.cleanup)
ui.run(title='Vibration Test App', port=8080)
