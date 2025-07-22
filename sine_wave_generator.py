from nicegui import ui
import pyaudio
import numpy as np
import asyncio
import time
from scipy.signal import welch
from collections import deque

class VibrationTestApp:
    def __init__(self):
        # audio parameters
        self.sample_rate       = 22050
        self.frames_per_buffer = 1024
        self.target_rms        = 6.9
        self.p                 = pyaudio.PyAudio()
        self.stream            = None

        # playback state
        self.playing           = False
        self.sequence_playing  = False

        # keep only the last 5 seconds of audio buffers
        max_secs     = 5
        max_buffers  = int(max_secs * self.sample_rate / self.frames_per_buffer)
        self.record_buffer = deque(maxlen=max_buffers)

        self.phase      = 0.0
        self.phase_inc  = 0.0
        self.noise_data = None
        self.noise_idx  = 0
        self.start_time = 0.0
        self.duration   = 0.0     # None = indefinite
        self.volume     = 1.0     # real‐time updated by slider
        self.elapsed    = 0.0

        # sequence presets: duration, volume, frequency
        self.presets = [
            {'duration': 2.0, 'volume': 0.5, 'frequency': 200.0},
            {'duration': 1.5, 'volume': 0.7, 'frequency': 440.0},
            {'duration': 2.5, 'volume': 0.3, 'frequency': 800.0},
            {'duration': 1.0, 'volume': 1.0, 'frequency': 1000.0},
            {'duration': 3.0, 'volume': 0.6, 'frequency': 1500.0},
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
            self.freq_input = ui.input('Frequency (Hz)', value='440.0')\
                                .props('type=number step=1')

        with ui.row():
            self.duration_input = ui.input('Duration (s)', value='8.0')\
                                     .props('type=number step=0.1')

        with ui.row():
            self.volume_input = ui.input('Volume (0–1)', value='1.0')\
                                  .props('type=number step=0.01')
            self.volume_slider = ui.slider(min=0, max=1, step=0.01)\
                .bind_value(self.volume_input, 'value')\
                .bind_value(self, 'volume')

        ui.button('Start', on_click=self.start_playback)
        ui.button('Stop', on_click=self.stop_playback)\
          .props('color=negative')\
          .bind_visibility_from(self, 'playing')
        ui.button('Reset', on_click=self.reset_time)\
          .bind_visibility_from(self, 'playing')

        ui.separator()
        self.time_label = ui.label('Time: 0.0 s')
        self.rms_label  = ui.label('RMS (g): –')
        self.psd_label  = ui.label('PSD err (%): –')
        ui.separator()

        ui.label('Sequence presets (s, vol, freq):')
        self.preset_inputs = []
        for p in self.presets:
            with ui.row():
                d_in = ui.input()\
                         .props('type=number step=0.1 placeholder=duration (s)')\
                         .bind_value(p, 'duration')
                v_in = ui.input()\
                         .props('type=number step=0.01 placeholder=volume')\
                         .bind_value(p, 'volume')
                f_in = ui.input()\
                         .props('type=number step=1 placeholder=frequency (Hz)')\
                         .bind_value(p, 'frequency')
                f_slider = ui.slider(min=20, max=2000, step=1)\
                             .bind_value(f_in, 'value')\
                             .bind_value(p, 'frequency')
                self.preset_inputs.append((d_in, v_in, f_in, f_slider))

        ui.button('Play sequence', on_click=lambda: asyncio.create_task(self.play_sequence()))
        self.seq_label = ui.label('')

    def validate_inputs(self, sine_mode=True):
        try:
            f = None
            if sine_mode:
                fs = self.freq_input.value
                if not fs:
                    raise ValueError('Podaj częstotliwość')
                f = float(fs)
                if f <= 0:
                    raise ValueError('Frequency must be > 0')

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
        freqs = np.fft.rfftfreq(M, 1/self.sample_rate)
        tgt = np.array([
            [20,   0.01], [50,   0.03], [100,  0.05],
            [200,  0.06], [500,  0.04], [1000, 0.02],
            [2000, 0.01]
        ])
        logF, logP = np.log10(tgt[:,0]), np.log10(tgt[:,1])
        psd_ref = 10**np.interp(
            np.log10(np.clip(freqs, 20, None)),
            logF, logP,
            left=10**logP[0], right=10**logP[-1]
        )
        df = self.sample_rate / M
        mag = np.sqrt(psd_ref * df)
        phase = np.exp(1j * 2*np.pi * np.random.rand(len(freqs)))
        spectrum = mag * phase
        noise = np.fft.irfft(spectrum, n=M)[:N].astype(np.float32)
        noise -= np.mean(noise)
        rms_cur = np.sqrt(np.mean(noise**2))
        if rms_cur > 0:
            noise /= rms_cur
        noise *= self.target_rms
        return noise

    def audio_callback(self, in_data, frame_count, time_info, status):
        if not self.playing:
            silence = np.zeros(frame_count, dtype=np.float32).tobytes()
            return silence, pyaudio.paComplete

        vol = self.volume
        if self.mode.value == 'sine':
            t     = np.arange(frame_count, dtype=np.float32)
            block = (vol * np.sin(self.phase + self.phase_inc * t)).astype(np.float32)
            self.phase = (self.phase + self.phase_inc * frame_count) % (2*np.pi)
        else:
            s, e  = self.noise_idx, self.noise_idx + frame_count
            block  = vol * self.noise_data[s:e]
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

        self.duration   = d
        self.volume     = v
        self.start_time = time.time()
        self.record_buffer.clear()
        self.playing    = True
        self.elapsed    = 0.0

        if sine_mode:
            self.phase     = 0.0
            self.phase_inc = 2*np.pi * f / self.sample_rate
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

    def update_time_and_metrics(self):
        self.time_label.set_text(f'Time: {self.elapsed:.1f} s')
        if self.playing and self.elapsed >= 1.0 and self.record_buffer:
            data = np.concatenate(list(self.record_buffer))
            rms  = np.sqrt(np.mean(data**2))
            self.rms_label.set_text(f'RMS (g): {rms:.3f}')
            f, Pxx = welch(data, fs=self.sample_rate, nperseg=1024)
            tgt = np.array([
                [20,   0.01], [50,   0.03], [100,  0.05],
                [200,  0.06], [500,  0.04], [1000, 0.02],
                [2000, 0.01]
            ])
            logF, logP = np.log10(tgt[:,0]), np.log10(tgt[:,1])
            mask    = (f >= 20) & (f <= 2000)
            interp  = 10**np.interp(np.log10(f[mask]), logF, logP)
            err     = np.abs(Pxx[mask] - interp) / interp * 100
            self.psd_label.set_text(f'PSD err (%): {np.mean(err):.1f}')

    async def play_sequence(self):
        if self.playing or self.sequence_playing:
            return
        self.sequence_playing = True
        for i, p in enumerate(self.presets, start=1):
            self.seq_label.set_text(
                f'Preset {i}: {p["duration"]}s, vol={p["volume"]}, freq={p["frequency"]}Hz'
            )
            self.mode.value           = 'sine'
            self.freq_input.value     = p['frequency']
            self.duration_input.value = p['duration']
            self.volume_input.value   = p['volume']
            self.start_playback()
            while self.playing:
                await asyncio.sleep(0.1)
        self.seq_label.set_text('Sequence completed')
        self.sequence_playing = False

    def cleanup(self):
        self.playing = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.p.terminate()

app = VibrationTestApp()
ui.on('shutdown', app.cleanup)
ui.run(title='Vibration Test App', port=8080)
