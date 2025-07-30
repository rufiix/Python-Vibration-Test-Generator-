# Wersja kodu bez nieudanej próby wyciszenia
import nicegui
from nicegui import ui
import pyaudio
import numpy as np
import asyncio
import time
from scipy.signal import welch
from collections import deque

class VibrationTestApp:
    INFINITE_LOOP_DURATION = 10.0

    def __init__(self):
        # Parametry audio / DSP
        self.sample_rate = 22050
        self.frames_per_buffer = 1024
        self.target_rms = 6.9
        # Inicjalizacja PyAudio z powrotem wewnątrz klasy
        self.p = pyaudio.PyAudio()
        self.stream = None

        # ... cała reszta klasy bez zmian ...
        # (wklej tutaj resztę klasy z wersji, która poprawnie obliczała błąd PSD)
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
        self.target_psd_profile = None
        self.sequence_playing = False
        self.seq_paused = False
        self.seq_current = 0
        self.seq_elapsed = 0.0
        self.record_buffer = deque(maxlen=int(5 * self.sample_rate / self.frames_per_buffer))
        self.presets = [
            {'duration': 2.0, 'frequency': 200.0, 'volume': 0.5},
            {'duration': 1.5, 'frequency': 440.0, 'volume': 0.7},
            {'duration': 2.5, 'frequency': 800.0, 'volume': 0.3},
            {'duration': 1.0, 'frequency': 1000.0, 'volume': 1.0},
            {'duration': 3.0, 'frequency': 1500.0, 'volume': 0.6},
        ]
        self.build_ui()
        ui.timer(0.1, self.update_time_and_metrics)

    def build_ui(self):
        ui.label('Vibration Test: Sine & Random Sloped').style('font-size:1.2em; margin-bottom:10px;')
        with ui.row():
            ui.label('Signal mode:')
            self.mode_radio = ui.radio(['sine', 'random'], value='sine')
        with ui.row():
            ui.label('Frequency (Hz)')
            self.freq_input = ui.input(value='440.0').props('type=number step=1')
        with ui.row():
            ui.label('Duration (s) (0 = infinite)')
            self.duration_input = ui.input(value='8.0').props('type=number step=0.1')
        with ui.row():
            ui.label('Volume (0–1)')
            self.volume_input = ui.input(value='1.0').props('type=number step=0.01')
            ui.slider(min=0, max=1, step=0.01).bind_value(self.volume_input, 'value').bind_value(self, 'volume')
        
        ui.button('Start', on_click=self.start_playback)
        self.stop_resume_btn = ui.button('Stop', on_click=self.toggle_playback).props('color=warning').bind_visibility_from(self, 'playing')
        self.cancel_btn = ui.button('Cancel', on_click=self.cancel_playback).props('color=negative').bind_visibility_from(self, 'playing')
        
        ui.separator()
        self.time_label = ui.label('Time: 0.0 s')
        self.rms_label = ui.label('RMS (g): –')
        self.psd_label = ui.label('PSD err (%): –')
        ui.separator()
        
        ui.label('Sequence presets:')
        with ui.row().style('gap:70px; margin-left: 20px;'):
            ui.label('Czas (s)').style('font-weight: bold;')
            ui.label('Częstotliwość (Hz)').style('font-weight: bold;')
            ui.label('Głośność').style('font-weight: bold;')

        for p in self.presets:
            with ui.row().style('align-items:center; gap:20px;'):
                ui.input(value=str(p['duration'])).props('type=number step=0.1').bind_value(p, 'duration')
                ui.input(value=str(p['frequency'])).props('type=number step=1').bind_value(p, 'frequency')
                vol_in = ui.input(value=str(p['volume'])).props('type=number step=0.01').bind_value(p, 'volume')
                ui.slider(min=0, max=1, step=0.01, value=p['volume']).bind_value(p, 'volume').bind_value(vol_in, 'value')

        ui.button('Play sequence', on_click=lambda: asyncio.create_task(self.play_sequence()))
        self.seq_stop_resume = ui.button('Stop Sequence', on_click=self.toggle_sequence).props('color=warning').bind_visibility_from(self, 'sequence_playing')
        self.seq_cancel = ui.button('Cancel Sequence', on_click=self.cancel_sequence).props('color=negative').bind_visibility_from(self, 'sequence_playing')
        self.seq_label = ui.label('')

    def validate_inputs(self, sine_mode=True):
        try:
            f = float(self.freq_input.value) if sine_mode else None
            d = float(self.duration_input.value)
            v = float(self.volume)
            if sine_mode and f <= 0: raise ValueError('Frequency must be > 0')
            if d < 0: raise ValueError('Duration cannot be negative')
            return f, d, v
        except Exception as e:
            ui.notify(str(e), color='negative')
            return None, None, None

    def generate_noise_sloped(self, duration):
        N = int(duration * self.sample_rate)
        if N == 0: return np.array([], dtype=np.float32)
        M = 1 << (N - 1).bit_length()
        freqs = np.fft.rfftfreq(M, 1 / self.sample_rate)
        tgt = np.array([[20,0.01],[50,0.03],[100,0.05],[200,0.06],[500,0.04],[1000,0.02],[2000,0.01]])
        logF, logP = np.log10(tgt[:, 0]), np.log10(tgt[:, 1])
        psd_ref = 10 ** np.interp(np.log10(np.clip(freqs, 20, None)), logF, logP, left=logP[0], right=logP[-1])
        
        self.target_psd_profile = {'freqs': freqs, 'psd': psd_ref}
        
        df = self.sample_rate / M
        mag = np.sqrt(psd_ref * df)
        phase = np.exp(1j * 2 * np.pi * np.random.rand(len(freqs)))
        noise = np.fft.irfft(mag * phase, n=M)[:N].astype(np.float32)
        noise -= noise.mean()
        rms_cur = np.sqrt(np.mean(noise**2))
        if rms_cur > 0: noise = (noise / rms_cur) * self.target_rms
        return noise

    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.paused: return np.zeros(frame_count, dtype=np.float32).tobytes(), pyaudio.paContinue
        if not self.playing: return np.zeros(frame_count, dtype=np.float32).tobytes(), pyaudio.paComplete

        if self.mode_radio.value == 'sine':
            t = np.arange(frame_count, dtype=np.float32)
            block = (self.volume * np.sin(self.phase + self.phase_inc * t)).astype(np.float32)
            self.phase = (self.phase + self.phase_inc * frame_count) % (2 * np.pi)
        else:
            if self.noise_data is None or len(self.noise_data) == 0:
                self.playing = False
                return np.zeros(frame_count, dtype=np.float32).tobytes(), pyaudio.paComplete
            
            if self.duration == 0:
                s = self.noise_idx
                e = s + frame_count
                if e < len(self.noise_data):
                    block = self.volume * self.noise_data[s:e]
                    self.noise_idx = e
                else:
                    rem_len = len(self.noise_data) - s
                    wrap_len = frame_count - rem_len
                    block = self.volume * np.concatenate((self.noise_data[s:], self.noise_data[:wrap_len]))
                    self.noise_idx = wrap_len
            else:
                rem = len(self.noise_data) - self.noise_idx
                if rem >= frame_count:
                    block = self.volume * self.noise_data[self.noise_idx : self.noise_idx + frame_count]
                    self.noise_idx += frame_count
                else:
                    last_chunk = self.volume * self.noise_data[self.noise_idx:]
                    block = np.pad(last_chunk, (0, frame_count - rem), 'constant')
                    self.playing = False
        
        self.record_buffer.append(block.copy())
        if self.duration > 0:
            current_elapsed = time.time() - self.start_time
            if current_elapsed >= self.duration: self.playing = False
        
        return block.tobytes(), pyaudio.paContinue if self.playing else pyaudio.paComplete

    def start_playback(self, offset: float = 0.0):
        if self.playing and not self.paused: return
        sine_mode = (self.mode_radio.value == 'sine')
        f, d, v = self.validate_inputs(sine_mode)
        if d is None: return

        if not self.sequence_playing:
            self.mode_radio.disable()
            self.freq_input.disable()
            self.duration_input.disable()

        self.playing = True
        self.paused = False
        self.duration = d
        self.volume = v
        self.start_time = time.time() - offset
        self.elapsed = offset
        if not self.sequence_playing: self.record_buffer.clear()
        
        if sine_mode:
            self.phase_inc = 2 * np.pi * f / self.sample_rate
            self.phase = (2 * np.pi * f * offset) % (2 * np.pi)
            self.noise_data = None
            self.target_psd_profile = None
        else:
            if d == 0:
                self.noise_data = self.generate_noise_sloped(self.INFINITE_LOOP_DURATION)
            else:
                self.noise_data = self.generate_noise_sloped(d)
            self.noise_idx = min(int(offset * self.sample_rate), len(self.noise_data))
        
        if self.stream is None or not self.stream.is_active():
            self.stream = self.p.open(format=pyaudio.paFloat32, channels=1, rate=self.sample_rate,
                                      output=True, frames_per_buffer=self.frames_per_buffer,
                                      stream_callback=self.audio_callback)

    def stop_playback(self):
        if self.stream: self.stream.close()
        self.stream = None
        self.playing = False
        self.paused = False
        self.target_psd_profile = None

        self.mode_radio.enable()
        self.freq_input.enable()
        self.duration_input.enable()
        self.stop_resume_btn.set_text('Stop')
        self.seq_stop_resume.set_text('Stop Sequence')
        
        self.psd_label.set_text('PSD err (%): –')


    def toggle_playback(self):
        if not self.playing: return
        
        self.paused = not self.paused
        if self.paused:
            self.elapsed = time.time() - self.start_time
            self.stop_resume_btn.set_text('Resume')
        else:
            self.start_time = time.time() - self.elapsed
            self.stop_resume_btn.set_text('Stop')

    async def play_sequence(self):
        if self.sequence_playing: return
        self.sequence_playing = True
        
        original_freq = self.freq_input.value
        original_duration = self.duration_input.value
        original_volume = self.volume_input.value
        
        self.seq_current = 0
        self.seq_elapsed = 0.0
        self.record_buffer.clear()

        try:
            while self.seq_current < len(self.presets) and self.sequence_playing:
                if self.seq_paused:
                    await asyncio.sleep(0.1)
                    continue

                p = self.presets[self.seq_current]
                if float(p.get('duration', 1)) <= 0:
                    self.seq_label.set_text(f'Preset {self.seq_current+1} ma nieprawidłowy czas trwania (<=0). Pomijanie.')
                    self.seq_current += 1
                    await asyncio.sleep(1)
                    continue

                self.seq_label.set_text(f'Preset {self.seq_current+1}: {p["duration"]}s, {p["frequency"]}Hz, vol={p["volume"]}')
                self.mode_radio.value = 'sine'
                self.freq_input.value = str(p['frequency'])
                self.duration_input.value = str(p['duration'])
                self.volume_input.value = str(p['volume'])
                
                self.start_playback(offset=self.seq_elapsed)
                
                while self.playing and self.sequence_playing:
                    await asyncio.sleep(0.05)
                
                if self.seq_paused: continue
                if not self.sequence_playing: break

                self.seq_elapsed = 0.0
                self.seq_current += 1
                await asyncio.sleep(0.1)
        finally:
            self.freq_input.value = original_freq
            self.duration_input.value = original_duration
            self.volume_input.value = original_volume
            
            self.stop_playback()
            self.sequence_playing = False
            self.seq_label.set_text('Sekwencja zakończona lub anulowana')

    def toggle_sequence(self):
        if not self.sequence_playing: return

        self.seq_paused = not self.seq_paused
        self.paused = self.seq_paused

        if self.seq_paused:
            self.seq_elapsed = time.time() - self.start_time
            self.seq_stop_resume.set_text('Wznów sekwencję')
        else:
            self.start_time = time.time() - self.seq_elapsed
            self.seq_stop_resume.set_text('Zatrzymaj sekwencję')

    def cancel_playback(self):
        if self.sequence_playing:
            self.cancel_sequence()
        else:
            self.stop_playback()
            self.elapsed = 0.0
            self.time_label.set_text('Time: 0.0 s')
            self.rms_label.set_text('RMS (g): –')
            self.psd_label.set_text('PSD err (%): –')

    def cancel_sequence(self):
        self.sequence_playing = False
        self.seq_paused = False

    def update_time_and_metrics(self):
        if self.playing and not self.paused:
            self.elapsed = time.time() - self.start_time
            if self.duration == 0:
                self.time_label.set_text(f'Time: {self.elapsed:.1f} s (Infinite)')
            else:
                self.time_label.set_text(f'Time: {self.elapsed:.1f} s')
            
            if self.elapsed > 0.5 and self.record_buffer:
                data = np.concatenate(list(self.record_buffer))
                if data.size == 0: return
                
                rms = np.sqrt(np.mean(data**2))
                self.rms_label.set_text(f'RMS (g): {rms:.3f}')
                
                if self.mode_radio.value == 'random' and self.target_psd_profile:
                    nperseg = min(data.size, 2048)
                    if data.size < nperseg:
                        self.psd_label.set_text('PSD err (%): zbieranie danych...')
                        return

                    f_actual, psd_actual = welch(
                        data, fs=self.sample_rate, nperseg=nperseg, scaling='density'
                    )

                    target_psd_interp = np.interp(
                        f_actual, self.target_psd_profile['freqs'], self.target_psd_profile['psd']
                    )

                    target_psd_interp *= (self.volume**2)
                    
                    idx_range = np.where((f_actual >= 20) & (f_actual <= 2000))
                    if idx_range[0].size > 0:
                        psd_actual_range = psd_actual[idx_range]
                        target_psd_interp_range = target_psd_interp[idx_range]
                        
                        denominator = np.where(target_psd_interp_range > 1e-12, target_psd_interp_range, 1e-12)
                        
                        percent_error = 100 * (psd_actual_range - target_psd_interp_range) / denominator
                        mean_abs_error = np.mean(np.abs(percent_error))
                        
                        self.psd_label.set_text(f'PSD err (%): {mean_abs_error:.2f}')
                    else:
                        self.psd_label.set_text('PSD err (%): –')
                else:
                    self.psd_label.set_text('PSD err (%): –')

    def cleanup(self):
        self.stop_playback()
        self.p.terminate()

# Główna część skryptu wraca do prostej formy
app = VibrationTestApp()
ui.on('shutdown', app.cleanup)
ui.run(title='Vibration Test App', port=8080, host='0.0.0.0', show=False)
