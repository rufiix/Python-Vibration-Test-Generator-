Sine Wave Generator
Overview
This Python application generates sine wave audio tones using PyAudio and provides a graphical user interface (GUI) built with Tkinter. Users can specify the frequency, duration, and volume of a single tone or play a sequence of preset tones. The application displays real-time playback information and supports editing and playing a sequence of up to five presets.
Features

Single Tone Playback: Generate a sine wave with user-defined frequency (Hz), optional duration (seconds), and volume (0.0 to 1.0).
Preset Sequence Playback: Play a sequence of up to five predefined or user-edited presets, each with frequency, duration, and volume.
Real-Time Feedback: Displays elapsed time during playback and the current preset during sequence playback.
Interactive GUI: Includes input fields, a volume slider, and buttons for starting, stopping, resetting time, and playing sequences.
Input Validation: Ensures valid frequency, duration, and volume inputs.
Clean Resource Management: Properly closes audio streams and terminates PyAudio on application exit.

Requirements

Python 3.6 or higher
Required libraries:
pyaudio (for audio playback)
numpy (for sine wave generation)
tkinter (included with standard Python for GUI)


Install dependencies:pip install pyaudio numpy



How to Run

Ensure the required libraries are installed.
Save the code in a file (e.g., sine_wave_generator.py).
Run the script:python sine_wave_generator.py


Usage:
Single Tone:
Enter a frequency (e.g., 440 Hz) and optional duration (e.g., 2 seconds).
Adjust the volume using the slider.
Click "Start" to play the tone, "Stop" to halt, or "Reset" to reset the elapsed time.


Preset Sequence:
Edit the preset fields (frequency, duration, volume) if desired.
Click "Play sequence" to play all presets in order.
The current preset and elapsed time are displayed during playback.


Close the window to exit, ensuring proper cleanup of audio resources.



GUI Layout

Inputs:
Frequency (Hz): Text field for the tone’s frequency (default: 440 Hz).
Duration (s, optional): Text field for playback duration (leave empty for continuous playback).
Volume: Slider from 0.0 to 1.0 (default: 1.0).


Preset Fields: Five rows of editable fields for frequency, duration, and volume (default presets provided).
Buttons:
"Start": Play a single tone based on input fields.
"Stop": Stop any ongoing playback.
"Reset": Reset the elapsed time counter.
"Play sequence": Play all presets sequentially.


Labels:
Elapsed time: Updates every 50ms during playback.
Current preset: Shows the active preset during sequence playback.



Technical Details

Audio Parameters:
Sample rate: 22,050 Hz
Buffer size: 1,024 frames
Format: 32-bit float, mono channel


Sine Wave Generation: Uses NumPy to compute sine wave samples in a callback function, ensuring smooth audio with continuous phase.
Sequence Playback: Schedules preset transitions using Tkinter’s after method, closing and reopening audio streams for each preset.
Error Handling: Validates inputs and displays error messages for invalid values.

Example Presets
The default presets are:

440 Hz, 2s, 0.5 volume
550 Hz, 1.5s, 0.7 volume
660 Hz, 2.5s, 0.3 volume
770 Hz, 1s, 1.0 volume
880 Hz, 3s, 0.6 volume

Users can edit these values in the GUI.
Limitations

Requires a working audio output device.
Presets must have positive frequency and duration, and volume between 0.0 and 1.0.
Continuous playback (no duration) runs until manually stopped.
Sequence playback cannot be paused; it must be stopped and restarted.

License
This project is for educational purposes and does not include a specific license. Use and modify at your own discretion.
