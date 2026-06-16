"""Synthetic polyphonic-instrument chords + a pitch-aligned feature.

Generates chords on the fly (root x quality x octave x voicing) as additive
tones WITH harmonics, so the detector has to learn the hard part: attributing
harmonics back to their fundamental pitch class (a naive chromagram can't).

Feature: a log-frequency (one-bin-per-semitone) spectrogram via a fixed
triangular filterbank on the STFT — pitch-aligned and, like the mel filterbank,
just a matrix multiply, so it ports to the C++/plugin path later.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

SR = 24000
N_FFT = 2048                 # ~11.7 Hz/bin @ 24 kHz (resolves down to ~C2)
MIDI_LO, MIDI_HI = 36, 96    # C2 .. C7 (covers bass/guitar/piano usefully)

# Chord qualities as semitone intervals from the root.
QUALITIES = {
    "maj":  (0, 4, 7),
    "min":  (0, 3, 7),
    "dom7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "sus4": (0, 5, 7),
    "5":    (0, 7),
}


def midi_to_hz(m: np.ndarray | float):
    return 440.0 * (2.0 ** ((np.asarray(m) - 69) / 12.0))


def build_logfreq_fb(sr: int = SR, n_fft: int = N_FFT,
                     midi_lo: int = MIDI_LO, midi_hi: int = MIDI_HI) -> np.ndarray:
    """Triangular semitone filterbank: [n_pitch_bins, n_fft//2+1]."""
    n_bins = n_fft // 2 + 1
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    midis = np.arange(midi_lo, midi_hi + 1)
    centres = midi_to_hz(midis)
    fb = np.zeros((len(midis), n_bins), np.float32)
    for i, _ in enumerate(midis):
        lo = midi_to_hz(midis[i] - 1)
        hi = midi_to_hz(midis[i] + 1)
        ce = centres[i]
        for k, f in enumerate(fft_freqs):
            if lo < f <= ce:
                fb[i, k] = (f - lo) / (ce - lo)
            elif ce < f < hi:
                fb[i, k] = (hi - f) / (hi - ce)
    return fb


N_PITCH = MIDI_HI - MIDI_LO + 1
_FB = build_logfreq_fb()
_WIN = np.hanning(N_FFT).astype(np.float32)


def synth_frame(midi_notes, sr: int = SR, n_fft: int = N_FFT) -> np.ndarray:
    """Additive synth of one analysis frame for a set of MIDI notes.

    Randomizes per-note gain, harmonic roll-off and a random overall level so
    the detector is robust to real-instrument timbre and (with the level-
    invariant feature) to playing level.
    """
    t = np.arange(n_fft) / sr
    sig = np.zeros(n_fft, np.float32)
    rolloff = np.random.uniform(0.7, 1.6)                # harmonic decay exponent
    for m in midi_notes:
        f0 = float(midi_to_hz(m))
        n_harm = max(1, int(min(10, (sr / 2) / f0)))     # harmonics under Nyquist
        amp = np.random.uniform(0.4, 1.0)
        phase = np.random.uniform(0, 2 * np.pi)
        for h in range(1, n_harm + 1):
            sig += (amp / (h ** rolloff)) * np.sin(2 * np.pi * f0 * h * t + phase)
    sig += np.random.normal(0, 0.01, n_fft).astype(np.float32)
    sig *= np.float32(10.0 ** np.random.uniform(-1.5, 0.5))   # random level (the feature normalizes it out)
    return sig


def frame_feature(sig: np.ndarray) -> np.ndarray:
    # Level-invariant: normalize to unit RMS so detection works at any input
    # gain (real guitar at normal levels, not just clipping-loud).
    rms = float(np.sqrt(np.mean(sig ** 2)))
    if rms > 1e-4:
        sig = sig / rms
    mag = np.abs(np.fft.rfft(sig * _WIN))
    return np.log(_FB @ mag + 1e-6).astype(np.float32)           # [N_PITCH]


def random_chord():
    """Return (midi_notes, pitch_class_label[12])."""
    root = np.random.randint(0, 12)
    quality = QUALITIES[np.random.choice(list(QUALITIES))]
    octave = np.random.randint(0, 4)                              # which register
    base = MIDI_LO + 12 * octave + root
    notes = [base + iv for iv in quality]
    notes = [n + 12 * np.random.randint(0, 2) for n in notes]     # random voicing
    notes = [n for n in notes if MIDI_LO <= n <= MIDI_HI]
    if not notes:
        notes = [base]
    label = np.zeros(12, np.float32)
    for n in notes:
        label[n % 12] = 1.0
    return notes, label


class ChordDataset(Dataset):
    """Infinite random chords (epoch length is virtual)."""

    def __init__(self, length: int = 20000):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        notes, label = random_chord()
        feat = frame_feature(synth_frame(notes))
        return {"feat": torch.from_numpy(feat), "label": torch.from_numpy(label)}
