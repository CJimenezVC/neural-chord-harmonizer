"""End-to-end offline harmonizer: voice auto-tuned to a chord progression.

Synthesizes an instrument chord progression, runs a real voice through the
trained ChordNet (active pitch classes per frame) + YIN F0, snaps the voice to
the nearest chord tone, and pitch-shifts it with a formant-preserving phase
vocoder (the reference algorithm for the C++ plugin).

Outputs WAVs and reports how well the tuned voice lands on the chord tones.

    python harmonizer_demo.py --voice <wav>
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

import chord_synth as cs
from model import ChordNet

SR = cs.SR


# --- instrument: a sustained chord progression -----------------------------

def synth_progression(chords, dur=1.0):
    """chords: list of (root_pc, quality). Returns audio + per-time MIDI sets."""
    out = []
    seg = int(SR * dur)
    t = np.arange(seg) / SR
    for root, qual in chords:
        midis = [cs.MIDI_LO + 24 + root + iv for iv in cs.QUALITIES[qual]]   # ~C4 register
        sig = np.zeros(seg, np.float32)
        for m in midis:
            f0 = float(cs.midi_to_hz(m))
            for h in range(1, 7):
                sig += (0.7 / h) * np.sin(2 * np.pi * f0 * h * t)
        env = np.minimum(1.0, np.minimum(t * 20, (dur - t) * 8)).astype(np.float32)
        out.append(sig * env)
    return np.concatenate(out)


# --- formant-preserving phase-vocoder pitch shift --------------------------

def smooth_env(logmag, width=21):
    k = np.ones(width) / width
    return np.convolve(logmag, k, mode="same")


def pitch_shift(x, ratios, n_fft=1024, hop=256):
    """ratios: pitch ratio per analysis hop (1.0 = unchanged)."""
    win = np.hanning(n_fft).astype(np.float32)
    nb = n_fft // 2 + 1
    omega = 2 * np.pi * hop * np.arange(nb) / n_fft
    n = 1 + (len(x) - n_fft) // hop
    out = np.zeros(len(x) + n_fft, np.float32)
    wsum = np.zeros(len(x) + n_fft, np.float32)
    prev = np.zeros(nb)
    syn = np.zeros(nb)
    for i in range(n):
        seg = x[i * hop:i * hop + n_fft] * win
        S = np.fft.rfft(seg)
        mag, ph = np.abs(S), np.angle(S)
        d = ph - prev - omega
        prev = ph
        d = np.mod(d + np.pi, 2 * np.pi) - np.pi
        inst = omega + d                                   # true phase advance/hop
        r = float(ratios[min(i, len(ratios) - 1)])
        env = np.exp(smooth_env(np.log(mag + 1e-8)))       # formant envelope
        flat = mag / (env + 1e-8)                          # whitened excitation
        nf = np.zeros(nb)
        ni = np.zeros(nb)
        idx = np.round(np.arange(nb) * r).astype(int)
        for k in range(nb):
            kr = idx[k]
            if 0 <= kr < nb:
                nf[kr] += flat[k]
                ni[kr] = inst[k] * r
        new_mag = nf * env                                 # re-apply ORIGINAL formants
        syn = syn + ni
        Y = new_mag * np.exp(1j * syn)
        out[i * hop:i * hop + n_fft] += (np.fft.irfft(Y, n=n_fft) * win).astype(np.float32)
        wsum[i * hop:i * hop + n_fft] += win ** 2
    out /= (wsum + 1e-8)
    return out[:len(x)]


# --- harmonizer ------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default=None)
    ap.add_argument("--out", default="/tmp/harmonized.wav")
    args = ap.parse_args()

    pt = "../models/pytorch"
    info = json.load(open(pt + "/chord_info.json"))
    net = ChordNet(info["in_dim"]).eval()
    net.load_state_dict(torch.load(pt + "/chordnet.pt", map_location="cpu"))
    midis = np.arange(cs.MIDI_LO, cs.MIDI_HI + 1)

    voice = args.voice or sorted(glob.glob("../data/datasets/vcc2020/source/SEF1/*.wav"))[0]
    v, _ = librosa.load(voice, sr=SR, mono=True)
    v = v / (np.abs(v).max() + 1e-9) * 0.9                           # normalize input

    # Instrument: loop C - G - Am - F to cover the full vocal.
    base = [(0, "maj"), (7, "maj"), (9, "min"), (5, "maj")]
    dur = 1.5
    n_chords = int(np.ceil(len(v) / (SR * dur)))
    prog = [base[i % len(base)] for i in range(n_chords)]
    instr_raw = synth_progression(prog, dur=dur)[:len(v)]            # loud, for detection
    instr = 0.7 * instr_raw / (np.abs(instr_raw).max() + 1e-9)       # normalized, for audio

    hop = 256
    n = 1 + (len(v) - cs.N_FFT) // hop
    # voice F0
    f0, _, _ = librosa.pyin(v, fmin=80, fmax=600, sr=SR, frame_length=cs.N_FFT, hop_length=hop)
    f0 = np.nan_to_num(f0)

    ratios = np.ones(n)
    landed = 0
    voiced = 0
    for i in range(n):
        s = i * hop
        # detect chord pitch classes from the (loud) instrument frame
        feat = cs.frame_feature(instr_raw[s:s + cs.N_FFT])
        with torch.no_grad():
            act = torch.sigmoid(net(torch.from_numpy(feat))).numpy()
        pcs = np.where(act > 0.5)[0]
        fv = f0[min(i, len(f0) - 1)]
        if fv < 50 or len(pcs) == 0:
            continue
        voiced += 1
        # nearest chord tone (in any octave) to the voice's current pitch
        v_midi = 69 + 12 * np.log2(fv / 440.0)
        cands = [m for m in midis if (m % 12) in pcs]
        target = min(cands, key=lambda m: abs(m - v_midi))
        ratios[i] = float(cs.midi_to_hz(target) / fv)
        if abs(12 * np.log2(cs.midi_to_hz(target) / fv)) < 6:
            landed += 1

    tuned = pitch_shift(v, ratios, n_fft=cs.N_FFT, hop=hop)

    def norm(a, peak=0.9):
        return (peak * a / (np.abs(a).max() + 1e-9)).astype(np.float32)

    tuned = norm(tuned)
    sf.write(args.out, tuned, SR)
    sf.write(args.out.replace(".wav", "_mix.wav"),
             norm(0.85 * tuned + 0.5 * instr[:len(tuned)]), SR)
    sf.write(args.out.replace(".wav", "_instrument.wav"), instr, SR)

    # Verify: F0 of the tuned voice should sit on chord tones.
    f0o, _, _ = librosa.pyin(tuned, fmin=80, fmax=600, sr=SR,
                             frame_length=cs.N_FFT, hop_length=hop)
    f0o = np.nan_to_num(f0o)
    errs = []
    for i in range(min(n, len(f0o))):
        if f0o[i] < 50:
            continue
        m = 69 + 12 * np.log2(f0o[i] / 440.0)
        errs.append(abs(m - round(m)))     # cents-ish distance to nearest semitone
    print(f"frames: {n}  voiced: {voiced}  shifts within 6 st: {landed}")
    print(f"tuned-voice pitch: median |dist to nearest semitone| = {np.median(errs)*100:.1f} cents")
    print(f"wrote {args.out} (+ _mix, _instrument)")


if __name__ == "__main__":
    main()
