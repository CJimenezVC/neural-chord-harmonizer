# DSP Fundamentals

Background on the signal-processing techniques used in the Chord Harmonizer.
This is a reference, not a tutorial.

## Short-Time Fourier Transform (STFT)

Both paths slice audio into overlapping windows and compute a DFT of each:

```
X[m, k] = Σ_n x[n + m·H] · w[n] · e^(−j 2π k n / N)
```

| Use                 | `N` (window) | `H` (hop) | Rate    |
| ------------------- | ------------ | --------- | ------- |
| Detector feature    | 2048         | 512       | 24 kHz  |
| Voice pitch shifter | 1024         | 256       | host    |

`w[n]` is a Hann window; the 75 % overlap (hop = N/4) satisfies the
constant-overlap-add (COLA) condition for clean reconstruction. The transform
itself is a JUCE-free radix-2 FFT (`DSP/SimpleFFT.h`) so the exact same code runs
in the plugin and in the offline unit tests.

## Log-Frequency Feature

Western pitch is logarithmic: each octave doubles in frequency, and a semitone is
a fixed *ratio* (2^(1/12)). For chord detection we want **one bin per semitone**,
not the linear bins of a raw FFT. We project the magnitude spectrum onto a bank
of triangular filters centered on equal-tempered semitones:

```
feature[p] = log( Σ_k  fb[p, k] · |X[k]|  + 1e-6 )
```

- 61 filters spanning MIDI 36–96 (C2–C7), one per semitone.
- The filterbank `fb` is baked at training time and shipped in `chord_info.json`,
  so the C++ and Python features are identical.

**Level invariance.** Before the FFT, each frame is normalized to unit RMS. A
quiet pick and a clipping strum then produce the same feature, so the detector
works at any input gain instead of only when the instrument is loud.

## YIN Pitch Detection

The voice path needs the singer's fundamental F0. YIN (de Cheveigné &
Kawahara, 2002) estimates it via the difference function:

```
d[τ] = Σ_n (x[n] − x[n + τ])²
```

then a cumulative mean normalized difference:

```
d'[τ] = d[τ] / ( (1/τ) Σ_{j=1..τ} d[j] )
```

F0 is the first `τ` where `d'[τ]` dips below a threshold, refined by parabolic
interpolation; confidence is `1 − d'[τ*]`. YIN is accurate, noise-robust, cheap,
and streaming-friendly — and unlike the detector it is monophonic, which is all
the voice path needs.

## Phase-Vocoder Pitch Shifting (formant-preserving)

To move the voice onto a chord tone without turning it into a chipmunk, we shift
pitch while **preserving formants** (the vocal-tract resonances that carry vowel
identity and timbre):

1. **Analysis.** STFT the voice; for each bin estimate the *instantaneous
   frequency* from the phase difference between consecutive hops — more accurate
   than the bin's nominal center frequency.
2. **Envelope whitening.** Divide the magnitude by a smoothed spectral envelope,
   separating *what notes* (fine structure) from *what timbre* (envelope).
3. **Resynthesis.** Re-bin the fine structure to the target pitch ratio,
   accumulate synthesis phase for coherence, then **re-apply the original
   envelope** at the new pitch. The formants stay put, so it still sounds like
   the singer.

This is validated against a Python reference at corr ≈ 0.996.

## Overlap-Add Reconstruction

Each synthesized frame is windowed, overlapped by the hop, and summed
(`DSP/OverlapAddBuffer.h`). With a COLA-compliant window the overlapping tapers
sum to unity, giving artifact-free reconstruction.

## Chord Stabilization

Per-frame detector output flickers. Two simple, classical tools smooth it:

- **Noise gate:** zero the activations when the instrument's frame RMS is below
  the `Gate` threshold (no reading notes out of noise).
- **Peak-hold with decay:** `held = max(0.97 · held, activation)` per hop, so a
  strummed chord rings down smoothly instead of stuttering.

## Further Reading

- de Cheveigné & Kawahara, "YIN, a fundamental frequency estimator for speech
  and music," JASA 2002.
- Laroche & Dolson, "Improved phase vocoder time-scale modification of audio,"
  IEEE TSAP 1999.
- Smith, *Spectral Audio Signal Processing* (CCRMA, online).
