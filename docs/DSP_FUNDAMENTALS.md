# DSP Fundamentals

Background on the signal-processing techniques used in the feature-extraction
front end. This is a reference, not a tutorial.

## Short-Time Fourier Transform (STFT)

The STFT slices audio into overlapping windows and computes a DFT of each:

```
X[m, k] = Σ_n x[n + m·H] · w[n] · e^(−j 2π k n / N)
```

- `N` = window length = 512 samples
- `H` = hop = 128 samples (75 % overlap)
- `w[n]` = analysis window (Hann)

Overlap of 75 % satisfies the constant-overlap-add (COLA) condition for clean
reconstruction.

## Mel-Spectrogram

Human pitch perception is roughly logarithmic. The mel scale warps frequency:

```
mel(f) = 2595 · log10(1 + f / 700)
```

We project the linear power spectrum onto 128 triangular mel filters spanning
20 Hz – 8 kHz, then take the log. This yields a perceptually-motivated,
compact representation that the neural network operates on.

## YIN Pitch Detection

YIN (de Cheveigné & Kawahara, 2002) estimates F0 via the difference function:

```
d[τ] = Σ_n (x[n] − x[n + τ])²
```

then a cumulative mean normalized difference:

```
d'[τ] = d[τ] / ( (1/τ) Σ_{j=1..τ} d[j] )
```

F0 is found at the first `τ` where `d'[τ]` dips below a threshold (0.1), refined
by parabolic interpolation. A confidence value `1 − d'[τ*]` is returned.

| Parameter      | Value                       |
| -------------- | --------------------------- |
| Threshold      | 0.1                         |
| Min period     | 50 samples (≈ 960 Hz max)   |
| Max period     | 4000 samples (≈ 12 Hz min)  |

YIN is more accurate than raw autocorrelation, robust to noise, cheap, and
streaming-friendly.

## Formant Analysis

Formants are the resonant frequencies of the vocal tract. We estimate them via
linear predictive coding (LPC): fit an all-pole model and take the angles of
its roots as formant frequencies. The first four formants (F1–F4) carry most
vowel identity and are shifted by the `FormantShift` parameter.

## Overlap-Add Reconstruction

To resynthesize, we window each output frame, overlap them by the hop size, and
sum. With a COLA-compliant window the overlapping tapers sum to unity, giving
artifact-free reconstruction.

## Further Reading

- de Cheveigné & Kawahara, "YIN, a fundamental frequency estimator for speech
  and music," JASA 2002.
- Rabiner & Schafer, *Theory and Application of Digital Speech Processing*.
- Smith, *Spectral Audio Signal Processing* (CCRMA, online).
