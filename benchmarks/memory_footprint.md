# Memory Footprint

The Chord Harmonizer is light: the only model is the small dense ChordNet, and
the DSP buffers are modest.

| Component                  | Approx. size | Notes                                  |
| -------------------------- | ------------ | -------------------------------------- |
| ChordNet weights           | _TBD_        | dense 61→256→256→12 (FP32)             |
| Log-freq filterbank        | _TBD_        | 61 × n_fft_bins (baked, pre-allocated) |
| Detector FFT + frame       | _TBD_        | 2048-pt @ 24 kHz                       |
| Pitch shifters (× up to 6) | _TBD_        | 1024-pt FFT + OLA buffers per voice    |
| Sidechain FIFO + resampler | _TBD_        | pre-allocated                          |
| JUCE + UI                  | _TBD_        |                                        |

All audio-thread buffers are pre-allocated in `prepareToPlay`; there is no
runtime allocation in `processBlock`. The per-voice pitch-shifter buffers are
the largest DSP allocation and scale with `maxVoices` (6). Populate measured
values after the first profiled build.
