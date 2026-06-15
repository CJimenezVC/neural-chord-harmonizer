# Memory Footprint

Target: ~200 MB resident.

| Component              | Approx. size | Notes                        |
| ---------------------- | ------------ | ---------------------------- |
| Encoder weights        | _TBD_        | ~150K params (FP32)          |
| Decoder weights        | _TBD_        | ~200K params (FP32)          |
| Vocoder weights        | _TBD_        | ~150K params (FP32)          |
| Mel filterbank + FFT   | _TBD_        | pre-allocated                |
| Ring / OLA buffers     | _TBD_        | pre-allocated in prepareToPlay |
| JUCE + UI              | _TBD_        |                              |

All audio-thread buffers are pre-allocated; no runtime allocation in
`processBlock`. Populate measured values after the first profiled build.
