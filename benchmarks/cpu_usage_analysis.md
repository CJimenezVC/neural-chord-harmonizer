# CPU Usage Analysis

Measured in the Standalone host at 48 kHz, 512-sample blocks, with a looped
vocal on the main input and an instrument on the sidechain.

## What drives CPU

| Stage                  | Cost driver                                          |
| ---------------------- | --------------------------------------------------- |
| Voice pitch shifters   | **dominant** — one phase vocoder per active voice (scales with Polyphony) |
| Voice F0 (YIN)         | per-block difference function on a rolling window    |
| Detector feature + net | 2048-pt FFT @ 24 kHz every hop + a tiny dense net    |
| Sidechain SRC          | Lagrange resampler (host → 24 kHz)                   |

The harmony voices are the hot path: CPU scales roughly linearly with the
**Polyphony** setting and how many chord tones are actually active.

## Methodology

1. Build the plugin in Release.
2. Run the Standalone host; route a looped instrument to the sidechain.
3. Sweep **Polyphony** 1 → 6 and record process CPU (Activity Monitor / `perf` /
   Task Manager) at each setting.

## Results

| Polyphony | Active voices | CPU % | Notes |
| --------- | ------------- | ----- | ----- |
| 1         | bass line     | _TBD_ |       |
| 4         | typical chord | _TBD_ | default |
| 6         | full guitar   | _TBD_ | worst case |

## Observations

_Fill in after profiling._
