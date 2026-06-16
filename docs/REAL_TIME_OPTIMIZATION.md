# Real-Time Optimization

Strategies used to keep the Chord Harmonizer responsive and light on CPU.

## Where the latency is

The only latency in the **audible** path is the voice pitch shifter's analysis
window. The detector path adds no audible delay — the voice is shifted
continuously and simply follows the most recent detected chord.

| Stage                  | Latency / cost                                   |
| ---------------------- | ------------------------------------------------ |
| Voice pitch shift      | 1024-sample FFT @ host rate (~21 ms @ 48 kHz) ← reported to host |
| Voice F0 (YIN)         | per-block, on a rolling window                   |
| Sidechain → 24 kHz SRC | Lagrange resampler + small FIFO (not in voice path) |
| Detector feature       | 2048-pt FFT @ 24 kHz, every 512-sample hop       |
| ChordNet forward       | 3 dense layers (61→256→256→12) — negligible      |

Report the true value to the host with `setLatencySamples()` (the pitch-shifter
FFT size) so the DAW can compensate.

## 1. Keep the voice path at host rate

Resampling only the **sidechain** (not the voice) means the audible signal never
passes through a rate converter — no SRC artifacts on the harmony, and the
pitch shifter's latency is the only reported delay.

## 2. Cheap detector

- ChordNet is dense-only and tiny; a forward pass is a few matrix-vector
  products. No recurrence, no autoregression.
- The detector runs at 24 kHz with a 512-sample hop (~187 frames/s), far below
  the audio block rate, and drains a FIFO rather than blocking.

## 3. Streaming, allocation-free DSP

- Overlap-add streaming pitch shifters (`PitchShifter`), not batch processing.
- Pre-allocate **all** buffers in `prepareToPlay`; never `new`/`malloc` on the
  audio thread.
- Reuse scratch arrays across the per-voice loop.

## 4. Polyphony scales the cost

Each harmony voice is an independent phase-vocoder pitch shifter. CPU scales with
the **Polyphony** setting (and how many chord tones are active): 1 voice for a
bass line is far cheaper than 6 voices for a full guitar chord. Idle voices
still run their shifter but are not summed — set Polyphony to your instrument's
realistic chord size.

## 5. SIMD / threading (optional)

- `SimpleFFT` is a clear radix-2 implementation; swap in `juce::dsp::FFT` or an
  optimized FFT if profiling shows the transforms dominate.
- Lock-free FIFOs already isolate the message thread (model load, chord readout)
  from the audio thread.

## Measuring

Profile in the Standalone host with a looped vocal + instrument, and record
findings in [`../benchmarks/cpu_usage_analysis.md`](../benchmarks/cpu_usage_analysis.md)
and [`../benchmarks/memory_footprint.md`](../benchmarks/memory_footprint.md).

## Checklist before release

- [ ] No allocations on the audio thread (verify with a RT allocator hook).
- [ ] `setLatencySamples` matches the pitch-shifter FFT size.
- [ ] CPU acceptable at max Polyphony (6 voices) at 48 kHz / 512-sample blocks.
- [ ] `pluginval` strictness 8 passes.
- [ ] No denormals (enable FTZ/DAZ via `ScopedNoDenormals`).
