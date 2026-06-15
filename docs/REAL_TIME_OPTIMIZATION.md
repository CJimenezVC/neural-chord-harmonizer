# Real-Time Optimization

Strategies used to keep total latency in the 40–50 ms range and CPU under ~25 %
on a modern 4-core processor.

## Latency Budget

| Stage              | Budget | Source of cost                       |
| ------------------ | ------ | ------------------------------------ |
| Input buffering    | 10 ms  | 512-sample frame @ 24 kHz (~21 ms window, hop-paced) |
| Down/up resampling | 2 ms   | Lagrange SRC host ↔ 24 kHz + group delay |
| STFT analysis      | 5 ms   | 512-point FFT @ 24 kHz               |
| Feature norm       | 2 ms   | memory bandwidth                     |
| Encoder            | 8 ms   | 150 K params @ 24 kHz                |
| Decoder            | 9 ms   | 200 K params @ 24 kHz                |
| Vocoder            | 9 ms   | autoregressive GRU @ 24 kHz          |
| Overlap-add        | 3 ms   | windowing + sum                      |
| Output delay       | 2 ms   | ring buffer                          |
| **Total**          | **50 ms** |                                   |

Inference at 24 kHz (instead of 48 kHz) roughly halves the per-frame neural
cost; the resampler trades a small fixed overhead for that saving. Report the
true value to the host with `setLatencySamples()` so the DAW can compensate.

## 1. Model Compression

- **Pruning** — remove low-magnitude connections (target < 5 % accuracy loss).
- **Knowledge distillation** — train a smaller student against the full model.
- **Quantization** — INT8 inference where the measured loss stays < 1 dB MCD.

## 2. Streaming Inference

- Overlap-add instead of batch processing.
- Persist RNN state across frames (`VocoderNetwork` carries the GRU hidden state).
- Pre-compute the vocoder conditioning stack once per frame.

## 3. SIMD / Multithreading

- SIMD FFT (KISS FFT / FFTW / `juce::dsp::FFT`).
- Optional worker thread for the vocoder feeding a safety buffer (lock-free).
- Lock-free ring buffers for cross-thread communication.

## 4. Memory Management

- Pre-allocate **all** buffers in `prepareToPlay`.
- Reuse scratch arrays; never `new`/`malloc` on the audio thread.
- Cache-aligned data structures for hot inner loops.

## Measuring

Use `Utilities/Performance.h` to time each stage and dump to
`benchmarks/latency_measurements.csv`:

```cpp
ScopedTimer t{ "encoder" };   // RAII; logs elapsed µs on scope exit
```

Aggregate with `scripts/` tooling and record findings in
[`../benchmarks/cpu_usage_analysis.md`](../benchmarks/cpu_usage_analysis.md).

## Checklist before release

- [ ] No allocations on the audio thread (verify with a RT allocator hook).
- [ ] `setLatencySamples` matches measured latency.
- [ ] CPU < 25 % at 48 kHz / 512-sample blocks.
- [ ] `pluginval` strictness 8 passes.
- [ ] No denormals (enable FTZ/DAZ).
