# Architecture

This is the detailed companion to [`../TECHNICAL.md`](../TECHNICAL.md). It maps
each stage of the Chord Harmonizer to the concrete C++ class that implements it.

## Module Map

| Stage                  | Class                          | Location                     |
| ---------------------- | ------------------------------ | ---------------------------- |
| Sidechain → 24 kHz SRC | `Resampler`, `SampleFifo`      | `plugin/Source/DSP/`         |
| Log-frequency feature  | `LogFreqFeature` (+ `SimpleFFT`)| `plugin/Source/DSP/`        |
| Voice F0               | `YINPitchDetector`             | `plugin/Source/DSP/`         |
| Formant-preserving shift | `PitchShifter`               | `plugin/Source/DSP/`         |
| Streaming buffers      | `OverlapAddBuffer`, `SampleFifo`| `plugin/Source/DSP/`        |
| Inference engine       | `NNModel` (+ `NNMath.h`)       | `plugin/Source/ML/`          |
| Chord detector         | `ChordDetector`                | `plugin/Source/ML/`          |
| Plugin host glue       | `PluginProcessor`, `PluginEditor`| `plugin/Source/`           |

> **Legacy (deprecated, not in the build):** `FeatureExtractor`,
> `FormantAnalyzer`, `SpectrogramProcessor`, `EncoderNetwork`,
> `DecoderNetwork`, `VocoderNetwork`, `ModelManager`, `NeuralAudioProcessor`
> belong to the old voice-conversion engine and are kept for reference only.

## Data Flow (audio thread)

The host runs at any sample rate. Only the sidechain is resampled to the
detector's fixed 24 kHz; the voice path stays at host rate.

```
processBlock(buffer)                                    // host rate
  voiceMono = mono-sum(mainInput)
  sideMono  = mono-sum(sidechainInput)

  // --- detector path (24 kHz) ---
  downsampler: sideMono → 24 kHz → scFifo.push
  runDetector():
    while scFifo has a full 2048 frame:
        rms gate?  → if open: feature = LogFreqFeature.compute(frame)
                              act     = ChordNet.forward(feature)   // 12 sigmoids
                     else:    act = 0
        chordHeld  = max(0.97 · chordHeld, act)         // peak-hold + decay
        chordMask  = threshold(chordHeld)
        scFifo.consume(512)

  // --- voice path (host rate) ---
  (f0, conf) = YIN.detect(voice rolling window)
  targets    = collectTargets(f0)                       // strongest `Polyphony` pcs
  mix = 0
  for v in 0..maxVoices:
      ratio = active ? (targetHz/f0)^Tune : 1
      PitchShifter[v].setRatio(smoothed ratio)
      out = PitchShifter[v].process(voiceMono)
      if active: mix += out
  buffer = mix · (1/√activeCount)                       // equal-power; silent if no chord
```

## Threading & Real-Time Safety

- **No allocations on the audio thread.** All buffers are pre-allocated in
  `prepareToPlay` (detector frame, scratch, FIFOs, per-voice shifters).
- **No locks on the audio thread.** Parameter changes flow through
  `AudioProcessorValueTreeState` atomics. The detected chord is published to the
  editor via a single `std::atomic<int>` bitmask (`getChordMask`).
- **Model loading** suspends processing briefly (`suspendProcessing`) while
  `ChordDetector::load` swaps in new weights — safe to call while playing.

## Model Loading

`ChordDetector::load` reads two files from `models/pretrained/`:

- `chordnet.rtneural` — the dense ChordNet weights (loaded by `NNModel`).
- `chord_info.json` — `in_dim`, `n_fft`, `sample_rate`, `midi_lo`, `midi_hi`, and
  the baked `logfreq_fb` filterbank used by `LogFreqFeature`.

Load via the editor's **Load Models...** button or the `AVT_MODELS_DIR`
environment variable.

See [`API_REFERENCE.md`](API_REFERENCE.md) for class-level interface docs.
