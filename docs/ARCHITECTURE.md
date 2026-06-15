# Architecture

This is the detailed companion to [`../TECHNICAL.md`](../TECHNICAL.md). It maps
each pipeline stage to the concrete C++ class that implements it.

## Module Map

| Stage              | Class                              | Location                          |
| ------------------ | ---------------------------------- | --------------------------------- |
| Host ↔ 24 kHz SRC  | `Resampler`, `SampleFifo`          | `plugin/Source/DSP/`              |
| Feature extraction | `FeatureExtractor`                 | `plugin/Source/DSP/`              |
| Pitch detection    | `YINPitchDetector`                 | `plugin/Source/DSP/`              |
| Formant analysis   | `FormantAnalyzer`                  | `plugin/Source/DSP/`              |
| STFT / spectra     | `SpectrogramProcessor`             | `plugin/Source/DSP/`              |
| Streaming buffers  | `OverlapAddBuffer`, `CircularAudioBuffer` | `plugin/Source/DSP/`       |
| Inference engine   | `NNModel` (+ `NNMath.h`)           | `plugin/Source/ML/`               |
| Model loading      | `ModelManager`                     | `plugin/Source/ML/`               |
| Encoder            | `EncoderNetwork`                   | `plugin/Source/ML/`               |
| Decoder            | `DecoderNetwork`                   | `plugin/Source/ML/`               |
| Vocoder            | `VocoderNetwork`                   | `plugin/Source/ML/`               |
| Unified inference  | `NeuralAudioProcessor`             | `plugin/Source/ML/`               |
| Style control      | `StyleInterpolation`               | `plugin/Source/Parameters/`       |
| Plugin host glue   | `PluginProcessor`, `PluginEditor`  | `plugin/Source/`                  |

## Data Flow (audio thread)

The host runs at any sample rate; the neural chain runs at a fixed 24 kHz, so
the block is resampled on the way in and out.

```
processBlock(buffer)                                   // host rate
  → hostInFifo.push(buffer)
  → downsampler: hostInFifo → 24 kHz → CircularAudioBuffer.push
  → while a full frame is available:                   // all @ 24 kHz
        frame    = CircularAudioBuffer.pop(frameSize, hop)
        features = FeatureExtractor.process(frame)      // mel, F0, formants
        style    = EncoderNetwork.encode(features.mel)
        style    = StyleInterpolation.apply(style, params)
        mel'     = DecoderNetwork.decode(features.mel, style)
        wave     = VocoderNetwork.synthesize(mel')
        OverlapAddBuffer.add(wave)
        modelOutFifo.push(OverlapAddBuffer.read(hop))
  → upsampler: modelOutFifo → host rate → buffer        // fills numSamples
```

## Threading & Real-Time Safety

- **No allocations on the audio thread.** All buffers are pre-allocated in
  `prepareToPlay`.
- **No locks on the audio thread.** Parameter changes flow through the
  `AudioProcessorValueTreeState` atomics; model swaps use a lock-free pointer
  exchange in `ModelManager`.
- **State carried across frames** for the GRU vocoder lives in `VocoderNetwork`.

## Model Loading

`ModelManager` loads three RTNeural JSON files (`encoder`, `decoder`,
`vocoder`) plus `model_info.json` (architecture metadata + normalization
stats). Models are loaded on the message thread and published to the audio
thread atomically.

See [`API_REFERENCE.md`](API_REFERENCE.md) for class-level interface docs.
