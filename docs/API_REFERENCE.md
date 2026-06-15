# C++ API Reference

Interface-level documentation for the plugin's core classes. Signatures mirror
the headers under `plugin/Source/`. Implementations are in progress; this
documents the intended contract.

## DSP

### `FeatureExtractor`
```cpp
struct Features {
    std::vector<float> mel;       // 128 bins
    float f0 = 0.0f;              // Hz
    std::array<float, 4> formants{};
    float voicedConfidence = 0.0f;
};

void  prepare(double sampleRate, int frameSize, int hopSize);
Features process(const float* frame, int numSamples);
void  reset();
```

### `YINPitchDetector`
```cpp
void  prepare(double sampleRate, int bufferSize);
// Returns {f0Hz, confidence in [0,1]}
std::pair<float, float> detect(const float* buffer, int numSamples);
void  setThreshold(float t);          // default 0.1
```

### `FormantAnalyzer`
```cpp
void  prepare(double sampleRate, int order);   // LPC order
std::array<float, 4> estimate(const float* frame, int numSamples);
```

### `OverlapAddBuffer` / `CircularAudioBuffer`
```cpp
void  prepare(int frameSize, int hopSize, int channels);
void  add(const float* frame, int numSamples);   // OLA accumulate
int   read(float* dst, int numSamples);          // pop reconstructed audio
// CircularAudioBuffer:
void  push(const float* src, int numSamples);
bool  pop(float* dst, int frameSize, int hopSize);
```

## ML

### `ModelManager`
```cpp
bool  loadFromDirectory(const juce::File& dir);  // encoder/decoder/vocoder + info
bool  isLoaded() const noexcept;
EncoderNetwork&  encoder() noexcept;
DecoderNetwork&  decoder() noexcept;
VocoderNetwork&  vocoder() noexcept;
const ModelInfo& info() const noexcept;          // norm stats, dims
```

### `NNModel` (+ `NNMath.h`)
Self-contained feed-forward engine that loads an exported `.rtneural` JSON file
and runs a single-frame (streaming) forward pass. Supported layers: dense,
conv1d (single-frame centre tap), global_mean_pool (identity for one frame),
gru (state carried across calls). The math lives in the JUCE-free `NNMath.h`
and is unit-tested (`Tests/test_nn.cpp`) and validated against PyTorch to ~1e-8
(`training/verify_rtneural_export.py`).
```cpp
bool  loadFromFile(const juce::File& jsonFile);
void  forward(const float* input, int inputLen, float* out);  // carries GRU state
void  reset();
int   inputSize() const noexcept;   int outputSize() const noexcept;
```

> A single-frame conv1d with `padding=1, kernel=3` reduces exactly to the centre
> kernel tap — which is why the streaming path is correct without buffering past
> frames (the ±1 temporal taps are unused in this mode).

### `EncoderNetwork` / `DecoderNetwork` / `VocoderNetwork`
Thin wrappers over `NNModel`:
```cpp
bool  loadModel(const juce::File& jsonFile);     // all three
void  EncoderNetwork::encode(const float* mel, int numFrames, float* styleOut);
void  DecoderNetwork::decode(const float* melFrame, const float* style, float* melOut);
int   VocoderNetwork::synthesize(const float* melFrame, float* audioOut, int maxSamples);
void  VocoderNetwork::reset();   // clear recurrent state
```

> **Vocoder caveat:** the trained WaveRNN head emits one 256-way mu-law
> categorical value *per mel frame* (frame-rate), not one sample per step. The
> wrapper decodes the argmax to an amplitude and fills the hop with a click-free
> ramp — so it runs the model end-to-end, but output is a frame-rate envelope,
> not speech-quality audio. A sample-rate vocoder (or mel inversion) is the next
> step for real audio.

### `NeuralAudioProcessor`
```cpp
void  prepare(double sampleRate, int frameSize);
void  setModels(ModelManager* models) noexcept;
void  setStyleParams(const StyleParams& p) noexcept;  // real-time safe
// Full encode → modulate → decode → vocode for one frame
int   processFrame(const Features& feats, float* audioOut, int maxSamples);
```

## Parameters

### `StyleInterpolation`
```cpp
struct StyleParams {
    float styleShift = 0.0f;       // [0,1]
    float brightness = 0.0f;       // [-1,1]
    float formantShift = 0.0f;     // semitones [-12,12]
    float pitchShift = 0.0f;       // semitones [-24,24]
};
void apply(const float* sourceStyle, const float* targetStyle,
           const StyleParams& p, float* styleOut, int styleDim);
```

## Host Integration

### `PluginProcessor : juce::AudioProcessor`
Standard JUCE lifecycle: `prepareToPlay`, `processBlock`, `releaseResources`,
state save/load via `AudioProcessorValueTreeState`. Reports latency via
`setLatencySamples`.

```cpp
// Load RTNeural models and re-prepare at the model's sample rate
// (read from model_info.json). Suspends processing during the swap;
// safe to call while playing.
bool loadModels (const juce::File& dir);
```

The neural chain runs at `model_info.json`'s `sample_rate` (24 kHz by default).
`prepareToPlay` adopts that rate if models are already loaded; `loadModels`
re-prepares the resamplers/buffers if models arrive later. The host stream is
resampled to/from this rate (see `DSP/Resampler.h`).

### `PluginEditor : juce::AudioProcessorEditor`
Hosts `StyleKnob`, brightness/formant/pitch sliders, `PresetManager`, and two
`SpectrumAnalyzer` views (before/after).
