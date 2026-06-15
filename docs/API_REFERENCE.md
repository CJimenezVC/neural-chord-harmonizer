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

### `EncoderNetwork`
```cpp
void  loadModel(const nlohmann::json& weights);
// mel (T x 128) -> style vector (styleDim)
void  encode(const float* mel, int numFrames, float* styleOut);
```

### `DecoderNetwork`
```cpp
void  loadModel(const nlohmann::json& weights);
// (mel frame ⊕ style) -> transformed mel frame
void  decode(const float* melFrame, const float* style, float* melOut);
```

### `VocoderNetwork`
```cpp
void  loadModel(const nlohmann::json& weights);
// mel frame -> waveform samples; carries GRU state across calls
int   synthesize(const float* melFrame, float* audioOut, int maxSamples);
void  reset();   // clear recurrent state
```

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

### `PluginEditor : juce::AudioProcessorEditor`
Hosts `StyleKnob`, brightness/formant/pitch sliders, `PresetManager`, and two
`SpectrumAnalyzer` views (before/after).
