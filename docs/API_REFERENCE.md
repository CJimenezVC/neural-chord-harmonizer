# C++ API Reference

Interface-level documentation for the Chord Harmonizer's core classes.
Signatures mirror the headers under `plugin/Source/`.

## DSP

### `LogFreqFeature` (`DSP/LogFreqFeature.h`)
Log-frequency (one-bin-per-semitone) spectrogram feature for the detector.
JUCE-free (`SimpleFFT` + a baked filterbank), level-invariant, and a bit-for-bit
match to `training/chord_synth.py` (validated to ~3.4e-5).
```cpp
void prepare(int fftSize);
void setFilterbank(const float* fb, int nPitch, int nBins);  // from chord_info.json
int  getNumPitch() const noexcept;
// unit-RMS normalize → Hann → FFT → |·| → filterbank → log; writes nPitch values
void compute(const float* frame, int numSamples, float* out);
```

### `PitchShifter` (`DSP/PitchShifter.h`)
Streaming, formant-preserving phase-vocoder pitch shifter. JUCE-free
(`SimpleFFT` + `SampleFifo` + `OverlapAddBuffer`). Validated vs the Python
reference at corr ≈ 0.996.
```cpp
void prepare(int fftSize, int hop);     // 1024 / 256 in the plugin
void setRatio(float ratio);             // target / source pitch ratio
void process(const float* in, float* out, int numSamples);  // streaming
int  getLatencySamples() const;         // == fftSize
void reset();
```

### `YINPitchDetector` (`DSP/YINPitchDetector.h`)
```cpp
void prepare(double sampleRate, int bufferSize);
// Returns {f0Hz, confidence in [0,1]}
std::pair<float, float> detect(const float* buffer, int numSamples);
```

### `Resampler` / `SampleFifo` (`DSP/`)
```cpp
// Resampler: arbitrary-ratio SRC (Lagrange) used for sidechain host → 24 kHz
void prepare(double inRate, double outRate);
void process(const float* in, float* out, int numOut);
// SampleFifo: lock-free single-thread FIFO feeding the detector
void  push(const float* src, int n);   int size() const;
const float* data() const;             void consume(int n);
```

### `SimpleFFT` (`DSP/SimpleFFT.h`)
JUCE-free in-place radix-2 complex FFT, so the DSP compiles identically into the
plugin and the standalone unit tests.
```cpp
void prepare(int fftSize);
void transform(float* re, float* im, bool inverse);
```

## ML

### `NNModel` (+ `NNMath.h`) (`ML/`)
Self-contained feed-forward engine that loads an exported `.rtneural` JSON file
and runs a forward pass. Supported layers: dense, ReLU, sigmoid. JUCE-free math
in `NNMath.h`, unit-tested (`Tests/test_nn.cpp`) and validated against PyTorch to
~1e-8.
```cpp
bool loadFromFile(const juce::File& jsonFile);
void forward(const float* input, int inputLen, float* out);
int  inputSize() const noexcept;   int outputSize() const noexcept;
void reset();
```

### `ChordDetector` (`ML/ChordDetector.h`)
Wraps `LogFreqFeature` + `NNModel` into the polyphonic pitch-class detector.
```cpp
// Loads chordnet.rtneural + chord_info.json (dims, n_fft, midi range, filterbank)
bool load(const juce::File& chordnetFile, const juce::File& chordInfoFile);
bool isLoaded() const noexcept;
int  getFftSize() const noexcept;   // detector frame size (2048)
int  getMidiLo() const noexcept;    int getMidiHi() const noexcept;
// feature.compute → net.forward (sigmoid); writes 12 pitch-class activations
void detect(const float* frame, int numSamples, float* pc12);
```

## Host Integration

### `AdaptiveVoiceTransformProcessor : juce::AudioProcessor`
Standard JUCE lifecycle (`prepareToPlay`, `processBlock`, `releaseResources`,
APVTS state save/load). Declares a stereo main I/O plus a **sidechain** input
bus. Reports the pitch-shifter latency via `setLatencySamples`.
```cpp
bool loadModels(const juce::File& dir);     // chordnet.rtneural + chord_info.json
bool modelsLoaded() const noexcept;
int  getChordMask() const noexcept;         // 12-bit mask of detected pitch classes
juce::AudioProcessorValueTreeState& getValueTreeState() noexcept;
```
Parameters (APVTS): `tune` (0–1), `gate` (−80…−10 dB), `polyphony` (int 1–6).

Private helpers: `runDetector()` drains the sidechain FIFO and updates the held
chord; `collectTargets(voiceHz, midiOut)` selects the strongest `Polyphony`
chord tones near the voice and orders them lead-first.

### `AdaptiveVoiceTransformEditor : juce::AudioProcessorEditor`
Hosts the **Tune**, **Gate**, and **Polyphony** knobs, a live 12-note chord
readout (polled from `getChordMask` via a `Timer`), a **Load Models...** button,
and a model-status label. All UI strings are ASCII-only (JUCE's
`juce::String(const char*)` asserts on bytes > 127).
