// pybind11 binding exposing the plugin's exact feature extraction to Python.
//
// The point: training preprocessing calls the *same* C++ SpectrogramProcessor
// the plugin uses, so there is no Python/C++ feature drift by construction.
// Streaming framing (left-aligned Hann frames, hop advance) matches the plugin,
// not librosa's centered STFT.
//
// Build (see bindings/build.sh):
//   c++ -O3 -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) \
//       -I ../plugin/Source avtdsp.cpp ../plugin/Source/DSP/SpectrogramProcessor.cpp \
//       -o avtdsp$(python3-config --extension-suffix)

#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "DSP/SpectrogramProcessor.h"

namespace py = pybind11;

// audio: 1-D float32; mel_fb: [n_mels, n_bins] float32 (row-major).
// Returns log-mel features [T, n_mels] using the plugin's streaming framing.
py::array_t<float> extract_logmel (py::array_t<float, py::array::c_style | py::array::forcecast> audio,
                                   int n_fft, int hop_length, int n_mels,
                                   py::array_t<float, py::array::c_style | py::array::forcecast> mel_fb,
                                   double sample_rate)
{
    const int len   = (int) audio.shape (0);
    const int nBins = n_fft / 2 + 1;
    if ((int) mel_fb.shape (0) != n_mels || (int) mel_fb.shape (1) != nBins)
        throw std::runtime_error ("mel_fb shape must be [n_mels, n_fft/2+1]");

    SpectrogramProcessor sp;
    sp.prepare (sample_rate, n_fft, n_mels);
    sp.setMelFilterbank (mel_fb.data(), n_mels, nBins);

    const int T = len >= n_fft ? 1 + (len - n_fft) / hop_length : 0;
    py::array_t<float> out ({ T, n_mels });
    auto o = out.mutable_unchecked<2>();
    const float* x = audio.data();

    std::vector<float> power ((size_t) nBins), mel ((size_t) n_mels);
    for (int t = 0; t < T; ++t)
    {
        sp.computeMagnitude (x + (size_t) (t * hop_length), n_fft, power.data());  // power
        sp.magnitudeToLogMel (power.data(), mel.data());                           // log-mel
        for (int m = 0; m < n_mels; ++m)
            o (t, m) = mel[(size_t) m];
    }
    return out;
}

PYBIND11_MODULE (avtdsp, m)
{
    m.doc() = "Adaptive Voice Transform DSP — the plugin's feature extraction, for training.";
    m.def ("extract_logmel", &extract_logmel,
           py::arg ("audio"), py::arg ("n_fft"), py::arg ("hop_length"),
           py::arg ("n_mels"), py::arg ("mel_fb"), py::arg ("sample_rate") = 24000.0,
           "Streaming log-mel features [T, n_mels] using the plugin's SpectrogramProcessor.");
}
