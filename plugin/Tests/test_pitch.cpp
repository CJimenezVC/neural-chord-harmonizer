// Standalone check: the C++ PitchShifter must match the Python reference
// (training/harmonizer_demo.py:pitch_shift) for the same input + ratio.
//
//   python -c "...generate /tmp/ps_in.f32 /tmp/ps_ref.f32..."   (see commit msg)
//   c++ -std=c++20 -I plugin/Source plugin/Tests/test_pitch.cpp \
//       plugin/Source/DSP/OverlapAddBuffer.cpp -o /tmp/test_pitch && /tmp/test_pitch
#include <cmath>
#include <cstdio>
#include <vector>

#include "DSP/PitchShifter.h"

static std::vector<float> readF32 (const char* path)
{
    FILE* f = std::fopen (path, "rb");
    if (! f) { std::printf ("cannot open %s\n", path); return {}; }
    std::fseek (f, 0, SEEK_END);
    const long n = std::ftell (f) / 4;
    std::fseek (f, 0, SEEK_SET);
    std::vector<float> v ((size_t) n);
    [[maybe_unused]] auto r = std::fread (v.data(), 4, (size_t) n, f);
    std::fclose (f);
    return v;
}

int main()
{
    auto in  = readF32 ("/tmp/ps_in.f32");
    auto ref = readF32 ("/tmp/ps_ref.f32");
    if (in.empty() || ref.empty()) return 2;

    PitchShifter ps;
    ps.prepare (1024, 256);
    ps.setRatio (1.3f);

    std::vector<float> out (in.size(), 0.0f);
    const int blk = 256;
    for (size_t i = 0; i + (size_t) blk <= in.size(); i += (size_t) blk)
        ps.process (&in[i], &out[i], blk);

    // The streaming shifter is delayed ~fftSize; find the lag that best aligns
    // C++ output with the (non-delayed) Python reference, then report correlation.
    const int N = (int) in.size();
    double bestC = -2.0; int bestLag = 0;
    for (int lag = 0; lag < 2048; ++lag)
    {
        double sx = 0, sy = 0, sxy = 0, sxx = 0, syy = 0; int cnt = 0;
        for (int t = 4000; t < N - 2048; ++t)
        {
            const float a = out[(size_t) t];
            const float b = ref[(size_t) (t - lag)];
            sx += a; sy += b; sxy += a * b; sxx += a * a; syy += b * b; ++cnt;
        }
        const double cov = sxy - sx * sy / cnt;
        const double va = sxx - sx * sx / cnt, vb = syy - sy * sy / cnt;
        const double c = cov / (std::sqrt (va * vb) + 1e-9);
        if (c > bestC) { bestC = c; bestLag = lag; }
    }
    std::printf ("best lag=%d  correlation=%.4f\n", bestLag, bestC);
    std::printf ("%s\n", bestC > 0.9 ? "PASS: C++ shifter matches Python reference"
                                     : "FAIL: divergence from reference");
    return bestC > 0.9 ? 0 : 1;
}
