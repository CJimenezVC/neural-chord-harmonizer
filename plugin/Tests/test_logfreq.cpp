// Standalone check: C++ LogFreqFeature must match the Python training feature
// (training/chord_synth.py:frame_feature) for the same frame + filterbank.
#include <cmath>
#include <cstdio>
#include <vector>

#include "DSP/LogFreqFeature.h"

static std::vector<float> readF32 (const char* p)
{
    FILE* f = std::fopen (p, "rb");
    if (! f) return {};
    std::fseek (f, 0, SEEK_END); long n = std::ftell (f) / 4; std::fseek (f, 0, SEEK_SET);
    std::vector<float> v ((size_t) n);
    [[maybe_unused]] auto r = std::fread (v.data(), 4, (size_t) n, f); std::fclose (f);
    return v;
}

int main()
{
    auto frame = readF32 ("/tmp/lf_frame.f32");   // 2048
    auto fb    = readF32 ("/tmp/lf_fb.f32");       // 61*1025
    auto ref   = readF32 ("/tmp/lf_ref.f32");      // 61
    if (frame.empty() || fb.empty() || ref.empty()) { std::printf ("missing input\n"); return 2; }

    const int nPitch = (int) ref.size();
    const int nBins  = (int) fb.size() / nPitch;

    LogFreqFeature feat;
    feat.prepare ((int) frame.size());
    feat.setFilterbank (fb.data(), nPitch, nBins);

    std::vector<float> out ((size_t) nPitch, 0.0f);
    feat.compute (frame.data(), (int) frame.size(), out.data());

    double maxd = 0.0;
    for (int i = 0; i < nPitch; ++i) maxd = std::max (maxd, (double) std::fabs (out[(size_t) i] - ref[(size_t) i]));
    std::printf ("max|Δ| vs python = %.2e\n", maxd);
    std::printf ("%s\n", maxd < 1e-3 ? "PASS: feature matches python" : "FAIL");
    return maxd < 1e-3 ? 0 : 1;
}
