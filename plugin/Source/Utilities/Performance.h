#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

/**
    Minimal profiling helpers for measuring per-stage processing time.

    Use ScopedTimer to accumulate elapsed microseconds into a PerfCounter, then
    dump aggregates to benchmarks/latency_measurements.csv.
*/
class PerfCounter
{
public:
    void add (std::uint64_t micros) noexcept
    {
        total.fetch_add (micros, std::memory_order_relaxed);
        count.fetch_add (1, std::memory_order_relaxed);
        std::uint64_t prev = peak.load (std::memory_order_relaxed);
        while (micros > prev && ! peak.compare_exchange_weak (prev, micros)) {}
    }

    double averageMicros() const noexcept
    {
        const auto c = count.load();
        return c ? (double) total.load() / (double) c : 0.0;
    }

    std::uint64_t peakMicros() const noexcept { return peak.load(); }
    void reset() noexcept { total = 0; count = 0; peak = 0; }

private:
    std::atomic<std::uint64_t> total { 0 };
    std::atomic<std::uint64_t> count { 0 };
    std::atomic<std::uint64_t> peak  { 0 };
};

/** RAII timer: accumulates its lifetime into a PerfCounter on destruction. */
class ScopedTimer
{
public:
    explicit ScopedTimer (PerfCounter& c) noexcept
        : counter (c), start (std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer()
    {
        using namespace std::chrono;
        const auto us = duration_cast<microseconds> (high_resolution_clock::now() - start).count();
        counter.add ((std::uint64_t) us);
    }

private:
    PerfCounter& counter;
    std::chrono::high_resolution_clock::time_point start;
};
