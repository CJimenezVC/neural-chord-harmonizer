# CPU Usage Analysis

Measured with `Utilities/Performance.h` (`ScopedTimer` per stage) at 48 kHz,
512-sample blocks. Populate after the first end-to-end build.

## Target

< 25 % of one core on a modern 4-core processor (2021+).

## Methodology

1. Build the plugin in Release.
2. Run in the Standalone host with a looped speech file.
3. Sample per-stage timings into `latency_measurements.csv`.
4. Sample total process CPU via OS tools (`Activity Monitor` / `perf` / Task Manager).

## Results

| Configuration         | CPU % | Notes        |
| --------------------- | ----- | ------------ |
| FP32, single-thread   | _TBD_ |              |
| FP32, vocoder thread  | _TBD_ | safety buffer |
| INT8                  | _TBD_ | optional     |

## Observations

_Fill in after profiling._
