**IMU: `consumer-mems`** with scan-to-map localization (6 windows per cell)

| aiding during outage | 10 s | 30 s | 60 s | 120 s |
|---|---|---|---|---|
| `ins+zupt+nhc` | 0.32 / 0.92 | 0.62 / 2.49 | 0.79 / 2.96 | 1.21 / 1.37 |
| **`ins+zupt+nhc+map`** | 0.12 / 0.16 | 0.05 / 0.08 | 0.02 / 0.03 | 0.01 / 0.02 |
| `ins+zupt+nhc+map(phantom)` | 0.11 / 0.15 | 0.04 / 0.08 | 0.02 / 0.04 | 0.01 / 0.02 |
