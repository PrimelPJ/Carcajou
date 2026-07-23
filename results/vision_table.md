**IMU: `consumer-mems`** with stereo VO aiding (24 windows per cell)

| aiding during outage | 10 s | 30 s | 60 s | 120 s |
|---|---|---|---|---|
| `ins+zupt+nhc` | 0.45 / 1.17 | 0.39 / 2.64 | 1.21 / 3.65 | 1.28 / 20.14 |
| `ins+zupt+nhc+vo(mask off)` | 0.45 / 1.17 | 0.39 / 2.64 | 1.32 / 3.65 | 1.52 / 19.36 |
| **`ins+zupt+nhc+vo(mask on)`** | 0.49 / 0.68 | 0.37 / 0.65 | 0.37 / 0.74 | 0.42 / 0.92 |
