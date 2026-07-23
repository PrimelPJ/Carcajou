### GNSS-outage drift, percent of distance travelled (median / p95)

GNSS grade `spp` before the outage. 4 sensor-noise seeds per configuration. Target: **< 1.0 %**.

**IMU: `consumer-mems`**

| aiding during outage | 10 s | 30 s | 60 s | 120 s |
|---|---|---|---|---|
| `ins-only` | 0.52 / 1.13 | 1.80 / 4.97 | 5.12 / 15.70 | 17.39 / 51.56 |
| `ins+zupt` | 0.52 / 1.13 | 1.10 / 4.94 | 3.78 / 15.70 | 15.73 / 51.56 |
| `ins+nhc` | 0.45 / 1.17 | 0.39 / 2.64 | 1.22 / 3.65 | 1.34 / 20.14 |
| `ins+zupt+nhc` | 0.45 / 1.17 | 0.39 / 2.64 | 1.21 / 3.65 | 1.28 / 20.14 |

**IMU: `industrial-mems`**

| aiding during outage | 10 s | 30 s | 60 s | 120 s |
|---|---|---|---|---|
| `ins-only` | 0.36 / 0.73 | 0.86 / 1.58 | 2.03 / 3.92 | 5.36 / 10.24 |
| `ins+zupt` | 0.36 / 0.73 | 0.67 / 1.45 | 1.45 / 3.92 | 3.92 / 10.24 |
| `ins+nhc` | 0.35 / 0.73 | 0.49 / 1.03 | 0.48 / 1.21 | 0.40 / 1.15 |
| `ins+zupt+nhc` | 0.35 / 0.73 | 0.21 / 1.03 | 0.22 / 0.64 | 0.22 / 1.15 |

**IMU: `tactical`**

| aiding during outage | 10 s | 30 s | 60 s | 120 s |
|---|---|---|---|---|
| `ins-only` | 0.15 / 0.35 | 0.18 / 0.34 | 0.26 / 0.49 | 0.48 / 0.90 |
| `ins+zupt` | 0.15 / 0.35 | 0.15 / 0.30 | 0.08 / 0.45 | 0.08 / 0.28 |
| `ins+nhc` | 0.15 / 0.35 | 0.14 / 0.28 | 0.11 / 0.22 | 0.08 / 0.18 |
| `ins+zupt+nhc` | 0.15 / 0.35 | 0.12 / 0.22 | 0.05 / 0.13 | 0.04 / 0.10 |
