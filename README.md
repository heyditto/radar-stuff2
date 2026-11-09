# Radar gap filling

This project downloads public NEXRAD volumes from KLGX (Langley Hill, Washington) and trains a small network to fill holes in a reflectivity sweep. The unblocked target, adjacent elevation tilts, scans from 30 minutes before through 30 minutes after, the atmospheric environment, time-aligned Pacific buoy observations, and GOES infrared imagery are model inputs.

The question: can spatial, vertical, temporal, marine, and satellite context reconstruct an artificially blocked part of a radar sweep?

## What you get

Figure: `outputs/gapfill_holdout_gaps.png`. It shows the held-out 15 November 2021 atmospheric-river scan, its vertical neighbors, artificial holes, and the fill.

Categories, applied only to the sweep being filled:

- **Observed** — the radar sample was left visible
- **Estimated** — the pixel was hidden and sits within 12 km of a visible sample
- **Uncertain** — the pixel was hidden and is farther than 12 km from any visible sample

The 12 km cutoff is a distance rule, not a second model. A neighbor tilt is not an observed sample. Cells the radar never sampled stay blank.

The archive bucket is `s3://unidata-nexrad-level2`. It replaced `noaa-nexrad-level2` and is public.

## Training days

Training days are all KLGX. The 15 November holdout center and its context files are not downloaded until training has finished.

| Day | Role | Volumes (UTC) |
| --- | --- | --- |
| 11 November 2021 | training | 12:05, first atmospheric-river pulse |
| 12 November 2021 | training | 12:07, continuing first pulse |
| 25 November 2021 | training | 17:56, late-November AR sequence |
| 27 November 2021 | training | 17:56, late-November AR sequence |
| 28 November 2021 | training | 18:00, late-November AR sequence |

Held-out volume: 12:02 UTC on 15 November 2021, `KLGX20211115_120233_V06`. No center or context scan from that UTC day is in training. The mask seed is `20211115`.

Training uses every distinct reflectivity tilt in those volumes. Split cuts and SAILS rescans within 0.15° are one tilt, and the cut with more unmasked gates represents it. The lowest tilt has no sweep below. The highest tilt has no sweep above. A missing side is flag 0 and reflectivity 0. A below-threshold gate is 0 dBZ with the flag set. Range-folded and uncollected gates stay invalid.

For each center, archive volumes are assigned uniquely to fixed slots at −30, −25, …, −5, +5, …, +30 minutes. A slot may be absent because KLGX volume cadence is not exactly five minutes; its reflectivity and validity channels are then zero. Each temporal volume is matched to the nearest target elevation within 0.18°. Grids are not motion-corrected.

Each step cuts new holes. Half the crops also punch a disk of 28 km radius. Hidden pixels at or above 20 dBZ count four times in the loss. Clear air stays in the loss at weight one. Where a vertical neighbor was sampled, the fill starts as the vertical-neighbor average and learns a correction using all 73 input channels.

GOES ABI Band 13 (10.3 µm infrared) supplies cloud pattern context through day and night. For KLGX, the run uses GOES-17 before its January 2023 operational handoff and GOES-18 afterward; KHGX uses GOES-16. It selects the latest completed CONUS image ending at or before the radar volume start, within 20 minutes. Native GOES pixels are projected onto the radar grid. Brightness temperature is scaled to a cold-cloud feature, and a separate flag marks valid, usable pixels. This is infrared brightness temperature, not a cloud-only mask: warm surfaces also appear in the image. Aligned images are cached under `data/satellite/`.

Those fields are not in Level II data. They come from the Quillayute (`UIL`) radiosonde, the nearest routine coastal balloon to KLGX. Each target uses the latest available 00:00 or 12:00 UTC launch at or before its start, falling back when a scheduled profile is absent. A height the balloon did not cover is flagged off and left at zero.

The surface marine inputs come from NDBC standard-meteorological archives. The configured stations are 46005 West Washington, 46002 West Oregon, and 46050 Stonewall Bank, the three Pacific buoys verified to report on every experiment date. The originally requested 46087 Neah Bay has no 2021 archive, while 46041 Cape Elizabeth stopped reporting in March 2021. Each radar volume uses only the latest buoy record at or before its start, no more than 20 minutes old.

For each buoy, wind direction and speed are converted to eastward/northward components. Dew point, sea-level pressure, sea-surface temperature, and air temperature are also inputs. Every value has a separate validity channel: 46002 lacks dew point and 46050 lacks air temperature and dew point in this period, so those values remain flagged off rather than imputed. PWV and rainfall are not included because they are absent from NDBC standard-met archives.

## Scores

The past-only motion experiment compares two visible, motion-aligned prior
scans with the earlier no-scan and raw two-scan runs on three held-out storm
days. Mean middle/lowest tilt MAE is 2.138/3.113 dBZ with alignment,
2.103/2.702 without past scans, and 2.800/3.712 with raw past scans.
Alignment helps the raw two-scan variant but does not beat the no-scan
baseline overall. See [the matched experiment](outputs/temporal_recency/README.md)
and [its scores](outputs/motion_alignment/summary.csv).

Matched 20-epoch runs with the same seed and 73-channel architecture, changing only whether the GOES channels contain imagery or zeros:

| Tilt | GOES Band 13 on | GOES Band 13 off | Neighbor average |
| --- | ---: | ---: | ---: |
| 0.48° middle | 2.36 | 2.80 | 2.40 |
| 0.18° lowest | 2.44 | 3.52 | 2.34 (above-tilt copy) |

Values are holdout MAE in dBZ on the same artificial holes. This is one held-out storm, so it establishes an improvement in this experiment rather than across weather regimes. The satellite-enabled run writes `outputs/gapfill_holdout_gaps.png`; the ablation writes `outputs/gapfill_holdout_no_satellite.png`. The aligned holdout image is shown in `outputs/holdout_goes_band13.png`.

Identical seeded runs, with and without buoy channels, **before satellite imagery was added**:

| Tilt | Inputs | U-Net | Neighbor | Temporal mean | Nearest |
| --- | --- | --- | --- | --- | --- |
| 0.48° middle | buoys on | 2.27 | 2.40 | 3.63 | 4.51 |
| 0.48° middle | buoys off | 2.21 | 2.40 | 3.63 | 4.51 |
| 0.18° lowest | buoys on | 3.74 | 2.34 | 3.63 | 4.69 |
| 0.18° lowest | buoys off | 2.16 | 2.34 | 3.63 | 4.69 |

With buoys, middle-tilt U-Net MAE was **3.14** on precipitation, **2.15** on estimated pixels, and **2.70** on uncertain pixels. Without buoys those were **3.11**, **2.12**, and **2.50**. Adding the NDBC channels did not improve holdout MAE. Scores in older KHGX figures are not comparable and are intentionally not carried forward.

## Run it

Use Python 3.10 or newer. A CPU is enough. A GPU is optional.

From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_holdout_gaps.py
```

Downloads are cached under `data/`. NDBC annual files are stored under `data/buoys/`. The run trains for 20 epochs and writes `outputs/gapfill_holdout_gaps.png`. The held-out radar, buoy, and satellite observations are selected only after training finishes.

For the seeded no-buoy ablation:

```bash
PYTHONPATH=src python -c "from pathlib import Path; from radar.holdout_gaps import run; run(Path('outputs/gapfill_holdout_no_buoy.png'), use_buoy=False)"
```

For a same-seed satellite ablation, run `run(Path('outputs/gapfill_holdout_no_satellite.png'), use_satellite=False)` through the same Python entry point. The older buoy comparison table predates satellite integration.

The download is unsigned, the same idea as `aws s3 cp --no-sign-request`. Existing AWS credentials are not used, because a signed request to this public bucket can be rejected.

## How the pieces fit

- `src/radar/io.py` reads a Level II volume and lists distinct reflectivity tilts.
- `src/radar/grid.py` puts one sweep on a 1 km grid extending about 160 km from the radar.
- `src/radar/masks.py` builds the holes. Training draws new holes every step. The evaluation mask uses its own seed and is never passed into training.
- `src/radar/environment.py` reads the Quillayute sounding and samples it at each tilt's beam height.
- `src/radar/buoy.py` downloads, parses, validates, and causally aligns NDBC observations.
- `src/radar/temporal.py` discovers and aligns the twelve fixed temporal slots.
- `src/radar/satellite.py` loads, aligns, projects, and caches GOES Band 13 imagery.
- `src/radar/model.py` is the U-Net. The original 11 spatial/vertical/environment channels are followed by 12 temporal reflectivity/validity pairs, 18 buoy value/validity pairs, and one satellite value/validity pair, for 73 channels total.
- `src/radar/train.py` trains that network. `train_gapfill_gaps` is the run.
- `src/radar/classify.py` applies the 12 km rule and copies the nearest observed pixel. That copy is one baseline.
- `src/radar/plot.py` writes the figure.
- `src/radar/holdout.py` lists the training days and loads the lowest sweep.
- `src/radar/holdout_above.py` loads the next tilt above the lowest sweep.
- `src/radar/holdout_between.py` loads every distinct tilt and pairs each one with the tilts beside it.
- `src/radar/holdout_gaps.py` trains, then scores the held-out storm.

## What this leaves out

- Real blockage from terrain, buildings, or turbines
- Dense storm-motion correction between temporal frames; the past-only follow-up uses one translation per crop
- The radar's own Doppler wind. The wind input is the balloon profile, not the radial velocity in the volume.
- Buoy PWV and rainfall, which are not available in the selected NDBC archive product
- A SageMaker training job or endpoint
- Real-time nowcasting: positive-offset scans make this an offline interpolation task
- A FastAPI or Next.js dashboard
