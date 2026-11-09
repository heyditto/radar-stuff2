from __future__ import annotations

import unittest
from datetime import timedelta
from unittest.mock import patch

import numpy as np

from radar.grid import ReflectivityGrid
from radar.holdout import HOLDOUT_SCAN, Scan
from radar.io import volume_name_site, volume_name_time
from radar.model import (
    BUOY_FIELD_COUNT,
    BUOY_STATION_COUNT,
    INPUT_CHANNELS,
    RECENCY_AGE_START_CHANNEL,
    RECENCY_INPUT_CHANNELS,
    TEMPORAL_SLOT_COUNT,
    buoy_flag_channel,
    buoy_value_channel,
    temporal_dbz_channel,
    temporal_flag_channel,
)
from radar.temporal import (
    TEMPORAL_OFFSETS_MIN,
    assert_context_split,
    assign_temporal_scans,
    assign_recent_past_scans,
    load_temporal_contexts,
)
from radar.train import build_gap_input


def _name(timestamp) -> str:
    return f"KLGX{timestamp:%Y%m%d_%H%M%S}_V06"


class TemporalAssignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.center = Scan("2021-11-11", "stormy", "KLGX20211111_120000_V06")

    def test_archive_name_parser(self) -> None:
        self.assertEqual(volume_name_site(self.center.name), "KLGX")
        self.assertEqual(volume_name_time(self.center.name), self.center.timestamp)
        with self.assertRaises(ValueError):
            volume_name_time("KLGX20211111_120000_V06_MDM")

    def test_fixed_slots_are_unique_and_ordered(self) -> None:
        names = [_name(self.center.timestamp + timedelta(minutes=offset)) for offset in TEMPORAL_OFFSETS_MIN]
        names.append(self.center.name)
        assigned = assign_temporal_scans(self.center, names)
        self.assertEqual([scan.name for scan in assigned if scan], names[:-1])
        self.assertEqual(len({scan.name for scan in assigned if scan}), len(TEMPORAL_OFFSETS_MIN))

    def test_missing_slot_stays_none(self) -> None:
        names = [_name(self.center.timestamp - timedelta(minutes=5))]
        assigned = assign_temporal_scans(self.center, names)
        self.assertEqual(sum(scan is not None for scan in assigned), 1)
        self.assertIsNotNone(assigned[TEMPORAL_OFFSETS_MIN.index(-5)])

    def test_recent_past_order_is_stable_across_varying_cadence(self) -> None:
        names = [
            _name(self.center.timestamp - timedelta(minutes=7)),
            _name(self.center.timestamp - timedelta(minutes=15)),
            _name(self.center.timestamp + timedelta(minutes=7)),
            self.center.name,
        ]
        selected = assign_recent_past_scans(self.center, names)
        self.assertEqual([scan.name for scan in selected[:2]], names[:2])
        self.assertTrue(all(scan is None for scan in selected[2:]))

    def test_holdout_day_context_is_rejected(self) -> None:
        leaked = Scan(HOLDOUT_SCAN.day, "stormy", "KLGX20211115_115733_V06")
        with self.assertRaises(RuntimeError):
            assert_context_split({self.center.name: (leaked,)}, HOLDOUT_SCAN)

    def test_elevation_matching_masks_unmatched_tilt(self) -> None:
        def grid(elevation: float, value: float) -> ReflectivityGrid:
            dbz = np.full((8, 8), value, dtype=np.float32)
            return ReflectivityGrid(
                dbz=dbz,
                valid=np.ones((8, 8), dtype=bool),
                resolution_km=1.0,
                x_km=np.arange(8, dtype=np.float32),
                y_km=np.arange(8, dtype=np.float32),
                radar_name="KLGX",
                radar_lat=47.1,
                radar_lon=-124.1,
                sweep_elevation=elevation,
                timestamp="2021-11-11T12:00:00Z",
            )

        context_scan = Scan("2021-11-11", "stormy", "KLGX20211111_115500_V06")
        scans = (context_scan,) + (None,) * (len(TEMPORAL_OFFSETS_MIN) - 1)
        with patch("radar.temporal.load_tilts", return_value=[grid(0.5, 12.0)]):
            matched, unmatched = load_temporal_contexts(
                self.center, [grid(0.5, 0.0), grid(0.2, 0.0)], scans
            )
        self.assertTrue(matched.valid[0].all())
        self.assertEqual(matched.names[0], context_scan.name)
        self.assertFalse(unmatched.valid[0].any())
        self.assertIsNone(unmatched.names[0])


class TemporalChannelTests(unittest.TestCase):
    def test_persistent_blockage_hides_past_pixels_but_keeps_ages(self) -> None:
        shape = (8, 8)
        zeros = np.zeros(shape, dtype=np.float32)
        false = np.zeros(shape, dtype=bool)
        true = np.ones(shape, dtype=bool)
        hidden = false.copy()
        hidden[2:6, 2:6] = True
        temporal = np.ones((TEMPORAL_SLOT_COUNT, *shape), dtype=np.float32)
        temporal_valid = np.zeros_like(temporal, dtype=bool)
        temporal_valid[:2] = True
        buoy = np.zeros((BUOY_STATION_COUNT, BUOY_FIELD_COUNT), dtype=np.float32)
        buoy_valid = np.zeros_like(buoy, dtype=bool)
        stacked = build_gap_input(
            zeros, hidden, zeros, true, zeros, true, temporal, temporal_valid,
            zeros, zeros, zeros, zeros, false, buoy, buoy_valid,
            temporal_age_minutes=np.array([7.0, 14.0], dtype=np.float32),
            mask_temporal_inside_hole=True,
        )
        self.assertEqual(stacked.shape, (RECENCY_INPUT_CHANNELS, *shape))
        for slot in (0, 1):
            self.assertTrue(np.all(stacked[temporal_flag_channel(slot)][hidden] == 0))
            self.assertTrue(np.all(stacked[temporal_dbz_channel(slot)][hidden] == 0))
            self.assertTrue(np.all(stacked[temporal_flag_channel(slot)][~hidden] == 1))
        self.assertTrue(np.allclose(stacked[RECENCY_AGE_START_CHANNEL], 7 / 30))
        self.assertTrue(np.allclose(stacked[RECENCY_AGE_START_CHANNEL + 1], 14 / 30))

    def test_channel_layout_and_missing_flags(self) -> None:
        shape = (8, 8)
        zeros = np.zeros(shape, dtype=np.float32)
        false = np.zeros(shape, dtype=bool)
        true = np.ones(shape, dtype=bool)
        temporal = np.zeros((TEMPORAL_SLOT_COUNT, *shape), dtype=np.float32)
        temporal_valid = np.zeros_like(temporal, dtype=bool)
        temporal[3] = 0.75
        temporal_valid[3] = True
        buoy = np.zeros((BUOY_STATION_COUNT, BUOY_FIELD_COUNT), dtype=np.float32)
        buoy_valid = np.zeros_like(buoy, dtype=bool)
        buoy[0, 3] = 965.0
        buoy_valid[0, 3] = True
        stacked = build_gap_input(
            zeros,
            false,
            zeros,
            true,
            zeros,
            true,
            temporal,
            temporal_valid,
            zeros,
            zeros,
            zeros,
            zeros,
            false,
            buoy,
            buoy_valid,
        )
        self.assertEqual(stacked.shape, (INPUT_CHANNELS, *shape))
        self.assertEqual(INPUT_CHANNELS, 73)
        self.assertTrue(np.all(stacked[temporal_dbz_channel(3)] == 0.75))
        self.assertTrue(np.all(stacked[temporal_flag_channel(3)] == 1.0))
        self.assertTrue(np.all(stacked[temporal_dbz_channel(2)] == 0.0))
        self.assertTrue(np.all(stacked[temporal_flag_channel(2)] == 0.0))
        self.assertTrue(np.allclose(stacked[buoy_value_channel(0, 3)], 0.5))
        self.assertTrue(np.all(stacked[buoy_flag_channel(0, 3)] == 1.0))
        self.assertTrue(np.all(stacked[buoy_flag_channel(1, 3)] == 0.0))


if __name__ == "__main__":
    unittest.main()
