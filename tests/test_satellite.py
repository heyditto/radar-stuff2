from __future__ import annotations

import unittest
from datetime import datetime, timezone

import numpy as np

from radar.grid import ReflectivityGrid
from radar.model import SATELLITE_CHANNEL, SATELLITE_FLAG_CHANNEL
from radar.satellite import project_image, select_image
from radar.train import build_gap_input
from radar.model import BUOY_FIELD_COUNT, BUOY_STATION_COUNT, TEMPORAL_SLOT_COUNT


class SatelliteTests(unittest.TestCase):
    def test_select_completed_fresh_scene(self):
        when = datetime(2021, 11, 15, 12, 2, tzinfo=timezone.utc)
        keys = [
            "OR_ABI-L2-CMIPC-M6C13_G17_s20213191200000_e20213191201000_c.nc",
            "OR_ABI-L2-CMIPC-M6C13_G17_s20213191150000_e20213191159000_c.nc",
            "OR_ABI-L2-CMIPC-M6C13_G17_s20213191205000_e20213191206000_c.nc",
        ]
        selected = select_image(keys, when)
        self.assertEqual(selected[0], keys[0])
        self.assertEqual(selected[1].minute, 1)

    def test_project_native_pixel_and_quality(self):
        grid = ReflectivityGrid(
            np.zeros((2, 2), np.float32), np.ones((2, 2), bool), 1.0,
            np.array([-0.5, 0.5], np.float32), np.array([-0.5, 0.5], np.float32),
            "KLGX", 0.0, -137.0, 0.5, "2021-11-15T12:02:00Z",
        )
        projection = dict(grid_mapping_name="geostationary", perspective_point_height=35786023.0,
                          longitude_of_projection_origin=-137.0, latitude_of_projection_origin=0.0,
                          sweep_angle_axis="x", semi_major_axis=6378137.0, semi_minor_axis=6356752.31414)
        axis = np.linspace(-0.001, 0.001, 9)
        image = np.full((9, 9), 250.0, np.float32)
        quality = np.zeros((9, 9), np.uint8)
        brightness, valid = project_image(image, quality, axis, axis, projection, grid)
        self.assertTrue(valid.all())
        self.assertTrue(np.all(brightness == 250.0))
        quality[:] = 3
        _, valid = project_image(image, quality, axis, axis, projection, grid)
        self.assertFalse(valid.any())

    def test_channels_keep_missing_pixels_zero(self):
        shape = (8, 8)
        z = np.zeros(shape, np.float32)
        flags = np.zeros(shape, bool)
        temporal = np.zeros((TEMPORAL_SLOT_COUNT, *shape), np.float32)
        temporal_flags = np.zeros_like(temporal, bool)
        buoy = np.zeros((BUOY_STATION_COUNT, BUOY_FIELD_COUNT), np.float32)
        satellite = np.full(shape, 230.0, np.float32)
        satellite[0, 0] = np.nan
        stacked = build_gap_input(z, flags, z, flags, z, flags, temporal, temporal_flags,
                                  z, z, z, z, flags, buoy, np.zeros_like(buoy, bool),
                                  satellite, np.ones(shape, bool))
        self.assertEqual(stacked[SATELLITE_CHANNEL][0, 0], 0.0)
        self.assertEqual(stacked[SATELLITE_FLAG_CHANNEL][0, 0], 0.0)
        self.assertAlmostEqual(float(stacked[SATELLITE_CHANNEL][1, 1]), 2 / 3)


if __name__ == "__main__":
    unittest.main()
