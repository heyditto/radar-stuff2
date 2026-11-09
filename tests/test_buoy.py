from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from radar.buoy import (
    BUOY_FIELDS,
    BUOY_STATIONS,
    BuoyObservation,
    context_from_observations,
    parse_standard_met,
    select_observation,
    wind_components,
)


SAMPLE = """#YY MM DD hh mm WDIR WSPD GST WVHT DPD APD MWD PRES ATMP WTMP DEWP VIS TIDE
#yr mo dy hr mn degT m/s m/s m sec sec degT hPa degC degC degC mi ft
2021 11 15 11 50 180 10.0 12.0 99.00 99.00 99.00 999 1004.8 9.4 11.3 8.3 99.0 99.00
2021 11 15 12 00 999 99.0 99.0 99.00 99.00 99.00 999 1005.0 999.0 11.4 999.0 99.0 99.00
"""


class BuoyParsingTests(unittest.TestCase):
    def test_parser_converts_wind_and_missing_sentinels(self) -> None:
        observations = parse_standard_met(SAMPLE, "46005")
        self.assertEqual(len(observations), 2)
        first = observations[0]
        self.assertAlmostEqual(float(first.values[0]), 0.0, places=5)
        self.assertAlmostEqual(float(first.values[1]), 10.0, places=5)
        self.assertTrue(first.valid.all())
        second = observations[1]
        self.assertFalse(second.valid[0])
        self.assertFalse(second.valid[1])
        self.assertFalse(second.valid[2])
        self.assertFalse(second.valid[5])
        self.assertTrue(second.valid[3])
        self.assertTrue(second.valid[4])

    def test_wind_components_use_meteorological_direction(self) -> None:
        east, north = wind_components(270.0, 8.0)
        self.assertAlmostEqual(east, 8.0, places=5)
        self.assertAlmostEqual(north, 0.0, places=5)


class BuoyAlignmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target = datetime(2021, 11, 15, 12, 5, tzinfo=timezone.utc)
        self.observations = parse_standard_met(SAMPLE, "46005")

    def test_selection_is_causal_and_fresh(self) -> None:
        chosen = select_observation(self.observations, self.target)
        self.assertIsNotNone(chosen)
        self.assertEqual(chosen.timestamp.minute, 0)
        future_only = [self.observations[-1]]
        self.assertIsNone(
            select_observation(future_only, self.target - timedelta(minutes=10))
        )
        self.assertIsNone(
            select_observation(self.observations, self.target + timedelta(minutes=30))
        )

    def test_context_preserves_station_order_and_field_flags(self) -> None:
        observations = []
        for index, station in enumerate(BUOY_STATIONS):
            values = np.arange(len(BUOY_FIELDS), dtype=np.float32) + index
            valid = np.ones(len(BUOY_FIELDS), dtype=bool)
            if index == 1:
                valid[2] = False
                values[2] = np.nan
            observations.append(
                BuoyObservation(
                    station.station,
                    self.target - timedelta(minutes=index + 1),
                    values,
                    valid,
                )
            )
        context = context_from_observations(observations, self.target)
        self.assertEqual(context.values.shape, (3, 6))
        self.assertEqual(context.age_minutes, (1.0, 2.0, 3.0))
        self.assertFalse(context.valid[1, 2])
        self.assertEqual(context.values[1, 2], 0.0)
        self.assertTrue(context.observed_at[0].endswith("Z"))


if __name__ == "__main__":
    unittest.main()
