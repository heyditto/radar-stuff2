from __future__ import annotations

import unittest

import numpy as np

from radar.analog import Analog, retrieve_analogs


class AnalogRetrievalTests(unittest.TestCase):
    def test_query_hidden_truth_cannot_change_retrieval(self) -> None:
        rng = np.random.default_rng(3)
        field = np.zeros((64, 64), dtype=np.float32)
        field[10:55, 12:52] = rng.uniform(15, 40, (45, 40))
        valid = np.ones_like(field, dtype=bool)
        hidden = np.zeros_like(valid)
        hidden[25:40, 25:40] = True
        library = [
            Analog("a", "2021-11-11", field.copy(), valid, field.copy(), valid, 7.0),
            Analog("b", "2021-11-12", np.roll(field, 4, axis=0), valid, np.roll(field, 4, axis=0), valid, 7.0),
        ]
        first, first_flag, first_matches = retrieve_analogs(
            field, valid, field, valid, 7.0, hidden, library, temporal=True, k=1
        )
        changed = field.copy()
        changed[hidden] = 60.0
        second, second_flag, second_matches = retrieve_analogs(
            changed, valid, changed, valid, 7.0, hidden, library, temporal=True, k=1
        )
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first_flag, second_flag)
        self.assertEqual(first_matches, second_matches)
        self.assertEqual(first_matches[0]["scan"], "a")


if __name__ == "__main__":
    unittest.main()
