from __future__ import annotations

import unittest

import numpy as np
from scipy.ndimage import shift

from radar.motion import align_past_scans, estimate_translation


class MotionTests(unittest.TestCase):
    def test_recovers_past_pair_motion(self) -> None:
        rng = np.random.default_rng(4)
        older = np.zeros((64, 64), dtype=np.float32)
        older[14:45, 15:48] = rng.uniform(12, 45, (31, 33))
        recent = shift(older, (3, -2), order=0, mode="constant", cval=0)
        valid = np.ones_like(older, dtype=bool)
        self.assertEqual(estimate_translation(older, valid, recent, valid), (3, -2))

    def test_hidden_source_cannot_reappear_after_warp(self) -> None:
        rng = np.random.default_rng(8)
        older = np.zeros((64, 64), dtype=np.float32)
        older[12:49, 12:49] = rng.uniform(12, 45, (37, 37))
        recent = shift(older, (2, 1), order=0, mode="constant", cval=0)
        dbz = np.zeros((12, 64, 64), dtype=np.float32)
        dbz[0], dbz[1] = recent, older
        valid = np.zeros_like(dbz, dtype=bool)
        valid[:2] = True
        hidden = np.zeros((64, 64), dtype=bool)
        hidden[25:40, 25:40] = True
        aligned, flags, movement = align_past_scans(dbz, valid, hidden, np.array([5, 10]))
        self.assertEqual(movement, (2, 1))
        for slot, displacement in enumerate(((2, 1), (4, 2))):
            source_hidden_shifted = shift(hidden.astype(np.uint8), displacement, order=0, mode="constant", cval=0) > 0
            self.assertFalse(flags[slot][source_hidden_shifted].any())
            self.assertTrue(np.all(aligned[slot][~flags[slot]] == 0))
        self.assertTrue(flags[0][hidden].any())  # visible pixels can advect into the gap
        self.assertFalse(flags[2:].any())
        altered = dbz.copy()
        altered[:2, hidden] = 60.0
        altered_aligned, altered_flags, altered_movement = align_past_scans(
            altered, valid, hidden, np.array([5, 10])
        )
        self.assertEqual(altered_movement, movement)
        np.testing.assert_array_equal(altered_flags, flags)
        np.testing.assert_array_equal(altered_aligned, aligned)


if __name__ == "__main__":
    unittest.main()
