import numpy as np

from run_learned_analogs import SHAPE, blockage, descriptor, input_arrays


def test_hidden_radar_values_cannot_change_retrieval_descriptor():
    hidden = blockage()
    values = np.full(SHAPE, 20.0, np.float32)
    valid = np.ones(SHAPE, bool)
    changed = values.copy()
    changed[hidden] = 70.0
    a, af = descriptor(values, valid, hidden)
    b, bf = descriptor(changed, valid, hidden)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(af, bf)


def test_blockage_is_fixed_azimuth_sector():
    hidden = blockage()
    assert hidden.any() and (~hidden).any()
    assert not hidden[160, 160]
    assert hidden[80, 160]


def test_current_and_recent_past_share_the_obstruction():
    class Cache:
        def get(self, name):
            dbz = np.full((2, *SHAPE), 35.0 if name == 'center' else 50.0, np.float32)
            return dbz, np.ones((2, *SHAPE), bool)

    x, target, scored = input_arrays({'center': 'center', 'past': 'past'}, 0, Cache(), False)
    hidden = blockage()
    assert np.all(x[0, hidden] == 0)
    assert np.all(x[3, hidden] == 0)
    assert np.all(x[1, hidden] == 0)
    assert np.all(x[4, hidden] == 0)
    assert np.array_equal(scored, hidden)
    assert np.all(target[hidden] > 0)
