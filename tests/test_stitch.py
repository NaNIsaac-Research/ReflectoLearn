import numpy as np

from reflectolearn.stitch import stitch_scans


def test_stitch_demo():
    # Demo from xrr_combine: three attenuator segments, last one is the reference.
    s1 = np.array([[0.1, 400, 40], [0.2, 300, 30], [0.3, 200, 20], [0.4, 100, 10]])
    s2 = np.array([[0.5, 320, 32], [0.6, 240, 24], [0.7, 160, 16], [0.8, 80, 8]])
    s3 = np.array([[0.9, 280, 28], [1.0, 210, 21], [1.1, 140, 14], [1.2, 70, 7]])

    out = stitch_scans([s1, s2, s3])

    np.testing.assert_allclose(out[:, 0], [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.9, 1.0, 1.1, 1.2])
    np.testing.assert_allclose(out[:, 1], [4480, 3360, 2240, 1120, 840, 560, 280, 210, 140, 70])
    np.testing.assert_allclose(out[:, 2], out[:, 1] / 10)
    np.testing.assert_array_equal(s1[:, 1], [400, 300, 200, 100])  # inputs untouched
