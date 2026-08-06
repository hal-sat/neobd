import numpy as np

from neobd.selection import ModalRMSSelector


def test_modal_rms_selector_requires_every_channel() -> None:
    rms = [np.array([[0.95, 1.02, 1.8], [1.01, 0.98, 1.0]])]
    selected = ModalRMSSelector(0.15).select(rms)
    np.testing.assert_array_equal(selected, [0, 1])
