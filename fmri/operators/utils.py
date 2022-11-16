"""Utilities for fMRI Operators."""


def validate_smaps(shape, n_coils, smaps=None):
    """Raise Value Error if smaps does not fit dimensions."""
    if smaps is not None and n_coils != len(smaps) and smaps.shape[:-1] == tuple(shape):
        raise ValueError("smaps should  have dimension n_coils x shape")
    else:
        return True
