"""Vector helpers with no dependency on historical hair datasets."""
import numpy as np


def unit(v):
    return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-12)
