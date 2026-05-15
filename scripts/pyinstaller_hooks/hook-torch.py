"""Local PyInstaller hook for ``torch``: avoid optional ``tensorboard`` during analysis.

PyInstaller otherwise tries ``import torch.utils.tensorboard``, which requires the
``tensorboard`` package and emits WARNING when it is not installed. DanQing
inference does not need tensorboard.
"""

excludedimports = ["tensorboard"]
