"""
Garry – Stockfish-style carpet/rat agent.
"""
import os as _os
import sys as _sys

_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

_HERE   = _os.path.dirname(_os.path.abspath(__file__))
_ENGINE = _os.path.normpath(_os.path.join(_HERE, "..", "..", "engine"))
for _p in (_HERE, _ENGINE):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

from .agent import PlayerAgent
