import sys
import os

# Ensure project root is on sys.path so `app` and `mao` packages are importable
_root = os.path.abspath(os.path.dirname(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)
