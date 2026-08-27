"""MAO Clinical AI — HuggingFace Spaces entry point.

HF Spaces reads ``app_file: app.py`` from the README front-matter, so this file
must stay at the repo root. It is a shim: all HF-specific deployment glue lives
in deploy/huggingface/space_entrypoint.py (P1-16).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deploy.huggingface.space_entrypoint import main  # noqa: E402

main()
