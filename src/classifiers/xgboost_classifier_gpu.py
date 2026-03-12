"""
GPU entrypoint for XGBoost LOS/NLOS classifier.

This wrapper reuses xgboost_classifier.py and forces USE_GPU=1.
"""

import os
import runpy
from pathlib import Path


os.environ["USE_GPU"] = "1"

script_path = Path(__file__).with_name("xgboost_classifier.py")
runpy.run_path(str(script_path), run_name="__main__")
