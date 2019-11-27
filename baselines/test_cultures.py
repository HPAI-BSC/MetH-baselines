import sys
import os

if sys.version_info < (3, 0):
    print("Sorry, requires Python 3.x, not Python 2.x\n")
    sys.exit(1)

PROJECT_PATH = os.path.abspath(os.path.join(__file__, *(os.path.pardir for _ in range(2))))
sys.path.append(PROJECT_PATH)

import argparse
from baselines.src.testing import classification_test
from consts.paths import Paths

parser = argparse.ArgumentParser()
parser.add_argument("model_path", help="Path to the model chekpoint.", type=str)
args = parser.parse_args()

dataset_paths = Paths.MetHCultures
classification_test(dataset_paths.csv_path, dataset_paths.images_path, args.model_path)
