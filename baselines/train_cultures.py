import sys
import os

if sys.version_info < (3, 0):
    print("Sorry, requires Python 3.x, not Python 2.x\n")
    sys.exit(1)

PROJECT_PATH = os.path.abspath(os.path.join(__file__, *(os.path.pardir for _ in range(2))))
sys.path.append(PROJECT_PATH)

import argparse
import datetime
from baselines.src.training import classification_train
from consts.paths import Paths

parser = argparse.ArgumentParser()
parser.add_argument("--retrain", help="Retrain from already existing checkpoint.", type=str, default='')
args = parser.parse_args()

dataset_paths = Paths.MetHCultures

current_date = '{date:%Y-%m-%d_%H-%M-%S}'.format(date=datetime.datetime.now())
model_path = os.path.join(PROJECT_PATH, 'model_checkpoints', 'cultures_{}.ckpt'.format(current_date))
summaries_path = os.path.join(PROJECT_PATH, 'summaries', 'cultures_{}'.format(current_date))

if args.retrain:
    classification_train(dataset_paths.csv_path, dataset_paths.images_path, args.retrain, summaries_path, retrain=True)
else:
    classification_train(dataset_paths.csv_path, dataset_paths.images_path, model_path, summaries_path)
