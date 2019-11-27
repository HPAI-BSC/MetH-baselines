import sys
if sys.version_info < (3, 0):
    sys.stdout.write("Sorry, requires Python 3.x, not Python 2.x\n")
    sys.exit(1)

import os
PROJECT_PATH = os.path.abspath(os.path.join(__file__, *(os.path.pardir for _ in range(2))))
sys.path.append(PROJECT_PATH)

import wget
import tarfile
from consts.paths import Paths, DATASETS_PATH

csv_url = "https://storage.hpai.bsc.es/datasets/raw/MetH-datasets/MetH-Period.csv"
data_url = "https://storage.hpai.bsc.es/datasets/raw/MetH-datasets/MetH-Period_data.tar.gz"
dataset_paths = Paths.MetHPeriod

if not os.path.exists(DATASETS_PATH):
    os.makedirs(DATASETS_PATH)

os.chdir(DATASETS_PATH)

if not os.path.exists(dataset_paths.csv_path):
    dataset_dir = os.path.dirname(dataset_paths.csv_path)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    filename = wget.download(csv_url, out=dataset_dir)

if not os.path.exists(dataset_paths.images_path):
    filename = wget.download(data_url, out=DATASETS_PATH)
    tf = tarfile.open(os.path.join(DATASETS_PATH, filename))
    tf.extractall(path=DATASETS_PATH)
    os.remove(os.path.join(DATASETS_PATH, filename))
