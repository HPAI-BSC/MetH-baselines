# MetH-Baselines

This repository contains the code for replicating the baselines of the [MetH-datasets paper](https://arxiv.org/abs/1911.08953).

This code runs with Python 3.7.2. Its python packages requirements will be included soon in a `requirements.txt` file.

## How to run the baselines
First of all, we have to download the dataset that we aim to use. Downloading scripts are located in the [download_data](https://github.com/HPAI-BSC/MetH-baselines/tree/master/download_data) module. If we plan to use the MetH-Medium dataset, for example, go to terminal and run the corresponding script like following:

```shell
python download_data/get_MetH-Medium.py
```

Baselines can be trained and tested using its corresponding scripts inside the [baselines](https://github.com/HPAI-BSC/MetH-baselines/tree/master/baselines) module. Following the same example, if we would like to run the MetH-Medium baseline, we have to first train the model and, then, test it:

```shell
python baselines/train_medium.py
```

After training the model, the code will automatically save it under the `model_checkpoints/medium_DATE.ckpt` path (DATE corresponds to the current datetime). After that we can test the model:

```shell
python baselines/test_medium.py model_checkpoints/medium_DATE.ckpt
```

Running bash scripts will be added in the near future in order to make it as easy as possible.
