import os
from tensorboardX import SummaryWriter as Logger
from baselines.utils.consts import Split


class SummaryLogger:
    def __init__(self, logdir):
        self.logger = Logger(logdir)

    def add_scalar(self, name, scalar, global_step):
        self.logger.add_scalar(name, scalar, global_step)


class SummaryWriter:
    ACC = 'Accuracy'
    LOSS = 'Loss'

    def __init__(self, summaries_path):
        train_logdir = os.path.join(summaries_path, 'train')
        val_logdir = os.path.join(summaries_path, 'val')
        self.loggers = {
            Split.TRAIN: SummaryLogger(train_logdir),
            Split.VAL: SummaryLogger(val_logdir)
        }

    def __getitem__(self, subset):
        try:
            return self.loggers[subset]
        except KeyError:
            return None