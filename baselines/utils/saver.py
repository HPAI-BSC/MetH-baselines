import os
import torch


class Saver:

    def __init__(self, model_path):
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        self.model_path = model_path

    def save_checkpoint(self, model, optimizer, epoch, step):
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step
        }
        torch.save(state, self.model_path)
        print("=> saving checkpoint '{}'".format(self.model_path))

    def get_mapping_key(self, state_dict):
        mapping_dict = {}
        for key in state_dict:
            mapping_dict[key] = ".".join(key.split('.')[1:])
        return mapping_dict

    def map_keys(self, state_dict):
        """Remove 'module' from keys when model has been stored using DataParallel"""
        if 'module' not in state_dict.key()[0]:
            return state_dict
        mapping_dict = self.get_mapping_key(state_dict)
        all_keys = list(state_dict.keys())
        for key in all_keys:
            try:
                new_key = mapping_dict[key]
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
            except KeyError:
                continue
        return state_dict

    def load_checkpoint(self, model, optimizer=None):
        # Check working device
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")

        # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
        epoch = 0
        step = 0
        if os.path.isfile(self.model_path):
            print("=> loading checkpoint '{}'".format(self.model_path))
            checkpoint = torch.load(self.model_path, map_location=device)
            model.load_state_dict(self.map_keys(checkpoint['state_dict']))
            if optimizer:
                optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']
            step = checkpoint['step']
            print("=> loaded checkpoint '{}' (epoch {})".format(self.model_path, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(self.model_path))

        return model, optimizer, epoch, step
