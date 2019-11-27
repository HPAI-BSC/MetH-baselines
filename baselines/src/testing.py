import time
import torch
import torchvision.models as models
from torchvision.models import vgg16
from baselines.utils.consts import Split
from baselines.src.datasets import ClassificationDataset, RegressionDataset
from baselines.src.input import InputPipeline
from baselines.utils.saver import Saver


def _classification_testing(input_pipeline, model, loss_function, saver):
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if torch.cuda.device_count() > 1:
        print("Using {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model.to(device)

    model = saver.load_checkpoint(model)[0]

    # testing epoch
    correct = 0
    total = 0
    losses = 0
    t0 = time.time()
    for idx, (images_batch, labels_batch) in enumerate(input_pipeline[Split.TEST]):
        # Loading tensors in the used device
        images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

        outputs = model(images_batch)
        loss = loss_function(outputs, labels_batch)

        losses += loss.item()
        local_preds = torch.max(outputs.data, 1)[1]
        total += labels_batch.size(0)
        correct += (local_preds == labels_batch).sum().item()

    test_acc = correct / total
    test_loss = losses / total
    print('TESTING :: test Accuracy: {:.4f} - test Loss: {:.4f} in {}s'.format(
        test_acc, test_loss, int(time.time() - t0)
    ))
    
    # validation epoch
    correct = 0
    total = 0
    losses = 0
    t0 = time.time()
    for idx, (images_batch, labels_batch) in enumerate(input_pipeline[Split.VAL]):
        # Loading tensors in the used device
        images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

        outputs = model(images_batch)
        loss = loss_function(outputs, labels_batch)

        losses += loss.item()
        local_preds = torch.max(outputs.data, 1)[1]
        total += labels_batch.size(0)
        correct += (local_preds == labels_batch).sum().item()

    val_acc = correct / total
    val_loss = losses / total
    print('VALIDATION :: val Accuracy: {:.4f} - val Loss: {:.4f} in {}s'.format(
        val_acc, val_loss, int(time.time() - t0)
    ))


def classification_test(csv_path, data_folder, model_path):

    test_ds = ClassificationDataset(Split.TEST, csv_path, data_folder)
    val_ds = ClassificationDataset(Split.VAL, csv_path, data_folder)

    input_pipeline = InputPipeline(datasets_list=[test_ds, val_ds], batch_size=64)

    n_outputs = len(test_ds.get_idx2labels())
    model = vgg16(num_classes=n_outputs)

    loss_function = torch.nn.CrossEntropyLoss(reduction='sum')

    torch.backends.cudnn.benchmark = True

    saver = Saver(model_path)

    _classification_testing(
        input_pipeline=input_pipeline,
        model=model,
        loss_function=loss_function,
        saver=saver
    )


def _regression_testing(input_pipeline, model, loss_function, saver):
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if torch.cuda.device_count() > 1:
        print("Using {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model.to(device)
    model = saver.load_checkpoint(model)[0]

    # testing epoch
    total = 0
    losses = 0
    t0 = time.time()
    for idx, (images_batch, labels_batch) in enumerate(input_pipeline[Split.TEST]):
        # Loading tensors in the used device
        images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

        outputs = model(images_batch)
        loss = loss_function(outputs, labels_batch)

        losses += loss.item()
        total += labels_batch.size(0)

    test_loss = losses / total
    print('TESTING :: test Loss: {:.4f}in {}s'.format(
        test_loss, int(time.time() - t0)
    ))

    # validation epoch
    total = 0
    losses = 0
    t0 = time.time()
    for idx, (images_batch, labels_batch) in enumerate(input_pipeline[Split.VAL]):
        # Loading tensors in the used device
        images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

        outputs = model(images_batch)
        loss = loss_function(outputs, labels_batch)

        losses += loss.item()
        total += labels_batch.size(0)

    val_loss = losses / total
    print('VALIDATION :: val Loss: {:.4f} in {}s'.format(
        val_loss, int(time.time() - t0)
    ))


def regression_test(csv_path, data_folder, model_path):

    test_ds = RegressionDataset(Split.TEST, csv_path, data_folder)
    val_ds = RegressionDataset(Split.VAL, csv_path, data_folder)

    input_pipeline = InputPipeline(datasets_list=[test_ds, val_ds], batch_size=64)

    model = models.vgg16(num_classes=1)

    loss_function = torch.nn.MSELoss(reduction='mean')

    torch.backends.cudnn.benchmark = True

    saver = Saver(model_path)

    _regression_testing(
        input_pipeline=input_pipeline,
        model=model,
        loss_function=loss_function,
        saver=saver
    )
