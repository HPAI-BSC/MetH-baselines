import time
import torch
import torchvision.models as models
from baselines.src.models import ReducedVgg16
from baselines.utils.consts import Split
from baselines.src.datasets import ClassificationDataset, RegressionDataset
from baselines.src.input import InputPipeline
from baselines.utils.saver import Saver
from baselines.utils.tensorboard_writer import SummaryWriter


class TrainingLogger:

    def __init__(self, batches_per_epoch, total_prints=9):
        self.batches_per_epoch = batches_per_epoch
        self.total_prints = total_prints
        self.prev_stage = 0

    def get_stage(self, idx):
        return int(idx / self.batches_per_epoch * (self.total_prints + 1))

    def stage_to_string(self, stage):
        return '{:.1f}%'.format(stage / (self.total_prints + 1) * 100)

    def should_print(self, idx):
        stage = self.get_stage(idx)
        if stage == 0:
            return None
        if stage != self.prev_stage:
            self.prev_stage = stage
            return self.stage_to_string(stage)
        return None


def _classification_training(input_pipeline, model, loss_function, optimizer, saver, writer, retrain, max_epochs=100, total_prints=99):
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if torch.cuda.device_count() > 1:
        print("Using {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model.to(device)

    initial_epoch = 0
    step = 0

    if retrain:
        model, optimizer, last_epoch, step = saver.load_checkpoint(model, optimizer)
        initial_epoch = last_epoch + 1

    printer = TrainingLogger(batches_per_epoch=len(input_pipeline[Split.TRAIN]), total_prints=total_prints)

    for epoch in range(initial_epoch, max_epochs):

        # Training epoch
        correct = 0
        total = 0
        losses = 0
        t0 = time.time()
        for idx, (batch_images, batch_labels) in enumerate(input_pipeline[Split.TRAIN]):
            # Loading tensors in the used device
            step_images, step_labels = batch_images.to(device), batch_labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            step_output = model(step_images)
            loss = loss_function(step_output, step_labels)
            loss.backward()
            optimizer.step()
            step += 1

            step_preds = torch.max(step_output.data, 1)[1]
            step_correct = (step_preds == step_labels).sum().item()
            step_total = step_labels.size(0)
            step_loss = loss.item()
            step_acc = step_correct / step_total

            losses += step_loss
            total += step_total
            correct += step_correct

            print_stage = printer.should_print(idx)
            if print_stage:
                print('({}) Train Accuracy: {:.4f} - Train Loss: {:.4f} at {}s'.format(
                    print_stage, step_acc, step_loss, int(time.time() - t0)
                ))
            writer[Split.TRAIN].add_scalar(writer.ACC, step_acc, global_step=step)
            writer[Split.TRAIN].add_scalar(writer.LOSS, step_loss, global_step=step)

        train_acc = correct / total
        train_loss = losses / total
        print('EPOCH {} :: Train Accuracy: {:.4f} - Train Loss: {:.4f} in {}s'.format(
            epoch, train_acc, train_loss, int(time.time() - t0)
        ))

        saver.save_checkpoint(model, optimizer, epoch, step)

        # Validation epoch
        correct = 0
        total = 0
        losses = 0
        t0 = time.time()
        with torch.no_grad():
            for batch_images, batch_labels in input_pipeline[Split.VAL]:
                # Loading tensors in the used device
                step_images, step_labels = batch_images.to(device), batch_labels.to(device)

                step_output = model(step_images)
                loss = loss_function(step_output, step_labels)

                step_preds = torch.max(step_output.data, 1)[1]
                step_correct = (step_preds == step_labels).sum().item()
                step_total = step_labels.size(0)
                step_loss = loss.item()

                losses += step_loss
                total += step_total
                correct += step_correct

        val_acc = correct / total
        val_loss = losses / total
        print('EPOCH {} :: Validation Accuracy: {:.4f} - Validation Loss: {:.4f} in {}s'.format(
            epoch, val_acc, val_loss, int(time.time() - t0)
        ))
        writer[Split.VAL].add_scalar(writer.ACC, val_acc, global_step=step)
        writer[Split.VAL].add_scalar(writer.LOSS, val_loss, global_step=step)


def classification_train(csv_path, data_folder, model_path, summaries_path, retrain=False):

    train_ds = ClassificationDataset(Split.TRAIN, csv_path, data_folder)
    val_ds = ClassificationDataset(Split.VAL, csv_path, data_folder)

    input_pipeline = InputPipeline(datasets_list=[train_ds, val_ds], batch_size=256)

    n_outputs = len(train_ds.get_idx2labels())
    model = models.vgg16(num_classes=n_outputs)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    loss_function = torch.nn.CrossEntropyLoss(reduction='sum')

    # Model Saver
    saver = Saver(model_path)

    writer = SummaryWriter(summaries_path)

    torch.backends.cudnn.benchmark = True

    _classification_training(
        input_pipeline=input_pipeline,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        saver=saver,
        writer=writer,
        retrain=retrain
    )


def _regression_training(input_pipeline, model, loss_function, optimizer, saver, writer, retrain, max_epochs=100, total_prints=99):
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    if torch.cuda.device_count() > 1:
        print("Using {} GPUs!".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model.to(device)

    initial_epoch = 0
    step = 0

    if retrain:
        model, optimizer, last_epoch, step = saver.load_checkpoint(model, optimizer)
        initial_epoch = last_epoch + 1

    printer = TrainingLogger(batches_per_epoch=len(input_pipeline[Split.TRAIN]), total_prints=total_prints)

    for epoch in range(initial_epoch, max_epochs):

        # Training epoch
        total = 0
        losses = 0
        t0 = time.time()
        for idx, (images_batch, labels_batch) in enumerate(input_pipeline[Split.TRAIN]):
            # Loading tensors in the used device
            images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(images_batch)
            loss = loss_function(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            step += 1

            losses += loss.item()
            total += labels_batch.size(0)

            train_loss = losses / total
            print_stage = printer.should_print(idx)
            if print_stage:
                print('({}) Train Loss: {:.4f} at {}s'.format(
                    print_stage, train_loss, int(time.time() - t0)
                ))
            writer[Split.TRAIN].add_scalar(writer.LOSS, train_loss, global_step=step)

        train_loss = losses / total
        print('EPOCH {} :: Train Loss: {:.4f}in {}s'.format(
            epoch, train_loss, int(time.time() - t0)
        ))

        saver.save_checkpoint(model, optimizer, epoch, step)

        # Validation epoch
        total = 0
        losses = 0
        t0 = time.time()
        with torch.no_grad():
            for images_batch, labels_batch in input_pipeline[Split.VAL]:
                # Loading tensors in the used device
                images_batch, labels_batch = images_batch.to(device), labels_batch.to(device)

                outputs = model(images_batch)
                loss = loss_function(outputs, labels_batch)

                losses += loss.item()
                total += labels_batch.size(0)

        val_loss = losses / total
        print('EPOCH {} :: Validation Loss: {:.4f} in {}s'.format(
            epoch, val_loss, int(time.time() - t0)
        ))
        writer[Split.VAL].add_scalar(writer.LOSS, val_loss, global_step=step)


def regression_train(csv_path, data_folder, model_path, summaries_path, retrain=False):

    train_ds = RegressionDataset(Split.TRAIN, csv_path, data_folder)
    val_ds = RegressionDataset(Split.VAL, csv_path, data_folder)

    input_pipeline = InputPipeline(datasets_list=[train_ds, val_ds], batch_size=256)

    model = models.vgg16(num_classes=1)

    loss_function = torch.nn.MSELoss(reduction='mean')

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Model Saver
    saver = Saver(model_path)

    writer = SummaryWriter(summaries_path)

    torch.backends.cudnn.benchmark = True

    _regression_training(
        input_pipeline=input_pipeline,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        saver=saver,
        writer=writer,
        retrain=retrain,
        max_epochs=9
    )

