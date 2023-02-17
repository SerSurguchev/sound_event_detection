import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from catalyst.dl import SupervisedRunner, State, CallbackOrder, Callback, CheckpointCallback
from catalyst import dl, utils

from models import (
    Cnn14,
    Wavegram_Cnn14,
    MobileNetV2,
    Wavegram_Logmel_Cnn14
)

from pytorch_utils import (
    F1Callback,
    PANNsLoss,
    mAPCallback
)
from dataset import (
    set_seed,
    create_file_list,
    PANNsDataset
)

set_seed(42)

model_config = {
    "sample_rate": 32000,
    "window_size": 1024,
    "hop_size": 320,
    "mel_bins": 64,
    "fmin": 50,
    "fmax": 14000,
    "classes_num": 2
}


def train(model=Wavegram_Cnn14(**model_config)):
    train_file_list, val_file_list = create_file_list('serviceable', 'malfunction')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # loaders
    loaders = {
        "train": torch.utils.data.DataLoader(PANNsDataset(train_file_list, None),
                                             batch_size=8,
                                             shuffle=True,
                                             num_workers=2,
                                             pin_memory=True,
                                             drop_last=True),

        "valid": torch.utils.data.DataLoader(PANNsDataset(val_file_list, None),
                                             batch_size=8,
                                             shuffle=False,
                                             num_workers=2,
                                             pin_memory=True,
                                             drop_last=False)
    }

    model = model
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Loss
    criterion = PANNsLoss().to(device)

    # callbacks
    callbacks = [
        F1Callback(input_key="targets", output_key="logits", prefix="f1"),
        mAPCallback(input_key="targets", output_key="logits", prefix="mAP"),
    ]

    runner = SupervisedRunner(
        device=device,
        input_key="waveform",
        input_target_key="targets"
    )

    # Model training
    runner.train(
        model=model,
        criterion=criterion,
        loaders=loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=10,
        verbose=True,
        fp16=True,
        logdir="./logs",
        callbacks=callbacks,
        main_metric="epoch_f1",
        load_best_on_end=True,
        minimize_metric=False)

    # model evaluation
    metrics = runner.evaluate_loader(
        loader=loaders["valid"],
        callbacks=callbacks
    )

if __name__ == '__main__':
    train()
