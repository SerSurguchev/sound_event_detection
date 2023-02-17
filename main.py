import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from catalyst.dl import SupervisedRunner, CheckpointCallback
from catalyst import dl, utils

from models import (
    Wavegram_Cnn14,
    MobileNetV2,
    MobileNetV1,
    Wavegram_Logmel_Cnn14
)

from metrics import (
    F1Callback,
    PANNsLoss,
    mAPCallback,
)
from dataset import (
    set_seed,
    create_df,
    PANNsDataset
)

set_seed()

model_config = {
    "sample_rate": 32000,
    "window_size": 1024,
    "hop_size": 320,
    "mel_bins": 64,
    "fmin": 50,
    "fmax": 14000,
    "classes_num": 2
}


def train(model=MobileNetV2(**model_config)):
    train_name, val_name = create_df('serviceable', 'malfunction', save_csv=False)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # loaders
    loaders = {
        "train": torch.utils.data.DataLoader(PANNsDataset(train_name, None),
                                             batch_size=32,
                                             shuffle=True,
                                             num_workers=4,
                                             pin_memory=True,
                                             drop_last=True),

        "valid": torch.utils.data.DataLoader(PANNsDataset(val_name, None),
                                             batch_size=32,
                                             shuffle=False,
                                             num_workers=4,
                                             pin_memory=True,
                                             drop_last=False)
    }

    model = model
    model = torch.nn.DataParallel(model)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001,
                           betas=(0.9, 0.999), eps=1e-08, weight_decay=0., amsgrad=True)

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Loss
    criterion = PANNsLoss().to(device)

    # callbacks
    callbacks = [
        F1Callback(input_key="targets", output_key="logits", prefix="f1"),
        mAPCallback(input_key="targets", output_key="logits", prefix="mAP"),
        CheckpointCallback(save_n_best=0)
    ]

    runner = SupervisedRunner(
        device=device,
        input_key="waveform",
        input_target_key="targets")

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        loaders=loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=50,
        verbose=True,
        logdir=f"./MobileNetV2",
        callbacks=callbacks,
        main_metric="epoch_f1",
        minimize_metric=False)


if __name__ == '__main__':
    train()
