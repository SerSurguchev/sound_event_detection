import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from catalyst.dl import SupervisedRunner, CheckpointCallback

from models import (
    MobileNetV2,
    MobileNetV1,
    MobileNetV3
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
    "mel_bins": 128,
    "fmin": 50,
    "fmax": 14000,
    "classes_num": 2,
    "training": True,
}


def train(weights_path, model=None, pretrained=False, save_csv=False):
    train_name, val_name = create_df('serviceable', 'malfunction', save_csv=save_csv)
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

    if pretrained:
        print('Pretrained_weights loading...')
        assert weights_path, "Weights path shouldn't be None"
        checkpoint = torch.load(weights_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
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
        logdir=f"./MobileNetV3_spec_aug",
        callbacks=callbacks,
        main_metric="epoch_f1",
        minimize_metric=True)


if __name__ == '__main__':
    train(pretrained=False, save_csv=False, weights_path='./MobileNetV2_128/checkpoints/best.pth', model=MobileNetV3(**model_config))
