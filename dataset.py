import torch
import torch.nn as nn
import pandas as pd
import glob
import numpy as np
import soundfile as sf
import random
import os

PERIOD = 5


class PANNsDataset:
    def __init__(
            self,
            file_path,
            waveform_transforms=None):

        self.waveform_transforms = waveform_transforms
        df = pd.read_csv(file_path)
        self.file_list = df[['file_path', 'file_label']].values.tolist()  # list of list [file_path, label]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        wav_path, class_id = self.file_list[idx]

        y, sr = sf.read(wav_path)

        if self.waveform_transforms:
            y = self.waveform_transforms(y)
        else:
            len_y = len(y)
            effective_length = sr * PERIOD
            if len_y < effective_length:
                new_y = np.zeros(effective_length, dtype=y.dtype)
                start = np.random.randint(effective_length - len_y)
                new_y[start:start + len_y] = y
                y = new_y.astype(np.float32)
            elif len_y > effective_length:
                start = np.random.randint(len_y - effective_length)
                y = y[start:start + effective_length].astype(np.float32)
            else:
                y = y.astype(np.float32)

        labels = np.zeros(2, dtype="f")
        labels[class_id] = 1

        return {"waveform": y, "targets": labels}


def create_df(path1, path2,
              train_name='train_waves.csv',
              val_name='valid_waves.csv',
              save_csv=False):
    if save_csv:
        df = pd.DataFrame(
            list(glob.glob(f"{path1}/*.wav")) +
            list(glob.glob(f"{path2}/*.wav")),
            columns=["file_path"]
        )
        df = df.sample(frac=1).reset_index(drop=True)
        df["file_label"] = np.where(df['file_path'].str.contains(path1), 0, 1)
        df['is_valid'] = np.random.choice(a=[True, False], size=len(df), p=[0.25, 0.75])

        train, valid = df[['file_path', 'file_label']][df['is_valid'] == False], \
            df[['file_path', 'file_label']][df['is_valid'] == True]

        train.to_csv(train_name, index=False)
        valid.to_csv(val_name, index=False)

    return train_name, val_name


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
