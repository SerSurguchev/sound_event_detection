import os
import pathlib
import math
import random
import numpy as np
import torchaudio
import torch
import torch.nn as nn
import glob


class RandomClip:
    def __init__(self, sample_rate, clip_length):
        self.clip_length = clip_length
        self.vad = torchaudio.transforms.Vad(
            sample_rate=sample_rate, trigger_level=7.0)

    def __call__(self, audio_data):
        audio_length = audio_data.shape[0]
        if audio_length > self.clip_length:
            offset = random.randint(0, audio_length - self.clip_length)
            audio_data = audio_data[offset:(offset + self.clip_length)]

        return self.vad(audio_data)  # remove silences at the beggining/end


class RandomSpeedChange:
    def __init__(self, sample_rate, speed_factor=None):
        self.sample_rate = sample_rate
        self.speed_factor = speed_factor

    def __call__(self, audio_data):

        speed_factor = self.speed_factor
        if not self.speed_factor:
            speed_factor = random.choice([0.7, 0.8, 0.9, 1.1, 1.2, 1.3])
        if self.speed_factor == 1.0:  # no change
            return audio_data
        # change speed and resample to original rate:
        sox_effects = [
            ["speed", str(speed_factor)],
            ["rate", str(self.sample_rate)],
        ]
        transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
            audio_data, self.sample_rate, sox_effects)
        return transformed_audio


class RandomBackgroundNoise:
    def __init__(self, sample_rate, noise_dir, min_snr_db=0, max_snr_db=15):
        self.sample_rate = sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        if not os.path.exists(noise_dir):
            raise IOError(f'Noise directory `{noise_dir}` does not exist')
        # find all WAV files including in sub-folders:
        self.noise_files_list = list(pathlib.Path(noise_dir).glob('**/*.wav'))
        if len(self.noise_files_list) == 0:
            raise IOError(f'No .wav file found in the noise directory `{noise_dir}`')

    def __call__(self, audio_data):
        random_noise_file = random.choice(self.noise_files_list)
        effects = [
            ['remix', '1'],  # convert to mono
            ['rate', str(self.sample_rate)],  # resample
        ]
        noise, _ = torchaudio.sox_effects.apply_effects_file(random_noise_file, effects, normalize=True)
        audio_length = audio_data.shape[-1]
        noise_length = noise.shape[-1]
        if noise_length > audio_length:
            offset = random.randint(0, noise_length - audio_length)
            noise = noise[..., offset:offset + audio_length]
        elif noise_length < audio_length:
            noise = torch.cat([noise, torch.zeros((noise.shape[0], audio_length - noise_length))], dim=-1)

        snr_db = random.randint(self.min_snr_db, self.max_snr_db)
        snr = math.exp(snr_db / 10)
        audio_power = audio_data.norm(p=2)
        noise_power = noise.norm(p=2)
        scale = snr * noise_power / audio_power

        return (scale * audio_data + noise) / 2


class DropStripes(nn.Module):
    def __init__(self, dim, drop_width, stripes_num):
        """Drop stripes. 
        Args:
          dim: int, dimension along which to drop
          drop_width: int, maximum width of stripes to drop
          stripes_num: int, how many stripes to drop
        """
        super(DropStripes, self).__init__()

        assert dim in [2, 3]  # dim 2: time; dim 3: frequency

        self.dim = dim
        self.drop_width = drop_width
        self.stripes_num = stripes_num

    def forward(self, input):
        """
        input: (batch_size, channels, time_steps, freq_bins)
        """

        assert input.ndimension() == 4

        if self.training is False:
            return input

        else:
            batch_size = input.shape[0]
            total_width = input.shape[self.dim]

            for n in range(batch_size):
                self.transform_slice(input[n], total_width)

            return input

    def transform_slice(self, e, total_width):
        """e: (channels, time_steps, freq_bins)"""

        for _ in range(self.stripes_num):
            distance = torch.randint(low=0, high=self.drop_width, size=(1,))[0]
            bgn = torch.randint(low=0, high=total_width - distance, size=(1,))[0]

            if self.dim == 2:
                e[:, bgn: bgn + distance, :] = 0
            elif self.dim == 3:
                e[:, :, bgn: bgn + distance] = 0


class SpecAugmentation(nn.Module):
    def __init__(self, time_drop_width,
                 time_stripes_num,
                 freq_drop_width,
                 freq_stripes_num):
        """Spec augmetation.
        [ref] Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D. 
        and Le, Q.V., 2019. Specaugment: A simple data augmentation method 
        for automatic speech recognition. arXiv preprint arXiv:1904.08779.
        Args:
          time_drop_width: int
          time_stripes_num: int
          freq_drop_width: int
          freq_stripes_num: int
        """

        super(SpecAugmentation, self).__init__()

        self.time_dropper = DropStripes(dim=2, drop_width=time_drop_width,
                                        stripes_num=time_stripes_num)

        self.freq_dropper = DropStripes(dim=3, drop_width=freq_drop_width,
                                        stripes_num=freq_stripes_num)

    def forward(self, input):
        x = self.time_dropper(input)
        x = self.freq_dropper(x)
        return x


class ComposeTransform:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio_data):
        for t in self.transforms:
            audio_data = t(audio_data)
        return audio_data


def made_augmentation(path1, path2):
    path1 = list(glob.glob(f"{path1}/*.wav"))
    path2 = list(glob.glob(f"{path2}/*.wav"))

    for audio in path1:
        # Load audio
        audio_data, sample_rate = torchaudio.load(audio)

        # Made audio augmentation
        background_audio = RandomBackgroundNoise(sample_rate, './noise_dir',
                                                 min_snr_db=0, max_snr_db=15)(audio_data)

        for sp_factor in np.array([0.8, 1.2]):
            speed_audio = RandomSpeedChange(sample_rate, speed_factor=sp_factor)(audio_data)

            torchaudio.save(f"{audio.split('/')[0]}/speed_{sp_factor}_{audio.split('/')[1]}",
                            speed_audio,
                            sample_rate)

        compose_transform = ComposeTransform([
            RandomSpeedChange(sample_rate),
            RandomBackgroundNoise(sample_rate, './noise_dir')
        ])
        compose_audio = compose_transform(audio_data)

        # Save augmented audio
        torchaudio.save(f"{audio.split('/')[0]}/background_{audio.split('/')[1]}",
                        background_audio,
                        sample_rate)

        torchaudio.save(f"{audio.split('/')[0]}/compose_{audio.split('/')[1]}",
                        compose_audio,
                        sample_rate)

    for audio in path2:
        # Load audio
        audio_data, sample_rate = torchaudio.load(audio)

        # Made audio augmentation
        background_audio = RandomBackgroundNoise(sample_rate, './noise_dir',
                                                 min_snr_db=0, max_snr_db=15)(audio_data)

        for sp_factor in np.array([0.8, 1.2]):
            speed_audio = RandomSpeedChange(sample_rate, speed_factor=sp_factor)(audio_data)

            torchaudio.save(f"{audio.split('/')[0]}/speed_{sp_factor}_{audio.split('/')[1]}",
                            speed_audio,
                            sample_rate)
        compose_transform = ComposeTransform([
            RandomSpeedChange(sample_rate),
            RandomBackgroundNoise(sample_rate, './noise_dir')
        ])

        compose_audio = compose_transform(audio_data)

        # Save augmented audio
        torchaudio.save(f"{audio.split('/')[0]}/background_{audio.split('/')[1]}",
                        background_audio,
                        sample_rate)

        torchaudio.save(f"{audio.split('/')[0]}/compose_{audio.split('/')[1]}",
                        compose_audio,
                        sample_rate)
