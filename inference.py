import time
import torch
import numpy as np
import soundfile as sf
from pytorch_utils import move_data_to_device, append_to_dict
import pandas as pd
from main import model_config
from dataset import PANNsDataset
from models import (
    MobileNetV2,
    MobileNetV1,
    Wavegram_Logmel_Cnn14
)
from sklearn import metrics

# Model inference
model_config['training'] = False


def get_model(weights_path: str, model, config=model_config):
    model = model(**config)
    checkpoint = torch.load(weights_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Parallel
    if 'cuda' in str(device):
        model.to(device)
        print('GPU number: {}'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    else:
        print('Using CPU.')

    model.eval()
    return model


def forward(weights_path,
            model_name, generator,
            return_input=False,
            return_target=False):
    """Forward data to a model.

    Args:
      model: object
      generator: object
      return_input: bool
      return_target: bool
    Returns:
      audio_name: (audios_num,)
      clipwise_output: (audios_num, classes_num)
      (ifexist) segmentwise_output: (audios_num, segments_num, classes_num)
      (ifexist) framewise_output: (audios_num, frames_num, classes_num)
      (optional) return_input: (audios_num, segment_samples)
      (optional) return_target: (audios_num, classes_num)
    """
    output_dict = {}
    time1 = time.time()

    # Model
    model = get_model(weights_path=weights_path, model=model_name)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Forward data to a model in mini-batches
    for n, batch_data_dict in enumerate(generator):
        batch_waveform = move_data_to_device(batch_data_dict['waveform'], device)

        with torch.no_grad():
            batch_output = model(batch_waveform)

        append_to_dict(output_dict, 'clipwise_output',
                       batch_output['clipwise_output'].data.cpu().numpy())

        if 'framewise_output' in batch_output.keys():
            append_to_dict(output_dict, 'framewise_output',
                           batch_output['framewise_output'].data.cpu().numpy())

        if return_input:
            append_to_dict(output_dict, 'waveform', batch_data_dict['waveform'])

        if return_target:
            if 'targets' in batch_data_dict.keys():
                append_to_dict(output_dict, 'targets', batch_data_dict['targets'])

        if n % 5 == 0:
            print(' --- Inference time: {:.3f} s / 5 iterations ---'.format(
                time.time() - time1))
            time1 = time.time()

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)

    return output_dict


class Evaluator(object):
    def __init__(self, weights_path, model_name):
        """Evaluator.
        Args:
          model: object
        """
        self.weights_path = weights_path
        self.model_name = model_name

    def evaluate(self, data_loader):
        """Forward evaluation data and calculate statistics.
        Args:
          data_loader: object
        Returns:
          statistics: dict,
              {'average_precision': (classes_num,), 'auc': (classes_num,)}
        """

        # Forward
        output_dict = forward(
            weights_path=self.weights_path,
            model_name=self.model_name,
            generator=data_loader,
            return_target=True)

        clipwise_output = output_dict['clipwise_output']  # (audios_num, classes_num)
        target = output_dict['targets']  # (audios_num, classes_num)

        average_precision = metrics.average_precision_score(
            target, clipwise_output, average=None)

        auc = metrics.roc_auc_score(target, clipwise_output, average=None)

        statistics = {'average_precision': average_precision, 'auc': auc}

        return statistics


def audio_tagging(weights_path,
                  model_name,
                  audio_path,
                  targets,
                  save_df=False):
    """Inference audio tagging result of an audio clip.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    labels = [0, 1]

    # Model
    model = get_model(weights_path=weights_path, model=model_name)

    preds = [0 for i in range(len(targets))]
    probs = [0 for i in range(len(targets))]

    i = 0

    for audio in audio_path:
        # Load audio
        # (waveform, _) = librosa.core.load(audio, sr=model_config['sample_rate'], mono=True)
        waveform, _ = sf.read(audio, dtype=np.float32)

        waveform = waveform[None, :]  # (1, audio_length)
        waveform = move_data_to_device(waveform, device)

        # Forward
        with torch.no_grad():
            batch_output_dict = model(waveform, None)

        clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]

        sorted_indexes = np.argsort(clipwise_output)[::-1]

        preds[i] = sorted_indexes[0]
        probs[i] = clipwise_output

        # Print audio tagging top probabilities
        for k in range(2):
            print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]],
                                      clipwise_output[sorted_indexes[k]]))

        # Print embedding
        if 'embedding' in batch_output_dict.keys():
            embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
            print('embedding: {}'.format(embedding.shape))

        i += 1

    df = pd.DataFrame(
        {'audio_path': audio_path,
         'labels': targets,
         'prediction': preds,
         'probabilities': probs,
         })

    if save_df:
        df.to_csv(f"{weights_path.split('/')[0]}.csv", index=False)

    return clipwise_output, labels


if __name__ == '__main__':
    df = pd.read_csv('valid_waves.csv')
    audio_tagging(weights_path='MobileNetV2_spec_aug/checkpoints/best.pth',
                  model_name=MobileNetV2,
                  audio_path=df['file_path'].values.tolist(),
                  targets=df['file_label'].values.tolist(),
                  save_df=True)

    # data_loader = torch.utils.data.DataLoader(PANNsDataset('valid_waves.csv', None),
    #                                           batch_size=32,
    #                                           shuffle=False,
    #                                           num_workers=4,
    #                                           pin_memory=True,
    #                                           drop_last=False)
    #
    # statistics = Evaluator(weights_path='./MobileNetV2_spec_aug/checkpoints/best.pth',
    #                        model_name=MobileNetV2).evaluate(data_loader)

    # {'average_precision': array([0.88492542, 0.93000202]), 'auc': array([0.90491167, 0.9052417])}
