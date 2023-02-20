import torch
import numpy as np
import librosa
from pytorch_utils import move_data_to_device
import pandas as pd
from main import model_config
from models import (
    Wavegram_Cnn14,
    MobileNetV2,
    MobileNetV1,
    Wavegram_Logmel_Cnn14
)

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


def audio_tagging(audio_path):
    """Inference audio tagging result of an audio clip.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    classes_num = 2
    labels = [0, 1]

    # Model
    model = get_model(weights_path='./MobileNetV1_128/checkpoints/best.pth', model=MobileNetV1)

    for audio in audio_path:
        print(audio)

        # Load audio
        (waveform, _) = librosa.core.load(audio, sr=model_config['sample_rate'], mono=True)
        waveform = waveform[None, :]  # (1, audio_length)
        waveform = move_data_to_device(waveform, device)

        # Forward
        with torch.no_grad():
            # model.eval()
            batch_output_dict = model(waveform, None)

        clipwise_output = batch_output_dict['clipwise_output'].data.cpu().numpy()[0]
        print(clipwise_output)

        sorted_indexes = np.argsort(clipwise_output)[::-1]

        # Print audio tagging top probabilities
        for k in range(2):
            print('{}: {:.3f}'.format(np.array(labels)[sorted_indexes[k]],
                                      clipwise_output[sorted_indexes[k]]))

        # # Print embedding
        # if 'embedding' in batch_output_dict.keys():
        #     embedding = batch_output_dict['embedding'].data.cpu().numpy()[0]
        #     print('embedding: {}'.format(embedding.shape))

    return clipwise_output, labels


if __name__ == '__main__':
    df = pd.read_csv('valid_waves.csv')
    audio_tagging(df['file_path'].values.tolist())
