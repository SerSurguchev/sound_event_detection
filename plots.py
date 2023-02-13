import glob
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sklearn

def sound_wave(path):
    i = 0
    colors = ['#A300F9', '#4300FF', '#009DFF', '#00FFB0', '#D9FF00', '#7f7f7f']
    fig, ax = plt.subplots(6, figsize=(25, 16))
    fig.suptitle('Sound_waves', fontsize=16)
    
    for audio in glob.glob(f'{path}/*.wav'):
        print(audio)
        y, sr = librosa.load(audio)
        librosa.display.waveshow(y, sr=sr, ax=ax[i], color = colors[i])
        ax[i].set_ylabel(audio.split('/')[1], fontsize=13)
        i += 1


def plot_sound_waves(path1, path2):
    colors = ['#A300F9', '#4300FF', '#009DFF', '#00FFB0', '#D9FF00', '#7f7f7f']
    
    # Right side - malfunction audio
    fig, ax = plt.subplots(6, 2, figsize=(25, 16))
    fig.suptitle('Sound_waves', fontsize=16)
    
    path1 = list(glob.glob(f"{path1}/*.wav"))
    path2 = list(glob.glob(f"{path2}/*.wav"))
    print(path1)
    print(path2)
    
    for i in range(len(colors)):
        j = 0
        y1, sr1 = librosa.load(path1[i])
        y2, sr2 = librosa.load(path2[i])
        librosa.display.waveshow(y1, sr=sr1, ax=ax[i][j], color = colors[i])
        librosa.display.waveshow(y2, sr=sr2, ax=ax[i][j+1], color = colors[i])
        ax[i][j].set_ylabel(path1[i].split('/')[1], fontsize=13)
        ax[i][j+1].set_ylabel(path2[i].split('/')[1], fontsize=13)


def normalize(x, axis=0):
    # Function that normalizes the Sound Data
    return sklearn.preprocessing.minmax_scale(x, axis=axis)


def calc_centroid(y, sr):
    # Calculate the Spectral Centroids
    spectral_centroids = librosa.feature.spectral_centroid(y, sr=sr)[0]
    # Computing the time variable for visualization
    frames = range(len(spectral_centroids))
    # Converts frame counts to time (seconds)
    t = librosa.frames_to_time(frames)   
    return t, normalize(spectral_centroid)


def spectral_centroid(path1, path2):
    colors = ['#A300F9', '#4300FF', '#009DFF', '#00FFB0', '#D9FF00', '#7f7f7f']
    
    # Right side - malfunction audio
    fig, ax = plt.subplots(6, 2, figsize=(25, 16))
    fig.suptitle('Spectral Centroid', fontsize=16)
    
    path1 = list(glob.glob(f"{path1}/*.wav"))
    path2 = list(glob.glob(f"{path2}/*.wav"))

    print(path1)
    print(path2)
    
    for i in range(len(colors)):
        j = 0

        y1, sr1 = librosa.load(path1[i])
        y2, sr2 = librosa.load(path2[i])
        t1, cent1 = calc_centroid(y1, sr1)
        t2, cent2 = calc_centroid(y2, sr2)

        librosa.display.waveshow(y1, sr=sr1, ax=ax[i][j], color = colors[i])
        librosa.display.waveshow(y2, sr=sr2, ax=ax[i][j+1], color = colors[i])

        plt.plot(t1, cent1, lw=2, ax=ax[i][j], color=colors[i-1])
        plt.plot(t2, cent2, lw=2, ax=ax[i][j+1], color=color[i-1])

        ax[i][j].set_ylabel(path1[i].split('/')[1], fontsize=13)
        ax[i][j+1].set_ylabel(path2[i].split('/')[1], fontsize=13)

            
def plot_spectrograms(path1, path2, y_axis='log'):
    colors = ['#A300F9', '#4300FF', '#009DFF', '#00FFB0', '#D9FF00', '#7f7f7f']
    
    # Right side - malfunction audio
    fig, ax = plt.subplots(6, 2, figsize=(25, 16))
    fig.suptitle(f'{y_axis} Spectrogram', fontsize=16)
    
    path1 = list(glob.glob(f"{path1}/*.wav"))
    path2 = list(glob.glob(f"{path2}/*.wav"))
    print(path1)
    print(path2)
    
    for i in range(len(colors)):
        j = 0
        # Left spectrograms
        y1, sr1 = librosa.load(path1[i])
        X1 = librosa.stft(y1)
        X1db = librosa.amplitude_to_db(abs(X1))
        librosa.display.specshow(X1db, sr=sr1, x_axis='time', 
                                 y_axis=y_axis, ax=ax[i][j])
        # Right spectrograms
        y2, sr2 = librosa.load(path2[i])
        X2 = librosa.stft(y2)
        X2db = librosa.amplitude_to_db(abs(X2))
        librosa.display.specshow(X2db, sr=sr2, x_axis='time', 
                                 y_axis=y_axis, ax=ax[i][j+1])
        
        ax[i][j].set_ylabel(path1[i].split('/')[1], fontsize=13)
        ax[i][j+1].set_ylabel(path2[i].split('/')[1], fontsize=13)
