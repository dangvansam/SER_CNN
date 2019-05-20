import numpy as np
from scipy.io import wavfile
import os
from sklearn.model_selection import train_test_split
from python_speech_features import mfcc
import json
from scipy.signal import stft


def read_wav(x):
    sr, wav = wavfile.read(x) 
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    return sr,wav

def get_data(dataset_path, max_duration = 4.0):
    data = []
    labels = []
    emo2int = {}
    int2emo = {}

    list_emotion_dir = os.listdir(dataset_path)
    max_len_signal = 0
    print('Max signal len is',max_duration,',pad and slide signal all to max len')
    #print('Num dir:',len(list_emotion_dir))
    for i, directory in enumerate(list_emotion_dir):
        print ("Read file in folder:", directory)
        emo2int[directory] = i
        int2emo[i] = directory
        for filename in os.listdir(dataset_path + '/' + directory):
            sr,signal = read_wav(dataset_path + '/' + directory + '/'+ filename)
            s_len = len(signal)
            if s_len > max_len_signal:
                max_len_signal = s_len
            if s_len < int(max_duration*sr):
                pad_len = int(max_duration*sr - s_len)
                pad_rem = pad_len % 2
                pad_len /= 2
                pad_len=int(pad_len)
                signal = np.pad(signal, (pad_len, pad_len + pad_rem), 'constant', constant_values=0)
            else:
                pad_len = s_len - int(max_duration*sr)
                pad_len /= 2
                pad_len=int(pad_len)
                signal = signal[pad_len:pad_len + int(max_duration*sr)]

            mfcc_f = mfcc(signal,sr,numcep = 26) #trích xuất 26 MFCC feature
            data.append(mfcc_f)
            labels.append(i)
    print('max_len_signal:',max_len_signal)
    print('int2emo:',int2emo)
    with open('model/emotion_map.json', 'w') as fp:
        json.dump(int2emo, fp)
    print('Saved emotion map to model/emotion_map.json file!')
    '''
    with open('DATA_imocap_mfcc.pkl','wb') as f:
        pickle.dump(data,f)
    with open('LABEL_imocap_mfcc.pkl','wb') as f:
        pickle.dump(labels,f)

    with open('DATA.pkl','rb') as f:
        data = pickle.load(f)
    with open('LABEL.pkl','rb') as f:
        labels = pickle.load(f)
    '''
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=2019)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
