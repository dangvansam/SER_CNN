from keras.models import load_model
import keras
from keras.optimizers import Adam
import os
import numpy as np
from scipy.io import wavfile
from keras.models import model_from_json
from python_speech_features import mfcc
import json

def read_wav(x):
    sr, wav = wavfile.read(x) 
    # Normalize
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    return sr,wav

model_dir = 'model'
# load model from json
json_file = open(model_dir+'/model_json.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(model_dir+'/weights_best.h5')
#print("Loaded model from disk")

file_object = open(model_dir+'/emotion_map.json', 'r')
int2emo = json.load(file_object)
int2emo2 = sorted (int2emo.values())
#print('Emotion:',int2emo2)
print('Emotion map for predict:',int2emo)
# evaluate loaded model on test data
#opt = Adam(lr=0.0001)
loaded_model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

def predict_file(filename = 'test'):
       print('Testing file:',filename)
       #LOAD DATA
       data = []
       sr,signal = read_wav(filename)
       max_duration = 4.0
       s_len = len(signal)
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
       mfcc_f = mfcc(signal,sr,numcep = 26) #use 26 MFCC feature
       data.append(mfcc_f)
       data =np.array(data)

       c = loaded_model.predict(data,batch_size=32, verbose=0)
       int_pred =c.argmax(axis=1)
       print('==================================')
       #print(c[0].tolist())
       c2 = c[0].tolist()
       for i,va in enumerate(c2):
              print(int2emo2[i],'= {:.2f}%'.format(va*100))
       print('==================================')
       #print(c*100)
       #print(int(int_pred))
       print("Emotion detected:",int2emo2[int(int_pred)])
       print('==================================')
       return str(int2emo2[int(int_pred)])
test_path = 'test/sad/'
len_all_file = len(os.listdir(test_path))
count_true = 0
failed = []
for i,f in enumerate(os.listdir(test_path)):
       print('{}/{}'.format(i+1,len_all_file))
       emotion = predict_file(test_path+f)
       if emotion == 'tieu_cuc':
              count_true +=1
       else:
              failed.append([f,emotion])
print('Num true predict:',count_true,'/',len_all_file)
print('% true =',count_true/len_all_file*100)
print('Failed:',failed)
#{"0": "anger", "1": "boredom", "2": "disgust", "3": "fear", "4": "happiness", "5": "neutral", "6": "sadness"}

