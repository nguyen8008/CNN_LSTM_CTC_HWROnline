from keras.models import Model
from keras.layers import *
from keras import backend as K
from tqdm import tqdm
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.xception import Xception
import csv
import re
import os
import pickle
import cv2
from sklearn.preprocessing import LabelEncoder
import unicodedata
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
import itertools
#import unicode


IMAGE_HEIGHT = 64
IMAGE_WIDTH = None
NO_CHANNEL = 1
TRAIN_VAL_SPLIT = 0.8
BATCH_SIZE = 8
STRIDE = 12
FILTER_SIZE = 32
NO_CLASSES = 118 #66 + 1   # blank token - số lượng ký tự có trong thư viện từ (số lượng này phải = số ký tự có trong thư viện + 1)
DATA_FOLDER = './image/'
DATA_FOLDER_LABEL = './label/'
LABEL_ENCODER_PATH = 'label_encoder.txt'


class DataGenerator():
    def __init__(self, train_image_list, val_image_list, batch_size=BATCH_SIZE):
        self.train_image_list = train_image_list
        self.val_image_list = val_image_list
        self.batch_size = batch_size
        self.current_train_index = 0
        self.current_val_index = 0
#        self.load_label_encoder()
        self.load_label_index()
        #self.labels_to_text()
        #self.text_to_labels()

    def load_image(self, image_path):
        #print(type(image_path))
        image = cv2.imread(image_path, 0)
        image = cv2.resize(image, (128, 64)) 
        image = image / 255.0
        image = np.expand_dims(image, axis=-1)
        return image

    def load_label(self, image_path):
        #print(type(image_path))
        f = open(image_path, encoding="utf8")
        s = f.read()
        #print(type(s))
        return s

#    def load_label_encoder(self):
#        self.le = load_label_encoder()

    def load_label_index(self):
        self.le = load_label_index()

    # # Input data generator
    def labels_to_text(self, labels):     # letters의 index -> text (string)
        return ''.join(list(map(lambda x: self.le[int(labels)], labels)))

    def text_to_labels(self, text):      # text를 letters 배열에서의 인덱스 값으로 변환
        return list(map(lambda x: self.le.index(x), text)) #lấy vị trí của từng ký tự trong CHAR_VECTOR


    def get_batch(self, partition='train'):
        if partition == 'train':
            temp_image_list = self.train_image_list[self.current_train_index:self.current_train_index+self.batch_size]
            temp_image_list = [os.path.join(DATA_FOLDER, t) for t in temp_image_list]
        else:
            temp_image_list = self.val_image_list[self.current_val_index:self.current_val_index+self.batch_size]
            temp_image_list = [os.path.join(DATA_FOLDER, t) for t in temp_image_list]
        image_array = []
        label_array = []
        for ind in range(self.batch_size):
            image_array.append(self.load_image(temp_image_list[ind]))
            label_array.append(self.load_label('./label/'+temp_image_list[ind].split('/')[-1].split('.png')[0]+'.txt'))
        max_image_width = max([m.shape[1] for m in image_array])
        max_label_length = max(len(m) for m in label_array)
        input_image = np.ones((self.batch_size, IMAGE_HEIGHT, max_image_width, 1))
        input_true_label = np.ones((self.batch_size, max_label_length)) #* NO_CLASSES
        input_time_step = np.zeros((self.batch_size, 1))
        #print(input_time_step)
        input_label_length = np.zeros((self.batch_size, 1))
        for ind in range(self.batch_size):  
            real_width = image_array[ind].shape[1]
            #print(label_array[ind])
            #print(self.le)
            #print(type(self.le[0]))

            #print(self.le['p'.encode('ascii', 'ignore')])
            #tmp = [self.le.transform([t])[0] for t in label_array[ind]]
            tmp = [self.text_to_labels(t) for t in label_array[ind]] #chuyen tung ky tu sang index vi tri cua ky tu trong CHAR_VECTOR
            #print(tmp)
            tmp = list(itertools.chain.from_iterable(tmp)) #chuyen tu mang 2 chieu ([[x][y]]) ve mang 1 chieu ([x, y])
            #print(tmp)
            #print(temp)
            #print(tmp, ' - ', self.le.inverse_transform(tmp))
            if len(tmp) == 0:
                print('label: ' + label_array[ind])
            real_label_len = len(tmp)
            #print(real_label_len)
            input_image[ind, :, :real_width, :] = image_array[ind]            
            input_true_label[ind, :real_label_len] = tmp
            #print(input_true_label)
            input_time_step[ind] = compute_time_step(real_width) - 2
            #print(input_time_step[ind])
            input_label_length[ind] = real_label_len
        inputs = {
            'input_image': input_image,
            'input_true_label': input_true_label,
            'input_time_step': input_time_step, 
            'input_label_length': input_label_length}
        outputs = {'ctc': np.zeros((self.batch_size))}
        return (inputs, outputs)


    def next_train(self):
        while True:
            tmp = self.get_batch('train')
            self.current_train_index += self.batch_size
            if self.current_train_index >= len(self.train_image_list) - self.batch_size:
                self.train_image_list = shuffle(self.train_image_list)
                self.current_train_index = 0
            yield tmp


    def next_val(self):
        while True:
            tmp = self.get_batch('val')
            self.current_val_index += self.batch_size
            if self.current_val_index >= len(self.val_image_list) - self.batch_size:
                self.val_image_list = shuffle(self.val_image_list)
                self.current_val_index = 0
            yield tmp

def compute_time_step(image_width, stride=STRIDE//2):
    for i in range(2):
        tmp = (image_width - 1) // 2 + 1
    tmp = (tmp + stride - 1) // stride
    return tmp


def get_image_list(folder=DATA_FOLDER):
    image_list = os.listdir(folder)
    return image_list
      
def get_image_list_1(folder=DATA_FOLDER_LABEL):
    image_list = os.listdir(folder)
    return image_list

def get_all_character():
    all_character_list = []
    image_list = get_image_list_1()
    for i in image_list:
        f = open('./label/'+i, encoding="utf-8")
        s = f.read()
        all_character_list += s #.encode('utf8','ignore')
    return all_character_list

#def create_label_encoder(all_character_list):
#    all_character_list = list(set(all_character_list))
#    print(all_character_list, len(all_character_list))
#    le = LabelEncoder()
#    le.fit(all_character_list)
#    print(le.transform(['n']))
#    with open(LABEL_ENCODER_PATH, 'wb') as f:
#        pickle.dump(le, f)

def create_label_index(all_character_list):
    all_character_list = list(set(all_character_list))
    print(all_character_list, len(all_character_list))
    le = ""
    for i in all_character_list:
        le += i #.decode('utf-8')
    print(le)
    #le = all_character_list

    # Mở file
    file = open(str(LABEL_ENCODER_PATH), "w", encoding='UTF8')
    file.write(le)

    #letters = [letter for letter in CHAR_VECTOR]
    #le = LabelEncoder()
    #le.fit(all_character_list)
    #data = np.array(['a','b','c','d'])
    #le = pd.Series(all_character_list)#, index=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116])
    #print(le['q'])
    #with open(LABEL_ENCODER_PATH, 'wb') as f:
        #print("all charater list", f)
    #    pickle.dump(le, f) #viet 1 dai dien pickled cua "le" vao file "f" 
        #print("all charater list", f)


#def load_label_encoder():
#    with open(LABEL_ENCODER_PATH, 'rb') as f:
#        le = pickle.load(f)
#    return le

def load_label_index():
    # Mở file
    file = open(LABEL_ENCODER_PATH, "r", encoding="UTF8")
    CHAR_VECTOR = file.read()

    le = [le for le in CHAR_VECTOR]

    #with open(LABEL_ENCODER_PATH, 'rb') as f:        
    #    le = pickle.load(f, encoding="utf-8") #Doc 1 dai dien pickled tu file "f"
        #le = pd.Series(le)
        #le = f
    #print(le['q'])
    return le

def ctc_loss(args):
    y_pred, y_true, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)

def fake_loss(y_true, y_pred):
    return y_pred

def squeeze_layer(arr, axis=1):
    return K.squeeze(arr, axis)

def create_model(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NO_CHANNEL)):
    #input_shape = (img_w, img_h, 1)     # (128, 64, 1)

    # Make Network
    inputs = Input(name='input_image', shape=input_shape)  # (None, 128, 64, 1)

    # Convolution layer
    inner = Conv2D(64, (3, 3), padding='same', name='conv1', kernel_initializer='he_normal')(inputs)  # (None, 128, 64, 64)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max1')(inner)  # (None,64, 32, 64)

    inner = Conv2D(128, (3, 3), padding='same', name='conv2', kernel_initializer='he_normal')(inner)  # (None, 64, 32, 128)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(2, 2), name='max2')(inner)  # (None, 32, 16, 128)

    inner = Conv2D(256, (3, 3), padding='same', name='conv3', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = Conv2D(256, (3, 3), padding='same', name='conv4', kernel_initializer='he_normal')(inner)  # (None, 32, 16, 256)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max3')(inner)  # (None, 32, 8, 256)

    inner = Conv2D(512, (3, 3), padding='same', name='conv5', kernel_initializer='he_normal')(inner)  # (None, 32, 8, 512)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = Conv2D(512, (3, 3), padding='same', name='conv6')(inner)  # (None, 32, 8, 512)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)
    inner = MaxPooling2D(pool_size=(1, 2), name='max4')(inner)  # (None, 32, 4, 512)

    inner = Conv2D(512, (2, 2), padding='same', kernel_initializer='he_normal', name='con7')(inner)  # (None, 32, 4, 512)
    inner = BatchNormalization()(inner)
    inner = Activation('relu')(inner)

    # CNN to LSTM
    inner = Reshape(target_shape=((32, 2048)), name='reshape')(inner)  # (None, 32, 2048)
    inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)

    # LSTM layer
    lstm_1 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1')(inner)  # (None, 32, 512)
    lstm_1b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm1_b')(inner)
    reversed_lstm_1b = Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (lstm_1b)

    lstm1_merged = add([lstm_1, reversed_lstm_1b])  # (None, 32, 512)
    lstm1_merged = BatchNormalization()(lstm1_merged)
    
    lstm_2 = LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm2')(lstm1_merged)
    lstm_2b = LSTM(256, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='lstm2_b')(lstm1_merged)
    reversed_lstm_2b= Lambda(lambda inputTensor: K.reverse(inputTensor, axes=1)) (lstm_2b)

    lstm2_merged = concatenate([lstm_2, reversed_lstm_2b])  # (None, 32, 1024)
    lstm2_merged = BatchNormalization()(lstm2_merged)

    # transforms LSTM output to character activations:
    inner = Dense(units=NO_CLASSES,name='dense2')(lstm2_merged) #(None, 32, 63)
    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='input_true_label', shape=(None,)) # (None ,8)
    input_length = Input(name='input_time_step', shape=(1,))     # (None, 1)
    label_length = Input(name='input_label_length', shape=(1,))     # (None, 1)

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_loss, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length]) #(None, 1)

    return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
    
# le = load_label_encoder()
# print (le.classes_)
