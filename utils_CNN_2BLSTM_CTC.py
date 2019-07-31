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


IMAGE_HEIGHT = 64
IMAGE_WIDTH = None
NO_CHANNEL = 1
TRAIN_VAL_SPLIT = 0.8
BATCH_SIZE = 8
STRIDE = 12
FILTER_SIZE = 32
NO_CLASSES = 117 #66 + 1   # blank token - số lượng ký tự có trong thư viện từ (số lượng này phải = số ký tự có trong thư viện + 1)
DATA_FOLDER = './image/'
DATA_FOLDER_LABEL = './label/'
LABEL_ENCODER_PATH = 'label_encoder.pkl'


class DataGenerator():
    def __init__(self, train_image_list, val_image_list, batch_size=BATCH_SIZE):
        self.train_image_list = train_image_list
        self.val_image_list = val_image_list
        self.batch_size = batch_size
        self.current_train_index = 0
        self.current_val_index = 0
        self.load_label_encoder()

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

    def load_label_encoder(self):
        self.le = load_label_encoder()

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
        input_label_length = np.zeros((self.batch_size, 1))
        for ind in range(self.batch_size):  
            real_width = image_array[ind].shape[1]
            tmp = [self.le.transform([t])[0] for t in label_array[ind]]
            #print(tmp, ' - ', self.le.inverse_transform(tmp))
            if len(tmp) == 0:
                print('label: ' + label_array[ind])
            real_label_len = len(tmp)
            input_image[ind, :, :real_width, :] = image_array[ind]
            input_true_label[ind, :real_label_len] = tmp
            #print(input_true_label)
            input_time_step[ind] = compute_time_step(real_width) - 2
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
        f = open('./label/'+i, encoding="utf8")
        s = f.read()
        all_character_list += s
    return all_character_list

def create_label_encoder(all_character_list):
    all_character_list = list(set(all_character_list))
    print(all_character_list, len(all_character_list))
    le = LabelEncoder()
    le.fit(all_character_list)
    print(le.transform(['n']))
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(le, f)


def load_label_encoder():
    with open(LABEL_ENCODER_PATH, 'rb') as f:
        le = pickle.load(f)
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

    # Make Networkw
    inputs = Input(name='input_image', shape=input_shape)  # (None, 128, 64, 1)

    # Convolution layer (VGG)
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

    # CNN to RNN
    inner = Reshape(target_shape=((32, 2048)), name='reshape')(inner)  # (None, 32, 2048)
    inner = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense1')(inner)  # (None, 32, 64)

    # RNN layer
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

    # transforms RNN output to character activations:
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