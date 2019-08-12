#kiểm tra nhận dạng 

from keras.models import load_model, Model
from utils_CNN_2BLSTM_CTC import *
import os
import itertools
from keras import backend as K


MODEL_PATH = './model/xception_model_9.72528.h5'
test_folder = './image/'
model = load_model(MODEL_PATH,
                   custom_objects={'squeeze_layer': squeeze_layer,
                                   'ctc_loss': ctc_loss,
                                   'fake_loss': fake_loss})
input_layer = model.inputs[0]
output_layer = model.layers[-5].output

model_new = Model(input_layer, output_layer)
# print (model_new.summary())

test_image_list = os.listdir(test_folder)
#le = load_label_encoder()
le = load_label_index()

# # Input data generator
def labels_to_text(labels):     # letters의 index -> text (string)
    return ''.join(list(map(lambda x: le[int(labels)], labels))) #lay ky tu cua tung vi tri trong CHAR_VECTOR

def text_to_labels(text):      # text를 letters 배열에서의 인덱스 값으로 변환
    return list(map(lambda x: le.index(x), text)) #lấy vị trí của từng ký tự trong CHAR_VECTOR 

for t in test_image_list:
    image = cv2.imread(os.path.join(test_folder, t), 0)    
    image = cv2.resize(image, (None, 64))     
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image / 255., axis=-1)
    result = model_new.predict(image)
    result = np.squeeze(result)[2:, :]
    result = np.argmax(result, 1)

    result = [k for k, _ in itertools.groupby(result) if k != NO_CLASSES-1]
    print(result)
    ketqua = ""
    for i in result:
        #print(str(i))
        ketqua += labels_to_text(str(i))
    ketqua = ''.join(ketqua)
        #print(ketqua)
    #result = le.inverse_transform(result)
    #result = [labels_to_text(t) for t in result]
    #result = ''.join(result)

    image_path = './label/'+t.split('/')[-1].split('.png')[0]+'.txt'
    f = open(image_path, encoding="utf8")
    s = f.read()
    
    #print ("True label:", t.split('_')[0])
    print ("True label:", s)
    print ("Predicted :", ketqua) #result[:len(s)])
    print ('_'*150)
