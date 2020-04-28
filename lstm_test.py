import keras
from keras.models import load_model
import numpy as np
import pandas as pd
import os
import cv2

model = load_model('tracking_model.h5')

means = pd.Series([883, 265, 194, 462], dtype=float)
max_min = pd.Series([1802, 456, 675, 942], dtype=float)
X_testO = []

for data in os.listdir('s_test'):
    Data = []
    Data = pd.read_csv('s_test/' + data)
    Data = Data.values.tolist()

    for i in range(len(Data)):
        X_testO.append((Data[i]))

X_testO = pd.DataFrame(X_testO)
X_test = X_testO.apply(lambda x: (x - means) / max_min, axis=1)
X_test = np.array(X_test).reshape(len(X_test), 1, 4)
print(len(X_test))
predict = model.predict(X_test)
predict = np.array(predict).reshape(1482, 4)
predict = pd.DataFrame(predict)
predict = predict.apply(lambda x: x * max_min + means, axis=1)

path = 'C:/Users/user/Desktop/DATA/img_bb_test'
line = 0

for i in os.listdir(path):
    for j in range(len(os.listdir(path + '/' + i))):
        image = cv2.imread(path + '/' + i + '/' + str(j + 1) + '.jpg')
        # if j != 0:
        image = cv2.rectangle(image, (int(predict.values[line, 0]), int(predict.values[line, 1])),
                              (int(predict.values[line, 0] + predict.values[line, 2]),
                              int(predict.values[line, 1] + predict.values[line, 3])), (255, 0, 0), 4)
        cv2.imwrite('C:/Users/user/Desktop/DATA/YOLO_LSTM/'+str(i)+str(j+1)+'.jpg', image)
        cv2.imshow('img', image)
        cv2.waitKey(10)
        line += 1
