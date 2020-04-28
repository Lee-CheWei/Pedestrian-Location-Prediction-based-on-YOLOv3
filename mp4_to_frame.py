import cv2
import os

path = 'C:/Users/user/Desktop/DATA 2.0'

for i in range(39):
    vc = cv2.VideoCapture('{}/test data/test ({}).mp4'.format(path, str(i + 1)))  # 讀入視訊檔案
    os.mkdir('{}/test_frame({})'.format(path, str(i + 1)))
    c = 1

    if vc.isOpened():  # 判斷是否正常開啟
        rval, frame = vc.read()
        print('nice')
    else:
        rval = False
        print('false')

    timeF = 1/2  # 視訊幀計數間隔頻率，根據需要影象的差異性大小調整

    while rval:  # 迴圈讀取視訊幀
        rval, frame = vc.read()
        if (c % timeF == 0):  # 每隔timeF幀進行儲存操作
            cv2.imwrite(path+'/train_frame('+str(i+1)+')/' + str(c) + '.jpg', frame)  # 儲存為影象
            print('save')
        c = c + 1
        cv2.waitKey(1)
    vc.release()