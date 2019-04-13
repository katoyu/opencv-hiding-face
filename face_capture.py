import cv2
import os

if __name__ == '__main__':
	#定数定義
    ESC = 27
    INTERVAL = 33
    FRAME_RATE = 30

    ORG_WINDOW_NAME = "org"
    GAUSSIAN_WINDOW_NAME = "gaussian"

    DEVICE_ID = 0

    #分類器指定
    cascade_file = "haarcascade_frontalface_default 2.xml"
    cascade = cv2.CascadeClassifier(cascade_file)

    #カメラ画像取得
    cap = cv2.VideoCapture(DEVICE_ID)

    #初期フレーム読み込み
    end_flag, c_frame = cap.read()
    H, W, C = c_frame.shape

    #ウィンドウ準備
    cv2.namedWindow(ORG_WINDOW_NAME)
    cv2.namedWindow(GAUSSIAN_WINDOW_NAME)

    #変換処理ループ
    while end_flag == True:
        #画像取得+顔検出
        img = c_frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_list = cascade.detectMultiScale(img_gray, minSize=(100, 100))
        #検出した顔を矩形で囲う
        for (x, y, w, h) in face_list:
            color =(0, 0, 255)
            pen_w = 3
            cv2.rectangle(img_gray, (x, y), (x+w, h+y), color, thickness= pen_w)
        #フレーム表示
        cv2.imshow(ORG_WINDOW_NAME, c_frame)
        cv2.imshow(GAUSSIAN_WINDOW_NAME, img_gray)
        #Escで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC:
            break
        #次のフレーム読み込み
        end_flag, c_frame = cap.read()

    #終了処理
    cv2.destroyAllWindows()
    cap.release()