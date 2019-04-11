import cv2
import os

if __name__ == '__main__':
    # 定数定義
    ESC_KEY = 27     # Escキー
    INTERVAL= 33     # 待ち時間
    FRAME_RATE = 30  # fps

    ORG_WINDOW_NAME = "org"
    GAUSSIAN_WINDOW_NAME = "gaussian"

    DEVICE_ID = 0

    # 分類器の指定
    cascade_file = "haarcascade_frontalface_default 2.xml"
    cascade = cv2.CascadeClassifier(cascade_file)

    # カメラ映像取得
    cap = cv2.VideoCapture(DEVICE_ID)

    # 初期フレームの読込
    end_flag, c_frame = cap.read()
    height, width, channels = c_frame.shape

    # ウィンドウの準備
    cv2.namedWindow(ORG_WINDOW_NAME)
    cv2.namedWindow(GAUSSIAN_WINDOW_NAME)

    assert os.path.isfile(cascade_file), 'haarcascade_frontalface_default.xml がない'

    # 変換処理ループ
    while end_flag == True:

        # 画像の取得と顔の検出
        img = c_frame
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_list = cascade.detectMultiScale(img_gray, minSize=(100, 100))

        # 検出した顔に印を付ける
        """
        for (x, y, w, h) in face_list:
            color = (0, 0, 225)
            pen_w = 3
            cv2.rectangle(img_gray, (x, y), (x+w, y+h), color, thickness = pen_w)
        """
        """
        for rect in face_list:
            color = (0, 0, 225)
            pen_w = 3
            cv2.rectangle(img_gray, rect[0:2], rect[1:3], (0,0,255), 3)
            img_face = img_gray[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
            img_face = cv2.resize(img_face, (rect[2]//20, rect[3]//20))
            img_face = cv2.resize(ing_face, (rect[2], rect[3]), cv2.INTER_NEAREST)
        """
        for rect in face_list:
            cv2.rectangle(img_gray, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), (255,255,255),3)
            #顔部分切り出し
            cut_img = img_gray[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]
            #20分の1サイズに縮小
            cut_img = cv2.resize(cut_img,(rect[2]//30, rect[3]//30))
            #元のサイズに戻す cv2.INTER_NEARESTを使わないと滑らかに
            cut_img = cv2.resize(cut_img,(rect[2], rect[3]),cv2.INTER_NEAREST)
            #切り出した画像で元の画像を置き換え
            img_gray[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2]]=cut_img

        # フレーム表示
        cv2.imshow(ORG_WINDOW_NAME, c_frame)
        cv2.imshow(GAUSSIAN_WINDOW_NAME, img_gray)

        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # 次のフレーム読み込み
        end_flag, c_frame = cap.read()

    # 終了処理
    cv2.destroyAllWindows()
    cap.release()
