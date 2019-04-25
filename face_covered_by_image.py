import cv2
import os
import numpy as np
from PIL import Image


class CvOverlayImage(object):
    """
    [summary]
      OpenCV形式の画像に指定画像を重ねる
    """

    def __init__(self):
        pass

    @classmethod
    def overlay(
            cls,
            cv2_background_image,
            cv2_overlay_image,
            point,
    ):
        """
        [summary]
          OpenCV形式の画像に指定画像を重ねる
        Parameters
        ----------
        cv2_background_image : [OpenCV Image]
        cv2_overlay_image : [OpenCV Image]
        point : [(x, y)]
        Returns : [OpenCV Image]
        """
        overlay_height, overlay_width = cv2_overlay_image.shape[:2]

        # OpenCV形式の画像をPIL形式に変換(α値含む)
        # 背景画像
        cv2_rgb_bg_image = cv2.cvtColor(cv2_background_image, cv2.COLOR_BGR2RGB)
        pil_rgb_bg_image = Image.fromarray(cv2_rgb_bg_image)
        pil_rgba_bg_image = pil_rgb_bg_image.convert('RGBA')
        # オーバーレイ画像
        cv2_rgb_ol_image = cv2.cvtColor(cv2_overlay_image, cv2.COLOR_BGRA2RGBA)
        pil_rgb_ol_image = Image.fromarray(cv2_rgb_ol_image)
        pil_rgba_ol_image = pil_rgb_ol_image.convert('RGBA')

        # composite()は同サイズ画像同士が必須のため、合成用画像を用意
        pil_rgba_bg_temp = Image.new('RGBA', pil_rgba_bg_image.size,(255, 255, 255, 0))
        # 座標を指定し重ね合わせる
        pil_rgba_bg_temp.paste(pil_rgba_ol_image, point, pil_rgba_ol_image)
        result_image = Image.alpha_composite(pil_rgba_bg_image, pil_rgba_bg_temp)

        # OpenCV形式画像へ変換
        cv2_bgr_result_image = cv2.cvtColor(
            np.asarray(result_image), cv2.COLOR_RGBA2BGRA)

        return cv2_bgr_result_image  


if __name__ == '__main__':
    # 定数定義
    ESC_KEY = 27     # Escキー
    INTERVAL= 33     # 待ち時間
    FRAME_RATE = 30  # fps

    ORG_WINDOW_NAME = "org"
    GAUSSIAN_WINDOW_NAME = "gaussian"

    DEVICE_ID = 0
    PIC = cv2.imread("image1.png")

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

        # フレーム表示
        for rect in face_list:
            color = (0, 0, 225)
            pen_w = 3
            #cv2.rectangle(img_gray, tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]), (0,0,0), 3)

            cv2_background_image = img_gray
#           cv2_overlay_image = cv2.imread("image1.png", cv2.IMREAD_UNCHANGED)  # IMREAD_UNCHANGEDを指定しα込みで読み込む　#透過なし
            cv2_overlay_image = cv2.imread("book.png", cv2.IMREAD_UNCHANGED)   #透過あり
            cv2_overlay_image = cv2.resize(cv2_overlay_image, (rect[2],rect[3]))

            point = (rect[0], rect[1])
            img_gray = CvOverlayImage.overlay(cv2_background_image, cv2_overlay_image, point)

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
