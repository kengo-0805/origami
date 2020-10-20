import cv2 as cv
import numpy as np
import sys
# 白黒に閾値処理するプログラム
def main():
    # ファイルを読み込み
    image_file = 'fig/target.png'
    src = cv.imread(image_file, cv.IMREAD_COLOR)
    # グレースケール化
    img_gray = cv.cvtColor(src, cv.COLOR_RGB2GRAY)
    # しきい値指定によるフィルタリング
    retval, dst = cv.threshold(img_gray, 127, 255, cv.THRESH_TOZERO_INV )
    cv.imwrite('fig/debug_1.png', dst)
    # 白黒の反転
    dst = cv.bitwise_not(dst)
    cv.imwrite('fig/debug_2.png', dst)
    # 再度フィルタリング
    retval, dst = cv.threshold(dst, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    # 結果を保存
    cv.imwrite('fig/result.png', dst)

    # 画像のサイズを取得
    global height
    global width
    height, width, channels = src.shape[:3]
    print("width: " + str(width))
    print("height: " + str(height))




# 最初に当たる黒い部分を探す関数
def search():
    img = cv.imread('fig/result.png')
    
    for i in range(width):
        for j in range(height):
            pixelValue = img[i, j]
            if (pixelValue == [0,0,0]).all():
                kadoX = i
                kadoY = j
            # print('座標[{},{}] = '.format(i,j) + str(pixelValue))
                print(kadoX,kadoY)
    # else:
        # 格納した角の値をprint

    

    
if __name__ == '__main__':
    main()
    search()