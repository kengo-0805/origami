import cv2 as cv
import numpy as np
import sys
# 白黒に閾値処理するプログラム
def main():
    # ファイルを読み込み
    image_file = 'fig/ipad.png'
    src = cv.imread(image_file, cv.IMREAD_COLOR)
    # イメージを読み込めなかった場合
    if image_file is None:
        sys.exit("File not found.")
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

    # 角検出
    src = 255 - src
    gray = cv.cvtColor(src,cv.COLOR_BGR2GRAY)
    gray= np.float32(gray)

    # Detect corner
    dst = cv.cornerHarris(gray,2,9,0.16)

    # Expand corner
    # dst = cv.dilate(dst, None) 

    # Draw red points
    src[dst>0.01*dst.max()]= [0,0,255]

    # Detect red pixel
    coord = np.where(np.all(src == (0, 0, 255), axis=-1))

    # Print coordinate
    for i in range(len(coord[0])):
        print("X:%s Y:%s"%(coord[1][i],coord[0][i]))

    cv.imshow('imgWithCorner',src)
    cv.waitKey(0)

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
    # search()