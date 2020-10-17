import cv2 as cv
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
    
if __name__ == '__main__':
    main()