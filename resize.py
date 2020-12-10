from PIL import Image
# 画像ファイル名指定
file_name = "fig/dot.png"

# 入力画像の読み込み
img = Image.open(file_name)

width, height = img.size

# 画像の幅を表示
print('width:', width)

# 画像の高さを表示
print('height:',height)

img_resize = img.resize((660, 900))
img_resize.save('fig/dot_risize.png')
