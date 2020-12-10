from re import X
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.lib.type_check import imag
import pyglet
from realsensecv import RealsenseCapture
from PIL import Image
import os 
import pyglet
import pyglet.gl as gl
import math
import time


# ===============================
# 定数
# ===============================
DATA_DIRNAME = "data"
DATA_DIRPATH = os.path.join(os.path.dirname(__file__), DATA_DIRNAME)
if not os.path.exists(DATA_DIRPATH):
    os.makedirs(DATA_DIRPATH)

CHESS_HNUM = 7  # 水平方向個数
CHESS_VNUM = 10  # 垂直方向個数
CHESS_MARGIN = 50  # [px]
CHESS_BLOCKSIZE = 80  # [px]

BOARD_WIDTH = 0.02  # chessboard の横幅 [m]
BOARD_HEIGHT = 0.01  # chessboard の縦幅 [m]
BOARD_X = 0.  # chessboard の3次元位置X座標 [m]（右手系）
BOARD_Y = 0.  # chessboard の3次元位置Y座標 [m]（右手系）
BOARD_Z = 0.41  # chessboard の3次元位置Z座標 [m]（右手系）

# OpenGL の射影のパラメータ
class Params:
    def __init__(self, zNear=0.0001, zFar=20.0, fovy=21.0):
        self.Z_NEAR = zNear  # 最も近い点 [m]
        self.Z_FAR = zFar  # 最も遠い点 [m]
        self.FOVY = fovy  # 縦の視野角 [deg]


PARAMS = Params(zNear=0.0001,  # [m]
                zFar=20.0,  # [m]
                fovy=21.0  # [deg]
                )

'''
# ストリーム(Color/Depth)の設定
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# ストリーミング開始
pipeline = rs.pipeline()
profile = pipeline.start(config)
'''

# ===============================
# グローバル変数
# ===============================
window = None  # pyglet の Window　クラスのインスタンス
state = None  # アプリの状態を管理する変数（AppState）
cam_w, cam_h = 0, 0  # 画面解像度


class AppState:
    def __init__(self, params):
        self.params = params
        self.pitch = math.radians(0)
        self.yaw = math.radians(0)
        self.translation = np.array([0, 0, 0], np.float32)
        self.distance = 0
        self.lighting = False
        self.zNear = self.params.Z_NEAR



# ===============================
# 画像処理の関数
# ===============================


# def VideoCapture():
#   cap = RealsenseCapture()
#   # プロパティの設定
#   cap.WIDTH = 640
#   cap.HEIGHT = 480
#   cap.FPS = 30
#   # cv2.VideoCapture()と違ってcap.start()を忘れずに
#   cap.start()

#   while True:
#       ret, frames = cap.read()  # frames[0]にRGB、frames[1]にDepthの画像がndarrayが入っている
#       color_frame = frames[0]
#       images = color_frame
#       cv2.imshow('RealSense', images)

#       if cv2.waitKey(1) & 0xFF == ord('q'):
#           break

#   # ストリーミング停止
#   cap.release()
#   cv2.destroyAllWindows()

def changePosition():
  aruco = cv2.aruco
  p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
  img = cv2.imread("pic/picture{:0=3}".format(num)+".png")
  # img = cv2.imread("fig/square_risize.png")
  print("画像を読み込んだ")
  corners, ids, rejectedImgPoints = aruco.detectMarkers(img, p_dict) # 検出

  # 時計回りで左上から順にマーカーの「中心座標」を m に格納
  m = np.empty((4,2)) # 空の行列を作る
  for i,c in zip(ids.ravel(), corners):
    m[i] = c[0].mean(axis=0)

  corners2 = [np.empty((1,4,2))]*4
  for i,c in zip(ids.ravel(), corners):
    corners2[i] = c.copy()
  m[0] = corners2[0][0][2]
  m[1] = corners2[1][0][3]
  m[2] = corners2[2][0][0]
  m[3] = corners2[3][0][1]
  
  width, height = (1600,900) # 変形後画像サイズ
  marker_coordinates = np.float32(m)
  true_coordinates   = np.float32([[0,0],[width,0],[width,height],[0,height]])
  trans_mat = cv2.getPerspectiveTransform(marker_coordinates,true_coordinates)
  img_trans = cv2.warpPerspective(img,trans_mat,(width, height))
  img_trans = cv2.cvtColor(img_trans, cv2.COLOR_BGR2RGB)
  print(m[0],m[1],m[2],m[3])
  cv2.imwrite("fig/img_trans.png",img_trans)
  # plt.imshow(img_trans)
  # plt.show()

def resize():
  # 画像ファイル名指定
  file_name = "fig/img_trans.png"

  # 入力画像の読み込み
  img = Image.open(file_name)

  width, height = img.size

  # 画像の幅を表示
  print('Original Width:', width)

  # 画像の高さを表示
  print('Original Height:', height)

  # 任意のサイズに指定 
  img_resize = img.resize((160, 90))
  width, height = img_resize.size
  print('Resized Width:', width)
  print('Resized Hight:', height)
  img_resize.save('fig/image_risize.png')

def cornerDetect():
  # 画像の読み込み
  image = cv2.imread("fig/image_risize.png")
  # グレイスケール
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
  # エッジ処理
  edge = cv2.Canny(gray,400,20)
  # エッジ処理後の画像を書き込む
  cv2.imwrite("fig/edge_resize.png", edge)
  # 32bit化？
  edge = np.float32(edge)
  # コーナー検出
  dst = cv2.cornerHarris(edge,2,3,0.11)
  # 膨張処理
  dst = cv2.dilate(dst,None)
  # 赤い点をつける
  image[dst>0.01*dst.max()] = [255,0,0]
  # 赤い点の検知
  coord = np.where(np.all(image == (255, 0, 0), axis=-1))
  # 座標の表示
  corner_X = []
  corner_Y = []
  for i in range(len(coord[0])):
      # print("X:%s Y:%s"%(coord[1][i],coord[0][i]))
      original = coord[1][i] + coord[0][i] 
      old = coord[1][i-1] + coord [0][i-1]
      if abs(original - old) > 15:  # XYを足した値が前の値から15以上変化していたら
        corner_X.append(coord[1][i])
        corner_Y.append(coord[0][i])
  for j in range(len(corner_X)):
    print("角X:%s 角Y:%s"%(corner_X[j],corner_Y[j]))
  # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # 保存
  cv2.imwrite("fig/image_corner_resize.png",image)
  return corner_X, corner_Y

  plt.imshow(image)
  plt.title('cornerHarris image')
  plt.show()


def webcam():
  zahyouX, zahyouY = cornerDetect()
  print(zahyouX,zahyouY)


# ===============================
# 描画の関数
# ===============================

# テクスチャの自作
def make_chessboard(num_h, num_v, margin, block_size):
    chessboard = np.ones((block_size * num_v + margin * 2, block_size * num_h + margin * 2, 3), dtype=np.uint8) * 255

    for y in range(num_v):
        for x in range(num_h):
            if (x + y) % 2 == 0:
                sx = x * block_size + margin
                sy = y * block_size + margin
                chessboard[sy:sy + block_size, sx:sx + block_size, 0] = 0
                chessboard[sy:sy + block_size, sx:sx + block_size, 1] = 0
                chessboard[sy:sy + block_size, sx:sx + block_size, 2] = 0
    return chessboard




# 描画する画像の用意
def load_chessboard():
    global texture_ids, chessboard_image

    # chessboard = make_chessboard(CHESS_HNUM, CHESS_VNUM, CHESS_MARGIN, CHESS_BLOCKSIZE)

    # filepath = os.path.join(DATA_DIRPATH, 'fig/ipad.png')
    filepath = ("fig/dotdotdot.png")
    # cv2.imwrite(filepath, chessboard)
    chessboard_image = Image.open(filepath)

    tw, th = chessboard_image.width, chessboard_image.height
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_ids[0])
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, tw, th, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, chessboard_image.tobytes())



#   座標の指定と描画
def board():
    global chessboard_image, texture_ids

    gl.glEnable(gl.GL_TEXTURE_2D)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_ids[0])
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexEnvi(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_REPLACE)

    gl.glMatrixMode(gl.GL_TEXTURE)
    gl.glPushMatrix()
    gl.glLoadIdentity()
    gl.glTranslatef(0.5 / chessboard_image.width, 0.5 / chessboard_image.height, 0)
    # テクスチャの頂点の指定
    gl.glBegin(gl.GL_QUADS)

# 座標たち
    gl.glTexCoord2i(0, 0) # テクスチャ座標（左上）id = 0
    gl.glVertex3f(-0.0670625, -0.136888889, BOARD_Z)
    gl.glTexCoord2i(0, 1) # テクスチャ座標（左下）id = 3
    gl.glVertex3f(-0.0525625, -0.023111111, BOARD_Z)
    gl.glTexCoord2i(1, 1) # テクスチャ座標（右下）id = 2
    gl.glVertex3f(0.0598125, -0.035555556, BOARD_Z)
    gl.glTexCoord2i(1, 0) # テクスチャ座標（右上）id = 1
    gl.glVertex3f(0.0489375, -0.147555556, BOARD_Z)
    gl.glEnd()
    gl.glPopMatrix()


# # 座標たち
#     gl.glTexCoord2i(0, 0) # テクスチャ座標（左上）id = 0
#     gl.glVertex3f((X*0.29/160) - 0.145, -(0.16 - Y*0.16/90), BOARD_Z)
#     gl.glTexCoord2i(0, 1) # テクスチャ座標（左下）id = 3
#     gl.glVertex3f((X*0.29/160) - 0.145, -(0.16 - Y*0.16/90), BOARD_Z)
#     gl.glTexCoord2i(1, 1) # テクスチャ座標（右下）id = 2
#     gl.glVertex3f((X*0.29/160) - 0.145, -(0.16 - Y*0.16/90), BOARD_Z)
#     gl.glTexCoord2i(1, 0) # テクスチャ座標（右上）id = 1
#     gl.glVertex3f((X*0.29/160) - 0.145, -(0.16 - Y*0.16/90), BOARD_Z)
#     gl.glEnd()
#     gl.glPopMatrix()


# 描画の世界を作っている
def on_draw_impl():
    window.clear()

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_LINE_SMOOTH)

    width, height = window.get_size()
    # print(width,height)
    # width = 2560
    # height = 1600
    gl.glViewport(0, 0, width, height)

    # 射影行列の設定
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    aspect = width / float(height * 2)
    bottom = 0
    top = state.zNear * np.tan(np.radians(PARAMS.FOVY))
    left = - top * aspect
    right = top * aspect
    gl.glFrustum(left, right, bottom, top, state.zNear, PARAMS.Z_FAR)  # 視野錐台の大事なやつ

    # 視点の設定
    gl.glMatrixMode(gl.GL_MODELVIEW)
    gl.glLoadIdentity()
    gl.gluLookAt(0.0, 0.0, -0.1,
                 0.0, 0.0, 1.0,
                 0.0, -1.0, 0.0)

    gl.glTranslatef(0, 0, state.distance)
    gl.glRotated(state.pitch, 1, 0, 0)
    gl.glRotated(state.yaw, 0, 1, 0)

    gl.glTranslatef(0, 0, -state.distance)
    gl.glTranslatef(*state.translation)
    # * は分解して渡すことを意味している
    # gl.glTranslatef(*[a,b,c]) は gl.glTranslatef(a,b,c) と同じ

    if state.lighting:
        ldir = [0.5, 0.5, 0.5]  # world-space lighting
        ldir = np.dot(state.rotation, (0, 0, 1))  # MeshLab style lighting
        ldir = list(ldir) + [0]  # w=0, directional light
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (gl.GLfloat * 4)(*ldir))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE,
                     (gl.GLfloat * 3)(1.0, 1.0, 1.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT,
                     (gl.GLfloat * 3)(0.75, 0.75, 0.75))
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_NORMALIZE)
        gl.glEnable(gl.GL_LIGHTING)

    # comment this to get round points with MSAA on
    gl.glEnable(gl.GL_POINT_SPRITE)
    board()
    # line()


# -------------------------------
# ここからがメイン部分（画像処理）
# -------------------------------


# # cap = VideoCapture()
# video_path = 0
# cap = cv2.VideoCapture(video_path)

# num = 0
# global frame
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     if ret == True:
#         cv2.imwrite("pic/picture{:0=3}".format(num)+".png",frame)
#         # print("save picture{:0=3}".format(num)+".png")
#         changePosition()
#         resize()
#         cornerDetect()
#         num += 1
#     else:
#         break
#     time.sleep(3)

# cap.release()




# -------------------------------
# ここからがメイン部分（描画）
# -------------------------------

texture_ids = (pyglet.gl.GLuint * 1) ()
gl.glGenTextures(1, texture_ids)
load_chessboard()

webcam()
# アプリクラスのインスタンス
state = AppState(PARAMS)
platform = pyglet.window.get_platform()
display = platform.get_default_display()
screens = display.get_screens()
window = pyglet.window.Window(
    config=gl.Config(
        double_buffer=True,
        samples=8  # MSAA
    ),
    resizable=True,
    vsync=True,
    fullscreen=True,
    screen=screens[0])

@window.event
def on_draw():
    on_draw_impl()
video_path = 1
cap = cv2.VideoCapture(video_path)
# cap.isOpened()
# ret, frame = cap.read()
pyglet.app.run()

# elapsed_time = time.time() - start
# if @window.event = 1
# print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")



# cv2.destroyAllWindows()