from OpenGL.GL import *
from OpenGL.GLUT import *




def resize(w, h):
    glViewport(0, 0, w, h)
    glLoadIdentity()
    glOrtho(-0.5, float(w)-0.5, float(h)-0.5, -0.5, -1.0, 1.0)

def init():
    glClearColor(1.0, 1.0, 1.0, 1.0)

def line():
  # glClear(GL_COLOR_BUFFER_BIT)
  glLineWidth(5)
  glBegin(GL_LINES)                           
  #  線分の描画
  glVertex2f(50, 50)
  glVertex2f(100, 100)
  glVertex2f(50,50)
  glVertex2f(100,50)
  glEnd()
  glFlush()


glutInitWindowPosition(100, 100)
glutInitWindowSize(320, 240)
glutInit(sys.argv)
glutInitDisplayMode(GLUT_RGBA)
glutCreateWindow("sample")

glutReshapeFunc(resize)
# glutMouseFunc(mouse)

init()
glutDisplayFunc(line)

glutMainLoop()