import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tensorflow.python.keras.models import load_model
import sys
import sys
import PySimpleGUI as sg
import io
import os

face_cascade_path = '/Users/makishima/opt/anaconda3/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

path_list = []
img_list = []


def get_img_data(f, face_box, maxsize=(100, 120), first=False):
  """Generate image data using PIL
  """
  # print("open file:", f)
  # print("face_box.shape:" + str(face_box.shape))
  img = Image.open(f)
  img = img.crop(face_box)
  img.thumbnail(maxsize)
  if first:  # tkinter is inactive the first time
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()
  return ImageTk.PhotoImage(img)


def get_face_position(img_path):
  img = cv2.imread(img_path)
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  face_box = face_cascade.detectMultiScale(img_gray)
  # face_box = face_cascade.detectMultiScale(img_gray, scaleFactor=1.05, minNeighbors=3, minSize=(120, 120))
  # print(face_box)
  for i in range(face_box.shape[0]):
    face_box[i][2] += face_box[i][0]
    face_box[i][3] += face_box[i][1]
  return face_box


for img_path in sys.argv:
  if img_path == 'test.py':
    continue
  path_list.append(img_path)


for path in path_list:
  face_box = get_face_position(path)
  print(face_box)
  for i in range(face_box.shape[0]):
    img_list.append(sg.Image(data=get_img_data(path, face_box[i], first=True)))


if len(img_list) > 1:
  candidate_layout = [[sg.Text('Please enter a number between 1 and ' + str(face_box.shape[0]) + '.'), sg.InputText()],
                      [sg.Button('OK'), sg.Button('Cancel')],
                      img_list]
else:
  candidate_layout = [[sg.Text('Is this the face?')],
                      [sg.Button('Yes'), sg.Button('No')],
                      img_list]


candidate_window = sg.Window('Choose the face', candidate_layout, return_keyboard_events=True, use_default_focus=False)
# loop reading the user input and displaying image, filename
while True:
  event, values = candidate_window.read()
  if event == sg.WIN_CLOSED or event == 'Cancel' or event == 'No':
    break
  elif event == 'OK' or event == 'Yes':
    print('What you entered：', values[0])  # emotions[int(values[0])]

candidate_window.close()


result_layout = [[sg.Text('ここは1行目')],
                 [sg.Text('ここは2行目：適当に文字を入力してください'), sg.InputText()],
                 [sg.Button('OK'), sg.Button('キャンセル')]]

# ウィンドウの生成
result_window = sg.Window('サンプルプログラム', result_layout)

# イベントループ
while True:
  event, values = result_window.read()
  if event == sg.WIN_CLOSED or event == 'キャンセル':
    break
  elif event == 'OK':
    print('あなたが入力した値： ', values[0])  # emotions[int(values[0])]

result_window.close()
