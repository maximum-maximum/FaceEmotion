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

box_list = []
img_list = []


def get_img_data(f, maxsize=(600, 450), first=False):
  """Generate image data using PIL
  """
  print("open file:", f)
  img = Image.open(f)
  img.thumbnail(maxsize)
  if first:  # tkinter is inactive the first time
    bio = io.BytesIO()
    img.save(bio, format="PNG")
    del img
    return bio.getvalue()
  return ImageTk.PhotoImage(img)


def get_face_position(img_path):
  img = cv2.imread(img_path)
  print(img_path)
#     plt.imshow(img, cmap='gray')
  img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # face_box = face_cascade.detectMultiScale(img_gray)
  face_box = face_cascade.detectMultiScale(img_gray, scaleFactor=1.05, minNeighbors=3, minSize=(150, 150))

  for i in range(face_box.shape[0]):
    face_box[i][2] += face_box[i][0]
    face_box[i][3] += face_box[i][1]
  return face_box


for img_path in sys.argv:
  if img_path == 'test.py':
    continue
  box_list.append(img_path)
  get_face_position(img_path)


for box in box_list:
  img_list.append(sg.Image(data=get_img_data(box, first=True)))


# layout = [col_read_file, img_list]
layout = [[sg.Text('Please enter a number.'), sg.InputText()],
          [sg.Button('OK'), sg.Button('Cancel')],
          img_list]

window = sg.Window('Choose face', layout, return_keyboard_events=True, use_default_focus=False)

# loop reading the user input and displaying image, filename
while True:
  event, values = window.read()
  if event == sg.WIN_CLOSED or event == 'Cancel':
    break
  elif event == 'OK':
    print('What you entered：', values[0])  # emotions[int(values[0])]

window.close()
