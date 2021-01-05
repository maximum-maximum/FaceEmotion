import numpy as np
import cv2
from PIL import Image, ImageTk
from tensorflow.python.keras.models import load_model
import sys
import PySimpleGUI as sg
import io

face_cascade_path = '/Users/makishima/opt/anaconda3/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
path_list = []
img_list = []
# face_box = []

model = load_model('face_emotion.h5')


def get_img_data(f, face_box, maxsize=(120, 120), first=False):
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


def predict(path, face_box):
  gray = Image.open(path).convert('L').crop(face_box)
  img_resize = gray.resize((48, 48))
  img = np.array(img_resize).reshape(1, 48, 48, 1)

  pre = model.predict(img).reshape(7)
  arg = np.argsort(pre)[::-1]
  pre_result = []
  for i in range(3):
    pre_result.append(str(i + 1) + '.' + str(emotions[arg[i]]) + ': ' + str(pre[arg[i]]))
  return pre_result


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


def show_result(values):
  index = int(values) - 1
  result_img = sg.Image(data=get_img_data(path, face_box[index], first=True))
  pre_result = predict(path, face_box[index])

  result_layout = [[sg.Text('Here are the results.'), sg.Button('OK')],
                   [result_img],
                   [sg.Text(pre_result[0])],
                   [sg.Text(pre_result[1])],
                   [sg.Text(pre_result[2])]]
# ウィンドウの生成
  result_window = sg.Window('Result', result_layout)

# イベントループ
  while True:
    event, values = result_window.read()
    if event == sg.WIN_CLOSED or event == 'OK':
      break

  result_window.close()


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
  elif event == 'Yes':
    values[0] = 1
    show_result(values[0])
    break
  elif event == 'OK':
    print('What you entered：', values[0])  # emotions[int(values[0])]
    show_result(values[0])
    break
candidate_window.close()
