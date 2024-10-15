from tkinter import *
import numpy as np
from PIL import ImageGrab
from tensorflow import keras
from keras.models import load_model
import win32gui

window = Tk()
window.title("Handwritten digit recognition")
l1 = Label()
m=load_model('model.h5')
def mypro():
	HWND = cv.winfo_id()
	rect = win32gui.GetWindowRect(HWND)
	im = ImageGrab.grab(rect)
	digit = predict_digit(im)
	l1 = Label(window, text="Digit = " + str(digit),font=(20))
	l1.place(x=230, y=420)
def predict_digit(img):
    img = img.resize((28,28))
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(1,28,28,1)
    img = img/255.0
    res = m.predict([img])[0]
    return np.argmax(res)
lastx, lasty = None, None
def clear_widget():
	global cv, l1
	cv.delete("all")
	l1.destroy()

def event_activation(event):
	global lastx, lasty
	cv.bind('<B1-Motion>', draw_lines)
	lastx, lasty = event.x, event.y

def draw_lines(event):
	global lastx, lasty
	x, y = event.x, event.y
	cv.create_line((lastx, lasty, x, y), width=30, fill='white', capstyle=ROUND, smooth=TRUE, splinesteps=12)
	lastx, lasty = x, y

L1 = Label(window, text="Handwritten Digit Recoginition", font=('Algerian', 25), fg="blue")
L1.place(x=35, y=10)

b1 = Button(window, text="Clear", bg="orange", fg="black",font=(25),command=clear_widget)
b1.place(x=120, y=370)

b2 = Button(window, text="Predict", bg="orange", fg="black",font=(25),command=mypro)
b2.place(x=320, y=370)

cv = Canvas(window, width=350, height=290, bg='black')
cv.place(x=120, y=70)

cv.bind('<Button-1>', event_activation)
window.geometry("600x500")
window.mainloop()
