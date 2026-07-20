## Importing needed libraries
from ultralytics import YOLO
import tkinter as tk
from PIL import ImageTk,Image
import cv2

## Function to create a display window after saving image
def show_image(path):
	image_window=tk.Tk()
	img=ImageTk.PhotoImage(Image.open(path))
	panel=tk.Label(image_window,image=img)
	panel.image=img
	panel.pack(side="bottom",fill="both",expand="yes")
	image_window.mainloop()

## Loading the model
model = YOLO("best.pt")

## Selecting image to run prediction on
## Select the path of image present in the the dataset
img_source=input("Enter path to your image:")
#img_source="/path/to/image.jpeg"

results=model(img_source)
processed_img=results[0].plot()

for i in results:
	print("New ",i)
##Save the image as "processed_image.jpg"
save_path="processed_image.jpg"
cv2.imwrite(save_path, processed_img)

##Open the image in a tkinter window
show_image(save_path)
