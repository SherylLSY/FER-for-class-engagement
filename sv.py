#convert video to frames and resize

import cv2
import glob
from PIL import Image

cap = cv2.VideoCapture('cut5.mp4')
count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        cv2.imwrite('cut5/frame{:d}.jpg'.format(count), frame)
        count += 150 # i.e. at 150 fps, this advances 5 second
        cap.set(1, count)
    else:
        cap.release()
        break

#Retriving all image names and it's path with .jpg extension from given directory path in imageNames list
imageNames = glob.glob(r"C:\Users\user\Desktop\testing\charlie\cut5\*.jpg")

#Defining width and height of image
new_width  = 1280
new_height = 720

#Count variable to show the progress of image resized
count=0

#Creating for loop to take one image from imageNames list and resize
for i in imageNames:
	#opening image for editing
	img = Image.open(i)
	#using resize() to resize image
	img = img.resize((new_width, new_height), Image.ANTIALIAS)
	#save() to save image at given path and count is the name of image eg. first image name will be 0.jpg
	img.save(r"C:\Users\user\Desktop\testing\charlie\cut5r\\"+str(count)+".jpg") 
	#incrementing count value
	count+=1
	#showing image resize progress
	print("Images Resized " +str(count)+"/"+str(len(imageNames)),end='\r')
