import pytesseract as tess
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

tess.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
img=cv2.imread('cod.png')
fix_img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img=fix_img[410:550,0:300]
#cv2.rectangle(img,pt1=(0,410),pt2=(300,550),color=(0,255,0),thickness=5)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh1 = cv2.threshold(gray, 0, 100, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1) 

contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

im2 = img.copy()

file = open("recognized.txt", "w+") 
file.write("") 
file.close()

text = tess.image_to_string(im2,lang="eng") 
print(text)

for cnt in contours: 
    x, y, w, h = cv2.boundingRect(cnt) 
      
    # Drawing a rectangle on copied image 
   # rect = cv2.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2) 
      
    # Cropping the text block for giving input to OCR 
    cropped = im2 
      
    # Open the file in append mode 
    file = open("recognized.txt", "a") 
      
    # Apply OCR on the cropped image 
    text = tess.image_to_string(cropped,lang="eng") 
      
    # Appending the text into file 
    file.write(text) 
    print(text)
    file.write("\n") 
      
    # Close the file 
    file.close 

# img=Image.open('sample.jpg')

# text=tess.image_to_string(img)
# print(text)