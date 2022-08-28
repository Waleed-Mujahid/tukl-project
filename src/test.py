import cv2
import tensorflow as tf
from utils import faceRecog

model = tf.keras.models.load_model('content\saved_model\my_model')
classes = {0: 'afra', 1: 'asfand', 2: 'not US', 3: 'omer', 4: 'saad', 5: 'talha', 6: 'waleed', 7: 'wasay'}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while True:  
    ret, img = cap.read()  
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray_img, 1.25, 4) 
  
    for (x,y,w,h) in faces: 
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)  
        rec_gray = gray_img[y:y+h, x:x+w] 
        rec_color = img[y:y+h, x:x+w] 
        face = img[y:y + h, x:x + w]
        faceRecog(face)
  
    cv2.imshow('Face Recognition',img) 

    k = cv2.waitKey(30) & 0xff
    if k == 27: 
        break

  
cap.release() 
cv2.destroyAllWindows()