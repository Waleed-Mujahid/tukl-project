import time
import tensorflow as tf
import cv2

def faceRecog(faces, labels, model):
    faces = tf.image.resize(faces, size = (160,160))
    faces = tf.expand_dims(faces/255., axis = 0)

    # Making prediction and getting result
    preds = model.predict(faces)
    label = preds.argmax()
    probs = preds[0][label]

    # if probs > 0.5:
    print(f"Person detected: {labels[label]} \nConfidence level: {probs*100:.2f}")
    # else :
    #     print(f"No Authorized induvidual detected in this image. \nConfidence level: {probs*100:.2f}")

    time.sleep(1.5)

def liveFeed(model, face_cascade, labels) :
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
            faceRecog(face, labels, model)
    
        cv2.imshow('Face Recognition',img) 

        k = cv2.waitKey(30) & 0xff
        if k == 27: 
            break

    cap.release() 
    cv2.destroyAllWindows()

def predOnImage(path, face_cascade, model, labels):
    img = cv2.imread(path)
    # first we use OpenCV to get the cropped image of the face
    # Convert into grayscale for face detection using face detecion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.25, 4)
    
    # Crop face from image
    for (x, y, w, h) in faces:
        # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        faces = img[y:y + h, x:x + w]

    # Making prediction and getting result
    faceRecog(faces, labels, model)