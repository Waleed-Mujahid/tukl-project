# Some changes added
import cv2
import tensorflow as tf
import sys
from utils import liveFeed, predOnImage

# Loading our trained model
model = tf.keras.models.load_model('my_model')

# Labels for our classes
classes = {0: 'afra', 1: 'asfand', 2: 'Unauthorized Personnel', 3: 'omer', 4: 'saad', 5: 'talha', 6: 'waleed', 7: 'wasay'}

#loading the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read image from command line
if (len(sys.argv) == 1):
    liveFeed(model, face_cascade, classes)
else:
    path = sys.argv[1]
    predOnImage(path, face_cascade, model, classes)
    