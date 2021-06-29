import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import cv2
from PIL import Image
import PIL.ImageOps

X, Y = fetch_openml('mnist_784', version=1, return_X_y=True)
print(pd.Series(Y).value_counts())
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
numberofclasses = len(classes)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    X, Y, test_size=2500, train_size=7500, random_state=9)
Xtrain = Xtrain/255.0
Xtest = Xtest/255.0
classifier = LogisticRegression(
    random_state=9, solver='saga', multi_class='multinomial')
classifier.fit(Xtrain, Ytrain)
prediction = classifier.predict(Xtest)
accuracy = accuracy_score(Ytest, prediction)
print(accuracy)

cam = cv2.VideoCapture(0)

while(True):
  # Capture frame-by-frame
  try:
    ret, frame = cam.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Drawing a box in the center of the video
    height, width = gray.shape
    upper_left = (int(width / 2 - 56), int(height / 2 - 56))
    bottom_right = (int(width / 2 + 56), int(height / 2 + 56))
    cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

    #To only consider the area inside the box for detecting the digit
    #roi = Region Of Interest
    roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

    #Converting cv2 image to pil format
    im_pil = Image.fromarray(roi)

    # convert to grayscale image - 'L' format means each pixel is 
    # represented by a single value from 0 to 255
    image_bw = im_pil.convert('L')
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)

    image_bw_resized_inverted = PIL.ImageOps.invert(image_bw_resized)
    pixel_filter = 20
    min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
    max_pixel = np.max(image_bw_resized_inverted)
    image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
    test_pred = classifier.predict(test_sample)
    print("Predicted class is: ", test_pred)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  except Exception as e:
    pass
cam.release()
cv2.destroyAllWindows()