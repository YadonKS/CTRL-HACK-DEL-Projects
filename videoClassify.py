import cv2
import numpy as np
from PIL import Image
from keras import models
import os
import tensorflow as tf

model = models.load_model('model.h5')
video = cv2.VideoCapture(0)

while True:
        _, frame = video.read()

        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')

        #Resizing into dimensions you used while training
        im = im.resize((150,150))
        img_array = np.array(im)

        #Expand dimensions to match the 4D Tensor shape.
        img_array = np.expand_dims(img_array, axis=0)

        #Calling the predict function using keras
        prediction = model.predict(img_array)#[0][0]
        print(prediction)
        #Customize this part to your liking...
        if(prediction >= 0.5):
            print("Recyclable")
        else:
            print("Organic")

        cv2.imshow("Prediction", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
video.release()
cv2.destroyAllWindows()




# // This function triggers the file input when the label (camera icon) is clicked
#         document.querySelector('label[for="file-upload"]').addEventListener('click', function() {
#             document.getElementById('file-upload').click();  // Simulate file input click
#         });
#
#         // This function submits the form automatically once a file is selected
#         function submitForm() {
#             const form = document.getElementById('upload-form');
#             form.submit(); // Automatically submit the form after a file is selected
#         }