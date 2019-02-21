import numpy as np
import cv2
import tensorflow as tf
from matplotlib.image import imread


# Constants
WIDTH = 480
HEIGHT = 640
BOX = 28*2
img_name = "temp.jpg"

# Initialize Webcam
cap = cv2.VideoCapture(0)
ret = cap.set(3, WIDTH)
ret = cap.set(4, HEIGHT)


# Load Model and start session
model = tf.keras.models.load_model('MnistModel_1.h5')
sess = tf.Session()

def inv(image):
    image = (255-image)
    return image


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    input_height = 28
    input_width = 28

    # Our operations on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Crop and Make black and white
    crop = gray[WIDTH//2:WIDTH//2+BOX, HEIGHT//2:HEIGHT//2+BOX]
    thresh = 127
    bw = cv2.threshold(crop, thresh, 255, cv2.THRESH_BINARY)[1]
    save = cv2.resize(bw, (input_width, input_height))

    # Draw Rectangle
    show = cv2.rectangle(frame, ((HEIGHT//2), (WIDTH//2)), ((HEIGHT//2+BOX), (WIDTH//2+BOX)), (0, 255, 0), 3)

    # Preprocess the image
    result = inv(save)
    result = result.reshape((1, 28, 28, 1))
    result = result.astype('float32') / 255

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Make Predictions
    predictions = model.predict(result)
    # print(np.amax(predictions))
    if np.amax(predictions) > .7:
        print(np.argmax(predictions))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

