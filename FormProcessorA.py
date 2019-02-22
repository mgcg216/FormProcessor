import numpy as np
import cv2
import tensorflow as tf
from matplotlib.image import imread


def main():

    # Constants
    WIDTH = 480
    HEIGHT = 640
    BOXH = 100 # Box Height
    BOXW = 70  # Box Width

    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    ret = cap.set(3, WIDTH)
    ret = cap.set(4, HEIGHT)

    # Load Model and start session
    model = tf.keras.models.load_model('MnistModel_3.h5')

    # Processes items
    coor_box = [[100, 170, 240, 310, 380], [150, 150, 150, 150, 150]]  # x, y

    def preImg(inp):
        re = cv2.bitwise_not(inp)
        re = re.reshape((1, 28, 28, 1))
        re = re.astype('float32') / 255
        return re

    def horRec(fram, boxes):
        for i in range(len(boxes[0])):
            cv2.rectangle(fram, (boxes[0][i], boxes[1][i]), ((boxes[0][i] + BOXW), (boxes[1][i] + BOXH)), (0, 255, 0), 3)

    def cropper(gry, boxes, borw=False, threzero=False):
        img = []
        eye = []
        for i in range(len(boxes[0])):
            crop = gry[boxes[1][i]:(boxes[1][i] + BOXH), boxes[0][i]:(boxes[0][i] + BOXW)]  # Cropping the  image
            if borw is True:  # Makes the image monochrome
                thresh = 100
                crop = cv2.threshold(crop, thresh, 255, cv2.THRESH_BINARY)[1]
            if threzero is True:
                thresh = 127
                crop = cv2.threshold(crop, thresh, 255, cv2.THRESH_TRUNC)[1]
            resz = (cv2.resize(crop, (input_width, input_height)))  # Resize to 28, 28
            if len(eye) == 0:
                eye = (cv2.resize(resz, (input_width*5, input_height*5)))
            else:
                eye = np.concatenate((eye, (cv2.resize(resz, (input_width*5, input_height*5)))), axis=1)
            img.append(preImg(resz))
        return img, eye

    def pred(img, thre=.9, runs=1):
        out = []
        n = 0
        if len(img) == len(coor_box[0]):
            for i in range(len(img)):
                pred = model.predict(img[i])
                if np.amax(pred > thre):
                    n += 1
                    if n >= runs:
                        out.append(np.argmax(pred))
                else:
                    out.append('_')
        print(out)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        input_height = 28
        input_width = 28

        # Our operations on the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Crop and Make black and white
        result, see = cropper(gray, coor_box, borw=True, threzero=False)

        # Draw Rectangle
        horRec(frame, coor_box)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        cv2.imshow('eye', see)

        # Make Predictions
        pred(result, thre=.9, runs=1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
