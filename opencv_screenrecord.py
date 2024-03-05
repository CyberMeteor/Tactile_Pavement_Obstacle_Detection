import PySimpleGUI as sg
import cv2
import numpy as np
import torch


def read_camera():
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        if ret:
            yield frame
        else:
            print(" Cannot Read Camera")
            break
def detect():
    i = 0
    # model = torch.hub.load("ultralytics/yolov5","yolov5")
    frames = read_camera()

    for image in frames:
        results = Model(image)

        # img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # im = decodeDisplay(img, camera)

        # cv2.imshow("camera", results)
        cv2.imshow("camera", image)
        key = cv2.waitKey(5)  # press esc to exit
        if key == 27:
            break
        elif key == ord('s'):  # press 's' to save pictures
            cv2.imwrite('/Users/ruiyangchen/PycharmProjects/pythonProject2/ScreenShot/'+str(i)+'.jpg',frame)
            i += 1
        # if cv2.getWindowProperty('camera', cv2.WND_PROP_AUTOSIZE) < 1:
        #     break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect()

