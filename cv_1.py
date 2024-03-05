import cv2
import numpy as np
import torch
from ultralytics import YOLO

def read_camera(camera):
    while True:
        ret, frame = camera.read()
        if ret:
            yield frame
        else:
            print(" Cannot Read Camera")
            break

def draw_box(image, boxes):
    for box in boxes:
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        color = (0, 0, 255)  # Red color in the format of (B, G, R)
        thickness = 2  # Thickness of the box
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    return image

def detect():
    i = 0
    Model = YOLO('obstacle.pt')  # change the path of yolo model
    camera = cv2.VideoCapture(0)
    frames = read_camera(camera)
    for image in frames:
        results = Model(image)

        # input: seg results  output:obstacle red box coordinates (cor_list)
        cor_list = []

        final_pic = image.draw_box(image, cor_list)
        cv2.imshow("camera", final_pic)
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