import cv2
import numpy
from shapely.geometry import Polygon
from ultralytics import YOLO

def read_camera(camera):
    while True:
        ret, frame = camera.read()
        if ret:
            yield frame
        else:
            print(" Cannot Read Camera")
            break

def draw_box(image, tuple_list):
    for tup in tuple_list:
        obstacle_class = tup[0]
        intersection_polygon = tup[1]

        # List of coordinates for polygons
        polygon_points_float = numpy.array(intersection_polygon, numpy.float32)

        # Convert floating-point coordinates to integer coordinates
        polygon_points = polygon_points_float.astype(int)

        # Calculate the center position of a polygon
        centroid = numpy.mean(polygon_points, axis=0).astype(int)

        # Draw the mask of the polygon onto the image
        cv2.polylines(image, [polygon_points], isClosed=True, color=(0, 0, 255), thickness=2)  # Red color in the format of (B, G, R)
        cv2.fillPoly(image, [polygon_points], color=255)

        # Calculate the dynamic offset of obstacle_class_name position
        text_offset_x = int(0.1 * (max(polygon_points[:, 0]) - min(polygon_points[:, 0])))
        text_offset_y = int(0.1 * (max(polygon_points[:, 1]) - min(polygon_points[:, 1])))
        text_position = (centroid[0] - text_offset_x, centroid[1] - text_offset_y)

        # Label the name of the polygon diagonally above it
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, obstacle_class, text_position, font, 1, 255, 2, cv2.LINE_AA)

    return

def polygons_intersection(polygon1: numpy.ndarray, polygon2: numpy.ndarray) -> list:
    return Polygon(polygon1).intersection(Polygon(polygon2))


def pred_result_analyze(pred_result: list) -> list:
    """
    Analyze if any tactile pavement intersects with obstacles.

    Parameter:
        predResult (list): The result of the prediction of one frame.

    Returns:
        list: the list of the intersection information in which
              the element of the list is a tuple in the form (obstacle_class, intersection_polygon)
    """
    pred_res = pred_result[0]  # the length of predResult equals one, because only one frame is predicated in a time

    cls_list = pred_res.boxes.cls.cpu().tolist()  # class tensor to list
    cls_list_length = len(cls_list)

    tactile_pavement_index_list = []  # the list of indexes of tactile pavement in boxes and masks

    for i in range(0, cls_list_length):
        if abs(tactile_pavement_cls - cls_list[i]) < 1e-9:
            tactile_pavement_index_list.append(i)

    intersection_information_list = []

    for tactile_pavement_index in tactile_pavement_index_list:  # iterates tactile pavement
        for obstacle_index in range(0, cls_list_length):
            if obstacle_index not in tactile_pavement_index_list:  # iterates obstacles
                intersection = polygons_intersection(pred_res.masks.xy[tactile_pavement_index], pred_res.masks.xy[obstacle_index])
                if intersection:  # not empty
                    intersection_information_list.append((cls_list[obstacle_index], intersection))

    return intersection_information_list

def detect():
    i = 0
    Model = YOLO('/Users/ruiyangchen/PycharmProjects/pythonProject2/train26/weights/best.pt')  # change the path of yolo model
    camera = cv2.VideoCapture(0)
    frames = read_camera(camera)
    for image in frames:
        results = Model(image)

        # Input: seg results  Output:obstacle red box coordinates (cor_list)
        intersection_tuple_list = pred_result_analyze(results)
        print(intersection_tuple_list)

        final_pic = draw_box(image, intersection_tuple_list)

        cv2.imshow("camera", final_pic)
        key = cv2.waitKey(5)  # press esc to exit
        if key == 27:
            break
        elif key == ord('s'):  # press 's' to save pictures
            cv2.imwrite('/Users/ruiyangchen/PycharmProjects/pythonProject2/ScreenShot/'+str(i)+'.jpg',image)
            i += 1
        # if cv2.getWindowProperty('camera', cv2.WND_PROP_AUTOSIZE) < 1:
        #     break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    tactile_pavement_cls = 80
    detect()