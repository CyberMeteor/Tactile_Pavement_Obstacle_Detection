import cv2
import numpy
import time
from shapely.geometry import Polygon
from ultralytics import YOLO
import pyttsx3 as pt3

def read_camera(camera):
    while True:
        ret, frame = camera.read()
        if ret:
            yield frame
        else:
            print(" Cannot Read Camera")
            break

def draw_all_box(image, clsIdNumber, diagonal_boxes):
    for i in range(len(clsIdNumber)):
        diag = diagonal_boxes[i]
        p1 = diag[0]
        p2 = diag[1]
        # 0:person 1:bicycle 2:car 80:tactile pavement  81:chair 82:bin
        COLORS = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 255), 80: (255, 0, 0), 81: (0, 255, 255), 82: (255, 255, 0)}
        color = COLORS.get(clsIdNumber[i], (255, 0, 0))  # Red color in the format of (B, G, R)
        thickness = 2  # Thickness of the box
        cv2.rectangle(image, p1, p2, color, thickness)

        # draw a bounding box rectangle and label on the image
        classId = Model.names.get(clsIdNumber[i])
        # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        text_position = (p1[0], p1[1] - 10)
        cv2.putText(image, classId, text_position, cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)

        # put the class name
        # class_name = box[0]  # boxclass.classname?
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # text_position = (p1[0], p1[1]-10)
        # cv2.putText(image, class_name, text_position, font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return


def draw_boxes_masks(image, prediction_result):
    # draw the boxes of all identified objects
    draw_all_box(image, prediction_result[0], prediction_result[1])

    # predict the intersection of obstacle and pavement
    pred_result_tuple_list = pred_result_analyze(prediction_result[0], prediction_result[2])

    zeros_1 = numpy.zeros((image.shape), dtype=numpy.uint8)  # transparent mask
    # zeros_2 = numpy.zeros((image.shape), dtype=numpy.uint8)
    # draw the mask of intersection
    for tup in pred_result_tuple_list:
        # obstacle_class = tup[0]

        polygonCoords_1 = tup[0]
        polygonCoords_2 = tup[1]
        cls_name = tup[2]

        # List of coordinates for polygons
        polygon_points_float_1 = numpy.array(polygonCoords_1, numpy.float32)
        polygon_points_float_2 = numpy.array(polygonCoords_2, numpy.float32)

        # Convert floating-point coordinates to integer coordinates
        polygon_points_1 = polygon_points_float_1.astype(int)
        polygon_points_2 = polygon_points_float_2.astype(int)

        # Calculate the center position of a polygon
        # centroid = numpy.mean(polygon_points, axis=0).astype(int)

        # Draw the mask of the polygon onto the image
        cv2.polylines(zeros_1, [polygon_points_1], isClosed=True, color=(255, 165, 0), thickness=2)  # Orange color in the format of (R, G, B)
        cv2.polylines(zeros_1, [polygon_points_2], isClosed=True, color=(255, 0, 165),
                      thickness=2)
        cv2.fillPoly(zeros_1, [polygon_points_1], color=(255, 165, 0))
        cv2.fillPoly(zeros_1, [polygon_points_2], color=(255, 0, 165))

        alpha = 0.2
        image = cv2.addWeighted(image, 1 - alpha, zeros_1, alpha, 0)


        # Calculate the dynamic offset of obstacle_class_name position
        # text_offset_x = int(0.1 * (max(polygon_points[:, 0]) - min(polygon_points[:, 0])))
        # text_offset_y = int(0.1 * (max(polygon_points[:, 1]) - min(polygon_points[:, 1])))
        # text_position = (centroid[0] - text_offset_x, centroid[1] - text_offset_y)

        # Label the name of the polygon diagonally above it
        # font = cv2.FONT_HERSHEY_SIMPLEX
        # cv2.putText(image, obstacle_class, text_position, font, 1, 255, 2, cv2.LINE_AA)

    return image

def rectangle_diagonal_to_vertices(diagnal_coor_tensor_list: list) -> list:
    diagnal_coor_list = diagnal_coor_tensor_list.cpu().tolist()
    x1 = diagnal_coor_list[0]
    y1 = diagnal_coor_list[1]
    x2 = diagnal_coor_list[2]
    y2 = diagnal_coor_list[3]
    midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
    delta_x = abs(x1 - x2) / 2
    delta_y = abs(y1 - y2) / 2
    vertex1 = (int(midpoint[0] - delta_x), int(midpoint[1] - delta_y))
    vertex2 = (midpoint[0] + delta_x, midpoint[1] - delta_y)
    vertex3 = (int(midpoint[0] + delta_x), int(midpoint[1] + delta_y))
    vertex4 = (int(midpoint[0] - delta_x), int(midpoint[1] + delta_y))
    # return [vertex1, vertex2, vertex3, vertex4]
    print("vertex:", [vertex1, vertex3])
    return [vertex1, vertex3]

def boxes_masks(pred_result: list) -> list:
    """
    Parameter:
        predResult (list): The result of the prediction of one frame.

    Return a list with 2 elements.
    The first element is a list ((class, coordinates_list), ..., (class, coordinates_list)) of boxes.
    The second element is a list (coordinates_list, ..., coordinates_list) of masks.
    """
    res = [[], [], [], []]   # first list is box class, second list is box xyxy, third list is mask, fourth list is obstacle box xyxy
    pred_res = pred_result[0]  # the length of predResult equals one, because only one frame is predicated in a time
    cls_list = pred_res.boxes.cls.cpu().tolist()  # class tensor to list
    cls_list_length = len(cls_list)
    conf_list = pred_res.boxes.conf.cpu().tolist()

    for i in range(0, cls_list_length):
        if conf_list[i] > threshold:
            # res[0].append([cls_list[i], rectangle_diagonal_to_vertices(pred_res.boxes.xyxy[i])])
            res[0].append(cls_list[i])
            res[1].append(rectangle_diagonal_to_vertices(pred_res.boxes.xyxy[i]))  # return two diagonal point [[x1,y1],[x2,y2]]
            res[2].append(pred_res.masks.xy[i])
            if abs(cls_list[i] - 80.0) > 1e-9:
                res[3].append(rectangle_diagonal_to_vertices(pred_res.boxes.xyxy[i]))
    return res

def polygons_intersection(polygon1: numpy.ndarray, polygon2: numpy.ndarray) -> list:
    polygon1_geo = Polygon(polygon1).buffer(0)  # TP
    polygon2_geo = Polygon(polygon2).buffer(0)  # Obstacle
    intersection_polygon_geo = polygon1_geo.intersection(polygon2_geo)
    xx, yy = intersection_polygon_geo.exterior.coords.xy
    intersection_polygon_list = []
    xx_len = len(xx)
    for i in range(0, xx_len):
        intersection_polygon_list.append(list([xx[i], yy[i]]))
    print("intersection coordinates:", intersection_polygon_list)
    return intersection_polygon_list

def polygons_too_close(polygon1, polygon2):
    """
    Analyze if two polygons are too close.

    Parameters:
        polygon1 (numpy.ndarray): Coordinates of the vertices of the first polygon.
        polygon2 (numpy.ndarray): Coordinates of the vertices of the second polygon.

    Returns:
        bool: True if the two polygons are too close, False otherwise.
    """
    polygon1_geo = Polygon(polygon1).buffer(0)
    polygon2_geo = Polygon(polygon2).buffer(0)
    distance = polygon1_geo.distance(polygon2_geo)
    print("distance:", distance)
    return distance < threshold_distance

def pred_result_analyze(cls_list, predict_mask):
    """
    Analyze if any two polygons are too close.

    Parameters:
        pred_result (list): The result of the prediction of one frame.

    Returns:
        list: List of tuples containing information about polygons that are too close.
              Each tuple is in the form (polygon1_index, polygon2_index).
    """
    # pred_res = pred_result[0]  # the length of predResult equals one, because only one frame is predicated in a time

    # cls_list = pred_res.boxes.cls.cpu().tolist()  # class tensor to list
    cls_list_length = len(cls_list)

    tactile_pavement_index_list = []

    for i in range(0, cls_list_length):
        if abs(tactile_pavement_cls - cls_list[i]) < 1e-9:
            tactile_pavement_index_list.append(i)

    too_close_information_list = []

    for tactile_pavement_index in tactile_pavement_index_list:
        for obstacle_index in range(0, cls_list_length):
            if obstacle_index not in tactile_pavement_index_list:

                polygon1_coords = predict_mask[tactile_pavement_index]
                polygon2_coords = predict_mask[obstacle_index]

                if polygons_too_close(polygon1_coords, polygon2_coords):
                    too_close_information_list.append((polygon1_coords, polygon2_coords, cls_list[obstacle_index]))

    return too_close_information_list

def play_audio(message):
    voice_engine.setProperty('rate', 300)  # Set voice speed
    voice_engine.setProperty('volume', 1)  # Set voice volume
    voice_engine.say(message)  # Convert text to speech
    voice_engine.runAndWait()
    return

def detect():
    i = 0
    last_detection_time = 0
    # Model = YOLO('/Users/ruiyangchen/PycharmProjects/pythonProject2/train26/weights/best.pt')  # change the path of yolo model
    camera = cv2.VideoCapture(0)
    # frames = read_camera(camera)
    print(Model.names.get(82))
    while (True):
        ret, image = camera.read()
        height, width = image.shape[:2]
        model_results = Model(image)

        # Input: seg results  Output:obstacle red box coordinates (cor_list)
        # intersection_tuple_list = pred_result_analyze(model_results)
        # print(intersection_tuple_list)

        # transform the model results into boxes, coordinates and masks
        trans_results = boxes_masks(model_results)

        # Draw the boxes and masks, including analyse the intersection
        final_image = draw_boxes_masks(image, trans_results)
        cv2.imshow("image", final_image)

        # voice prompt
        if len(trans_results[3]) != 0:
            center_x_values = []
            for cor_1, cor_2 in trans_results[3]:
                x1 = cor_1[0]
                x2 = cor_2[0]
                center_x_values.append((x1 + x2) // 2)

            # Determine which side the obstacle is on
            if all(x < width // 2 for x in center_x_values):
                message = "Obstacle on your left side"
            elif all(x > width // 2 for x in center_x_values):
                message = "Obstacle on your right side"
            else:
                message = "Obstacles on both your left and right sides"

            # Determine whether to trigger voice prompts
            current_time = time.time()
            time_difference = current_time - last_detection_time

            if time_difference > 10:
                play_audio(message)
                last_detection_time = current_time

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
    threshold = 0.4
    threshold_distance = 10  # Set the threshold distance globally
    Model = YOLO('/Users/ruiyangchen/PycharmProjects/pythonProject2/best.pt')  # change the path of yolo model
    voice_engine = pt3.init()
    detect()