import ultralytics
import numpy
from shapely.geometry import Polygon

tactile_pavement_cls = 80

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


