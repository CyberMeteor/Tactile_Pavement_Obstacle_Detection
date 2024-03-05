import numpy as np
from shapely.geometry import Polygon

tactile_pavement_cls = 80
threshold_distance = 10  # Set the threshold distance globally


def polygons_too_close(polygon1: np.ndarray, polygon2: np.ndarray) -> bool:
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
    print(distance)
    return distance < threshold_distance


def pred_result_analyze(pred_result: list) -> list:
    """
    Analyze if any two polygons are too close.

    Parameters:
        pred_result (list): The result of the prediction of one frame.

    Returns:
        list: List of tuples containing information about polygons that are too close.
              Each tuple is in the form (polygon1_index, polygon2_index).
    """
    pred_res = pred_result[0]  # the length of predResult equals one, because only one frame is predicated in a time

    cls_list = pred_res.boxes.cls.cpu().tolist()  # class tensor to list
    cls_list_length = len(cls_list)

    too_close_information_list = []

    for i in range(0, cls_list_length - 1):
        for j in range(i + 1, cls_list_length):
            if abs(tactile_pavement_cls - cls_list[i]) < 1e-9 and abs(tactile_pavement_cls - cls_list[j]) < 1e-9:
                # both polygons are tactile pavement
                continue

            polygon1_index = i
            polygon2_index = j

            polygon1_coords = pred_res.masks.xy[polygon1_index]
            polygon2_coords = pred_res.masks.xy[polygon2_index]

            if polygons_too_close(polygon1_coords, polygon2_coords):
                too_close_information_list.append((polygon1_coords, polygon2_coords))

    return too_close_information_list

