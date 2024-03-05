threshold = 0.8

def rectangle_diagonal_to_vertices(diagnal_coor_list: list) -> list:
    x1 = diagnal_coor_list[0][0]
    y1 = diagnal_coor_list[0][1]
    x2 = diagnal_coor_list[1][0]
    y2 = diagnal_coor_list[1][1]
    midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)
    delta_x = abs(x1 - x2) / 2
    delta_y = abs(y1 - y2) / 2
    vertex1 = (midpoint[0] - delta_x, midpoint[1] - delta_y)
    vertex2 = (midpoint[0] + delta_x, midpoint[1] - delta_y)
    vertex3 = (midpoint[0] + delta_x, midpoint[1] + delta_y)
    vertex4 = (midpoint[0] - delta_x, midpoint[1] + delta_y)
    return [vertex1, vertex2, vertex3, vertex4]


def boxes_masks(pred_result: list) -> list:
    """
    Parameter:
        predResult (list): The result of the prediction of one frame.

    Return a list with 2 elements.
    The first element is a list ((class, coordinates_list), ..., (class, coordinates_list)) of boxes.
    The second element is a list (coordinates_list, ..., coordinates_list) of masks.
    """
    res = [[], []]
    pred_res = pred_result[0]  # the length of predResult equals one, because only one frame is predicated in a time
    cls_list = pred_res.boxes.cls.cpu().tolist()  # class tensor to list
    cls_list_length = len(cls_list)
    conf_list = pred_res.boxes.conf.cpu().tolist()

    for i in range(0, cls_list_length):
        if conf_list[i] > threshold:
            res[0].append([cls_list[i], rectangle_diagonal_to_vertices(pred_res.boxes.xyxy[i])])
            res[1].append(pred_res.masks.xy[i])
    return res