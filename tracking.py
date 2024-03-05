import cv2
import numpy as np
from pathlib import Path

from boxmot import DeepOCSORT
from ultralytics import YOLO

tracker = DeepOCSORT(
    model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
    device='mps',
    fp16=True,
)

vid = cv2.VideoCapture(0)
color = (0, 0, 255)  # BGR
thickness = 2
fontscale = 0.5

Model = YOLO('obstacle.pt')

while True:
    ret, im = vid.read()

    # substitute by your object detector, input to tracker has to be N X (x, y, x, y, conf, cls)
    # dets = np.array([[144, 212, 578, 480, 0.82, 0],
    #                 [425, 281, 576, 472, 0.56, 65]])
    #
    # tracks = tracker.update(dets, im) # --> (x, y, x, y, id, conf, cls, ind)
    #
    # xyxys = tracks[:, 0:4].astype('int') # float64 to int
    # ids = tracks[:, 4].astype('int') # float64 to int
    # confs = tracks[:, 5]
    # clss = tracks[:, 6].astype('int') # float64 to int
    # inds = tracks[:, 7].astype('int') # float64 to int

    results = Model(im)
    for result in results:
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        cls = result.boxes.cls
        # convert PyTorch to NumPy
        boxes_np = boxes.cpu().numpy()
        confs_np = confs.cpu().numpy()
        cls_np = cls.cpu().numpy()
        detection_results = np.column_stack((boxes_np, confs_np, cls_np))
    tracks = tracker.update(detection_results, im)  # --> (x, y, x, y, id, conf, cls, ind)

    xyxys = tracks[:, 0:4].astype('int') # float64 to int
    ids = tracks[:, 4].astype('int') # float64 to int
    confs = tracks[:, 5]
    clss = tracks[:, 6].astype('int') # float64 to int
    inds = tracks[:, 7].astype('int') # float64 to int


    # in case you have segmentations or poses alongside with your detections you can use
    # the ind variable in order to identify which track is associated to each seg or pose by:
    # segs = segs[inds]
    # poses = poses[inds]
    # you can then zip them together: zip(tracks, poses)


    # print bboxes with their associated id, cls and conf
    if tracks.shape[0] != 0:
        for xyxy, id, conf, cls in zip(xyxys, ids, confs, clss):
        # for boxes, confs, cls in zip(tracks):
            im = cv2.rectangle(
                im,
                (xyxy[0], xyxy[1]),
                (xyxy[2], xyxy[3]),
                # (boxes[0], boxes[1]),
                # (boxes[2], boxes[3]),
                color,
                thickness
            )
            cv2.putText(
                im,
                f'id: {id}, conf: {conf}, c: {cls}',
                (xyxy[0], xyxy[1]-10),
                # f'id: {id}, conf: {confs}, c: {cls}',
                # (boxes[0], boxes[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontscale,
                color,
                thickness
            )

    # show image with bboxes, ids, classes and confidences
    cv2.imshow('frame', im)

    # break on pressing q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()