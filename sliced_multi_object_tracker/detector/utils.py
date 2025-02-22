import os,sys,shutil

from pathlib import Path
import numpy as np
from typing import List, Dict
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

from sliced_multi_object_tracker.core import *

def cxywh2xyxy(bboxes:np.ndarray):
    if bboxes.shape[0] == 0:
        return bboxes.copy()
    else:
        xyxy = np.zeros(bboxes.shape)
        xyxy[:, 0] = bboxes[:, 0] - bboxes[:, 2] / 2
        xyxy[:, 1] = bboxes[:, 1] - bboxes[:, 3] / 2
        xyxy[:, 2] = bboxes[:, 0] + bboxes[:, 2] / 2
        xyxy[:, 3] = bboxes[:, 1] + bboxes[:, 3] / 2
        return xyxy

def download_labels(path:Path):
    labels: Dict[str, List[np.ndarray]] = {}
    for p in path.glob("*.txt"):
        if os.path.getsize(p) != 0:
            array = np.loadtxt(p, delimiter=" ")
            if array.ndim == 1:
                array = array.reshape(-1, 5) # cls, x, y, w, h
            gt_cls, gt_bbox = array[:, 0].reshape(-1, 1), cxywh2xyxy(array[:, 1:])
            labels[p.stem] = np.hstack((gt_bbox, gt_cls))
        else:
            labels[p.stem] = np.empty((0, 5))
    return labels

def true_positive_detections(detections:np.ndarray, targets:np.ndarray, min_iou:float, conf_thres:float, nc=1):
    ''''
    Arguments
    ---------
    detections: [N, 6]: x1n, y1n, x2n, y2n, conf, cls
    targets: [N, 5]: x1n, y1n, x2n, y2n, cls
    
    Returns
    -------
    cm_detections: Dict(key: tp, fp, fn, value: ndarray[-1 ,4])
    tp: ndarray [N',]: 0 or 1
    conf: ndarray [N',]
    pred_cls: ndarray [N', ] (0 or 1 or ... or nc-1)
    '''
    cm_detections = {
        key: [] for key in ['tp', 'fp', 'fn']
    }
    if detections.shape[0] == 0:
        cm_detections = {
            key: np.array(value) if value != [] else np.empty((0, 4)) for key, value in cm_detections.items()
        }
        return cm_detections, np.empty((0,nc)), np.empty((0,)), np.empty((0,)), np.zeros((4,))
    
    # thresholding
    detections = detections[detections[:, -2] > conf_thres]
    
    # get class candidates
    cls = list(range(nc))
    
    true_positive = np.zeros((detections.shape[0], nc))
    
    tp, fp, tn, fn = 0, 0, 0, 0
    

    
    for c in cls:
        c_det_idx = np.where(detections[:, -1] == c)[0]
        c_det = [d[:4] for d in detections[c_det_idx]]
        c_tar = [t[:4] for t in targets[targets[:, -1] == c]]
        
        # matching 
        match_idx, unmatch_pos, unmatch_true = hungarian_matching(c_det, c_tar, f=intersection_of_union, threshold=min_iou)
        for i, _ in match_idx:
            true_positive[c_det_idx[i]] = 1
        tp += len(match_idx)
        tps = [c_det[i] for i in [row[0] for row in match_idx]]
        cm_detections['tp'].extend(tps)
        fp += len(unmatch_pos)
        fps = [c_det[i] for i in unmatch_pos]
        cm_detections['fp'].extend(fps)
        fn += len(unmatch_true)
        fns = [c_tar[i] for i in unmatch_true]
        cm_detections['fn'].extend(fns)
        
    cm_detections = {
        key: np.array(value) if value != [] else np.empty((0, 4)) for key, value in cm_detections.items()
    }
            
    return cm_detections, true_positive, detections[:, -2], detections[:, -1], np.array([tp, fp, tn, fn])

def metrics_visualize(curves, path:Path):
    _, f1, p, r = curves
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.set_xlim(-.1,1.1)
    ax.set_ylim(-.1,1.1)
    ax.plot(f1[0], f1[1].reshape(-1), "black", linewidth=2, label="F1 score")
    ax.plot(p[0], p[1].reshape(-1), "r", linewidth=1, label="precision")
    ax.plot(r[0], r[1].reshape(-1), "b", linewidth=1, label="recall")
    ax.set_xlabel("Confidence threshold", fontsize=20)
    ax.set_ylabel("Metrics", fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(loc="lower left", fontsize=15)
    ax.set_xticks(np.linspace(-0.1, 1.1, 13), minor=True)
    ax.set_yticks(np.linspace(-0.1, 1.1, 13), minor=True)
    ax.grid(True, which='both')
    plt.tight_layout()
    fig.savefig(path)
    return

def myplot(image:Path, dest:Path, boxes:np.ndarray):
    image = Image.open(image).convert('RGB')
    draw = ImageDraw.Draw(image)
    boxes[:, ::2] = boxes[:, ::2] * image.size[0]
    boxes[:, 1::2] = boxes[:, 1::2] * image.size[1]
    for b in boxes:
        draw.rectangle(b, outline="red", width=2)
    image.save(dest)
    
def myplot_with_false(image:Path, dest:Path, boxes:Dict[str, np.ndarray]):
    image = Image.open(image).convert('RGB')
    draw = ImageDraw.Draw(image)
    for cm, bs in boxes.items():
        bs[:, ::2] = bs[:, ::2] * image.size[0]
        bs[:, 1::2] = bs[:, 1::2] * image.size[1]
        if cm == 'tp':
            for b in bs:
                draw.rectangle(b, outline="red", width=2)
        elif cm == 'fp':
            for b in bs:
                draw.rectangle(b, outline="blue", width=4)
        elif cm == 'fn':
            for b in bs:
                draw.rectangle(b.tolist(), outline="green", width=4)
    image.save(dest)