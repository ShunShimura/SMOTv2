import os, sys
import argparse
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm

from sliced_multi_object_tracker.detector.ultralytics import YOLOv10
from sliced_multi_object_tracker.detector.ultralytics.utils.metrics import DetMetrics
from sliced_multi_object_tracker.detector.ultralytics.engine.results import Results
from sliced_multi_object_tracker.detector.utils import download_labels, true_positive_detections, metrics_visualize, myplot, myplot_with_false
from sliced_multi_object_tracker.utils import create_or_clear_directory

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

'''
CAUTION: This script must be runned on SMOT directory. and for single GPU

library: (conda 24.9.2)
    conda install pytorch torchvision torchaudio cudatoolkit=12.1
    opencv: CAUTION; opencv may change pytorch version from one for GPU to one for CPU
    psutil
    matplotlib
    tqdm 
    pandas
    huggingface_hub (Please pay attension to "_" ! Not "-" !)
    -> please install simultanously !
    
'''

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, help="destination to target source path (required).")
    parser.add_argument("--conf", type=float, help="confidence threshold (required).")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str)
    parser.add_argument("--output", type=Path, default=Path('sliced_mulit_object_tracker')/'detector'/'runs'/'valid'/'default')
    parser.add_argument("--weights", type=str, default='sliced_multi_object_tracker/detector/ultralytics/pretrained/yolov10n.pt')
    return parser.parse_args()

if __name__=='__main__':
    opt = parse_opt()
    opt.device = [int(val) for val in opt.device.split(",")]
    
    create_or_clear_directory(opt.output)
    annotated_path = opt.output/'annotated_images'
    create_or_clear_directory(annotated_path)
    
    model = YOLOv10(opt.weights)
    
    results: List[Results] = model(
        source=opt.source / 'images',
        conf=opt.conf,
        imgsz=opt.imgsz,
        device=opt.device,
        stream=True,
        verbose=False
        )
    
    labels = download_labels(path=opt.source / 'labels')
    metrics = DetMetrics(save_dir=Path(opt.output), names={0:'sphere'})
    
    confusion_matrix_with_conf = {c: np.zeros((4,)) for c in range(10)} # tp, fp, tn, fn
    tp, conf, pred_cls, tar_cls = np.empty((0, 1)), np.empty((0,)), np.empty((0,)), np.empty((0,))
    fp_list, fn_list = set(), set()
    runtimes = []
    for r in results: 
        print("\n", Path(r.path).stem)
        runtimes.append(sum(r.speed.values()))
        r.save_txt(dir=opt.output/'labels', basename=Path(r.path).stem, save_conf=True)
        detections = r.boxes.data.cpu().numpy()
        xyxyn = r.boxes.xyxyn.cpu().numpy()
        detections[:, :4] = xyxyn
        # myplot(image=r.path, dest=(annotated_path / Path(r.path).stem).with_suffix(".pdf"), boxes=xyxyn)
        if Path(r.path).stem in labels.keys():
            targets = labels[Path(r.path).stem]
            for c in confusion_matrix_with_conf.keys():
                cm_dets, _, _, _, cm = true_positive_detections(detections, targets, min_iou=0.5, conf_thres=c*0.1, nc=1)
                confusion_matrix_with_conf[c] = confusion_matrix_with_conf[c] + cm
                if c == 2:
                    myplot_with_false(image=r.path, dest=(annotated_path / Path(r.path).stem).with_suffix(".pdf"), boxes=cm_dets)
            _, tp_, conf_, pred_cls_, cm = true_positive_detections(detections, targets, min_iou=0.5, conf_thres=0.0, nc=1)
            if cm[1] != 0:
                fp_list.add(Path(r.path).name)
            if cm[3] != 0:
                fn_list.add(Path(r.path).name)
            tp = np.vstack((tp, tp_))
            conf = np.hstack((conf, conf_))
            pred_cls = np.hstack((pred_cls, pred_cls_))
            tar_cls = np.hstack((tar_cls, targets[:, -1]))
            
    metrics.process(tp, conf, pred_cls, tar_cls)
    
    # visualize
    results = metrics.curves_results
    metrics_visualize(results, path=opt.output/'metrics.pdf')
    with open(opt.output/'runtime.txt', 'w') as f:
        f.write(f'{sum(runtimes)/len(runtimes)}\t{metrics.maps[0]}')
    np.savetxt(opt.output/'confusion.txt', np.array([
        confusion_matrix_with_conf[c] for c in range(10)
    ]), fmt="%d", delimiter="\t")
    