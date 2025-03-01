import os,sys,shutil

import logging, argparse
from tqdm import tqdm
from typing import List, Dict, Set
from pathlib import Path
import numpy as np

from sliced_multi_object_tracker.core import *
from sliced_multi_object_tracker.io import *
from sliced_multi_object_tracker.utils import *
from sliced_multi_object_tracker.detector import *
from sliced_multi_object_tracker.filter import *
from sliced_multi_object_tracker.tracker import *


# parser
def parse_hyp(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path('.')/'data', help="Path to input folder of images.")
    parser.add_argument("--output", type=Path, default=Path('.')/'output'/'default', help="Path to output folder.")
    parser.add_argument("--ns", type=int, help="Number of slice (required).")
    parser.add_argument("--nf", type=int, help="Number of frame for time-direction (required).")
    parser.add_argument("--mode", type=str, default='SD', help="Mode of Mulit Sliced Sphere Tracker")
    parser.add_argument("--dv", type=List[float], default=[100.0, 100.0, 1.0], help="Scale on real space with respect to 1.0 on Bbox and Slice")
    parser.add_argument("--conf-thres", type=float, default=.5, help="Confidence threshold on detection (default 0.5)")
    parser.add_argument("--time-stale-thres", type=int, default=3, help="Time-directional threshold on sort (default 3)")
    parser.add_argument("--space-stale-thres", type=int, default=1, help="Depth-directional threshold on sort (default 1)")
    parser.add_argument("--init-var", type=float, default=100.0, help="Elements of P_0 on estimation (default 100.0)")
    parser.add_argument("--q-var", type=float, default=1.0, help="Elements of Q on estimation (default 100.0)")
    parser.add_argument("--r-var", type=float, default=1.0, help="Elements of R on estimation (default 100.0)")
    parser.add_argument("--log", type=Path, default=Path('logs/default.log'), help="Path to files where log is written.")
    parser.add_argument("--gt-detect", action="store_true", default=False, help="Whether use ground_truth of detection or not")
    parser.add_argument("--gt-reid", action="store_true", default=False, help="Whether use ground_truth of reidentification or not")
    return parser.parse_known_args()[0] if known else parser.parse_args()
    
# logger
def determine_log(name, log:Path=Path(".")/'out.log') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log, mode='w')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    np.set_printoptions(formatter={'float': '{:.8f}'.format}, linewidth=1e6)
    return logger

if __name__ == '__main__':
    
    # parser hyper-parameters
    hyp = parse_hyp()
    # logger
    logger = determine_log('mylogger', hyp.log)
    
    # make directories
    output_folder = hyp.output
    detection_folder = output_folder / 'detection'
    reid_folder = output_folder / 'reid'
    estimation_folder = output_folder / 'estimation'
    create_or_clear_directory(detection_folder)
    create_or_clear_directory(reid_folder)
    create_or_clear_directory(estimation_folder)
    
    # for yolo detection
    detection_tmp_folder = detection_folder / 'tmp'
    
    # constract
    data_loader = DataLoader(
        ns=hyp.ns, 
        nf=hyp.nf,
        root=hyp.input,
    )
    detector_weight = Path('.')/'runs'/'detect'/'test_size-b'/'weights'/'best.pt' # for demo
    detector = YOLOv10(detector_weight)
    sliced_model = SlicedModel(
        state_pos_order=1,
        state_size_order=0,
        number_of_slice=hyp.ns,
        dv=hyp.dv,
    )
    bbox_model=BboxModel(
        state_pos_order=1,
        state_size_order=0,
    )
    tracker = SlicedMultiObjectTracker(
        ns=hyp.ns,
        nf=hyp.nf,
        temporal_max_staleness=hyp.space_stale_thres,
        spacio_max_staleness=hyp.space_stale_thres,
        model=sliced_model,
        bbox_model=bbox_model,
        mode=hyp.mode
    )
    
    ### Main process ###
    
    for t in tqdm(range(hyp.nf)):
        
        logger.critical(enclosed_by_dash_lines(f"Step {str(t).zfill(3)} starts", factor=100))

        ## Detection ## 
        
        # use ground truth
        if hyp.gt_detect:
            detected_bboxes: Dict[int, np.ndarray] = data_loader.load_detection_labels(time=t)
            
        # use detector and images
        else:
            # copy images to tmp_folder
            image_paths: List[Path] = data_loader.load_image_paths(time=t)
            create_or_clear_directory(detection_tmp_folder)
            for ip in image_paths:
                shutil.copy(ip, detection_tmp_folder)
                
            # save results
            detection_results: List[Results] = detector(
                source=detection_tmp_folder,
                conf=0.1,
                imgsz=640,
                stream=False,
                verbose=False,
                device=1
            )
            detected_bboxes: Dict[int, np.ndarray] = {}
            for s, r in enumerate(detection_results):
                r.save(dir=detection_folder/'annotated') # no need of preliminary makedir
                boxes = r.boxes.xywhn.detach().cpu().numpy().astype(float)
                confs = r.boxes.conf.detach().cpu().numpy().astype(float)
                idx = non_maximum_suppression(boxes, confs, iou_threshold=0.5)
                if not Path(detection_folder / 'text').exists():
                    os.makedirs(detection_folder / 'text')
                np.savetxt(
                    detection_folder/'text'/f"t{str(t).zfill(3)}s{str(s).zfill(3)}.txt",
                    np.hstack((boxes[idx], confs[idx].reshape(-1, 1))), 
                    fmt="%.8f"
                    )
                detected_bboxes[s] = boxes[idx]
                
        ## Tracking ## 
        
        # use ground truth
        if hyp.gt_reid:
            gt_bboxes = data_loader.load_reidentification_labels(time=t)
            tracker.gt_step(gt_bboxes)
            
        # use reidentifier
        else:
            # convert array-like to Bbox
            sliced_bboxes = {
                s: [
                    Bbox(b, t, s) for b in detected_bboxes[s]
                ] for s in range(hyp.ns)
            }
            tracker.step(sliced_bboxes)
            
        logger.info(f"Tracking {len(tracker.single_trackers)} objects.")
            
    all_bboxes:List[Set[Bbox]] = tracker.all_bboxes
    all_predictions:List[Dict[int, np.ndarray]] = tracker.all_predictions
    all_filterings:List[Dict[int, List[np.ndarray]]] = tracker.all_filterings
    # save_prediction(all_predictions, hyp.nf, dir=estimation_folder)
    # save_meanIoU(all_predictions, hyp.nf, data_loader, dir=estimation_folder)
    # save_motmetric(all_bboxes, hyp.nf, data_loader, dir=reid_folder)
    # save_detection_map(all_bboxes, hyp.ns, hyp.nf, dir=reid_folder)
    save_posterior(all_predictions, all_filterings, hyp.nf, data_loader, dir=estimation_folder)