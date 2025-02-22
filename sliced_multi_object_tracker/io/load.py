import os,sys,shutil

from pathlib import Path
from typing import Dict, Set, List
import numpy as np

from sliced_multi_object_tracker.core import *

class DataLoader:
    '''
    To handle input datas, and load them.
    
    Attributes
    ----------
    ns: number of slice
    nf: length of shooting time
    root: dataset's root path
    
    Methods
    -------
    _check_path: check the dataset folder format and if ns and nf correspond
    load_image_paths: return image paths at time-t to detect
    load_detection_labels: if you use ground-truth detection, load ideal detections
    load_reidentification_labels: if you use ground-truth ReID, load ideal ReID
    load_estimation_labels: load ground-truth positions to assess estimation performance 
    '''
    def __init__(self, ns:int, nf:int, root:Path):
        self.root = root
        self.ns = ns
        self.nf = nf
        
    def _check_path(self) -> bool:
        '''
        Check the dataset folder format and if ns and nf correspond
            Inputs consists of three part, which are detection, reid and estimation
            And they have 'labels' folder that is ground-truth
            
        Returns
        -------
        True/False: If there are any missing data or number of slice and number of frame are not adaptive, return False.
        '''
        detection_path = self.root / 'detection'
        if not ((detection_path / 'labels').exists() and (detection_path / 'images').exists()):
            return False
        else:
            if not (len([f for f in (detection_path / 'labels').glob("*.txt")]) == self.ns * self.nf and len([f for f in (detection_path / 'images').glob("*.jpg")]) == self.ns * self.nf):
                return False
        reid_path = self.root / 'reid'
        if not ((reid_path / 'labels').exists() and (reid_path / 'bboxes').exists()):
            return False
        else:
            if not (len([f for f in (reid_path / 'labels').glob("*.txt")]) == self.ns * self.nf and len([f for f in (reid_path / 'bboxes').glob("*.txt")]) == self.ns * self.nf):
                return False
        estimation_path = self.root / 'estimation'
        if not ((estimation_path / 'labels').exists() and (estimation_path / 'observations').exists()):
            return False
        else:
            if not (len([f for f in (estimation_path / 'labels').glob("*.txt")]) == self.nf and len([f for f in (estimation_path / 'observations').glob("*.txt")]) == self.ns * self.nf):
                return False
        return True
    
    def load_image_paths(self, time:int) -> List[Path]:
        '''
        Load image path on time-t to detect (In each time, there are n_s images)
        
        Parameters
        ----------
        time: discrete time that corresponding n_s images
        
        Returns
        -------
        List of image path (Path) where the order is ascending
        '''
        return sorted(list((self.root / 'detection' / 'images').glob("*.jpg")))[time * self.ns: (time + 1) * self.ns]
    
    def load_detection_labels(self, time:int) -> Dict[int, np.ndarray]:
        '''
        Load detected bounding-boxes on time-t
        
        Parameters
        ----------
        time: discrete time that corresponding n_s images
        
        Returns
        -------
        bboxes: Dictonary where key is slice-index and value is stacked bouding-boxes.
            bounding-boxes are the format of normalized (cx, cy, w, h)
        '''
        detection_label_paths = sorted([p for p in (self.root / 'detection' / 'labels').glob("*.txt")])[self.ns * time: self.ns * (time + 1)]
        bboxes: Dict[int, Bbox] = {}
        for s, p in enumerate(detection_label_paths):
            if p.stat().st_size > 0:
                slice_bboxes = np.loadtxt(p, dtype=float, delimiter=" ")
                if slice_bboxes.ndim == 1:
                    slice_bboxes = slice_bboxes.reshape(-1, 5)
            else:
                slice_bboxes = np.empty((0, 5))
            bboxes[s] = slice_bboxes[:, 1:]
        return bboxes
    
    def load_reidentification_labels(self, time:int) -> Dict[int, Set[Bbox]]:
        '''
        Load ground-truth bounding-boxes with ID on time-t
        
        Parameters
        ----------
        time: discrete time that corresponding n_s images
        
        Returns
        -------
        bboxes: Dictonary where key is ID and value is set of bounding-boxes whose id is the ID
            bounding-boxes are the format of normalized (cx, cy, w, h)
        '''
        reidentification_label_paths = sorted([p for p in (self.root / 'reid' / 'labels').glob("*.txt")])[self.ns * time: self.ns * (time + 1)]
        identified_bboxes:Dict[int, Set[Bbox]] = {}
        for s, p in enumerate(reidentification_label_paths):
            with open(p, "r") as f:
                for line in f:
                    line.strip()
                    contents = line.split(",")
                    id = int(contents[1])
                    box = np.array([float(val) for val in contents[2:6]])
                    if id in identified_bboxes.keys():
                        identified_bboxes[id].add(Bbox(box, t=time, s=s, id=id))
                    else:
                        identified_bboxes[id] = {Bbox(box, t=time, s=s, id=id)}
        return identified_bboxes
    
    def load_estimation_labels(self) -> Dict[int, Dict[int, np.ndarray]]:
        '''
        Load ground-truth object positions and size
            Used for assessment of estimation
            
        Returns
        -------
        ground_truth: Dictionary where the key is ID and value is tracks of each object.
            Each value consists of dictionary whose key is the time and value is (x, y, z, r)
        '''
        dir_path = Path(self.root / 'estimation' / 'labels')
        labels_path = sorted(list(dir_path.glob("*.txt")))
        ground_truth: Dict[int, Dict[int, np.ndarray]] = {}
        for t, lp in enumerate(labels_path):
            with open(lp, 'r') as f:
                for line in f:
                    line.strip()
                    items = [float(v) for v in line.split(",")]
                    x, y, z, r = items[1::2]
                    id = int(items[0])
                    if not id in ground_truth.keys():
                        ground_truth[id] = {t:np.array([x, y, z, r])}
                    else:
                        ground_truth[id][t] = np.array([x, y, z, r])
        ground_truth = list(ground_truth.values())  
        return ground_truth