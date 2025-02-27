import sys,os,shutil

from typing import List, Set, Dict
from pathlib import Path
import numpy as np

from sliced_multi_object_tracker.core import *
from sliced_multi_object_tracker.utils import *
from sliced_multi_object_tracker.io.load import DataLoader

EPS = 1e-6

def save_reid(bboxes_with_id:List[Set[Bbox]]):
    return

def save_prediction(all_prediction:List[Dict[int, np.ndarray]], nf:int, dir:Path) -> None:
    ''' 
    Save prediction of tracked spheres by 3D-like KalmanFilter at each time.
    
    Parameters
    ----------
    all_prediction: Each shpere's prediction time-series. Key of dict means time.
    nf: number of frame
    dir: Directory path where prediction text files are saved.
    '''
    folder = Path(dir) / 'prediction'
    create_or_clear_directory(folder)
    # save predictions (include vel, size and so on)
    for t in range(nf):
        predictions = all_prediction[t]
        with open(folder / f't{str(t).zfill(3)}.txt', 'w') as f:
            for id, vec in predictions.items():
                f.write(
                    ",".join([str(id)] + [str(val) for val in vec]) + "\n"
                )