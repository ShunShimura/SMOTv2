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
                
def save_meanIoU(all_predictions:List[Dict[int, np.ndarray]], nf:int, data_loader:DataLoader, dir:Path) -> None:
    '''
    Save prediction performace by text formats.
        IoU means how predicted position overraps ground-truth position.
        'mean' indicates that values are averaged over all tracked spheres.
    
    Parameters
    ----------
    all_predictions: Each shpere's prediction time-series. Key of dict means time.
    nf: number of frames
    data_loader: instance of DataLoader used to extract ground-truth positions
    dir: Directory path where output text file is saved.
    '''
    # GT : [x, y, z, r] (key is time and id)
    ground_truth_positions = data_loader.load_estimation_labels()
    
    # convert predictions to List[int, Dict[np.ndarray]] ( id: {time: array} )
    predictios_with_id: Dict[int, Dict[np.ndarray]] = {}
    for t, predictions in enumerate(all_predictions):
        for id, pred in predictions.items():
            if id in predictios_with_id.keys():
                predictios_with_id[id][t] = pred
            else:
                predictios_with_id[id] = {t: pred}
    predicted_positions: List[Dict[int, np.ndarray]] = list(predictios_with_id.values())
    
    # match preID and gtID to maximize score
    
    n = len(ground_truth_positions)
    m = len(predicted_positions)
    if n == 0 or m == 0:
        with open(dir / 'meanIoU.txt', 'w') as f:
            f.write("gt or pred is None.")
    else:
        match_idx, _, _ = hungarian_matching(
            instance1=ground_truth_positions,
            instance2=predicted_positions,
            f=sumIoU,
            threshold=EPS
        )
        scores = []
        for t in range(nf):
            count = 0
            score_t = 0
            for i, j in match_idx:
                if (not t in ground_truth_positions[i].keys()) or (not t in predicted_positions[j].keys()):
                    continue
                elif (t in ground_truth_positions[i].keys()) and (not t in predicted_positions[j].keys()):
                    count += 1
                else:
                    iou = sphere_iou([ground_truth_positions[i][t]], [predicted_positions[j][t]])[0, 0]
                    score_t += iou
                    count += 1
            scores.append([t, score_t / count])
        with open(dir / 'meanIoU.txt', 'w') as f:
            f.write("\n".join([
                ",".join([str(val) for val in row]) for row in scores
            ]))