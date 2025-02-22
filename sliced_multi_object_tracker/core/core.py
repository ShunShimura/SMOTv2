import os,sys,shutil
import numpy as np

class Bbox:
    def __init__(self, bbox:np.ndarray, t:int, s:int, conf:float=1.0, id=None):
        '''
        Parameters
        ----------
        bbox: (center-x, center-y, width, height) (normalized)
        t: time
        s: slice index
        conf: confidence at detection (e.g. YOLO)
        id: when re-identified, it is given
        '''
        self.bbox = bbox
        self.conf = conf
        self.time = t
        self.slice = s
        self.id = id
        
    def __repr__(self):
        return f'time: {self.time}, slice: {self.slice}, bbox: {self.bbox}'
    
    @property
    def xyxyn(self):
        '''
        Returns
        -------
        nb: np.ndarray (left-x(x_min), top-y(y_min), right-x(x_max), bottom-y(y_max)), all values are normailized
        '''
        nb = np.zeros((4,))
        nb[:2] = self.bbox[:2] - self.bbox[2:] / 2
        nb[2:] = self.bbox[:2] + self.bbox[2:] / 2
        return nb
    
    @property
    def xywhn(self):
        '''
        Returns
        -------
        nb: np.ndarray (left-x, top-y, width, height), all values are normalized
        '''
        nb = np.zeros((4,))
        nb[:2] = self.bbox[:2] - self.bbox[2:] /2
        nb[2:] = self.bbox[2:]
        return nb    