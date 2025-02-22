import os,sys,shutil
import numpy as np
from numpy import dot
import scipy
from typing import Set, Tuple

from sliced_multi_object_tracker.filter.state_space_model import *
from sliced_multi_object_tracker.core import *

class KalmanFilter:
    def __init__(self, model:BaseModel, obs0:np.ndarray):
        self.model:BaseModel = model
        self.F = self.model.build_F() # transition
        self.H = self.model.build_H() # observation
        self.Q = self.model.build_Q() # system noise
        self.R = self.model.build_R() # observation noise
        self.P = self.model.build_P() # predicted state error's covariances
        self.x = self.model.obs0_to_state(obs0)
        self.inv = scipy.linalg.inv
        
    def state_predict(self) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Calclate state prediction (\mu_{t+1|t}, \Sigma_{t+1|t}) 
        
        Returns
        -------
        self.x.copy(): Mean of state prediction
        self.P.copy(): Covariance of state prediction
        '''
        # x = Fx
        self.x = dot(self.F, self.x)
        
        # P = FPF' + Q
        self.P = dot(dot(self.F, self.P), self.F.T) + self.Q
        
        return self.x.copy(), self.P.copy()
    
    def state_update(self, y:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Calculate state filtering (= posterior, \mu_{t|t}, \Sigma_{t|t})
        
        Parameters
        ----------
        y: Observation at time-t
        
        Returns
        -------
        self.x.copy(): Mean of state filtering
        self.P.copy(): Covariance of state filtering
        '''
        # residual r = y - Hx
        r = y - dot(self.H, self.x)
        
        # kalman gain K = PH' / (HPH' + R)
        self.K = dot(
            dot(self.P, self.H.T),
            self.inv(
                dot(self.H, dot(self.P, self.H.T)) + self.R
            )
        )
        
        # x = x + Kr ( = (1-KH)x + Ky)
        self.x = self.x + dot(self.K, r)
        
        # P = (I - KH) P
        self.P = dot(
            np.eye(self.model.state_length) - dot(self.K, self.H),
            self.P
            )
        
        return self.x.copy(), self.P.copy()
    
    def observe_predict(self) -> np.ndarray:
        '''
        Predict observation from state prediction (\mu_{t+1|t})
        
        Returns
        -------
        (M,) np.array: M is length of observation variables
        '''
        # y = Hx
        return dot(self.H, self.x)
    
class ExtendedKalmanFilter(KalmanFilter):
    def __init__(self, model:ExtendedModel, obs0:np.ndarray):
        super().__init__(model, obs0)
        self.model: ExtendedModel = model
        # constract non-linear maps
        if self.model.nonlinear['f']:
            self.f = self.model.build_f
        if self.model.nonlinear['h']:
            self.h = self.model.build_h
            
    def state_predict(self) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Calclate state prediction (\mu_{t+1|t}, \Sigma_{t+1|t}) 
        But here, when transition equation is non-linear, use non-linear 'f' to calculate prediction.
        
        Returns
        -------
        self.x.copy(): Mean of state prediction
        self.P.copy(): Covariance of state prediction
        '''
        if self.model.nonlinear['f']:
            # rebuild F if f(x) is non-linear
            self.F = self.model.rebuild_F()
            
            # use non-linear function f(x)
            self.x = self.f(self.x)
            self.P = dot(
                dot(self.F, self.P),
                self.F.T
            ) + self.Q
            
            return self.x.copy(), self.P.copy()
        
        else:
            # use linear matrix
            return super().state_predict()
        
    def state_update(self, y:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Calculate state filtering (= posterior, \mu_{t|t}, \Sigma_{t|t})
        But here, when observation equation is non-linear, use non-linear function 'h' to predict observation
        
        Parameters
        ----------
        y: Observation at time-t
        
        Returns
        -------
        self.x.copy(): Mean of state filtering
        self.P.copy(): Covariance of state filtering
        '''
        if self.model.nonlinear['h']:
            # rebuild H if h is non-linear (use taylor expansion at \hat{x})
            self.H = self.model.rebuild_H(self.x)
            
            # r = y - h(x)
            r = y - self.h(self.x)
        
            # from now on, left updating procedure is the same to linear-model
            
            # S = HPH' + R
            self.S = dot(
                self.H,
                dot(self.P, self.H.T)
            ) + self.R
            
            # K = PH' / S
            self.K = dot(
                dot(self.P, self.H.T),
                self.inv(self.S)
            )
            
            # x = x + Kr (= (1 - KH)x + Ky)
            self.x = self.x + dot(self.K, r)
            
            # P = (I - KH) P
            self.P = dot(
                np.eye(self.model.state_length) - dot(self.K, self.H),
                self.P
                )
            
            return self.x.copy(), self.P.copy()
        
        else:
            return super().state_update(y)
    
    def observe_predict(self) -> np.ndarray:
        '''
        Predict observation from state prediction (\mu_{t+1|t})
        But here, when observation equation is non-linear, use non-linear function 'h' to predict observation
        
        Returns
        -------
        (M,) np.array: M is length of observation variables
        '''
        return super().observe_predict() if not self.model.nonlinear['h'] else self.h(self.x)
    
class SliceKalmanFilter(ExtendedKalmanFilter):
    def __init__(self, model:SlicedModel, obs0:Set[Bbox]):
        self.model: SlicedModel = model
        super().__init__(model, obs0)
        
    def state_update(self, y:Set[Bbox]) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Calculate filtering \mu_{t|t}, \Sigma_{t|t}
        In SliceKalmanFilter, H, R and y consists of slices that exists in observation and prediction. 
        (See SlicedModel on filter.state_space_model.py)
        
        Parameters
        ----------
        y: Observations at time-t. Observations consists of bounding-boxes that are re-identified as what is derived from identical object
        
        Returns
        -------
        self.x.copy(): Mean of state filtering
        self.P.copy(): Covariance of state filtering        
        '''
        
        # rebuilding H, R depends on shape of y
        self.H = self.model.rebuild_H(self.x, y)
        self.R = self.model.rebuild_R(self.x, y)
        
        # if slices that exists in prediction and observation is None, return
        if self.H is None:
            return self.x.copy(), self.P.copy()
        
        # constract array-like y that limits slices (where both pred and obs exists)
        y_array = self.model.y_array(self.x, y)
        
        # resuidual r = y - h(x)
        r = y_array - self.h(self.x, y)
        
        # ※ procedure below is the same to ExtendedKalmanFilter.state_update()
        
        # S = HPH' + R
        self.S = dot(
            self.H,
            dot(self.P, self.H.T)
        ) + self.R
        
        # K = PH' / S
        self.K = dot(
            dot(self.P, self.H.T),
            self.inv(self.S)
        )
        
        # x = x + Kr (= (1 - KH)x + Ky)
        self.x = self.x + dot(self.K, r)
        
        # P = (I - KH) P
        self.P = dot(
            np.eye(self.model.state_length) - dot(self.K, self.H),
            self.P
            )
        
        return self.x.copy(), self.P.copy()        
    
    def observe_predict(self) -> np.ndarray:
        '''
        Predict observation from state prediction (\mu_{t+1|t})
        In SliceKalmanFilter, use slice-wise observation map h_s(x) and then returns dict of it.
        
        Returns
        -------
        Dictionary of predicted bounding-boxes. Key indicates slice-index.
        '''
        # to predict without y, use slice-wise observation map, hs(x)
        return {s: self.model._hs(self.x, s) for s in self.model.ns}
            
        