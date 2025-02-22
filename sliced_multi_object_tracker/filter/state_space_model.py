import os,sys,shutil
import numpy as np
from typing import Dict, List, Set, Tuple
from scipy.linalg import block_diag
from numpy import dot

from sliced_multi_object_tracker.core import *

def _base_dim_block(dt:float, order:int=1) -> np.ndarray:
    '''
    Make transition matrix (to be multiplied to pos, vel and acc.)
    Note that is only compatible until order = 2
    
    Parameters
    ----------
    dt: time increment
    order: the order of model (0 means steady, 1 means constant velocity and 2 means having acceleration)
    
    Returns
    -------
    Array of transition matrix corresponding (x, x', x'')^{T}. The matrix product gives (x, x', x'')^{T} at dt latter.
    '''
    block = np.array([
        [1, dt, dt**2/2],
        [0, 1, dt],
        [0, 0, 1]
    ])
    cutoff = order + 1
    return block[:cutoff, :cutoff]

def Q_continuous_white_noise(order:str, dt:float, var:float) -> np.ndarray:
    '''
    Calculate sysytem noise covariance in the case that acceleration is white noise.
    In motion of point mass, position, velocity and acceleration have covariance each other.
    And this function calculates continuous white noise because mass point moves continuously in real space.
    (It is obtaind by integrating discrete white noise)
    
    Parameters
    ----------
    order: the order of model (0 means steady, 1 means constant velocity and 2 means having acceleration)
    dt: time increment
    var: population variance of acceleration (this is white noise)
    
    Returns
    -------
    Covariance array. Index corresponds pos, vel and acceleration in order.
    '''
    block = np.array([
        [dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
        [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
        [dt ** 3 / 6, dt ** 2 / 2, dt],
    ])
    
    return var * block[:order, :order]

class BaseModel:
    def __init__(
        self, 
        state_pos_dim: int = 0,
        state_size_dim: int = 0,
        state_pos_order: int = 0,
        state_size_order: int = 0,
        obs_pos_dim: int = 0,
        obs_size_dim: int = 0,
        dt: float = 1.0,
        q_var_pos: float = 1.0,
        q_var_size: float = 1.0,
        r_var_pos: float = 1.0,
        r_var_size: float = 1.0,
        p_var_p0: float = 100.0,
        ):
        
        # shape of state and observe vector
        self.state_pos_dim = state_pos_dim
        self.state_size_dim = state_size_dim
        self.state_pos_order = state_pos_order
        self.state_size_order = state_size_order
        self.obs_pos_dim = obs_pos_dim
        self.obs_size_dim = obs_size_dim
        # numerical value parameter
        self.dt = dt
        self.q_var_pos = q_var_pos
        self.q_var_size = q_var_size
        self.r_var_pos = r_var_pos
        self.r_var_size = r_var_size
        self.p_var_p0 = p_var_p0
        
        self.state_length = (self.state_pos_order + 1) * self.state_pos_dim + (self.state_size_order + 1) * self.state_size_dim
        self.obs_length = self.obs_pos_dim + self.obs_size_dim        
        
    # transition matrix
    def build_F(self) -> np.ndarray:
        '''
        Build state transition matrix F
        
        Retunrs
        -------
        Array of transition matrix F (whose shape is (state-lenth, state-length))
        '''
        block_pos = _base_dim_block(self.dt, self.state_pos_order)
        block_size = _base_dim_block(self.dt, self.state_size_order)
        diag_components = [block_pos] * self.state_pos_dim + [block_size] * self.state_size_dim
        return block_diag(*diag_components)
        
    # system noise covariance
    def build_Q(self) -> np.ndarray:
        '''
        Build system noise covarinace Q
        
        Returns
        -------
        Array of system noise covarinace Q (whose shape is (state-lenth, state-length))
        '''
        q_pos = self.q_var_pos if self.state_pos_order == 0 else Q_continuous_white_noise(order=self.state_pos_order + 1, dt=self.dt, var=self.q_var_pos)
        q_size = self.q_var_size if self.state_size_order == 0 else Q_continuous_white_noise(dim=self.state_size_order + 1, dt=self.dt, var=self.q_var_size)
        diag_components = [q_pos] * self.state_pos_dim + [q_size] * self.state_size_dim
        return block_diag(*diag_components)
    
    # observation matrix
    def build_H(self) -> np.ndarray:
        '''
        Build observation matrix H
        
        Returns
        -------
        Array of observation matrix H (whose shape is (observe-length, state-length))
        '''
        def _base_block(state_order):
            return np.array([1] + [0] * state_order)
        diag_components = \
            [_base_block(self.state_pos_order)] * self.obs_pos_dim +\
            [_base_block(self.state_size_order)] * self.obs_size_dim
        return block_diag(*diag_components)
    
    # observation noise covarinace
    def build_R(self) -> np.ndarray:
        '''
        Build observation noise covariance
        
        Returns
        -------
        Array of observation noise covariance (whose shape is (observe-length, observe-length))
        '''
        block_pos = np.eye(self.obs_pos_dim) * self.r_var_pos
        block_size = np.eye(self.obs_size_dim) * self.r_var_size
        return block_diag(block_pos, block_size)
    
    # initial state-prediction error covariance
    def build_P(self) -> np.ndarray:
        '''
        Build initial state-prediction error covariance
            state-prediction is \hat{s} and state-prediction error is defined as \hat{s} - s (we can't calculate this because s is oracle)
            So the covariance P is E[(\hat{s} - s)^2] - {E[\hat{s} - s]}^2
            (And especially in linear kalman-filter, the covarinace equals state-prediction covariace.)
            
        Returns
        -------
        Array of state-prediction error covariance (whose shape is (state-length, state-length))
        '''
        return np.eye(self.state_length) * self.p_var_p0
             
    # initialize \mu_{0|0} from y_{0}
    def obs0_to_state(self, obs0:np.ndarray) -> np.ndarray:
        '''
        Make initial state-estimation from initial observation
            Leverage only position and size (so except their velocity or acceleration).
            
        Parameters
        ----------
        obs0: array of initial observation (y_{0})
        
        Returns
        -------
        state: array of initial esitmated-state (\mu_{0|0})
        '''
        state = np.zeros(self.state_length)
        # pos
        state[:(self.state_pos_order + 1) * self.state_pos_dim: self.state_pos_order + 1] = obs0[:self.obs_pos_dim]
        # size
        state[(self.state_pos_order + 1) * self.state_pos_dim:: self.state_size_order + 1] = obs0[self.obs_pos_dim:]
        return state
    
class BboxModel(BaseModel):
    def __init__(
        self,
        state_pos_order: int,
        state_size_order: int,
        dt: float = 1.0,
        q_var_pos: float = 1.0,
        q_var_size: float = 1.0,
        r_var_pos: float = 1.0,
        r_var_size: float = 1.0,
        p_var_p0: float = 100.0,
        ):
        '''
        BboxModel means state-vector consists of (x, y, w, h) and their differenctial terms 
        and observe-vector is (x, y, w, h).
        Note that you can select the state-vector's order. 
        '''

        super().__init__(
            state_pos_order = state_pos_order,
            state_size_order = state_size_order,
            dt = dt,
            q_var_pos = q_var_pos,
            q_var_size = q_var_size,
            r_var_pos = r_var_pos,
            r_var_size = r_var_size,
            p_var_p0 = p_var_p0,
            state_pos_dim = 2, # (x,y)
            state_size_dim = 2, # (w,h)
            obs_pos_dim = 2, # (x,y)
            obs_size_dim = 2, # (w,h)
        )
        
    @property
    def state_to_obs_index(self) -> np.ndarray:
        '''
        To sample observation elements from state vector.
            You can use as obs_variables = state_vector[ret] (ret means returned value by this function)
        
        Returns
        -------
        Array of indexes that indicates the index on state-vector of variables on observe-vector
        '''
        pos_idx = [idx for i in range(self.state_pos_dim) if (idx := i * (self.state_pos_order + 1)) < self.state_pos_dim * (self.state_pos_order + 1)]
        size_idx = [self.state_pos_dim * (self.state_pos_order + 1) + idx for i in range(self.state_size_dim) if (idx := i * (self.state_size_order + 1)) < self.state_size_dim * (self.state_size_order + 1)]
        return np.array(pos_idx + size_idx)  
    
class ExtendedModel(BaseModel):
    def __init__(
        self, 
        nonlinear: Dict[str, bool],
        state_pos_dim: int,
        state_size_dim: int,
        state_pos_order: int,
        state_size_order: int,
        obs_pos_dim: int,
        obs_size_dim: int,
        dt: float = 1.0,
        q_var_pos: float = 1.0,
        q_var_size: float = 1.0,
        r_var_pos: float = 1.0,
        r_var_size: float = 1.0,
        p_var_p0: float = 100.0,
    ):
        '''
        ExtendedModel corresponds ExtendedKalmanFilter to adopt non-linear system or observation.
        In ExtendedKalmanFilter, we need non-linear functions f (state-transition) or h (observation). 
        And to calculate some procedures on KalmanFilter, you need to rebuild F or H by linearization.
        
        Parameters
        ----------
        nonlinear: {map: True/False}; "f: True" means transition is non-linear. And "h: True" means observation is non-linear.
        '''
        super().__init__(
            state_pos_dim = state_pos_dim,
            state_size_dim = state_size_dim,
            state_pos_order = state_pos_order,
            state_size_order = state_size_order,
            obs_pos_dim = obs_pos_dim,
            obs_size_dim = obs_size_dim,
            dt = dt,
            q_var_pos = q_var_pos,
            q_var_size = q_var_size,
            r_var_pos = r_var_pos,
            r_var_size = r_var_size,
            p_var_p0 = p_var_p0
        )
        self.nonlinear = nonlinear
        
    def build_f(self, x:np.ndarray) -> np.ndarray:
        if self.nonlinear['f']:
            raise NotImplementedError
        
    def rebuild_F(self)  -> np.ndarray:
        if self.nonlinear['f']:
            raise NotImplementedError
        
    def build_h(self, x:np.ndarray) -> np.ndarray:
        if self.nonlinear['h']:
            raise NotImplementedError
    
    def rebuild_H(self) -> np.ndarray:
        if self.nonlinear['h']:
            raise NotImplementedError
    
class SlicedModel(ExtendedModel):
    def __init__(
        self,
        state_pos_order: int,
        state_size_order: int,
        number_of_slice: int,
        dv: float,
        dt: float = 1.0,
        q_var_pos: float = 1.0,    
        q_var_size: float = 1.0,
        r_var_pos: float = 1.0,
        r_var_size: float = 1.0,
        p_var_p0: float = 100.0,
    ):
        '''
        In SlicedModel, the observation informations are given by set of bounding-boxes.
        (Multiple bounding-boxes indicates that they are given at different slices.)
        And the state-vector is defined as sphere (at least, consists of (x, y, z, r)).
        
        Parameters
        ----------
        dv = [dx, dy, dz]: indicates scale ratio of state-space to observe-space.
            for example, in the case that the images captures 2.0mm x 2.0mm space and bounding-boxes are normalized, dx and dy becomes 2.0mm
            (Note that you need not to explicit units.)
            And for depth-direction, dz indicates step between two consecutive slices.
        '''
        
        self.ns = number_of_slice
        self.nonlinear = {'f':False, 'h':True}
        self.dv = dv
        
        super().__init__(
            nonlinear = self.nonlinear,
            state_pos_order = state_pos_order,
            state_size_order = state_size_order,
            dt = dt,
            q_var_pos = q_var_pos,
            q_var_size = q_var_size,
            r_var_pos = r_var_pos,
            r_var_size = r_var_size,
            p_var_p0 = p_var_p0,
            state_pos_dim = 3, # (x,y,z)
            state_size_dim = 1, # (r)
            obs_pos_dim = 2, # (x,y)
            obs_size_dim = 2, # (w,h)            
        )

    def _slice_index(self, x:np.ndarray, y:Set[Bbox]):
        '''
        Take the intersection of slice-index on predictions and observations
            Because slice-index of two sets are required to be the same to calculate Kalman-Filtering. (res = y - h(x))
            
        Parameters
        ----------
        x: Predicted state-vector (\mu_{t|t-1})
        y: Set of observed bounding-boxes (y_{t})
        
        Returns 
        -------
        Slice-indexes that exists on both prediction and observation
            Leverage slice-wise observation map h_s(x) to see if each slice exists on prediction.
        '''
        return sorted([o.slice for o in y if self._hs(x, o.slice) is not None])
        
    def _hs(self, state:np.ndarray, s:int) -> np.ndarray:
        '''
        The function that calculate bounding-box obtained on slice-s from state-vector
            1. calculate sphere position when the sliced-image captured (t_dash indicates time shift)
            2. slice the sphere at the plane and obtain the bounding-box that circumscribed about the circle.
            (if the distance between the plane and center of sphere is larger than the radius, bounding box is not obtained.) 
            
        Parameters
        ----------
        state: Array of predicted state-vector (\mu_{t|t-1})
        s: The slice-index
        
        Returns
        -------
        normalized_obs: The bounding-box obtained on slice-s from the sphere 
            bounding-box is normalized (cx, cy, w, h)
            if the sphere doesn't sliced, return None.
        '''
        # first, correct state variables corresponding to the time at which slice-s is captured
        t_dash = s / self.ns * self.dt
        # x'_i = x_i + v_i * t_dash + a_i * 0.5 * t_dash^2
        # -> if order = 1, then x'_i = x_i + v_i * t_dash
        x, y, z = dot(
            state[:(self.state_pos_order + 1) * self.state_pos_dim].reshape(self.state_pos_dim, -1),
            np.array([1, t_dash, t_dash ** 2 / 2])[:self.state_pos_order + 1]
        )
        r = dot(
            state[(self.state_pos_order + 1) * self.state_pos_dim:], 
            np.array([1, t_dash, t_dash**2/2])[:self.state_size_order+1]
            )     
        
        # then, calculate distance between sphere and slice-s
        z_offset = abs(z - s * self.dv[2]) # sefl.dv means step between each slices on state-variable-space
        # the case the sphere is not sliced by slice-s
        if z_offset > r:
            return None
        # the case the sphere is sliced by slice-s
        else:
            sliced_rad = np.sqrt(r ** 2 - z_offset ** 2)
            unnormalized_obs = np.array([x, y, 2 * sliced_rad, 2 * sliced_rad])
            # scale obs from state-space to obs-space
            normalized_obs = unnormalized_obs.copy()
            normalized_obs[::2] /= self.dv[0]
            normalized_obs[1::2] /= self.dv[1]
            return normalized_obs

    def build_h(self, x:np.ndarray, y:Set[Bbox]) -> np.ndarray:
        '''
        This function is observation map from state-vector to set of observe-vector
            Return a set of only bounding-boxes that also exists on predictions y.
            
        Parameters
        ----------
        x: Array of predicted state-vector (\mu_{t|t-1})
        y: Set of observed bounding-boxes (y_{t})
        
        Returns
        -------
        Array of bounding-boxes (whose shape is (observe-length * number of the slices))
        '''
        # only return observation where prediction and observation exists
        return np.hstack([
            self._hs(x, s) for s in self._slice_index(x, y)
        ])
        
    def rebuild_H(self, x:np.ndarray, y:Set[Bbox]) -> np.ndarray:
        '''
        Rebuild H by linearizing h(x) to conduct kalman-filtering
            Each elements can be obtained by differentiation of each elements on bounding-box by each elements on state-vector.
        
        Parameters
        ----------
        x: Array of predicted state-vector (\mu_{t|t-1})
        y: Set of observed bounding-boxes (y_{t})
        
        Returns
        -------
        Array of observation matrix H (whose shape is (observe-length * number of the slices, state-length))
        '''
        
        # H_s: partial x / partial bbox_s 
        def _Hs(x:np.ndarray, s:int):
            # calculate corrected z and r
            t_dash = s / self.ns * self.dt
            transition = np.array([1, t_dash, t_dash ** 2 / 2])
            z = dot(
                x[(self.state_pos_order + 1) * (self.state_pos_dim - 1): (self.state_pos_order + 1) * self.state_pos_dim], 
                transition[:self.state_pos_order + 1]
            )
            r = dot(
                x[(self.state_pos_order + 1) * self.state_pos_dim:], 
                transition[:self.state_size_order + 1]
            )
            # calculate H
            z_offset = z - s * self.dv[2]
            sliced_rad = np.sqrt(r ** 2 - z_offset ** 2)
            block_pos = transition[:self.state_pos_order + 1]
            block_size = np.tile(
                np.hstack(
                    (
                        -2 * z_offset * transition[:self.state_pos_order + 1], 
                        2 * r * transition[:self.state_size_order + 1]
                    )
                ), 
                (2, 1)
            ) / sliced_rad
            diag_components = [block_pos] * (self.state_pos_dim - 1) + [block_size]
            unnormalized = block_diag(*diag_components)
            # scale obs from state-space to obs-space
            normalized = unnormalized.copy()
            normalized[::2] /= self.dv[0]
            normalized[1::2] /= self.dv[1]
            return normalized  
        
        multi_Hs = [_Hs(x, s) for s in self._slice_index(x, y)]
        return np.concatenate(multi_Hs, axis=0) if multi_Hs != [] else None
    
    def rebuild_R(self, x:np.ndarray, y:Set[Bbox]) -> np.ndarray:
        '''
        Rebuild R corresponding to the slices that exists on both prediction h(x) and observation y
        
        Parameters
        ----------
        x: Array of predicted state-vector (\mu_{t|t-1})
        y: Set of observed bounding-boxes (y_{t})
        
        Returns
        -------
        Array of observation noise covariace (whose shape is (observe-length * number of the slices, observe-length * number of the slices))
        '''
        block_pos = np.eye(self.obs_pos_dim) * self.r_var_pos / (self.dv[0] * self.dv[1])
        block_size= np.eye(self.obs_size_dim) * self.r_var_size / (self.dv[0] * self.dv[1])
        return block_diag(*[block_diag(block_pos, block_size) for _ in self._slice_index(x, y)])
    
    def obs0_to_state(self, obs0:Set[Bbox]) -> np.ndarray:
        '''
        Constract initial estimation \mu_{0|0} from y_{0}
            In SlicedModel, positions are obtained by mean() and sizes are obtained by max()
            
        Paramters
        ---------
        obs0: Initial(t=0) observation (y_0)
        
        Returns
        -------
        state: Initial(t=0) state-vector estimation \mu_{0|0}
        '''
        obs_array = np.array([b.bbox for b in obs0])
        # positions are calculated as mean
        xy_mean = np.sum(obs_array, axis=0)[:self.obs_pos_dim] / obs_array.shape[0] * np.array([self.dv[0], self.dv[1]])
        z_mean = sum([float(b.slice) for b in obs0]) * self.dv[2] / len(obs0)
        # size are calculated as max (because radius appears on equatorial plane)
        size_max = np.max(obs_array[:, self.obs_pos_dim:] * np.array([self.dv[0], self.dv[1]]))
        
        # insert estimated vector to state vector's position terms
        estimated_vector = np.array(xy_mean.tolist() + [z_mean, size_max / 2])
        state = np.zeros(self.state_length)
        state[self._primary_idx] = estimated_vector
        return state
        
    def y_array(self, x:np.ndarray, y:Set[Bbox]) -> np.ndarray:
        '''
        Convert Set of y to Array of y to conduct Kalman's procedures.
        
        Parameters
        ----------
        x: Array of predicted state-vector (\mu_{t|t-1})
        y: Set of observed bounding-boxes (y_{t})

        Returns
        -------
        Array of boundng-boxes
            The slices are limited one that exists on both prediction and observation
            The order is determined by slice-index (ascending)
        '''
        y_dict = {b.slice: b.bbox for b in y}
        return np.hstack([
            y_dict[s] for s in self._slice_index(x, y)
        ])
        
    @property
    def _primary_idx(self):
        '''
        Index of (x, y, z, r) to output predictions at each time
        
        Returns
        -------
        Array of the indexes of (x, y, z, r)
        '''
        pos_idx = [idx for i in range(self.state_pos_dim) if (idx := i * (self.state_pos_order + 1)) < self.state_pos_dim * (self.state_pos_order + 1)]
        size_idx = [self.state_pos_dim * (self.state_pos_order + 1) + idx for i in range(self.state_size_dim) if (idx := i * (self.state_size_order + 1)) < self.state_size_dim * (self.state_size_order + 1)]
        return np.array(pos_idx + size_idx)