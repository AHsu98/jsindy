import jax
import jax.numpy as jnp
from jsindy.util import check_is_partial_data,get_collocation_points
from jsindy.trajectory_model import TrajectoryModel
from jsindy.dynamics_model import FeatureLinearModel
from jsindy.residual_functions import (
    FullDataTerm,PartialDataTerm,CollocationTerm,
    JointResidual)
from jsindy.optim import DefaultOptimizer, LMSolver
default_optimizer = DefaultOptimizer()

class JSINDyModel():
    def __init__(
        self,
        trajectory_model:TrajectoryModel,
        dynamics_model:FeatureLinearModel,
        optimizer:LMSolver = LMSolver()
    ):
        self.traj_model = trajectory_model
        self.dynamics_model = dynamics_model
        self.optimizer = optimizer

    def initialize_fit(
        self,
        t,
        x,
        params = None,
        t_colloc = None
    ):
        if t_colloc is None:
            t_colloc = get_collocation_points(t)
        if params is None:
            params = dict()
        self.t = t
        self.x = x

        params = self.traj_model.initialize(
            self.t,self.x,t_colloc,params
            )
        
        params = self.dynamics_model.initialize(
            self.t,self.x,params
        )

        self.data_term = FullDataTerm(
            self.t,self.x,self.traj_model
        )
        self.colloc_term = CollocationTerm(
            t_colloc,self.traj_model,self.dynamics_model
        )
        self.residuals = JointResidual(self.data_term,self.colloc_term)
        return params
        
        
    def fit(
        self,
        t,
        x,
        params = None
    ):
        #TODO: Add a logs dictionary that's carried around in the same way that params is
        
        if params is None:
            params = dict()
        params = self.initialize_fit(t,x,params)
        z,theta,opt_result,params = self.optimizer.run(self,params)
        self.z = z
        self.theta = theta
        self.opt_result = opt_result
        self.params = params





    
        
