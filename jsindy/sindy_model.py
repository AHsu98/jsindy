import jax
import jax.numpy as jnp
from jsindy.util import check_is_partial_data
from jsindy.trajectory_model import TrajectoryModel
from jsindy.dynamics_model import FeatureLinearModel
from residual_functions import FullDataTerm,PartialDataTerm,CollocationTerm

default_optimizer = DefaultOptimizer()

class JSINDyModel():
    def __init__(
        self,
        trajectory_model:TrajectoryModel,
        dynamics_model:FeatureLinearModel,
        optimizer:Optimizer = default_optimizer
    ):
        self.traj_model = trajectory_model
        self.dynamics_model = dynamics_model
        self.optimizer = optimizer

    def initialize_fit(
        self,
        t,
        x,
        params = None
    ):
        if params is None:
            params = dict()
        self.t = t
        self.x = x

        params = self.traj_model.initialize_full(
            self.t,self.x,params
            )
        
        params = self.dynamics_model.initialize_full(
            self.t,self.x,params
        )

        self.data_term = FullDataTerm(
            self.t,self.x,self.traj_model
        )
        return params
        
        
    def fit(
        self,
        t,
        x,
        params = None
    ):
        if params is None:
            params = dict()
        params = self.initialize_fit(t,x,params)
        z,theta,opt_result,params = self.optimizer.run(self,params)
        self.z = z
        self.theta = theta
        self.opt_result = opt_result
        self.params = params
        




    
        
