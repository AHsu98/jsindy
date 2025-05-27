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
        x = None,
        y = None,
        v = None,
        ):
        self.t = t
        self.x = x
        self.y = y
        self.v = v

        is_partial_data = check_is_partial_data(t,x,y,v)
        if is_partial_data is True:
            self.traj_model = self.traj_model.initialize_partial(
                self.t,self.y,self.v
                )
            self.noise2_est = self.traj_model.noise2_est
            self.dynamics_model = self.dynamics_model.initialize_partial(
                self.t,self.y.self.v
            )
            self.data_term = PartialDataTerm(self.t,self.y,self.v,trajectory_model=self.traj_model)
        else:
            self.traj_model = self.traj_model.initialize_full(
                self.t,self.x
                )
            self.dynamics_model = self.dynamics_model.initialize_full(
                self.t,self.x
            )
            self.data_term = FullDataTerm(
                self.t,self.x,self.traj_model
            )
        
    def fit(
        self,t,
        x = None,y = None,v = None,
    ):
        self.initialize_fit(t,x,y,v)
        z,theta,opt_result = self.optimizer.run(self)
        self.z = z
        self.theta = theta
        self.opt_result = opt_result




    
        
