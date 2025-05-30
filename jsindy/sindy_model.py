import jax
import jax.numpy as jnp
from jsindy.util import check_is_partial_data,get_collocation_points,get_equations
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
        self.feature_names = None

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
    
    def print(self,theta=None, precision: int = 3, **kwargs) -> None:
        """Print the SINDy model equations.
        precision: int, optional (default 3)
            Precision to be used when printing out model coefficients.
        **kwargs: Additional keyword arguments passed to the builtin print function
        """
        if theta is None:
            theta = self.theta
        eqns = get_equations(
            coef = theta.T,
            feature_names = self.feature_names,
            feature_library = self.dynamics_model.feature_map,
            precision = precision
            )
        if self.feature_names is None:
            feature_names = [f"x{i}" for i in range(len(eqns))]
        else:
            feature_names = self.feature_names

        for name, eqn in zip(feature_names, eqns, strict=True):
            lhs = f"({name})'"
            print(f"{lhs} = {eqn}", **kwargs)






    
        
