import jax
import jax.numpy as jnp
from abc import ABC

class TrajectoryModel(ABC):
    system_dim:int
    def __call__(self, t, z):
        pass

    def derivative(self,t,z,order = 1):
        pass

