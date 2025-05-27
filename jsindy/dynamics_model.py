import jax
import jax.numpy as jnp
from abc import ABC

class FeatureLinearModel():
    def __init__(
        self,
        feature_map,
        in_dim,
        out_dim,
        ):
        self.shape = ...
        self.feature_map = feature_map
        self.regularization_weights = ...

    def featurize(self,X):
        return self.feature_map(X)
    
    def __call__(self, X,theta):
        FX = self.featurize(X)
        return FX@theta
    
