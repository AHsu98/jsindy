import jax
from jsindy.optim.solvers.lm_solver import CholeskyLM, LMSettings
from jax.scipy.linalg import block_diag
import jax.numpy as jnp
from jsindy.util import full_data_initialize

class LMSolver():
    def __init__(
            self, 
            beta_reg = 1.,
            solver_settings =  LMSettings()
        ):
        self.solver_settings = solver_settings
        self.beta_reg = beta_reg
    
    def run(self, model, params):
        # init_params = params["init_params"]
        params["data_weight"] = 1/(params["sigma2_est"]+0.01)
        params["colloc_weight"] = 10

        # z_theta_init = jnp.zeros(
        #     model.traj_model.tot_params + model.dynamics_model.tot_params
        # )
        z0,theta0 = full_data_initialize(model.t,model.x,model.traj_model,model.dynamics_model)
        z_theta_init = jnp.hstack([z0,theta0.flatten()])

        def resid_func(z_theta):
            z = z_theta[:model.traj_model.tot_params]
            theta = z_theta[model.traj_model.tot_params:].reshape(
                model.dynamics_model.param_shape
            )
            return model.residuals.residual(
                z,
                theta,
                params["data_weight"], 
                params["colloc_weight"]
            )   

        jac_func = jax.jacrev(resid_func)
        damping_matrix = block_diag(
            model.traj_model.regmat,
            model.dynamics_model.regmat
        )

        lm_prob = LMProblem(resid_func, jac_func, damping_matrix)
        z_theta, opt_results = CholeskyLM(
            z_theta_init, 
            lm_prob,
            self.beta_reg,
            self.solver_settings
        )
        z = z_theta[:model.traj_model.tot_params]
        theta = z_theta[model.traj_model.tot_params:].reshape(
            model.dynamics_model.param_shape
        )

        return z, theta, opt_results, params

class LMProblem():
    def __init__(self,resid_func, jac_func, damping_matrix):
        self.resid_func = resid_func
        self.jac_func = jac_func
        self.damping_matrix = damping_matrix