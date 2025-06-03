import jax
from jsindy.optim.solvers.lm_solver import CholeskyLM, LMSettings
from jsindy.optim.solvers.alt_active_set_lm_solver import AlternatingActiveSolve
from jsindy.trajectory_model import TrajectoryModel
from jax.scipy.linalg import block_diag
import jax.numpy as jnp
from jsindy.util import full_data_initialize
from dataclasses import dataclass

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

class AlternatingActiveSetLMSolver():
    def __init__(
            self, 
            beta_reg = 1.,
            solver_settings =  LMSettings()
        ):
        self.solver_settings = solver_settings
        self.beta_reg = beta_reg

    def run(self, model, params):
        params["data_weight"] = 1/(params["sigma2_est"]+0.01)
        params["colloc_weight"] = 10

        z_theta_init = jnp.zeros(
            model.traj_model.tot_params + model.dynamics_model.tot_params
        )

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
        print("Warm Start")
        z_theta, lm_opt_results = CholeskyLM(
            z_theta_init, 
            lm_prob,
            self.beta_reg,
            self.solver_settings
        )
        z = z_theta[:model.traj_model.tot_params]
        theta = z_theta[model.traj_model.tot_params:].reshape(
            model.dynamics_model.param_shape
        )

        print("Alternating Activeset Sparsifier")

        def F_split(z, theta):
            data_weight = params["data_weight"]
            colloc_weight = params["colloc_weight"]
            return model.residuals.residual(z,theta,data_weight,colloc_weight)

        # fix this later
        aaslm_prob = AASLMProblem(
            system_dim = model.traj_model.system_dim,
            num_features = model.dynamics_model.num_features,
            F_split = F_split,
            t_colloc = model.t_colloc,
            interpolant=model.traj_model,
            state_param_regmat=model.traj_model.regmat, 
            model_param_regmat=model.dynamics_model.regmat,
            feature_library=model.dynamics_model.feature_map
        )
        
        z, theta, aas_lm_opt_results = AlternatingActiveSolve(
            z0=z,
            theta0=theta,
            residual_objective=aaslm_prob,
            beta=self.beta_reg,
        )
        theta = theta.reshape(
            model.dynamics_model.param_shape
        )

        return z, theta, [lm_opt_results,aas_lm_opt_results,], params

@dataclass
class AASLMProblem():
    system_dim: int
    num_features: int
    F_split: callable
    t_colloc: jax.Array
    interpolant: TrajectoryModel
    state_param_regmat: jax.Array
    model_param_regmat: jax.Array
    feature_library: callable



