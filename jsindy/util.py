import jax
import jax.numpy as jnp

def validate_data_inputs(t,x,y,v):
    if x is None:
        assert y is not None
        assert v is not None
        assert len(t) == len(v)
        assert len(t) == len(y)
    if y is None:
        assert x is not None
        assert len(t) == len(x)
    if x is not None:
        assert y is None
        assert v is not None

def check_is_partial_data(t,x,y,v):
    validate_data_inputs(t,x,y,v)
    if v is None:
        return False
    else:
        return True
    
def get_collocation_points(t,num_colloc = 500):
    min_t = jnp.min(t)
    max_t = jnp.max(t)
    span = max_t - min_t
    lower = min_t - span/num_colloc
    upper = max_t + span/num_colloc
    return jnp.linspace(lower,upper,num_colloc)

@jax.jit
def l2reg_lstsq(A, y, reg=1e-10):
    U,sigma,Vt = jnp.linalg.svd(A, full_matrices=False)
    if jnp.ndim(y)==2:
        return Vt.T@(
            (sigma/(sigma**2+reg))[:,None]*(U.T@y)
            )
    else:
        return Vt.T@((sigma/(sigma**2+reg))*(U.T@y))

def tree_dot(tree, other):
    # Multiply corresponding leaves and sum each product over all its elements.
    vdots = jax.tree.map(lambda x, y: jnp.sum(x * y), tree, other)
    return jax.tree.reduce(lambda x, y: x + y, vdots, initializer=0.)

def tree_add(tree,other):
    return jax.tree.map(lambda x,y:x+y,tree,other)

def tree_scale(tree,scalar):
    return jax.tree.map(lambda x:scalar*x,tree)
