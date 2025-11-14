from matplotlib import pyplot as plt
plt.style.use('ggplot')

SIM_KW = {
    "color": "purple",
    'alpha': 0.4,
    "lw": 5,
    "linestyle": "-",
    "label": "Simulated Dynamics",
    "solid_capstyle": "butt",
}
EST_KW = {
    "color": "blue",
    "lw": 5,
    "alpha": 0.4,
    "linestyle": "-",
    "label": "State Estimation",
    "zorder": 1,
    "solid_capstyle": "butt",
}
TRUE_KW = {
    "color": "black",
    "linestyle": "-",
    "label": "True Trajectory",
    "zorder": 2,
}
TRUE_UNS_KW = {
    "color": "black",
    "linestyle": "--",
    "label": "True Unseen Trajectory",
}
OBS_KW = {
    "facecolors":"black",
    "edgecolors": "red",
    "lw":1,
    "s": 10,
    "label": "Observations",
    "zorder": 3,
}
