import numpy as np
from scipy.interpolate import LinearNDInterpolator

def obs_to_array(obs_data, xyz_topo):

    nx = 100
    ny = 100
    contourOpts={"cmap": "bwr"}
    xyz = xyz_topo
    data = obs_data
    v_max = np.max(np.abs(data))
    clim=(-v_max, v_max)
    vlimits = [np.min(clim), np.max(clim)]
    contourOpts = {}

    for i, key in enumerate(["vmin", "vmax"]):
        if key in contourOpts.keys():
            if vlimits[i] is None:
                vlimits[i] = contourOpts.pop(key)
            else:
                if not np.isclose(contourOpts[key], vlimits[i]):
                    raise Exception(
                        "The values provided in the colorbar limit, clim {} "
                        "does not match the value of {} provided in the "
                        "contourOpts: {}. Only one value should be provided or "
                        "the two values must be equal.".format(
                            vlimits[i], key, contourOpts[key]
                        )
                    )
                contourOpts.pop(key)
    vmin, vmax = vlimits[0], vlimits[1]

    # interpolate data to grid locations
    xmin, xmax = xyz[:, 0].min(), xyz[:, 0].max()
    ymin, ymax = xyz[:, 1].min(), xyz[:, 1].max()
    x = np.linspace(xmin, xmax, nx)
    y = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(x, y)
    xy = np.c_[X.flatten(), Y.flatten()]

    F = LinearNDInterpolator(xyz[:, :2], data)
    DATA = F(xy)
    DATA = DATA.reshape(X.shape)
    return DATA
