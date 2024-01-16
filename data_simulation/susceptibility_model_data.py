import numpy as np
from scipy.interpolate import LinearNDInterpolator
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

from discretize import TensorMesh
from discretize.utils import mkvc, active_from_xyz
from SimPEG.utils import plot2Ddata, model_builder
from SimPEG import maps
from SimPEG.potential_fields import magnetics

def define_topography():
    [x_topo, y_topo] = np.meshgrid(np.linspace(-200, 200, 41), np.linspace(-200, 200, 41))

    random_perturbation = np.random.normal(scale=1.5, size=x_topo.shape)
    z_topo = -15 * np.exp(-(x_topo**2 + y_topo**2) / 80**2) + random_perturbation
    x_topo, y_topo, z_topo = mkvc(x_topo), mkvc(y_topo), mkvc(z_topo)
    xyz_topo = np.c_[x_topo, y_topo, z_topo]
    return xyz_topo

def define_observation_locations(x_topo, y_topo, z_topo):
    x = np.linspace(-200.0, 200.0, 17)
    y = np.linspace(-200.0, 200.0, 17)
    x, y = np.meshgrid(x, y)
    x, y = mkvc(x.T), mkvc(y.T)
    fun_interp = LinearNDInterpolator(np.c_[x_topo, y_topo], z_topo)
    z = fun_interp(np.c_[x, y]) + 10  # Flight height 10 m above surface.
    receiver_locations = np.c_[x, y, z]
    return receiver_locations

def define_survey_and_receiver_list(receiver_locations):
    # Define the component(s) of the field we want to simulate as a list of strings.
    # Here we simulation total magnetic intensity data.
    components = ["tmi"]

    # Use the observation locations and components to define the receivers. To
    # simulate data, the receivers must be defined as a list.
    receiver_list = magnetics.receivers.Point(receiver_locations, components=components)

    receiver_list = [receiver_list]

    # Define the inducing field H0 = (intensity [nT], inclination [deg], declination [deg])
    inclination = 90
    declination = 0
    strength = 50000
    inducing_field = (strength, inclination, declination)

    source_field = magnetics.sources.SourceField(
        receiver_list=receiver_list, parameters=inducing_field
    )

    # Define the survey
    survey = magnetics.survey.Survey(source_field)
    return survey, receiver_list

def define_tensor_mesh():
    dh = 5.0
    hx = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
    hy = [(dh, 5, -1.3), (dh, 40), (dh, 5, 1.3)]
    hz = [(dh, 5, -1.3), (dh, 15)]
    mesh = TensorMesh([hx, hy, hz], "CCN")
    return mesh

def define_susceptibility_model(mesh, xyz_topo, num_spheres=3):
    # Define susceptibility values for each unit in SI
    background_susceptibility = 0.0001
    sphere_susceptibility = 0.01

    # Find cells that are active in the forward modeling (cells below surface)
    ind_active = active_from_xyz(mesh, xyz_topo)

    # Define mapping from model to active cells
    nC = int(ind_active.sum())
    model_map = maps.IdentityMap(nP=nC)  # model is a value for each active cell

    # Define model. Models in SimPEG are vector arrays
    model = background_susceptibility * np.ones(ind_active.sum())
    
    # Set minimum distance between sphere centers
    min_distance = 25.0  # Adjust based on your requirements
    
    x_locs, y_locs, z_locs, radii = [], [], [], []
    for i in range(num_spheres):
        # Generate random values for sphere location and radius
        x_loc = np.random.uniform(-150, 150)
        y_loc = np.random.uniform(-150, 150)
        z_loc = np.random.uniform(-120, -30)
        radius = np.random.uniform(7, 20)  # Adjust the range based on your needs

        # Check if the new sphere is too close to existing ones
        for j in range(i):
            try:
                distance = np.linalg.norm(np.array([x_loc, y_loc, z_loc]) - np.array([x_locs[j], y_locs[j], z_locs[j]]))
            except:
                i -= 1
                break
            if distance < min_distance:
                # If too close, regenerate the random values
                i -= 1
                break
        else:
            # If not too close, proceed to update sphere parameters
            x_locs.append(x_loc)
            y_locs.append(y_loc)
            z_locs.append(z_loc)
            radii.append(radius)

            # Get indices for the sphere in the active cells
            ind_sphere = model_builder.get_indices_sphere(
                np.r_[x_loc, y_loc, z_loc], radius, mesh.cell_centers
            )
            ind_sphere = ind_sphere[ind_active]
            
            # Assign susceptibility value for the sphere
            model[ind_sphere] = sphere_susceptibility

    return model, model_map, ind_active, np.column_stack((x_locs, y_locs, z_locs, radii))

def simulate_data(survey, mesh, model, model_map, ind_active):
    simulation = magnetics.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        model_type="scalar",
        chiMap=model_map,
        ind_active=ind_active,
        store_sensitivities="forward_only",
    )

    # Compute predicted data for a susceptibility model
    dpred = simulation.dpred(model)
    return dpred

def export_data(xyz_topo, dpred, receiver_locations, filenum):
    dir_path = os.path.dirname(__file__).split(os.path.sep)
    dir_path.extend(["outputs"])
    dir_path = os.path.sep.join(dir_path) + os.path.sep

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    fname_topo = dir_path + f"magnetics_topo_{filenum}.txt"
    np.savetxt(fname_topo, np.c_[xyz_topo], fmt="%.4e")

    #np.random.seed(211)
    maximum_anomaly = np.max(np.abs(dpred))
    noise = 0.02 * maximum_anomaly * np.random.randn(len(dpred))
    fname_data = dir_path + f"magnetics_data_{filenum}.obs"
    np.savetxt(fname_data, np.c_[receiver_locations, dpred + noise], fmt="%.4e")

def load_data(fname_topo, fname_data):
    topo_xyz = np.loadtxt(str(fname_topo))
    dobs = np.loadtxt(str(fname_data))

    receiver_locations = dobs[:, 0:3]
    dobs = dobs[:, -1]
    return dobs, receiver_locations, topo_xyz

def plot_loaded_data(dobs, receiver_locations):
    fig = plt.figure(figsize=(6, 5))
    v_max = np.max(np.abs(dobs))

    ax1 = fig.add_axes([0.1, 0.1, 0.75, 0.85])
    plot2Ddata(
        receiver_locations,
        dobs,
        ax=ax1,
        ncontour=30,
        clim=(-v_max, v_max),
        contourOpts={"cmap": "bwr"},
    )
    ax1.set_title("TMI Anomaly from loaded data")
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")

    ax2 = fig.add_axes([0.85, 0.05, 0.05, 0.9])
    norm = mpl.colors.Normalize(vmin=-np.max(np.abs(dobs)), vmax=np.max(np.abs(dobs)))
    cbar = mpl.colorbar.ColorbarBase(
        ax2, norm=norm, orientation="vertical", cmap=mpl.cm.bwr
    )
    cbar.set_label("$nT$", rotation=270, labelpad=15, size=12)

    plt.show()

def save_labels(sphere_labels, filenum):
    label_folder = "data_simulation/sphere_data_labels"
    os.makedirs(label_folder, exist_ok=True)
    label_filename = os.path.join(label_folder, f"sphere_label_{filenum}.npy")
    np.save(label_filename, sphere_labels)

def load_labels(file_path):
    sphere_data = np.load(file_path)
    return sphere_data

