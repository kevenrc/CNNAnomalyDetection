import numpy as np
import os
from susceptibility_model_data import (
    define_topography,
    define_observation_locations,
    define_survey_and_receiver_list,
    define_tensor_mesh,
    define_susceptibility_model,
    simulate_data,
    export_data,
    save_labels
)
    
    
if __name__ == "__main__":
    num_data = 200
    for i in range(num_data):
        num_spheres = np.random.randint(1, 5)
        xyz_topo = define_topography()
        x_topo = xyz_topo[:, 0]
        y_topo = xyz_topo[:, 1]
        z_topo = xyz_topo[:, 2]
        receiver_locations = define_observation_locations(x_topo, y_topo, z_topo)
        survey, receiver_list = define_survey_and_receiver_list(receiver_locations)
        mesh = define_tensor_mesh()
        model, model_map, ind_active, sphere_labels = define_susceptibility_model(mesh, xyz_topo, num_spheres=num_spheres)
        dpred = simulate_data(survey, mesh, model, model_map, ind_active)
        save_labels(sphere_labels, i)
        export_data(xyz_topo, dpred, receiver_locations, i)