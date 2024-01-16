from susceptibility_model_data import *
from obs_to_tensor import obs_to_array

file_numbers = np.random.randint(0, 200, 4)

# Manually look through random 4 simulations
for file_number in file_numbers:
    fname_topo = f"data_simulation/outputs/magnetics_topo_{file_number}.txt"
    fname_data = f"data_simulation/outputs/magnetics_data_{file_number}.obs"
    fname_labels = f"data_simulation/sphere_data_labels/sphere_label_{file_number}.npy"
    obs, receiver_locations, topo_xyz = load_data(fname_topo, fname_data)
    print(load_labels(fname_labels))
    #print(obs)
    #print(receiver_locations)
    data_array = obs_to_array(obs, receiver_locations)
    plot_loaded_data(obs, receiver_locations)