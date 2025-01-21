import ezc3d
import biorbd_casadi as biorbd
import numpy as np
import matplotlib.pyplot as plt
from casadi import MX, horzcat, DM
from bioptim import (
    BiorbdModel,
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    BoundsList,
    ObjectiveList,
    ObjectiveFcn,
    OdeSolver,
    OdeSolverBase,
    Node,
    Solver,
    BiMappingList,
    SolutionMerge,
)


def load_experimental_data(data_path, model_path):
    model = biorbd.Model(model_path)
    marker_names = [model.markerNames()[i].to_string() for i in range(model.nbMarkers())]
    # TODO: add muscles to the model
    # muscle_names = [model.muscleNames()[i].to_string() for i in range(model.nbMuscles())]
    c3d = ezc3d.c3d(data_path)
    marker_data = c3d["data"]["points"]
    marker_sampling_frequency = c3d["parameters"]["POINT"]["RATE"]["value"][0]  # Hz
    emg_data = c3d["data"]["analogs"]
    emg_sampling_frequency = c3d["parameters"]["ANALOG"]["RATE"]["value"][0]  # Hz
    n_frames = marker_data.shape[2]
    n_markers = marker_data.shape[1]
    time_vector = c3d["parameters"]["POINT"]["FRAMES"]["value"][0] * np.arange(n_frames)

    # Markers
    reordered_markers = np.zeros((3, n_markers, n_frames))
    for i_marker, marker_name in enumerate(marker_names):
        idx = c3d["parameters"]["POINT"]["LABELS"]["value"].index(marker_name.encode())
        reordered_markers[:, i_marker, :] = marker_data[idx, :, :]

    # # EMG
    # emg_reordered = np.zeros((len(muscle_names), n_frames))
    # for i_emg, muscle_name in enumerate(muscle_names):
    #     if muscle_name in c3d["parameters"]["ANALOG"]["LABELS"]["value"]:
    #         idx = c3d["parameters"]["ANALOG"]["LABELS"]["value"].index(muscle_name.encode())
    #         emg_reordered[i_emg, :] = emg_data[idx, :]

    # TODO: Check events for cycle start / stop and crop data

    exp_data = {
        "time_vector": time_vector,
        "marker_sampling_frequency": marker_sampling_frequency,
        "emg_sampling_frequency": emg_sampling_frequency,
        "markers": reordered_markers,
        # "emg": emg_reordered,
    }
    return  exp_data

def prepare_optimal_estimation(
    biorbd_model_path,
    time_ref,
    n_shooting,
    markers_ref,
):

    bio_model = BiorbdModel(biorbd_model_path)

    # Add objective functions
    objective_functions = ObjectiveList()
    objective_functions.add(
        ObjectiveFcn.Mayer.TRACK_MARKERS, node=Node.ALL, weight=100, target=markers_ref, quadratic=True
    )
    objective_functions.add(ObjectiveFcn.Mayer.MINIMIZE_TIME, weight=1, target=time_ref, quadratic=True)

    # Dynamics
    dynamics = DynamicsList()
    dynamics.add(DynamicsFcn.TORQUE_DRIVEN)

    # Path constraint
    x_bounds = BoundsList()
    x_bounds.add("q", bounds=bio_model.bounds_from_ranges("q"))
    x_bounds.add("qdot", bounds=bio_model.bounds_from_ranges("qdot"))

    # Define control path constraint
    n_tau = bio_model.nb_tau
    tau_min, tau_max = -5, 5
    u_bounds = BoundsList()
    u_bounds["tau"] = [tau_min] * n_tau, [tau_max] * n_tau

    return OptimalControlProgram(
        bio_model,
        dynamics,
        n_shooting,
        time_ref,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        objective_functions=objective_functions,
    )


def main():

    biorbd_model_path = "data/VIF_04.bioMod"
    experimental_data = load_experimental_data("data/VIF_04_Cond0001_processed6.c3d", biorbd_model_path)

    final_time = 3
    n_shooting = 30

    ocp_to_track = prepare_ocp_to_track(
        biorbd_model_path=biorbd_model_path, final_time=final_time, n_shooting=n_shooting
    )
    sol = ocp_to_track.solve()
    # sol.animate()
    # sol.graphs()

    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    controls = sol.decision_controls(to_merge=SolutionMerge.NODES)
    q, qdot, tau = states["q"], states["qdot"], controls["tau"]
    time = float(sol.decision_time(to_merge=SolutionMerge.NODES)[-1])

    model = biorbd.Model(biorbd_model_path)
    n_q = model.nbQ()
    n_marker = model.nbMarkers()

    symbolic_states = MX.sym("q", n_q, 1)
    markers_fun = biorbd.to_casadi_func("ForwardKin", model.markers, symbolic_states)
    markers_ref = np.zeros((3, n_marker, n_shooting + 1))
    for i_node in range(n_shooting + 1):
        markers_ref[:, :, i_node] = markers_fun(q[:, i_node])

    ocp = prepare_optimal_estimation(
        biorbd_model_path=biorbd_model_path,
        time_ref=time,
        n_shooting=n_shooting,
        markers_ref=markers_ref,
    )

    # --- Solve the program --- #
    sol = ocp.solve(Solver.IPOPT(show_online_optim=False))

    # --- Show results --- #
    states = sol.decision_states(to_merge=SolutionMerge.NODES)
    q = states["q"]

    markers_opt = np.zeros((3, n_marker, n_shooting + 1))
    for i_node in range(n_shooting + 1):
        markers_opt[:, :, i_node] = markers_fun(q[:, i_node])

    # --- Plot --- #
    plt.figure("Markers")
    for i in range(markers_opt.shape[1]):
        plt.plot(
            np.linspace(0, final_time, n_shooting + 1),
            markers_ref[:, i, :].T,
            "k",
        )
        plt.plot(
            np.linspace(0, final_time, n_shooting + 1),
            markers_opt[:, i, :].T,
            "r--",
        )
    plt.show()

    # sol.animate(show_tracked_markers=True)


if __name__ == "__main__":
    main()




