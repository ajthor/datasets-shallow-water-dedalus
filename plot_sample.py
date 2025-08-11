#!/usr/bin/env python3
"""
Plot a single sample from the shallow water dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from shallow_water_dataset import ShallowWaterDataset, build_s2_coord_vertices


def plot_shallow_water_sample(sample, save_path="sample_plot.png"):
    """Plot a 4-panel view of vorticity evolution"""
    # Extract data from comprehensive dataset output
    vorticity_traj = sample["vorticity_trajectory"]
    alpha = sample["alpha"]
    beta = sample["beta"]
    time_traj = sample["time_coordinates"]

    # Get shape directly from vorticity trajectory
    vorticity_shape = vorticity_traj.shape[1:]
    Nphi, Ntheta = vorticity_shape

    # Use 1D phi/theta for plotting, as in plot_sphere.py
    phi_1d = np.linspace(0, 2 * np.pi, Nphi, endpoint=False)
    theta_1d = np.linspace(0, np.pi, Ntheta)

    phi_vert, theta_vert = build_s2_coord_vertices(phi_1d, theta_1d)
    x = np.sin(theta_vert) * np.cos(phi_vert)
    y = np.sin(theta_vert) * np.sin(phi_vert)
    z = -np.cos(theta_vert)  # Flip poles

    # Select 4 time points: initial, two middle, and final
    n_times = len(time_traj)
    time_indices = [0, n_times // 3, 2 * n_times // 3, n_times - 1]
    time_labels = ["Initial", "Early", "Late", "Final"]

    # Get global vorticity range for consistent color scaling
    vmax_global = np.max(np.abs(vorticity_traj))

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(20, 5))
    axes = [fig.add_subplot(1, 4, i + 1, projection="3d") for i in range(4)]

    for i, (time_idx, label) in enumerate(zip(time_indices, time_labels)):
        ax = axes[i]
        vort_data = vorticity_traj[time_idx]
        # Use normalization as in plot_sphere.py
        import matplotlib

        norm = matplotlib.colors.Normalize(-vmax_global, vmax_global)
        fc = plt.cm.RdBu_r(norm(vort_data))
        ax.plot_surface(
            x,
            y,
            z,
            facecolors=fc,
            cstride=1,
            rstride=1,
            linewidth=0,
            antialiased=False,
            shade=False,
        )
        ax.set_title(f"{label} Vorticity (t={time_traj[time_idx]:.1f}h)", fontsize=12)
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])

    mappable = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
    mappable.set_array(vort_data)

    # Position colorbar on the right side
    plt.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(mappable, cax=cbar_ax, label="Vorticity (1/s)")

    fig.suptitle(
        f"Shallow Water Vorticity Evolution\\nα={alpha:.3f}, β={beta:.3f}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Sample visualization saved to {save_path}")
    print(f"Parameters: α={alpha:.3f}, β={beta:.3f}")
    print(f"Time steps saved: {len(time_traj)}")
    print(f"Vorticity grid size: {Nphi}×{Ntheta}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(1)

    # Create dataset with default parameters
    dataset = ShallowWaterDataset(
        Nphi=128,
        Ntheta=64,
        stop_sim_time=600,
        save_interval=1,
    )

    # Generate a single sample
    print("Generating shallow water sample...")
    sample = next(iter(dataset))

    print("Sample keys:", list(sample.keys()))
    for key, value in sample.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape {value.shape}")
        else:
            print(f"{key}: {type(value)}")

    # Create 4-panel plot
    plot_shallow_water_sample(sample)
