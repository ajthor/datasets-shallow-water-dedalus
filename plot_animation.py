#!/usr/bin/env python3
"""
Generate an animation GIF of a single shallow water sample time evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from PIL import Image
from shallow_water_dataset import ShallowWaterDataset, build_s2_coord_vertices


def create_shallow_water_animation(
    sample, save_path="shallow_water_animation.gif", fps=2
):
    """Create an animated GIF showing vorticity evolution over time"""
    # Extract data
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

    # Get global vorticity range for consistent color scaling
    vmax_global = np.max(np.abs(vorticity_traj))

    # Create temporary directory for frames
    temp_dir = "temp_gif_frames"
    os.makedirs(temp_dir, exist_ok=True)

    # Generate frames
    frame_paths = []

    from mpl_toolkits.mplot3d import Axes3D

    for i, t in enumerate(time_traj):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        vort_data = vorticity_traj[i]
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
        ax.set_title(
            f"Vorticity at t={t:.1f}h (α={alpha:.3f}, β={beta:.3f})",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])
        mappable = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm)
        mappable.set_array(vort_data)

        # Position colorbar on the right side
        plt.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
        fig.colorbar(mappable, cax=cbar_ax, label="Vorticity (1/s)")

        # Save frame
        frame_path = os.path.join(temp_dir, f"frame_{i:03d}.png")
        plt.savefig(frame_path, dpi=100, bbox_inches="tight", facecolor="white")
        frame_paths.append(frame_path)
        plt.close()

    # Create GIF from frames
    images = []
    for frame_path in frame_paths:
        img = Image.open(frame_path)
        images.append(img)

    # Calculate duration per frame (in milliseconds)
    frame_duration = int(1000 / fps)

    # Save as GIF
    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        duration=frame_duration,
        loop=0,  # Loop forever
    )

    # Clean up temporary files
    for frame_path in frame_paths:
        os.remove(frame_path)
    os.rmdir(temp_dir)

    print(f"Animated GIF saved to {save_path}")
    print(f"Animation duration: {len(time_traj) * frame_duration / 1000:.1f} seconds")
    print(f"Frames per second: {fps}")

    return save_path


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(1)

    # Create dataset with default parameters
    dataset = ShallowWaterDataset(
        Nphi=256,
        Ntheta=128,
        stop_sim_time=600,
        save_interval=1,
    )

    # Generate a single sample
    print("Generating shallow water sample...")
    sample = next(iter(dataset))

    print("Creating animation...")
    print(f"Time steps: {len(sample['time_coordinates'])}")
    print(f"Vorticity grid size: {sample['Nphi']}×{sample['Ntheta']}")

    # Create animation
    create_shallow_water_animation(sample)
