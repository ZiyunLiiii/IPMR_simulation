import numpy as np
import spekpy as sp
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from PIL import Image


def gen_spectrum(energy, filter_material, plot=False):
    # -----------------------
    # Generate spectrum
    # -----------------------
    s = sp.Spek(kvp=energy, th=12)
    s.filter(filter_material, 2.5)

    energies, spectrum = s.get_spectrum()

    # normalize to sum = 1
    weights = spectrum / np.sum(spectrum)

    energies = np.asarray(energies, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Remove possible zero or negative bins just in case
    mask = weights > 0
    energies = energies[mask]
    weights = weights[mask]

    # Weighted mean energy
    mean_E = np.sum(energies * weights) / np.sum(weights)

    # Weighted std
    std_E = np.sqrt(np.sum(weights * (energies - mean_E) ** 2) / np.sum(weights))

    print(f"Mean energy = {mean_E:.3f} keV")
    print(f"Std energy  = {std_E:.3f} keV")
    if plot:
        fig, ax = plt.subplots(figsize=(4.5, 3.0), dpi=300)
        ax.plot(energies, weights, lw=0.7, color='#4C72B0')

        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('Normalized weights')

        ax.set_xlim(0, 100)
        ax.set_ylim(0, weights.max() * 1.08)

        ax.tick_params(direction='out', length=4, width=0.8)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

        ax.grid(False)
        fig.tight_layout()
        plt.show()
    return energies, weights, mean_E, std_E



def get_material(material_name, config):
    m = config["materials"][material_name]

    E = np.array(m["energies"], dtype=float) * 1000  # MeV → keV
    mu = np.array(m["mu_over_rho"], dtype=float)
    rho = m["density"]

    return E, mu, rho



def align_energy_grid(E_src, mu_src, E_ref):
    """
    Align attenuation data mu(E) to a reference energy grid.

    Args:
        E_src : (N,) array
            Source energy grid (keV)
        mu_src : (N,) array
            Attenuation values (same length as E_src)
        E_ref : (M,) array
            Target energy grid (keV), e.g., spectrum energies

    Returns:
        mu_ref : (M,) array
            Interpolated mu on E_ref
    """

    # -------- ensure numpy --------
    E_src = np.asarray(E_src, dtype=float)
    mu_src = np.asarray(mu_src, dtype=float)
    E_ref = np.asarray(E_ref, dtype=float)

    # -------- remove duplicate energies (K-edge handling) --------
    E_unique = []
    mu_unique = []

    for e in np.unique(E_src):
        idx = np.where(E_src == e)[0]
        mu_unique.append(np.max(mu_src[idx]))  # keep post-edge
        E_unique.append(e)

    E_unique = np.array(E_unique)
    mu_unique = np.array(mu_unique)

    # -------- log-log interpolation --------
    interp_func = interp1d(
        np.log(E_unique),
        np.log(mu_unique),
        kind='linear',
        fill_value='extrapolate'
    )

    mu_ref = np.exp(interp_func(np.log(E_ref)))

    return mu_ref


def get_effective_attenuation(E_src, mu_src, energy_keV):
    """Evaluate attenuation at a single effective energy in keV."""
    return float(align_energy_grid(E_src, mu_src, np.array([energy_keV], dtype=float))[0])


def generate_polychromatic_sinogram(
    plastic_path_length,
    mu_plastic_interpolation,
    metal_path_lengths,
    metal_mu_interpolations,
    spectrum,
    I0=1.0,
):
    """Generate a beam-hardened sinogram for one plastic material and any number of metals."""

    if len(metal_path_lengths) != len(metal_mu_interpolations):
        raise ValueError("metal_path_lengths and metal_mu_interpolations must have the same length.")

    I = np.zeros_like(plastic_path_length, dtype=np.float32)

    for ie in range(len(spectrum)):
        line_integral = mu_plastic_interpolation[ie] * plastic_path_length
        for L_metal, mu_metal_interpolation in zip(metal_path_lengths, metal_mu_interpolations):
            line_integral += mu_metal_interpolation[ie] * L_metal
        I += spectrum[ie] * np.exp(-line_integral)

    I = np.clip(I, 1e-8, None)
    return -np.log(I / (I0 * spectrum.sum()))


def generate_cylinder_rod_phantom(
    nx=256,
    ny=256,
    nz=256,
    delta_voxel=0.2,
    plastic_radius=20.0,
    rod_radius=2.5,
    rod_centers=((-6.0, 0.0), (6.0, 0.0)),
    dtype=np.float32,
):
    """
    Generate a 3D phantom with one plastic cylinder and cylindrical metal rods.

    Returns:
        plastic_mask: array of shape (nx, ny, nz)
        metal_mask: array of shape (nx, ny, nz)
    """

    x = (np.arange(nx) - nx / 2 + 0.5) * delta_voxel
    y = (np.arange(ny) - ny / 2 + 0.5) * delta_voxel
    X, Y = np.meshgrid(x, y, indexing='ij')

    plastic_mask_2d = (X ** 2 + Y ** 2) <= plastic_radius ** 2

    metal_mask_2d = np.zeros((nx, ny), dtype=bool)
    for cx, cy in rod_centers:
        metal_mask_2d |= ((X - cx) ** 2 + (Y - cy) ** 2) <= rod_radius ** 2

    plastic_mask_2d &= ~metal_mask_2d

    plastic_mask = np.broadcast_to(plastic_mask_2d[:, :, None], (nx, ny, nz)).astype(dtype).copy()
    metal_mask = np.broadcast_to(metal_mask_2d[:, :, None], (nx, ny, nz)).astype(dtype).copy()

    return plastic_mask, metal_mask


def generate_cylinder_ring_rod_phantom(
    nx=256,
    ny=256,
    nz=256,
    delta_voxel=0.2,
    plastic_radius=20.0,
    rod_ring_radius=12.0,
    rod_radius=1.2,
    num_rods=16,
    angle_offset_deg=0.0,
    dtype=np.float32,
):
    """
    Generate a 3D phantom with one plastic cylinder and many thin metal rods
    placed uniformly on a circular ring.

    Returns:
        plastic_mask: array of shape (nx, ny, nz)
        metal_mask: array of shape (nx, ny, nz)
    """

    if num_rods < 1:
        raise ValueError("num_rods must be at least 1.")
    if rod_ring_radius + rod_radius > plastic_radius:
        raise ValueError("All rods must fit inside the plastic cylinder.")

    x = (np.arange(nx) - nx / 2 + 0.5) * delta_voxel
    y = (np.arange(ny) - ny / 2 + 0.5) * delta_voxel
    X, Y = np.meshgrid(x, y, indexing='ij')

    plastic_mask_2d = (X ** 2 + Y ** 2) <= plastic_radius ** 2

    angles_rad = np.linspace(0.0, 2.0 * np.pi, num_rods, endpoint=False)
    angles_rad += np.deg2rad(angle_offset_deg)

    metal_mask_2d = np.zeros((nx, ny), dtype=bool)
    for angle in angles_rad:
        cx = rod_ring_radius * np.cos(angle)
        cy = rod_ring_radius * np.sin(angle)
        metal_mask_2d |= ((X - cx) ** 2 + (Y - cy) ** 2) <= rod_radius ** 2

    plastic_mask_2d &= ~metal_mask_2d

    plastic_mask = np.broadcast_to(plastic_mask_2d[:, :, None], (nx, ny, nz)).astype(dtype).copy()
    metal_mask = np.broadcast_to(metal_mask_2d[:, :, None], (nx, ny, nz)).astype(dtype).copy()

    return plastic_mask, metal_mask


def generate_cylinder_ring_rod_two_metal_phantom(
    nx=256,
    ny=256,
    nz=256,
    delta_voxel=0.2,
    plastic_radius=20.0,
    rod_ring_radius=12.0,
    rod_radius=1.2,
    angle_offset_deg=0.0,
    metal_labels=("metal_1", "metal_2"),
    dtype=np.float32,
):
    """
    Generate a 3D phantom with one plastic cylinder and 6 thin rods on a ring.
    The rods alternate between two metal materials and are returned as a mask list.

    Returns:
        plastic_mask: array of shape (nx, ny, nz)
        metal_mask_list: list of arrays, one mask per metal material
    """

    if len(metal_labels) != 2:
        raise ValueError("metal_labels must contain exactly 2 entries.")
    if rod_ring_radius + rod_radius > plastic_radius:
        raise ValueError("All rods must fit inside the plastic cylinder.")

    x = (np.arange(nx) - nx / 2 + 0.5) * delta_voxel
    y = (np.arange(ny) - ny / 2 + 0.5) * delta_voxel
    X, Y = np.meshgrid(x, y, indexing='ij')

    plastic_mask_2d = (X ** 2 + Y ** 2) <= plastic_radius ** 2

    angles_rad = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    angles_rad += np.deg2rad(angle_offset_deg)

    metal_masks_2d = [np.zeros((nx, ny), dtype=bool) for _ in range(2)]
    for rod_index, angle in enumerate(angles_rad):
        cx = rod_ring_radius * np.cos(angle)
        cy = rod_ring_radius * np.sin(angle)
        mask = ((X - cx) ** 2 + (Y - cy) ** 2) <= rod_radius ** 2
        metal_masks_2d[rod_index % 2] |= mask

    combined_metal_mask_2d = np.zeros((nx, ny), dtype=bool)
    for metal_mask_2d in metal_masks_2d:
        combined_metal_mask_2d |= metal_mask_2d

    plastic_mask_2d &= ~combined_metal_mask_2d

    plastic_mask = np.broadcast_to(plastic_mask_2d[:, :, None], (nx, ny, nz)).astype(dtype).copy()
    metal_mask_list = [
        np.broadcast_to(metal_mask_2d[:, :, None], (nx, ny, nz)).astype(dtype).copy()
        for metal_mask_2d in metal_masks_2d
    ]

    return plastic_mask, metal_mask_list


def save_sinogram_gif(sinogram, output_path, duration_ms=40):
    """Save a cone-beam sinogram stack (views, rows, channels) as a GIF."""
    sino = np.asarray(sinogram, dtype=np.float32)
    sino_min = float(sino.min())
    sino_max = float(sino.max())
    if sino_max <= sino_min:
        sino_uint8 = np.zeros_like(sino, dtype=np.uint8)
    else:
        sino_uint8 = np.clip(255 * (sino - sino_min) / (sino_max - sino_min), 0, 255).astype(np.uint8)

    frames = [Image.fromarray(frame, mode="L") for frame in sino_uint8]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
