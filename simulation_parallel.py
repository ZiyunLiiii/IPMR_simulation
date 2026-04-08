import numpy as np
import mbirjax as mj
import spekpy as sp
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from utilities import gen_spectrum, get_material, align_energy_grid, generate_cylinder_rod_phantom, get_effective_attenuation



# ============================================================
# 1. Load YAML file that contains material specifications
#    Generate X-ray Spectrum
# ============================================================
repo_dir = Path(__file__).resolve().parent

with open(repo_dir / "material_attenuation.yaml", "r") as f:
    config = yaml.safe_load(f)

metal_name = 'Al'
plastic_name = 'C2H4'
tube_voltage = 90

# NIST mass attenuation data in cm^2/g; density in g/cm^3
E_metal_0, mu_over_rho_metal_0, rho_metal_0 = get_material(metal_name, config)
E_plastic, mu_over_rho_plastic, rho_plastic = get_material(plastic_name, config)

# Generate X-ray spectrum weighting
energies_keV, spectrum, mean_E, std_E = gen_spectrum(tube_voltage, 'Al')

# Convert to attenuation in mm^{-1}
mu_plastic_mm = rho_plastic * mu_over_rho_plastic / 10.0
mu_metal_mm      = rho_metal_0 * mu_over_rho_metal_0 / 10.0

mu_plastic_mm_interpolation = align_energy_grid(E_plastic, mu_plastic_mm, energies_keV)
mu_metal_mm_interpolation = align_energy_grid(E_metal_0, mu_metal_mm, energies_keV)

# plt.plot(energies_keV, mu_metal_mm_interpolation, label='interpolated')
# plt.plot(E_metal_0, mu_metal_mm, 'o', label='original')
# plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

# ============================================================
# 2. Geometry
#    Following the demo pattern: create angles and MBIRJAX model
# ============================================================

num_views = 360
num_det_rows = 1
num_det_channels = 256

angles = np.linspace(0, np.pi, num_views, endpoint=False, dtype=np.float32)

# Sinogram shape convention: (views, rows, channels)
sino_shape = (num_views, num_det_rows, num_det_channels)

# Start with parallel beam for simplicity
ct_model = mj.ParallelBeamModel(sino_shape, angles)


# ============================================================
# 3. Build a simple phantom:
#    one large plastic cylinder + two thin metal rods
# ============================================================

nx = 256
ny = 256
nz = 1

delta_voxel = 0.2

plastic_mask, metal_mask = generate_cylinder_rod_phantom(
    nx=nx,
    ny=ny,
    nz=nz,
    delta_voxel=delta_voxel,
    plastic_radius=20.0,
    rod_radius=2.5,
    rod_centers=((-6.0, 0.0), (6.0, 0.0)),
)

mu_plastic_eff = get_effective_attenuation(E_plastic, mu_plastic_mm, mean_E)
mu_metal_eff = get_effective_attenuation(E_metal_0, mu_metal_mm, mean_E)
ground_truth = mu_plastic_eff * plastic_mask + mu_metal_eff * metal_mask


# ============================================================
# 4. Forward-project material path lengths
#
#    Since the masks are binary and voxel values are 1,
#    the projector output is effectively in voxel-count units.
#    Multiply by delta_voxel to convert to mm path length.
# ============================================================

L_plastic = ct_model.forward_project(plastic_mask)[:, 0, :] * delta_voxel   # shape (views, channels)
L_metal   = ct_model.forward_project(metal_mask)[:, 0, :]   * delta_voxel


# ============================================================
# 5. Polychromatic forward model
# ============================================================

# Incident intensity
I0 = 1.0

# Accumulate transmitted intensity
I = np.zeros_like(L_plastic, dtype=np.float32)

for ie in range(len(energies_keV)):
    line_integral = mu_plastic_mm_interpolation[ie] * L_plastic + mu_metal_mm_interpolation[ie] * L_metal
    I += spectrum[ie] * np.exp(-line_integral)

# Normalize and take negative log
I = np.clip(I, 1e-8, None)
sino_bh_2d = -np.log(I / (I0 * spectrum.sum()))

# Add detector row dimension back: (views, rows, channels)
sino_bh = sino_bh_2d[:, None, :]


# ============================================================
# 6. Optional monochromatic comparison
# ============================================================

# Select reference mono energy
mono_energy_keV = mean_E
print(f"Using mean-energy attenuation at {mono_energy_keV:.3f} keV for ground truth.")

mono_line_integral = mu_plastic_eff * L_plastic + mu_metal_eff * L_metal
sino_mono = mono_line_integral[:, None, :]


# ============================================================
# 7. Reconstruct with MBIRJAX
# ============================================================

ct_model.set_params(sharpness=0.0)

recon_bh, recon_dict_bh = ct_model.recon(sino_bh, weights=None)
recon_mono, recon_dict_mono = ct_model.recon(sino_mono, weights=None)
mj.slice_viewer(ground_truth, recon_bh, recon_mono)
