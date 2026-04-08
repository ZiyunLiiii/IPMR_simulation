import numpy as np
import mbirjax as mj
import mbirjax.preprocess as mjp
import yaml
from pathlib import Path
import utilities


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
E_metal_0, mu_over_rho_metal_0, rho_metal_0 = utilities.get_material(metal_name, config)
E_plastic, mu_over_rho_plastic, rho_plastic = utilities.get_material(plastic_name, config)

# Generate X-ray spectrum weighting
energies_keV, spectrum, mean_E, std_E = utilities.gen_spectrum(tube_voltage, 'Al')

# Convert to attenuation in mm^{-1}
mu_plastic_mm = rho_plastic * mu_over_rho_plastic / 10.0
mu_metal_mm      = rho_metal_0 * mu_over_rho_metal_0 / 10.0

mu_plastic_mm_interpolation = utilities.align_energy_grid(E_plastic, mu_plastic_mm, energies_keV)
mu_metal_mm_interpolation = utilities.align_energy_grid(E_metal_0, mu_metal_mm, energies_keV)

# ============================================================
# 2. Geometry
#    Following the demo pattern: create angles and MBIRJAX model
# ============================================================

num_views = 360
num_det_rows = 256
num_det_channels = 256

# Cone geometry requires source_detector_dist > source_iso_dist.
source_detector_dist = 1024
source_iso_dist = 512
delta_voxel = 0.2

angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False, dtype=np.float32)

# Sinogram shape convention: (views, rows, channels)
sino_shape = (num_views, num_det_rows, num_det_channels)

# Use a cone-beam model with voxel size matched to the phantom definition.
ct_model = mj.ConeBeamModel(sino_shape, angles, source_detector_dist, source_iso_dist)
ct_model.set_params(delta_voxel=delta_voxel, sharpness=0.0)


# ============================================================
# 3. Build a simple phantom:
#    one large plastic cylinder + two thin metal rods
# ============================================================

nx = 256
ny = 256
nz = 256



plastic_mask, metal_mask = utilities.generate_cylinder_rod_phantom(
    nx=nx,
    ny=ny,
    nz=nz,
    delta_voxel=delta_voxel,
    plastic_radius=20.0,
    rod_radius=2.5,
    rod_centers=((-6.0, 0.0), (6.0, 0.0)),
)
# mj.slice_viewer(plastic_mask, metal_mask)

mu_plastic_eff = utilities.get_effective_attenuation(E_plastic, mu_plastic_mm, mean_E)
mu_metal_eff = utilities.get_effective_attenuation(E_metal_0, mu_metal_mm, mean_E)
ground_truth = mu_plastic_eff * plastic_mask + mu_metal_eff * metal_mask

# ============================================================
# 4. Forward-project material path lengths
#
#    MBIRJAX projectors already include the model voxel pitch in the
#    line-integral coefficients, so the output is already in path-length units
#    consistent with ct_model.get_params('delta_voxel').
# ============================================================

L_plastic = ct_model.forward_project(plastic_mask)   # shape (views, rows, channels)
L_metal   = ct_model.forward_project(metal_mask)

# mj.slice_viewer(L_plastic, L_metal)

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
sino_bh = -np.log(I / (I0 * spectrum.sum()))
utilities.save_sinogram_gif(sino_bh, "cone_sinogram.gif")

# mj.slice_viewer(sino_bh, slice_axis=1)


# ============================================================
# 6. Reconstruct with MBIRJAX and FDK
# ============================================================

weights_trans = ct_model.gen_weights(sino_bh, weight_type='transmission_root')

FDK_bh = ct_model.direct_recon(sino_bh)
mbir_bh, recon_dict_bh = ct_model.recon(sino_bh, weights=weights_trans)

# MAR parameters
sharpness = 1.0
order = 3
verbose = 1
num_metal = 1
alpha = 1
beta = 0 # default is 0.002
gamma = 0.1  # default is 0.1
num_constraint_update_iter = 15

recon_mar = mjp.recon_plastic_metal(
    ct_model, sino=sino_bh, weights=weights_trans,
    num_BH_iterations=3,
    num_metal=num_metal,
    num_constraint_update_iter=num_constraint_update_iter,
    order=order,
    verbose=1,
    alpha=alpha,
    beta=beta,
    gamma=gamma,
)

segment_plastic = False
visualize_recon = True

if segment_plastic:
    plastic_mask_fdk, _, _, _ = mjp.segment_plastic_metal(FDK_bh, num_metal=num_metal)
    plastic_mask_mbir, _, _, _  = mjp.segment_plastic_metal(mbir_bh, num_metal=num_metal)
    plastic_mask_mar, _, _, _  = mjp.segment_plastic_metal(recon_mar, num_metal=num_metal)
    mj.slice_viewer(
        plastic_mask_fdk,
        plastic_mask_mbir,
        plastic_mask_mar,
        title='Segmented Plastic Masks',
        slice_label=['FDK plastic', 'MBIR plastic', 'MAR plastic'],
    )

if visualize_recon:
    mj.slice_viewer(
        ground_truth,
        FDK_bh,
        mbir_bh,
        recon_mar,
        title='Cone MAR Reconstruction Comparison',
        slice_label=['Ground truth', 'FDK', 'MBIR', 'MAR'],
    )
