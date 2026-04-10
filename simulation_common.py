from pathlib import Path

import mbirjax as mj
import mbirjax.preprocess as mjp
import numpy as np
import yaml

import utilities


def load_material_setup(plastic_name, metal_names, tube_voltage, filter_material="Al"):
    repo_dir = Path(__file__).resolve().parent
    with open(repo_dir / "material_attenuation.yaml", "r") as f:
        config = yaml.safe_load(f)

    energies_keV, spectrum, mean_E, std_E = utilities.gen_spectrum(tube_voltage, filter_material)

    plastic = _prepare_material(plastic_name, config, energies_keV)
    metals = [_prepare_material(metal_name, config, energies_keV) for metal_name in metal_names]

    return {
        "plastic": plastic,
        "metals": metals,
        "energies_keV": energies_keV,
        "spectrum": spectrum,
        "mean_E": mean_E,
        "std_E": std_E,
    }


def _prepare_material(material_name, config, energies_keV):
    energies_src, mu_over_rho, density = utilities.get_material(material_name, config)
    mu_mm = density * mu_over_rho / 10.0
    mu_interp = utilities.align_energy_grid(energies_src, mu_mm, energies_keV)
    return {
        "name": material_name,
        "energies_src": energies_src,
        "mu_mm": mu_mm,
        "mu_interp": mu_interp,
    }


def create_cone_model(
    num_views=360,
    num_det_rows=256,
    num_det_channels=256,
    source_detector_dist=1024,
    source_iso_dist=512,
    delta_voxel=0.2,
    sharpness=1.0,
):
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False, dtype=np.float32)
    sino_shape = (num_views, num_det_rows, num_det_channels)
    ct_model = mj.ConeBeamModel(sino_shape, angles, source_detector_dist, source_iso_dist)
    ct_model.set_params(delta_voxel=delta_voxel, sharpness=sharpness)
    return ct_model


def build_single_metal_phantom(phantom_name, nx, ny, nz, delta_voxel):
    if phantom_name == "two_rods":
        plastic_mask, metal_mask = utilities.generate_cylinder_rod_phantom(
            nx=nx,
            ny=ny,
            nz=nz,
            delta_voxel=delta_voxel,
            plastic_radius=20.0,
            rod_radius=2.5,
            rod_centers=((-6.0, 0.0), (6.0, 0.0)),
        )
        return plastic_mask, [metal_mask]

    if phantom_name == "ring_rods":
        plastic_mask, metal_mask = utilities.generate_cylinder_ring_rod_phantom(
            nx=nx,
            ny=ny,
            nz=nz,
            delta_voxel=delta_voxel,
            plastic_radius=20.0,
            rod_ring_radius=12.0,
            rod_radius=1.2,
            num_rods=16,
            angle_offset_deg=0.0,
        )
        return plastic_mask, [metal_mask]

    raise ValueError(f"Unsupported single-metal phantom: {phantom_name}")


def build_two_metal_phantom(phantom_name, nx, ny, nz, delta_voxel):
    if phantom_name == "six_rods":
        return utilities.generate_cylinder_ring_rod_two_metal_phantom(
            nx=nx,
            ny=ny,
            nz=nz,
            delta_voxel=delta_voxel,
            plastic_radius=20.0,
            rod_ring_radius=12.0,
            rod_radius=0.8,
            angle_offset_deg=0.0,
        )

    raise ValueError(f"Unsupported two-metal phantom: {phantom_name}")


def build_mono_reference_volume(plastic_mask, metal_mask_list, plastic_material, metal_materials, mono_energy_keV):
    plastic_mu = utilities.get_effective_attenuation(
        plastic_material["energies_src"], plastic_material["mu_mm"], mono_energy_keV
    )
    reference_volume = plastic_mu * plastic_mask

    for metal_mask, metal_material in zip(metal_mask_list, metal_materials):
        metal_mu = utilities.get_effective_attenuation(
            metal_material["energies_src"], metal_material["mu_mm"], mono_energy_keV
        )
        reference_volume += metal_mu * metal_mask

    return reference_volume


def run_cone_metal_simulation(
    *,
    plastic_name,
    metal_names,
    phantom_name,
    is_two_metal,
    tube_voltage=90,
    filter_material="Al",
    delta_voxel=0.2,
    nx=256,
    ny=256,
    nz=256,
    sharpness=1.0,
    save_sinogram_path="cone_sinogram.gif",
    visualize_masks=False,
    visualize_recon=True,
):
    material_setup = load_material_setup(plastic_name, metal_names, tube_voltage, filter_material)
    ct_model = create_cone_model(delta_voxel=delta_voxel, sharpness=sharpness)

    if is_two_metal:
        plastic_mask, metal_mask_list = build_two_metal_phantom(phantom_name, nx, ny, nz, delta_voxel)
    else:
        plastic_mask, metal_mask_list = build_single_metal_phantom(phantom_name, nx, ny, nz, delta_voxel)

    if visualize_masks:
        mj.slice_viewer(plastic_mask, *metal_mask_list, title="Phantom Masks")

    mono_energy_keV = material_setup["mean_E"]
    mono_reference_volume = build_mono_reference_volume(
        plastic_mask,
        metal_mask_list,
        material_setup["plastic"],
        material_setup["metals"],
        mono_energy_keV,
    )

    plastic_path_length = ct_model.forward_project(plastic_mask)
    metal_path_lengths = [ct_model.forward_project(metal_mask) for metal_mask in metal_mask_list]

    sino_bh = utilities.generate_polychromatic_sinogram(
        plastic_path_length=plastic_path_length,
        mu_plastic_interpolation=material_setup["plastic"]["mu_interp"],
        metal_path_lengths=metal_path_lengths,
        metal_mu_interpolations=[metal["mu_interp"] for metal in material_setup["metals"]],
        spectrum=material_setup["spectrum"],
    )
    if save_sinogram_path:
        utilities.save_sinogram_gif(sino_bh, save_sinogram_path)

    weights_trans = ct_model.gen_weights(sino_bh, weight_type="transmission_root")
    fdk_bh = ct_model.direct_recon(sino_bh)
    mbir_bh, recon_dict_bh = ct_model.recon(sino_bh, weights=weights_trans)

    recon_mar = mjp.recon_plastic_metal(
        ct_model,
        sino=sino_bh,
        weights=weights_trans,
        num_BH_iterations=3,
        num_metal=len(metal_names),
        num_constraint_update_iter=15,
        order=3,
        verbose=1,
        alpha=1,
        beta=0,
        gamma=0.1,
    )

    if visualize_recon:
        mj.slice_viewer(
            mono_reference_volume,
            fdk_bh,
            mbir_bh,
            recon_mar,
            title="Cone MAR Reconstruction Comparison",
            slice_label=["Mono reference phantom", "FDK", "MBIR", "MAR"],
        )

    return {
        "ct_model": ct_model,
        "mono_energy_keV": mono_energy_keV,
        "mono_reference_volume": mono_reference_volume,
        "plastic_mask": plastic_mask,
        "metal_mask_list": metal_mask_list,
        "sino_bh": sino_bh,
        "weights_trans": weights_trans,
        "fdk_bh": fdk_bh,
        "mbir_bh": mbir_bh,
        "recon_mar": recon_mar,
        "recon_dict_bh": recon_dict_bh,
    }
