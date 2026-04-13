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


def create_cone_model(ct_model_config):
    angles = np.linspace(
        0,
        2 * np.pi,
        ct_model_config["num_views"],
        endpoint=False,
        dtype=np.float32,
    )
    sino_shape = (
        ct_model_config["num_views"],
        ct_model_config["num_det_rows"],
        ct_model_config["num_det_channels"],
    )
    ct_model = mj.ConeBeamModel(
        sino_shape,
        angles,
        ct_model_config["source_detector_dist"],
        ct_model_config["source_iso_dist"],
    )
    ct_model.set_params(
        delta_voxel=ct_model_config["delta_voxel"],
        sharpness=ct_model_config["sharpness"],
    )
    return ct_model


def build_single_metal_phantom(phantom_config, delta_voxel):
    phantom_name = phantom_config["phantom_name"]

    if phantom_name == "two_rods":
        plastic_mask, metal_mask = utilities.generate_cylinder_rod_phantom(
            nx=phantom_config["nx"],
            ny=phantom_config["ny"],
            nz=phantom_config["nz"],
            delta_voxel=delta_voxel,
            plastic_radius=phantom_config["plastic_radius"],
            rod_radius=phantom_config["rod_radius"],
            rod_centers=phantom_config["rod_centers"],
        )
        return plastic_mask, [metal_mask]

    if phantom_name == "ring_rods":
        plastic_mask, metal_mask = utilities.generate_cylinder_ring_rod_phantom(
            nx=phantom_config["nx"],
            ny=phantom_config["ny"],
            nz=phantom_config["nz"],
            delta_voxel=delta_voxel,
            plastic_radius=phantom_config["plastic_radius"],
            rod_ring_radius=phantom_config["rod_ring_radius"],
            rod_radius=phantom_config["rod_radius"],
            num_rods=phantom_config["num_rods"],
            angle_offset_deg=phantom_config["angle_offset_deg"],
        )
        return plastic_mask, [metal_mask]

    raise ValueError(f"Unsupported single-metal phantom: {phantom_name}")


def build_two_metal_phantom(phantom_config, delta_voxel):
    phantom_name = phantom_config["phantom_name"]

    if phantom_name == "six_rods":
        return utilities.generate_cylinder_ring_rod_two_metal_phantom(
            nx=phantom_config["nx"],
            ny=phantom_config["ny"],
            nz=phantom_config["nz"],
            delta_voxel=delta_voxel,
            plastic_radius=phantom_config["plastic_radius"],
            rod_ring_radius=phantom_config["rod_ring_radius"],
            rod_radius=phantom_config["rod_radius"],
            angle_offset_deg=phantom_config["angle_offset_deg"],
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
    material_config,
    phantom_config,
    ct_model_config,
    recon_config,
    output_config,
    is_two_metal,
):
    material_setup = load_material_setup(
        material_config["plastic_name"],
        material_config["metal_names"],
        material_config["tube_voltage"],
        material_config["filter_material"],
    )
    ct_model = create_cone_model(ct_model_config)
    delta_voxel = ct_model_config["delta_voxel"]

    if is_two_metal:
        plastic_mask, metal_mask_list = build_two_metal_phantom(phantom_config, delta_voxel)
    else:
        plastic_mask, metal_mask_list = build_single_metal_phantom(phantom_config, delta_voxel)

    if output_config["visualize_masks"]:
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
    if output_config["save_sinogram_path"]:
        utilities.save_sinogram_gif(sino_bh, output_config["save_sinogram_path"])

    weights_trans = ct_model.gen_weights(sino_bh, weight_type=recon_config["weight_type"])
    fdk_bh = ct_model.direct_recon(sino_bh)
    mbir_bh, recon_dict_bh = ct_model.recon(sino_bh, weights=weights_trans)

    recon_mar = mjp.recon_plastic_metal(
        ct_model,
        sino=sino_bh,
        weights=weights_trans,
        num_BH_iterations=recon_config["num_BH_iterations"],
        num_metal=len(material_config["metal_names"]),
        num_constraint_update_iter=recon_config["num_constraint_update_iter"],
        order=recon_config["order"],
        verbose=recon_config["verbose"],
        alpha=recon_config["alpha"],
        beta=recon_config["beta"],
        gamma=recon_config["gamma"],
    )

    if output_config["visualize_recon"]:
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
