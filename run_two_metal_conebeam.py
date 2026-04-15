import numpy as np
import os
from simulation_common import run_cone_metal_simulation


MATERIAL_CONFIG = {
    "plastic_name": "C2H4",
    "metal_names": ["Al", "Cu"],
    "tube_voltage": 90,
    "filter_material": "Al",
}

PHANTOM_CONFIG = {
    "phantom_name": "six_rods",
    "nx": 256,
    "ny": 256,
    "nz": 256,
    "plastic_radius": 20.0,
    "rod_ring_radius": 12.0,
    "rod_radius": 0.8,
    "angle_offset_deg": 0.0,
}

CT_MODEL_CONFIG = {
    "num_views": 360,
    "num_det_rows": 256,
    "num_det_channels": 256,
    "source_detector_dist": 1024,
    "source_iso_dist": 512,
    "delta_voxel": 0.2,
    "sharpness": 1.0,
}

RECON_CONFIG = {
    "weight_type": "transmission_root",
    "num_BH_iterations": 3,
    "num_constraint_update_iter": 15,
    "num_metal": 2,
    "order": 3,
    "verbose": 1,
    "alpha": 1,
    "beta": 0,
    "gamma": 0.1,
}

OUTPUT_CONFIG = {
    "save_sinogram_path": "cone_sinogram.gif",
    "visualize_masks": False,
    "visualize_recon": True,
}


if __name__ == "__main__":
    simulation_results = run_cone_metal_simulation(
        material_config=MATERIAL_CONFIG,
        phantom_config=PHANTOM_CONFIG,
        ct_model_config=CT_MODEL_CONFIG,
        recon_config=RECON_CONFIG,
        output_config=OUTPUT_CONFIG,
        is_two_metal=True
    )

    fdk_bh = simulation_results['fdk_bh']
    mbir_bh = simulation_results['mbir_bh']
    recon_mar = simulation_results['recon_mar']

    base_dir = ''
    save_dir = base_dir + f"{PHANTOM_CONFIG['phantom_name']}"
    os.makedirs(save_dir, exist_ok=True)

    np.save(save_dir + 'fdk_bh.npy', fdk_bh)
    np.save(save_dir + 'mbir_bh.npy', mbir_bh)
    np.save(save_dir + 'recon_mar.npy', recon_mar)
