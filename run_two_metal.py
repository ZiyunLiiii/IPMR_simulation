from simulation_common import run_cone_metal_simulation


PLASTIC_NAME = "C2H4"
METAL_NAMES = ["Al", "Cu"]
PHANTOM_NAME = "six_rods"


if __name__ == "__main__":
    run_cone_metal_simulation(
        plastic_name=PLASTIC_NAME,
        metal_names=METAL_NAMES,
        phantom_name=PHANTOM_NAME,
        is_two_metal=True,
        visualize_masks=False,
        visualize_recon=True,
    )
