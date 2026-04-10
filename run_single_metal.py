from simulation_common import run_cone_metal_simulation


PLASTIC_NAME = "C2H4"
METAL_NAME = "Al"
PHANTOM_NAME = "two_rods"


if __name__ == "__main__":
    run_cone_metal_simulation(
        plastic_name=PLASTIC_NAME,
        metal_names=[METAL_NAME],
        phantom_name=PHANTOM_NAME,
        is_two_metal=False,
        visualize_masks=False,
        visualize_recon=True,
    )
