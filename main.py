import torch
import logging
import toml
import csv
import sys

from src import utils
from src import gridgen
from src import experiment


def main():
    args = utils.get_cmdline_args(sys.argv)

    with open("config.toml", "r") as config_file:
        cfg = utils.toml_to_object(toml.load(config_file), "Config")

    logging.basicConfig(
        format=cfg.logging.format,
        level=args.log_level,
        filename=args.log_out,
    )

    # Initialize coefficient grids
    if not torch.cuda.is_available():
        logging.warn("CUDA acceleration not available, defaulting to CPU.")
        torch.set_default_tensor_type(torch.cpu.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    field_shape = (
        int(cfg.field.width // cfg.field.spatial_res),
        int(cfg.field.height // cfg.field.spatial_res),
        int(cfg.field.depth // cfg.field.spatial_res),
    )

    fractal_params = {
        "initial_length": cfg.fractal.initial_length,
        "depth": cfg.fractal.depth,
        "density": cfg.fractal.density,
        "transform": cfg.fractal.transform.scale
        * torch.Tensor(cfg.fractal.transform.rotation),
    }
    hemisphere_params = {"radius": cfg.hemisphere.radius}

    test_rng = torch.Generator(device="cuda:0")
    orig_state = test_rng.get_state()
    df_test = gridgen.gen_mask(
        field_shape,
        cfg.coeffs.bg.diffusion,
        cfg.coeffs.fg.diffusion,
        cfg.trial.population,
        gridgen.render_fractal,
        test_rng,
        **fractal_params,
    )
    test_rng.set_state(orig_state)
    ab_test = gridgen.gen_mask(
        field_shape,
        cfg.coeffs.bg.absorption,
        cfg.coeffs.fg.absorption,
        cfg.trial.population,
        gridgen.render_fractal,
        test_rng,
        **fractal_params,
    )

    control_rng = torch.Generator(device="cuda:0")
    orig_state = control_rng.get_state()
    df_control = gridgen.gen_mask(
        field_shape,
        cfg.coeffs.bg.diffusion,
        cfg.coeffs.fg.diffusion,
        cfg.trial.population,
        gridgen.render_hemisphere,
        control_rng,
        **hemisphere_params,
    )
    control_rng.set_state(orig_state)
    ab_control = gridgen.gen_mask(
        field_shape,
        cfg.coeffs.bg.absorption,
        cfg.coeffs.fg.absorption,
        cfg.trial.population,
        gridgen.render_hemisphere,
        control_rng,
        **hemisphere_params,
    )

    # Run the actual trials
    initial_grid = gridgen.gen_grid(field_shape, cfg.coeffs.initial_scale)

    logging.info("Running test trials...")
    test = experiment.run_trials(
        df_test,
        ab_test,
        initial_grid,
        cfg.field.spatial_res,
        cfg.trial.temporal_res,
        cfg.experiment.num_trials,
        cfg.trial.duration,
    )
    logging.info("Test trials complete.")
    logging.info("Running control trials...")
    control = experiment.run_trials(
        df_control,
        ab_control,
        initial_grid,
        cfg.field.spatial_res,
        cfg.trial.temporal_res,
        cfg.experiment.num_trials,
        cfg.trial.duration,
    )
    logging.info("Control trials complete.")

    # Output results
    with open(args.out, "w") as output_file:
        writer = csv.DictWriter(
            output_file, ["Control situation scores", "Test situation scores"]
        )
        writer.writeheader()
        for c, t in zip(control.tolist(), test.tolist()):
            writer.writerow({"Control situation scores": c, "Test situation scores": t})

    logging.info(f"p = {utils.welch_p(control.cpu(), test.cpu())}")


if __name__ == "__main__":
    main()
