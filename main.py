import torch
import scipy.stats
import logging
import math
import typing
import toml
import collections


# Basic helper functions


def laplacian(q: torch.Tensor) -> torch.Tensor:
    """Calculates the discrete Laplacian of a PyTorch tensor

    Args:
        q: the tensor to use
    Returns:
        the discrete Laplacian of q
    """
    return sum(
        [
            torch.gradient(
                torch.gradient(q, dim=x, edge_order=1)[0], dim=x, edge_order=1
            )[0]
            for x in range(3)
        ]
    )


def welch_p(x1: torch.Tensor, x2: torch.Tensor) -> float:
    """Determines the p-value using Welch's unequal variants t-test

    Args:
        x1: first batch of samples
        x2: second batch of samples
    Returns:
        p-value from applying Welch's t-test to x1 and x2
    """
    return scipy.stats.ttest_ind(x1.numpy(), x2.numpy(), equal_var=False).pvalue


def score(grid: torch.Tensor, ab_map: torch.Tensor) -> float:
    """Scores the final results of a trial

    Args:
        grid: final state of the simulation field
        ab_map: absorption coefficient grid for the trial
    Returns:
        mean value of grid divided by the mean value of ab_map
    """
    return (torch.mean(grid) / torch.mean(ab_map)).item()


def toml_to_object(fields: dict, typename: str):
    """Converts a dictionary to a Python object recursively

    Args:
        fields: dictionary returned from tomllib.load(1)
        typename: name of type to be returned for docstring purposes
    Returns:
        Python object with properties and values based on fields
    """
    sanitized_keys = [k.replace("-", "_") for k in fields.keys()]
    res = collections.namedtuple(typename, sanitized_keys)
    local_fields = fields.fromkeys(sanitized_keys)
    for k, v in fields.items():
        if isinstance(v, dict):
            local_fields[k.replace("-", "_")] = toml_to_object(
                v, typename + "_" + k.replace("-", "_")
            )
        else:
            local_fields[k.replace("-", "_")] = v
    return res(**local_fields)


# Functions used to render individuals onto absorption and diffusion masks


def render_hemisphere(
    grid: torch.Tensor, position: torch.Tensor, color: float, radius: float
):
    """Renders a hemisphere onto the simulation field

    Args:
        grid: simulation field to render onto
        position: 3D point where the center of the bottom half of the hemisphere will be
        color: float to use to 'color in' the grid
        radius: radius of the hemisphere
    Returns:
        nothing
    """
    for y in range(math.ceil(radius)):
        small_rad = math.sqrt(radius**2 - y**2)
        for r in range(math.ceil(small_rad)):
            for theta in range(math.ceil(2 * small_rad * math.pi)):
                grid[
                    math.floor(position[0] + r * math.cos(theta / small_rad)),
                    math.floor(position[1] + y),
                    math.floor(position[2] + r * math.sin(theta / small_rad)),
                ] = color


def fractalize(
    origin: torch.Tensor,
    extent: torch.Tensor,
    depth: int,
    transform: torch.Tensor,
    density: int,
):
    """Recursively branches out a line

    Args:
        origin: origin of the initial line
        extent: extent of the initial line on each axis
        depth: depth of recurrence
        transform: matrix to apply to the children of each branch relative
            to their parent
        density: number of child branches per parent
    Returns:
        list of (origin, extent) tuples representing the lines of the final fractal
    """
    if depth <= 0:
        return [(origin, extent)]
    else:
        origins = [origin + extent * (x + 1) / density for x in range(density)]
        extents = [torch.matmul(transform, extent) for _ in range(density)]
        leaves = zip(origins, extents)
        descendants = [
            fractalize(leaf[0], leaf[1], depth - 1, transform, density)
            for leaf in leaves
        ]
        flat = [leaf for branch in descendants for leaf in branch]
        return flat + [(origin, extent)]


def render_line(
    grid: torch.Tensor, origin: torch.Tensor, extent: torch.Tensor, color: float
):
    """Renders a line onto the simulation field

    Args:
        grid: simulation field to render onto
        origin: origin of the line to render
        extent: extent of the line to render on each axis
        color: float to use to 'color in' the grid
    Returns:
        nothing
    """
    longest = torch.max(extent)
    for i in range(math.ceil(longest)):
        t = i / longest
        grid[tuple((origin + extent * t).long())] = color


def render_fractal(
    grid: torch.Tensor,
    position: torch.Tensor,
    color: float,
    initial_length: float,
    depth: int,
    density: int,
    transform: torch.Tensor,
):
    """Renders a fractal onto the simulation field

    Args:
        grid: simulation field to render onto
        position: 3D point where the base of the fractal will be
        color: float to use to 'color in' the grid
        initial_length: length of the "backbone" of the fractal
        depth: recursive depth of the fractal
        density: number of child branches per parent branch
        transform: matrix to apply to the children of each branch relative to their parent
    Returns:
        nothing
    """
    lines = fractalize(
        position, torch.Tensor([0, initial_length, 0]), depth, transform, density
    )
    for line in lines:
        render_line(grid, line[0], line[1], color)


# Generators to create absorption and diffusion masks, since a lot of code is shared


def gen_mask(
    shape: tuple[int, int, int],
    bg: float,
    fg: float,
    population: int,
    render: typing.Callable,
    rng: torch.Generator,
    **kwargs,
) -> typing.Iterable[torch.Tensor]:
    """Generates a coefficient grid

    Args:
        shape: dimensions of the simulation field
        bg: 'background' float used for rendering
        fg: 'foreground' float used for rendering
        population: number of individuals to render
        render: function that takes a simulation field, a position, and keyword args,
            and renders an individual onto the field based on those specifications.
        rng: random number generator to use
        kwargs: keyword args to pass into the render function
    Returns:
        a generator that yields new coefficient grids according to the args provided
    """
    while True:
        grid = torch.ones(shape) * bg
        for _ in range(population):
            render(
                grid,
                (0.8 * torch.rand(3, generator=rng) + 0.1)
                * torch.Tensor([shape[0], 0, shape[2]]),
                fg,
                **kwargs,
            )
        yield grid


# Super basic generator for the initial concentration grid


def gen_grid(
    shape: tuple[int, int, int], scale: float
) -> typing.Iterable[torch.Tensor]:
    """Generates the initial concentration grid

    Args:
        shape: dimensions of the simulation field
        scale: maximum concentration value across the grid
    Returns:
        a generator that yields new concentration grids
    """
    while True:
        yield scale * torch.rand(shape)


# Actual code for running simulations!


def do_trial(
    duration: float,
    df_map: torch.Tensor,
    ab_map: torch.Tensor,
    grid: torch.Tensor,
    spatial_res: float,
    temporal_res: float,
) -> torch.Tensor:
    """Runs a single simulation trial

    Args:
        duration: duration of the trial
        df_map: grid of diffusion coefficients
        ab_map: grid of absorption coefficients
        grid: initial concentration grid
        spatial_res: spatial resolution (in meters) of the simulation field
        temporal_res: temporal resolution (in seconds)
    Returns:
        the final concentration grid
    """
    for _ in range(int(duration // temporal_res)):
        grid += (df_map * laplacian(grid) - ab_map * grid) * spatial_res * temporal_res
    return grid


def run_trials(
    df_gen: typing.Iterable[torch.Tensor],
    ab_gen: typing.Iterable[torch.Tensor],
    grid_gen: typing.Iterable[torch.Tensor],
    spatial_res: float,
    temporal_res: float,
    num_trials: int,
    trial_duration: float,
) -> torch.Tensor:
    """Runs and scores a series of trials

    Args:
        df_gen: generator for diffusion coefficient grids
        ab_gen: generator for absorption coefficient grids
        grid_gen: generator for initial concentration grids
        spatial_res: spatial resolution (in meters) of the simulation field
        temporal_res: temporal resolution (in seconds)
        num_trials: number of trials to run
        trial_duration: duration of each trial
    Returns:
        Array of length num_trials containing the computed score of each trial
    """
    results = torch.zeros(num_trials)
    for i in range(num_trials):
        df_map = next(df_gen)
        ab_map = next(ab_gen)
        grid = next(grid_gen)
        logging.info(f"Running trial {i + 1} of {num_trials}...")
        trial = do_trial(
            trial_duration, df_map, ab_map, grid, spatial_res, temporal_res
        )
        logging.info(f"Trial {i + 1} complete.")
        results[i] = score(trial, ab_map)
    return results


def main():
    logging.basicConfig(
        format="%(levelname)s\t%(asctime)s:\t%(message)s", level=logging.INFO
    )

    with open("config.toml", "r") as config_file:
        cfg = toml_to_object(toml.load(config_file), "Config")

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
    df_test = gen_mask(
        field_shape,
        cfg.coeffs.bg.diffusion,
        cfg.coeffs.fg.diffusion,
        cfg.trial.population,
        render_fractal,
        test_rng,
        **fractal_params,
    )
    test_rng.set_state(orig_state)
    ab_test = gen_mask(
        field_shape,
        cfg.coeffs.bg.absorption,
        cfg.coeffs.fg.absorption,
        cfg.trial.population,
        render_fractal,
        test_rng,
        **fractal_params,
    )

    control_rng = torch.Generator(device="cuda:0")
    orig_state = control_rng.get_state()
    df_control = gen_mask(
        field_shape,
        cfg.coeffs.bg.diffusion,
        cfg.coeffs.fg.diffusion,
        cfg.trial.population,
        render_hemisphere,
        control_rng,
        **hemisphere_params,
    )
    control_rng.set_state(orig_state)
    ab_control = gen_mask(
        field_shape,
        cfg.coeffs.bg.absorption,
        cfg.coeffs.fg.absorption,
        cfg.trial.population,
        render_hemisphere,
        control_rng,
        **hemisphere_params,
    )

    initial_grid = gen_grid(field_shape, cfg.coeffs.initial_scale)

    logging.info("Running test trials...")
    test = run_trials(
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
    control = run_trials(
        df_control,
        ab_control,
        initial_grid,
        cfg.field.spatial_res,
        cfg.trial.temporal_res,
        cfg.experiment.num_trials,
        cfg.trial.duration,
    )
    logging.info("Control trials complete.")
    print(f"Control situation scores:\n{control.tolist()}")
    print(f"Test situation scores:\n{test.tolist()}")
    print(f"p = {welch_p(control, test)}")


if __name__ == "__main__":
    main()
