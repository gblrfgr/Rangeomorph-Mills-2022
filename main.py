import torch
import scipy.stats
import logging
import math
import typing


# Basic helper functions

def laplacian(q: torch.Tensor) -> torch.Tensor:
    return sum([torch.gradient(torch.gradient(q, dim=x, edge_order=1)[0], dim=x, edge_order=1)[0]
                for x in range(3)])


def welch_p(x1: torch.Tensor, x2: torch.Tensor) -> float:
    return scipy.stats.ttest_ind(x1.numpy(), x2.numpy(), equal_var=False).pvalue


def score(grid: torch.Tensor, ab_map: torch.Tensor) -> float:
    return (torch.mean(grid) / torch.mean(ab_map)).item()


# Functions used to render individuals onto absorption and diffusion masks

def __render_semisphere(grid: torch.Tensor, position: torch.Tensor, color: float, radius: float):
    for y in range(math.ceil(radius)):
        lesser_radius = math.sqrt(radius**2 - y**2)
        for r in range(math.ceil(lesser_radius)):
            for theta in range(math.ceil(2 * lesser_radius * math.pi)):
                grid[math.floor(position[0] + r * math.cos(theta / lesser_radius)),
                     math.floor(position[1] + y),
                     math.floor(position[2] + r * math.sin(theta / lesser_radius))] = color


def __fractalize(origin: torch.Tensor, extent: torch.Tensor, depth: int, transform: torch.Tensor,
                 density: int):
    if depth <= 0:
        return [(origin, extent)]
    else:
        origins = [origin + extent * (x + 1) / density for x in range(density)]
        extents = [torch.matmul(transform, extent) for _ in range(density)]
        leaves = zip(origins, extents)
        descendants = [__fractalize(leaf[0], leaf[1], depth - 1, transform, density)
                       for leaf in leaves]
        flat = [leaf for branch in descendants for leaf in branch]
        return flat + [(origin, extent)]


def __line_draw(grid: torch.Tensor, origin: torch.Tensor, extent: torch.Tensor, color: float):
    longest = torch.max(extent)
    for i in range(math.ceil(longest)):
        t = i / longest
        grid[tuple((origin + extent * t).long())] = color


def __render_fractal(grid: torch.Tensor, position: torch.Tensor, color: float, initial_length: int,
                     depth: int, density: int, transform: torch.Tensor):
    lines = __fractalize(position, torch.Tensor([0, initial_length, 0]),
                         depth, transform, density)
    for line in lines:
        __line_draw(grid, line[0], line[1], color)


# Generators to create absorption and diffusion masks, since a lot of code is shared

def gen_mask(shape: tuple[int, int, int], bg: float, fg: float, population: int, render: typing.Callable, rng: torch.Generator,
             **kwargs) -> typing.Iterable[torch.Tensor]:
    while True:
        grid = torch.ones(shape) * bg
        for _ in range(population):
            render(grid, (0.8 * torch.rand(3, generator=rng) + 0.1) * torch.Tensor([shape[0], 0, shape[2]]),
                   fg, **kwargs)
        yield grid


# Super basic generator for the initial concentration grid

def gen_grid(shape: tuple[int, int, int], scale: float) -> typing.Iterable[torch.Tensor]:
    while True:
        yield scale * torch.rand(shape)


# Actual code for running simulations!

def run_simulation(duration: int, fps: int, df_map: torch.Tensor, ab_map: torch.Tensor,
                   grid: torch.Tensor, spatial_res: float) -> torch.Tensor:
    for s in range(duration):
        for f in range(fps):
            grid += (df_map*laplacian(grid) - ab_map*grid) * spatial_res / fps
    return grid


def run_trials(df_gen: typing.Iterable[torch.Tensor], ab_gen: typing.Iterable[torch.Tensor],
               grid_gen: typing.Iterable[torch.Tensor], spatial_res: float, num_trials: int,
               trial_duration: int, fps: int) -> torch.Tensor:
    results = torch.zeros(num_trials)
    for i in range(num_trials):
        df_map = next(df_gen)
        ab_map = next(ab_gen)
        grid = next(grid_gen)
        logging.info(f"Running trial {i + 1} of {num_trials}...")
        trial = run_simulation(trial_duration, fps, df_map, ab_map, grid, spatial_res)
        logging.info(f"Trial {i + 1} complete.")
        results[i] = score(trial, ab_map)
    return results


def main():
    logging.basicConfig(format="%(levelname)s\t%(asctime)s:\t%(message)s", level=logging.INFO)

    if not torch.cuda.is_available():
        logging.warn("CUDA acceleration not available, defaulting to CPU.")
        torch.set_default_tensor_type(torch.cpu.FloatTensor)
    else:
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    SHAPE = (500, 250, 500)
    FRACTAL_PARAMS = {"initial_length": 100,
                      "depth": 4,
                      "density": 6,
                      "transform": 0.1 * torch.Tensor([[0.9698463, 0.0301537, 0.2418448],
                                                       [0.0301537, 0.9698463, -0.2418448],
                                                       [-0.2418448, 0.2418448, 0.9396926]])}
    COLONY_PARAMS = {"radius": 4.1}
    KD_ALT = 0.3
    KA_ALT = 0.7
    KD_BASE = 1.0
    KA_BASE = 0.0
    POP = 20
    SCALE_C = 0.03
    SPATIAL_RES = 5e-3
    NUM_TRIALS = 32
    DURATION = 60
    FPS = 20

    test_rng = torch.Generator(device="cuda:0")
    orig_state = test_rng.get_state()
    df_test = gen_mask(SHAPE, KD_BASE, KD_ALT, POP, __render_fractal, test_rng, **FRACTAL_PARAMS)
    test_rng.set_state(orig_state)
    ab_test = gen_mask(SHAPE, KA_BASE, KA_ALT, POP, __render_fractal, test_rng, **FRACTAL_PARAMS)
    control_rng = torch.Generator(device="cuda:0")
    orig_state = control_rng.get_state()
    df_control = gen_mask(SHAPE, KD_BASE, KD_ALT, POP, __render_semisphere,
                          control_rng, **COLONY_PARAMS)
    control_rng.set_state(orig_state)
    ab_control = gen_mask(SHAPE, KA_BASE, KA_ALT, POP, __render_semisphere,
                          control_rng, **COLONY_PARAMS)
    initial_grid = gen_grid(SHAPE, SCALE_C)

    logging.info("Running test trials...")
    test = run_trials(df_test, ab_test, initial_grid, SPATIAL_RES, NUM_TRIALS, DURATION, FPS)
    logging.info("Test trials complete.")
    logging.info("Running control trials...")
    control = run_trials(df_control, ab_control, initial_grid,
                         SPATIAL_RES, NUM_TRIALS, DURATION, FPS)
    logging.info("Control trials complete.")
    print(f"Control situation scores:\n{control.tolist()}")
    print(f"Test situation scores:\n{test.tolist()}")
    print(f"p = {welch_p(control, test)}")


if __name__ == '__main__':
    main()
