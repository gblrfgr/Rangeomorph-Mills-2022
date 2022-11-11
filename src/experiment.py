import torch
import logging
from src import utils
from typing import Iterable


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
        grid += (
            (df_map * utils.laplacian(grid) - ab_map * grid)
            * spatial_res
            * temporal_res
        )
    return grid


def run_trials(
    df_gen: Iterable[torch.Tensor],
    ab_gen: Iterable[torch.Tensor],
    grid_gen: Iterable[torch.Tensor],
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
        results[i] = utils.score(trial, ab_map)
    return results
