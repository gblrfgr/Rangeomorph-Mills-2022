import torch
import scipy
import collections
import logging
import argparse


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


def get_cmdline_args(argv: list[str]) -> argparse.Namespace:
    """Extracts the command line arguments

    Args:
        argv: command line arguments passed in to the program
    Returns:
        argparse.Namespace containing the relevant fields for our program
    """
    arg_parser = argparse.ArgumentParser(
        prog="Rangeomorph Simulation (Mills 2022)",
        description="implementation of experiment described in Mills 2022",
    )

    arg_parser.add_argument(
        "config", nargs=1, help="file to source configuration information from"
    )
    arg_parser.add_argument(
        "-o",
        "--out",
        nargs="?",
        required=True,
        help="name of file to output results to (csv format)",
    )
    arg_parser.add_argument(
        "--log-out",
        nargs="?",
        help="file to output logged messages to",
    )
    arg_parser.add_argument(
        "-l",
        "--log-level",
        nargs="?",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=(lambda s: logging.getLevelName(s)),
        default=logging.DEBUG,
    )
    return arg_parser.parse_args(argv)
