import torch
import math
from typing import Iterable, Callable


def render_hemisphere(
    grid: torch.Tensor, position: torch.Tensor, color: float, radius: float
):
    """Renders a hemisphere onto the simulation field

    Args:
        grid: simulation field to render onto
        position: 3D point where the center of the bottom half of the
            hemisphere will be
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
        list of (origin, extent) tuples representing the lines of the final
            fractal
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
    render: Callable,
    rng: torch.Generator,
    **kwargs,
) -> Iterable[torch.Tensor]:
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


def gen_grid(shape: tuple[int, int, int], scale: float) -> Iterable[torch.Tensor]:
    """Generates the initial concentration grid

    Args:
        shape: dimensions of the simulation field
        scale: maximum concentration value across the grid
    Returns:
        a generator that yields new concentration grids
    """
    while True:
        yield scale * torch.rand(shape)
