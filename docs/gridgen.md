```python
render_hemisphere(
	grid: torch.Tensor,
	position: torch.Tensor,
	color: float,
	radius: float
)
```

Renders the top half of a sphere onto a PyTorch `Tensor`, using `color` as the value to set points inside the sphere to. `position` denotes the position of the center of the sphere.

```python
render_fractal(
	grid: torch.Tensor,
	position: torch.Tensor,
	color: float,
	initial_length: float,
	depth: int,
	density: int,
	transform: torch.Tensor,
)
```

```python
gen_mask(
	shape: tuple[int, int, int],
	bg: float,
	fg: float,
	population: int,
	render: Callable,
	rng: torch.Generator,
	**kwargs,
) -> Iterable[torch.Tensor]
```

```python
gen_grid(
	shape: tuple[int, int, int],
	scale: float
) -> Iterable[torch.Tensor]
```

# Private Functions

```python
render_line(
	grid: torch.Tensor,
	origin: torch.Tensor,
	extent: torch.Tensor,
	color: float
)
```

Renders a line onto a PyTorch `Tensor` using `color` as the value to set points along the line to. `origin` is the base or "tail" of the line, and `extent` is the distance the line goes in along every axis.

```python
fractalize(
	origin: torch.Tensor,
	extent: torch.Tensor,
	depth: int,
	transform: torch.Tensor,
	density: int,
)
```