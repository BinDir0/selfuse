import json
import pathlib

import numpy as np
import numpydantic
import pydantic


@pydantic.dataclasses.dataclass
class NormStats:
    mean: numpydantic.NDArray
    std: numpydantic.NDArray
    q01: numpydantic.NDArray | None = None  # 1st quantile
    q99: numpydantic.NDArray | None = None  # 99th quantile
    ignored_dims: numpydantic.NDArray | None = None  # 1.0 for dims to skip normalization


class RunningStats:
    """Compute running statistics of a batch of vectors."""

    def __init__(self):
        self._count = 0
        self._mean = None
        self._mean_of_squares = None
        self._min = None
        self._max = None
        self._histograms = None
        self._bin_edges = None
        self._num_quantile_bins = 5000  # for computing quantiles on the fly

    def update(self, batch: np.ndarray) -> None:
        """
        Update the running statistics with a batch of vectors.

        Args:
            vectors (np.ndarray): An array where all dimensions except the last are batch dimensions.
        """
        batch = batch.reshape(-1, batch.shape[-1])
        num_elements, vector_length = batch.shape
        if self._count == 0:
            self._mean = np.mean(batch, axis=0)
            self._mean_of_squares = np.mean(batch**2, axis=0)
            self._min = np.min(batch, axis=0)
            self._max = np.max(batch, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            if vector_length != self._mean.size:
                raise ValueError("The length of new vectors does not match the initialized vector length.")
            new_max = np.max(batch, axis=0)
            new_min = np.min(batch, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)

            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements

        batch_mean = np.mean(batch, axis=0)
        batch_mean_of_squares = np.mean(batch**2, axis=0)

        # Update running mean and mean of squares.
        self._mean += (batch_mean - self._mean) * (num_elements / self._count)
        self._mean_of_squares += (batch_mean_of_squares - self._mean_of_squares) * (num_elements / self._count)

        self._update_histograms(batch)

    def get_statistics(self) -> NormStats:
        """
        Compute and return the statistics of the vectors processed so far.

        Returns:
            dict: A dictionary containing the computed statistics.
        """
        if self._count < 2:
            raise ValueError("Cannot compute statistics for less than 2 vectors.")

        variance = self._mean_of_squares - self._mean**2
        stddev = np.sqrt(np.maximum(0, variance))
        q01, q99 = self._compute_quantiles([0.01, 0.99])
        return NormStats(mean=self._mean, std=stddev, q01=q01, q99=q99)

    def _adjust_histograms(self):
        """Adjust histograms when min or max changes."""
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            new_edges = np.linspace(self._min[i], self._max[i], self._num_quantile_bins + 1)

            # Redistribute the existing histogram counts to the new bins
            new_hist, _ = np.histogram(old_edges[:-1], bins=new_edges, weights=self._histograms[i])

            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def _update_histograms(self, batch: np.ndarray) -> None:
        """Update histograms with new vectors."""
        for i in range(batch.shape[1]):
            hist, _ = np.histogram(batch[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _compute_quantiles(self, quantiles):
        """Compute quantiles based on histograms."""
        results = []
        for q in quantiles:
            target_count = q * self._count
            q_values = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = np.searchsorted(cumsum, target_count)
                q_values.append(edges[idx])
            results.append(np.array(q_values))
        return results


class _NormStatsDict(pydantic.BaseModel):
    norm_stats: dict[str, NormStats]


def serialize_json(norm_stats: dict[str, NormStats]) -> str:
    """Serialize the running statistics to a JSON string."""
    return _NormStatsDict(norm_stats=norm_stats).model_dump_json(indent=2)


def deserialize_json(data: str) -> dict[str, NormStats]:
    """Deserialize the running statistics from a JSON string."""
    return _NormStatsDict(**json.loads(data)).norm_stats


def save(directory: pathlib.Path | str, norm_stats: dict[str, NormStats]) -> None:
    """Save the normalization stats to a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_json(norm_stats))


def load(directory: pathlib.Path | str) -> dict[str, NormStats]:
    """Load the normalization stats from a directory."""
    path = pathlib.Path(directory) / "norm_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    return deserialize_json(path.read_text())


class Normalizer:
    """Dict-aware normalizer with ignore_dim support, built on NormStats.

    Accepts both dict (keyed by "states"/"actions"/"motions"/...) and single
    array inputs.  Works with numpy arrays and torch tensors.
    """

    def __init__(
        self,
        norm_stats: dict[str, NormStats],
        output_max: float = 1.0,
        output_min: float = -1.0,
        range_eps: float = 1e-4,
    ):
        self.norm_stats = norm_stats
        self.output_max = output_max
        self.output_min = output_min
        self.range_eps = range_eps
        self._params: dict[str, dict[str, np.ndarray]] = {}
        for key, stats in norm_stats.items():
            self._params[key] = self._compute_params(stats)

    # ------------------------------------------------------------------
    # scale / offset (limits mode, same logic as LinearNormalizer._fit)
    # ------------------------------------------------------------------
    def _compute_params(self, stats: NormStats) -> dict[str, np.ndarray]:
        q01 = stats.q01 if stats.q01 is not None else (stats.mean - 2 * stats.std)
        q99 = stats.q99 if stats.q99 is not None else (stats.mean + 2 * stats.std)
        input_range = (q99 - q01).copy()
        small = input_range < self.range_eps
        input_range[small] = self.output_max - self.output_min
        scale = (self.output_max - self.output_min) / input_range
        offset = (self.output_min - scale * q01).copy()
        offset[small] = (self.output_max + self.output_min) / 2 - q01[small]

        if stats.ignored_dims is not None:
            mask = stats.ignored_dims.astype(bool)
            scale[mask] = 1.0
            offset[mask] = 0.0

        return {"scale": scale.astype(np.float32), "offset": offset.astype(np.float32)}

    # ------------------------------------------------------------------
    # ignore_dim  (same API as LinearNormalizer.ignore_dim)
    # ------------------------------------------------------------------
    def ignore_dim(self, key: str, dim: slice):
        """Ignore some dimensions when normalizing, e.g. the wrist rotation."""
        if key not in self.norm_stats:
            raise RuntimeError(f"Not initialized with key: {key}")
        stats = self.norm_stats[key]
        if stats.ignored_dims is None:
            stats.ignored_dims = np.zeros_like(stats.mean, dtype=np.float32)
        stats.ignored_dims[dim] = 1.0
        self._params[key] = self._compute_params(stats)

    # ------------------------------------------------------------------
    # normalize / unnormalize
    # ------------------------------------------------------------------
    def normalize(self, x):
        return self._apply(x, forward=True)

    def unnormalize(self, x):
        return self._apply(x, forward=False)

    def __call__(self, x):
        return self.normalize(x)

    def _apply(self, x, forward: bool):
        if isinstance(x, dict):
            return {
                key: self._normalize_single(val, self._params[key], forward)
                for key, val in x.items()
                if key in self._params
            }
        if len(self._params) == 1:
            key = next(iter(self._params))
            return self._normalize_single(x, self._params[key], forward)
        raise RuntimeError("Multiple keys in normalizer; pass a dict")

    @staticmethod
    def _normalize_single(x, params: dict, forward: bool):
        try:
            import torch
            is_torch = isinstance(x, torch.Tensor)
        except ImportError:
            is_torch = False

        scale = params["scale"]
        offset = params["offset"]
        if is_torch:
            scale = torch.from_numpy(scale).to(x.device)
            offset = torch.from_numpy(offset).to(x.device)

        src_shape = x.shape
        x = x.reshape(-1, scale.shape[0])
        if forward:
            x = x * scale + offset
        else:
            x = (x - offset) / scale
        return x.reshape(src_shape)