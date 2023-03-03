from .utils.plots import (
    plot_text,
    plot_diagnostic,
    plot_pairwise_scatter,
)
import importlib.metadata as importlib_metadata


__version__ = importlib_metadata.version(__name__)
