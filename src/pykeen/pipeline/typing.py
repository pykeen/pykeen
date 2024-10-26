from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

__all__ = [
    "PlotReturn",
]

PlotReturn = tuple["matplotlib.figure.Figure", list["matplotlib.axes.Axes"]]
