import numpy as np
import matplotlib
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

_DARK_BG    = "#0d0d0d"
_PANEL_BG   = "#1a1a1a"
_SPINE_COL  = "#3a3a3a"
_TICK_COL   = "#ffffff"
_LABEL_COL  = "#ffffff"
_TITLE_COL  = "#ffffff"

# Global matplotlib style — all text white so colorbars and annotations inherit it
matplotlib.rcParams.update({
    "text.color":        "#ffffff",
    "axes.labelcolor":   "#ffffff",
    "xtick.color":       "#ffffff",
    "ytick.color":       "#ffffff",
    "axes.edgecolor":    _SPINE_COL,
    "figure.facecolor":  _PANEL_BG,
    "axes.facecolor":    _DARK_BG,
    "grid.color":        _SPINE_COL,
})


class PlotCanvas(QWidget):
    """Reusable matplotlib canvas with dark theme, embedded toolbar."""

    def __init__(self, parent=None, nrows=1, ncols=1, figsize=(6, 4), dpi=100):
        super().__init__(parent)

        self.fig = Figure(figsize=figsize, dpi=dpi, facecolor=_PANEL_BG,
                          tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet(f"background-color: {_PANEL_BG};")

        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self._nrows = nrows
        self._ncols = ncols
        self._init_axes()

    # ------------------------------------------------------------------
    def _init_axes(self):
        self.fig.clear()
        raw = self.fig.subplots(self._nrows, self._ncols)
        # normalise to a flat list always
        if self._nrows == 1 and self._ncols == 1:
            self._axes_list = [raw]
        elif self._nrows == 1 or self._ncols == 1:
            self._axes_list = list(raw)
        else:
            self._axes_list = [ax for row in raw for ax in row]

        self.ax = self._axes_list[0]   # shortcut for single-axis use
        self.axes = raw                 # original shape

        for ax in self._axes_list:
            self._style_ax(ax)

        self.canvas.draw_idle()

    def _style_ax(self, ax):
        ax.set_facecolor(_DARK_BG)
        for spine in ax.spines.values():
            spine.set_color(_SPINE_COL)
        ax.tick_params(colors=_TICK_COL, which="both", labelsize=12)
        ax.xaxis.label.set_color(_LABEL_COL)
        ax.yaxis.label.set_color(_LABEL_COL)
        ax.title.set_color(_TITLE_COL)
        ax.grid(True, color=_SPINE_COL, linewidth=0.5, linestyle="--", alpha=0.5)

    # ------------------------------------------------------------------
    def clear(self):
        self._init_axes()

    def get_ax(self, index=0):
        return self._axes_list[index]

    def refresh(self):
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Convenience plot helpers
    def plot_line(self, x, y, ax_index=0, **kwargs):
        ax = self.get_ax(ax_index)
        line, = ax.plot(x, y, **kwargs)
        self.refresh()
        return line

    def plot_image(self, data, ax_index=0, cmap="inferno", aspect="auto", **kwargs):
        ax = self.get_ax(ax_index)
        im = ax.imshow(data, cmap=cmap, aspect=aspect,
                       origin="lower", **kwargs)
        self.refresh()
        return im
