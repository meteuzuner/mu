from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.io import fits
from astropy.wcs import WCS
from typing import Sequence, Tuple, Union, Optional
from mu.utils import assert_same_length

ColormapSpec = str | dict | list[str] | list[dict] | None

######################################################################################################### 
def plot_wcs(
    data_list:   list[np.ndarray],
    header_list: list[fits.Header],
    labels:      list[str],
    *,
    orientation: str = "horizontal",
    plot_style:  str | None = None,
    save:        bool = False,
    output_path: str | None = None,
    overwrite:   bool = False,
    colormaps:   ColormapSpec | None = "inferno",
    percentile_range: tuple[float, float] = (1, 99),
    vmin_vmax: Optional[
        Union[Tuple[float, float], Sequence[Tuple[float, float]]]
    ] = None,
    minimal:    bool  = False,
    pad_inches: float = 0.1,
) -> None:
    """
    Plot one or multiple 2D data arrays in their WCS coordinates, 
    each in a separate subplot. For each subplot, you can specify 
    either a standard continuous colormap or a custom discrete colormap 
    using boundaries.

    Parameters
    ----------
    data_list : list of np.ndarray
        A list of 2D data arrays to be plotted.
    header_list : list of fits.Header
        A list of corresponding FITS headers, describing each data's WCS.
    labels : list of str
        A list of labels for each data array (e.g., wavelengths or "Mass", "Temp").
    orientation : Literal["horizontal", "vertical"], optional
        Orientation of the subplots. If "horizontal", all subplots are in one row.
        If "vertical", all subplots are in one column. Default is "horizontal".
    plot_style : str, optional
        Path to a matplotlib style file (e.g., 'my-style.mplstyle'). If provided,
        `plt.style.use(plot_style)` will be applied.
    save : bool, optional
        If True, saves the final figure to `output_path`. Default is False.
    output_path : str, optional
        The file path where the figure will be saved. If save is True, this must be provided.
    overwrite : bool, optional
        If True, overwrite any existing file at `output_path`. Default is False.
    colormaps : str or dict or list of str/dict, optional
        This can be:
          - A single string (e.g. "viridis"): applies that continuous colormap to all subplots.
          - A single dict with "boundaries" and "colors": applies that custom discrete colormap to all subplots.
          - A list of strings (e.g. ["viridis", "plasma"]): each subplot uses a different built-in colormap.
          - A list of dicts (e.g. [{...}, {...}]): each subplot uses a different custom discrete colormap.
          - A mix of strings/dicts in a list. The length of the list must match len(data_list).
        Example of a dict for a custom discrete colormap:
          {
              "boundaries": [0.5, 1.0, 1.5, 999999],
              "colors": ["red", "green", "yellow"]
          }
        Default is "inferno".
    percentile_range : (float, float), optional
        The (low, high) percentile for stretching continuous colormaps,
        used when a discrete custom colormap isn't given. Default is (1, 99).
    vmin_vmax : tuple or list of tuples, optional
        The (min, max) values for the color scale. If it is none, percentile_range will be used.
    minimal : bool, optional
        If True, the plot will have minimal styling: no axis labels, ticks, or grid
    pad_inches : float, optional
        Amount of padding in inches around the figure when bbox_inches is 'tight'. Default is 0.1.
    Returns
    -------
    None
    """
    # --- 0. Inital setup and checks
    assert_same_length(data_list, header_list, labels)

    orientation = orientation.lower()
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError('orientation must be "horizontal" or "vertical"')

    # make vmin_vmax a list-of-pairs
    if vmin_vmax is not None:
        if isinstance(vmin_vmax, (tuple, list)) and len(vmin_vmax) == 2 \
           and not isinstance(vmin_vmax[0], (tuple, list)):
            vmin_vmax = [tuple(vmin_vmax)] * len(data_list)
        assert_same_length(vmin_vmax, data_list)


    # --- 1. (optional) Apply custom style if provided
    if plot_style is not None:
        p = Path(plot_style)
        if not p.exists():
            raise FileNotFoundError(p)
        plt.style.use(p)

    # --- 2. Figure size
    n_plots        = len(data_list)
    h_pix, w_pix   = data_list[0].shape
    aspect         = w_pix / h_pix          # width/height
    fig_w          = 10/3                   # MNRAS width in inches
    nrows, ncols   = (1, n_plots) if orientation=="horizontal" else (n_plots, 1)
    subplot_w      = fig_w / ncols
    subplot_h      = subplot_w / aspect
    fig_h          = subplot_h * nrows
    fig            = plt.figure(figsize=(fig_w, fig_h))


    # --- 3. Helper for colormap handling
    def _get_cmap(cmaps: Union[str, dict, list[str], list[dict], None], i: int):
        # If no colormap specified, fallback to "inferno"
        if cmaps is None:
            return "inferno"

        # If it's a single string or dict, apply it to all subplots
        if isinstance(cmaps, (str, dict)):
            return cmaps
        
        # If it's a list, pick the i-th element
        return cmaps[i]


    # --- 4. Iterate over each data/header pair
    for i, (data, hdr, lab) in enumerate(zip(data_list, header_list, labels)):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection=WCS(hdr))

        # 5a. Axes: labels, ticks, visibility
        if minimal:
            ax.set_frame_on(False)
            for c in ax.coords:
                c.set_axislabel('')
                c.set_ticklabel_visible(False)
                c.set_ticks_visible(False)
        else:
            if orientation == "horizontal":
                ax.set_xlabel("RA (J2000)")
                ax.tick_params(axis="x", top=False, bottom=True)
                if i == 0:
                    ax.set_ylabel("Dec (J2000)")
                    ax.tick_params(axis="y", left=True, right=False)
                else:
                    ax.coords['dec'].set_axislabel('')
                    ax.coords['dec'].set_ticklabel_visible(False)
                    ax.tick_params(axis="y", left=False, right=False)
            elif orientation == "vertical":
                #ax.set_ylabel("Dec (J2000)")
                ax.set_ylabel("")
                ax.tick_params(axis="y", left=True, right=False)
                if i == n_plots - 1:
                    ax.set_xlabel("RA (J2000)")
                    ax.tick_params(axis="x", top=False, bottom=True)
                else:
                    ax.coords['ra'].set_axislabel('')
                    ax.coords['ra'].set_ticklabel_visible(False)
                    ax.tick_params(axis="x", top=False, bottom=False)

        # 5b. Color limits
        if vmin_vmax is None:
            vmin = np.nanpercentile(data, percentile_range[0])
            vmax = np.nanpercentile(data, percentile_range[1])
        else:
            vmin, vmax = vmin_vmax[i]

        # 5c. Colorbar: If the user gave a dictionary with boundaries/colors => discrete colormap
        cmap_spec = _get_cmap(colormaps, i)
        if isinstance(cmap_spec, dict):
            cmap = ListedColormap(cmap_spec["colors"])
            norm = BoundaryNorm(cmap_spec["boundaries"], cmap.N)
            im   = ax.imshow(data, origin="lower", cmap=cmap, norm=norm)
        else:
            im   = ax.imshow(data, origin="lower", cmap=cmap_spec,
                             vmin=vmin, vmax=vmax)

        # Colorbar placement
        if orientation == "horizontal":
            cax = inset_axes(ax, width="100%", height="50%", loc="upper center",
                             bbox_to_anchor=(0.0, 0.95, 1, 0.1),
                             bbox_transform=ax.transAxes, borderpad=0)
            cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
            if not minimal:
                cbar.set_label(str(lab), labelpad=8)
            cbar.ax.xaxis.set_ticks_position('top')
            cbar.ax.xaxis.set_label_position('top')
            cbar.ax.tick_params(axis='x', pad=1)
        elif orientation== "vertical":
            cax = inset_axes(ax, width="7.5%", height="100%", loc="lower left",
                             bbox_to_anchor=(1.00, 0, 1, 1),
                             bbox_transform=ax.transAxes, borderpad=0)
            cbar = fig.colorbar(im, cax=cax, orientation="vertical")
            if not minimal:
                cbar.set_label(str(lab))


    # --- 6. Layout & Save 
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

    if save:
        if output_path is None:
            output_path = Path.home() / "Desktop" / "fig.png"
        save_fig(fig, output_path, overwrite=overwrite, pad_inches=pad_inches)

    plt.show()
    plt.close(fig)









#########################################################################################################            
def save_fig(
    fig: Figure | None = None,
    output_path: str | None = None,
    *,
    overwrite: bool = False,
    dpi: int = 300,
    pad_inches: float = 0.1,
) -> None:
    """
    Save a matplotlib figure to a file.

    Parameters
    ----------
    fig : Figure, optional
        The matplotlib figure to be saved. If None, the current figure (`plt.gcf()`) is used.
    output_path : str, optional
        The file path (including filename) where the figure will be saved.
        This can be any valid image or vector format supported by Matplotlib
        (e.g., '.png', '.pdf', '.svg'). Default is '~/Desktop/fig.png'.
    overwrite : bool, optional
        If True, an existing file at `output_path` will be overwritten.
        If False and the file exists, a FileExistsError is raised. Default is False.
    dpi : int, optional
        The resolution in dots per inch for the saved figure. Default is 300.
    pad_inches : float, optional
        Amount of padding in inches around the figure when bbox_inches is 'tight'. Default is 0.1.

    Returns
    -------
    None
        The function saves the figure to the specified path and does not return anything.
    """
    # --- 1. Set default output path if not provided
    if output_path is None:
        desktop_path = Path.home() / "Desktop"
        output_path = desktop_path / "fig.png"
    
    out_path = Path(output_path)  # Ensure it's a Path object

    # --- 2. If fig is None, use the current figure
    if fig is None:
        fig = plt.gcf()

    # --- 3. Validate the output path
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"File '{out_path}' already exists. "
            "Set 'overwrite=True' to overwrite it."
        )

    # --- 4. Save the figure
    try:
        fig.savefig(str(out_path), dpi=dpi, bbox_inches='tight', pad_inches=pad_inches)
        print(f"File saved: {out_path}")
    except Exception as e:
        raise RuntimeError(
            f"An error occurred while saving the figure to '{out_path}': {e}"
        ) from e