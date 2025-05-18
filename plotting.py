from typing import Optional, Tuple

#########################################################################################################      
def plot_wcs(
    data_list: list["np.ndarray"],
    header_list: list["fits.Header"],
    labels: list[str],
    plot_style: Optional[str] = None,
    save: bool = False,
    output_path: Optional[str] = None,
    overwrite: bool = False,
    colormaps: Optional[
        "ColormapSpec"
    ] = "viridis",
    percentile_range: tuple[float, float] = (1, 99),
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
    plot_style : str, optional
        Path to a matplotlib style file (e.g., 'my-style.mplstyle'). If provided,
        `plt.style.use(plot_style)` will be applied.
    save : bool, optional
        If True, saves the final figure to `output_path`. Default is False.
    output_path : str, optional
        The file path where the figure will be saved. Default path is '~Desktop/fig.png'.
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
        Default is "viridis".
    percentile_range : (float, float), optional
        The (low, high) percentile for stretching continuous colormaps,
        used when a discrete custom colormap isn't given. Default is (5, 95).

    Returns
    -------
    None
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.wcs import WCS
    from pathlib import Path
    from typing import Optional, Union, List
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.figure import Figure
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # --- 1. Validate inputs ---
    if not (len(data_list) == len(header_list) == len(labels)):
        raise ValueError("data_list, header_list, and labels must all have the same length.")

    # A helper function to parse either a single colormap spec or a list
    # and return the correct item for subplot i.
    def _get_colormap_spec(
        cmaps: Union[str, dict, List[str], List[dict], None],
        index: int,
        total: int
    ):
        """
        Returns (cmap_or_listedcmap, norm_or_None) for subplot i.
        If a single item, apply to all. If a list, pick the i-th entry.
        """
        if cmaps is None:
            # No colormap specified, fallback to "viridis"
            return "viridis", None

        # If it's a single string or dict, apply it to all subplots
        if isinstance(cmaps, (str, dict)):
            return cmaps, None
        
        # If it's a list, pick the i-th element
        if isinstance(cmaps, list):
            if len(cmaps) != total:
                raise ValueError(
                    f"Length of colormaps list ({len(cmaps)}) does not match "
                    f"the number of subplots ({total})."
                )
            return cmaps[index], None
        
        # Should never reach here if types are correct
        raise ValueError(
            f"Unsupported type for colormaps: {type(cmaps)}. "
            "Must be str, dict, list, or None."
        )

    # 2. Apply a custom style file if provided
    if plot_style is not None:
        style_path = Path(plot_style)
        if style_path.exists():
            plt.style.use(style_path)
        else:
            raise FileNotFoundError(f"Style file '{style_path}' does not exist.")

    # 3. Create figure and subplots
    n_plots = len(data_list)
    fig = plt.figure(figsize=(6 * n_plots, 6))

    # 4. Iterate over each data/header pair
    for i, (data, header, lab) in enumerate(zip(data_list, header_list, labels)):
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Element {i} in data_list is not a NumPy array. Got {type(data)}.")
        if data.ndim != 2:
            raise ValueError(f"Data at index {i} is not 2D. Got {data.ndim}D.")
        if not isinstance(header, fits.Header):
            raise TypeError(f"Element {i} in header_list is not a fits.Header. Got {type(header)}.")

        ax = fig.add_subplot(1, n_plots, i + 1, projection=WCS(header))
        ax.set_title(str(lab))
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")

        # Get the colormap spec for this subplot
        cmap_spec, _ = _get_colormap_spec(colormaps, i, n_plots)

        # If the user gave a dictionary with boundaries/colors => discrete colormap
        if isinstance(cmap_spec, dict):
            boundaries = cmap_spec["boundaries"]
            color_list = cmap_spec["colors"]
            if len(boundaries) != len(color_list) + 1:
                raise ValueError(
                    "len(boundaries) must be exactly one greater than len(colors). "
                    f"Got {len(boundaries)} boundaries vs. {len(color_list)} colors."
                )
            discrete_cmap = ListedColormap(color_list)
            discrete_norm = BoundaryNorm(boundaries, discrete_cmap.N)
            # Plot with discrete colormap
            im = ax.imshow(data, origin="lower", cmap=discrete_cmap, norm=discrete_norm)
        elif isinstance(cmap_spec, str):
            # It's a standard colormap name
            low_p, high_p = percentile_range
            vmin = np.nanpercentile(data, low_p)
            vmax = np.nanpercentile(data, high_p)
            im = ax.imshow(data, origin="lower", cmap=cmap_spec, vmin=vmin, vmax=vmax)
        else:
            # Fallback if somehow we get None or something else
            # Use "viridis" by default
            low_p, high_p = percentile_range
            vmin = np.nanpercentile(data, low_p)
            vmax = np.nanpercentile(data, high_p)
            im = ax.imshow(data, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)

        # Add colorbar
        #cbar = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.033, pad=0.09)
        # Add colorbar
        cax = inset_axes(
            ax,
            width="100%",
            height="5%",
            loc="lower left",
            bbox_to_anchor=(0, -0.1, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0)
        fig.colorbar(im, cax=cax, orientation="horizontal")

    fig.tight_layout()

    # 5. Optionally save the figure
    if save:
        from mu.plotting import save_fig
        save_fig(fig=fig, output_path=output_path, overwrite=overwrite)

    plt.show()
    plt.close(fig)












    

#########################################################################################################            
def save_fig(
    fig: Optional["Figure"] = None,
    output_path: Optional["str"] = None,
    overwrite: bool = False,
    dpi: int = 300
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

    Returns
    -------
    None
        The function saves the figure to the specified path and does not return anything.
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from typing import Optional
    from pathlib import Path

    if output_path is None:
        desktop_path = Path.home() / "Desktop"
        output_path = desktop_path / "fig.png"  # Keep as a Path object
    
    out_path = Path(output_path)  # Ensure it's a Path object

    # Use the current figure if none is provided.
    if fig is None:
        fig = plt.gcf()

    # Check if the file exists and whether we can overwrite it.
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"File '{out_path}' already exists. "
            "Set 'overwrite=True' to overwrite it."
        )

    try:
        # Cast out_path to str to avoid any compatibility issues with certain Matplotlib versions
        fig.savefig(str(out_path), dpi=dpi, bbox_inches='tight')
        print(f"File saved: {out_path}")
    except Exception as e:
        raise RuntimeError(
            f"An error occurred while saving the figure to '{out_path}': {e}"
        ) from e