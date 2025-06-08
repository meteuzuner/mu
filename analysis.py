import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from astropy.units import Quantity, UnitConversionError
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from pathlib import Path
from astropy.convolution import convolve_fft
from reproject import reproject_adaptive
from astropy.nddata import Cutout2D


def read_fits(data_path: str,
              data_ext: int = 0, 
              error_path: str | None = None,
              error_ext: int = 0, 
              SNR_cut: bool = False, 
              SNR_threshold: float | None = None,
              min_value: float | None = None,
              max_value: float | None = None,
              min_err_value: float | None = None,
              max_err_value: float | None = None
             ) -> tuple[np.ndarray, fits.Header, np.ndarray | None, fits.Header | None]:
    """
    Read the FITS data from a file and (optionally) an error file,
    applying an SNR cut if requested.
    
    Parameters
    ----------
    data_path : str
        Path to the FITS file containing the data.
    data_ext : int
        FITS extension index for the data file.
    error_path : str, optional
        Path to the FITS file containing the error data.
    error_ext : int
        FITS extension index for the error file.
    SNR_cut : bool
        If True, apply an SNR threshold to the data using the error map.
    SNR_threshold : float, optional
        Minimum SNR value for the cut. Required if SNR_cut is True.
    min_value : float, optional
        If provided, any pixel in the data that is < min_value will be replaced with NaN.
        The same mask is applied to the error (if present).
    max_value : float, optional
        If provided, any pixel in the data that is > max_value will be replaced with NaN.
        The same mask is applied to the error (if present).
    min_err_value : float, optional
        If provided, any pixel in both data and error for which error < min_err_value
        is replaced with NaN.
    max_err_value : float, optional
        If provided, any pixel in both data and error for which error > max_err_value
        is replaced with NaN.
    
    Returns
    -------
    data : np.ndarray
        Data values from the FITS file (potentially masked if SNR_cut is True).
    header : fits.Header
        Header object from the data FITS file.
    error : np.ndarray or None
        Error values from the error FITS file if provided; otherwise None.
    """
    # --- 1. Check data file and read data
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data, header = fits.getdata(data_path, ext=data_ext, header=True)
    

    # --- 2. Check error file and read error
    error = None
    error_header = None

    if error_path is not None:
        error_file = Path(error_path)
        if not error_file.exists():
            raise FileNotFoundError(f"Error file not found: {error_path}")
        
        error, error_header = fits.getdata(error_path, ext=error_ext, header=True)
        
        # If SNR cut is requested, apply threshold
        if SNR_cut:
            if SNR_threshold is None:
                raise ValueError("SNR_threshold must be provided if SNR_cut is True.")
            
            print(f"SNR threshold is set to {SNR_threshold}.\n")
            
            # Compare data/error units (optional, if your headers store them)
            data_unit  = header.get("SIGUNIT") or header.get("BUNIT")
            error_unit = error_header.get("SIGUNIT") or error_header.get("BUNIT")
            
            if data_unit and error_unit and (data_unit != error_unit):
                raise ValueError(
                    "Data and Error files have different units. "
                    "Cannot proceed with SNR cut!"
                )

            # Mask data based on SNR
            mask_snr = (data / error) < SNR_threshold
            data = np.where(mask_snr, np.nan, data)
            
            num_masked = np.count_nonzero(mask_snr)
            print(f"{num_masked} pixels (SNR < {SNR_threshold}) are changed to NaN.\n")
    

    # --- 3. Apply min_value / max_value masking
    # 3a. Mask by min_value / max_value (data + error)
    if min_value is not None:
        mask_min = data < min_value
        data = np.where(mask_min, np.nan, data)
        num_masked_min = np.count_nonzero(mask_min)
        print(f"{num_masked_min} pixels < {min_value} are changed to NaN in data.\n")

        if error is not None:
            error = np.where(mask_min, np.nan, error)

    if max_value is not None:
        mask_max = data > max_value
        data = np.where(mask_max, np.nan, data)
        num_masked_max = np.count_nonzero(mask_max)
        print(f"{num_masked_max} pixels > {max_value} are changed to NaN in data.\n")

        if error is not None:
            error = np.where(mask_max, np.nan, error)
        
    # 3b. Mask by min_err_value / max_err_value (error + data)
    # Only meaningful if we actually have an error file
    if (min_err_value is not None or max_err_value is not None) and error is None:
        raise ValueError("min_err_value/max_err_value given, but no error file was provided.")

    if min_err_value is not None and error is not None:
        mask_err_min = error < min_err_value
        data = np.where(mask_err_min, np.nan, data)
        error = np.where(mask_err_min, np.nan, error)
        num_masked_err_min = np.count_nonzero(mask_err_min)
        print(f"{num_masked_err_min} pixels have error < {min_err_value} => masked.\n")

    if max_err_value is not None and error is not None:
        mask_err_max = error > max_err_value
        data = np.where(mask_err_max, np.nan, data)
        error = np.where(mask_err_max, np.nan, error)
        num_masked_err_max = np.count_nonzero(mask_err_max)
        print(f"{num_masked_err_max} pixels have error > {max_err_value} => masked.\n")

        
    # --- 4. Return the results
    # Return either (data, header, error, error_header) or (data, header)
    if error is not None:
        return data, header, error, error_header
    else:
        return data, header














#########################################################################################################
def convolve_fits(data: np.ndarray,
                  kernel_data: np.ndarray,
                  error: np.ndarray | None = None,
                 ) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Convolve a FITS data and (optionally) an error file.
    
    Parameters
    ----------
    data : np.ndarray
        2D data array.
    kernel_data : np.ndarray
        2D kernel array used to convolve the data (and optionally the error).
        Must be normalized (sum ~ 1).
    error : np.ndarray, optional
        2D error array corresponding to the data, if available.
    
    Returns
    -------
    convolved_data : np.ndarray
        The convolved data array (same shape as `data`).
    convolved_error : np.ndarray, optional
        The convolved error array if an `error` was provided; otherwise `None`.
    """
    # --- 1. Check the data
    if not isinstance(data, np.ndarray):
        raise TypeError(f"'data' is not a NumPy array. Got {type(data)}.")
    if data.ndim != 2:
        raise ValueError(f"'data' must be 2D. Got {data.ndim}D.")

    # 1b. Check kernel file
    if not isinstance(kernel_data, np.ndarray):
        raise TypeError(f"'kernel_data' is not a NumPy array. Got {type(kernel_data)}.")
    if kernel_data.ndim != 2:
        raise ValueError(f"'kernel_data' must be 2D. Got {kernel_data.ndim}D.")
    
    # 1c. Check normalization of kernel data
    kernel_sum = np.nansum(kernel_data)
    if not np.isclose(kernel_sum, 1.0, atol=1e-3):
        raise ValueError(f"Kernel sum is {kernel_sum:.4g}, which is not close to 1. "
                         "Ensure the kernel is normalized.")  
    
    # --- 2. Convolve data
    print("Convolving data with the provided kernel ...")
    convolved_data = convolve_fft(
        data,
        kernel_data,
        allow_huge=True,
        normalize_kernel=False,  # already normalized
        boundary="fill",
        fill_value=0.0,
        nan_treatment="fill",
        preserve_nan=True,
    )

    
    # --- 3. Convolve error (optional)

    convolved_error = None  # Default if no error map is provided.

    # 3a. Check error map
    if error is not None:
        if not isinstance(error, np.ndarray):
            raise TypeError(f"'error' is not a NumPy array. Got {type(error)}.")
        if error.ndim != 2:
            raise ValueError(f"'error' must be 2D. Got {error.ndim}D.") 
    
        # Convolve error map
        print("Convolving error with the squared kernel ...")
        # For error propagation in convolution, we convolve error^2 with kernel^2,
        # then take the square root of that result.
        convolved_variance = convolve_fft(
            error**2,
            kernel_data**2,
            allow_huge=True,
            normalize_kernel=False,
            boundary="fill",
            fill_value=0.0,
            nan_treatment="fill",
            preserve_nan=True,
        )
        convolved_error = np.sqrt(convolved_variance)
        
    print("Completed.")
    # Return either (convolved_data) or (convolved_data, convolved_error)
    if convolved_error is not None:
        return convolved_data, convolved_error
    else:
        return convolved_data














#########################################################################################################       
def reproject_fits(data: np.ndarray,
                   data_header: fits.Header,
                   target_header: fits.Header,
                   error: np.ndarray | None = None,
                  ) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Reproject a FITS data and (optionally) an error file.
    
    Parameters
    ----------
    data : np.ndarray
        2D data array to be reprojected.
    data_header : fits.Header
        FITS header describing the WCS of the input `data`.
    target_header : fits.Header
        Target header.
    error : np.ndarray, optional
        2D error array corresponding to the data, if available.
    
    Returns
    -------
    reprojected_data : np.ndarray
        The reprojected data array (same shape as `data`).
    reprojected_error : np.ndarray, optional
        The reprojected error array if an `error` was provided; otherwise `None`.
    """  
    # --- 1. Check data
    if not isinstance(data, np.ndarray):
        raise TypeError(f"'data' is not a NumPy array. Got {type(data)}.")
    if data.ndim != 2:
        raise ValueError(f"'data' must be 2D. Got {data.ndim}D.")

    # 1b. Check headers
    if not isinstance(data_header, fits.Header):
        raise TypeError(f"'data_header' is not a fits.Header object. Got {type(data_header)}.")
    if not isinstance(target_header, fits.Header):
        raise TypeError(f"'target_header' is not a fits.Header object. Got {type(target_header)}.")
    
    
    # --- 2. Reproject data
    print("Reprojecting data to the provided target WCS...\n")
    reprojected_data, _ = reproject_adaptive(
        input_data=(data, data_header),
        output_projection=target_header,
        conserve_flux=True
    )
    

    # --- 3. Reproject error (optional)

    reprojected_error = None  # Default if no error map is provided.
    # If provided
    if error is not None:
        # 3a. Check error file
        if not isinstance(error, np.ndarray):
            raise TypeError(f"'error' is not a NumPy array. Got {type(error)}.")
        if error.ndim != 2:
            raise ValueError(f"'error' must be 2D. Got {error.ndim}D.") 
    
        # 3b. Reproject the error map
        print("Reprojecting error to the target WCS...\n")
        # For error propagation, reproject error**2, then take sqrt
        reprojected_variance, _ = reproject_adaptive(
            input_data=(error**2, data_header),
            output_projection=target_header,
            conserve_flux=True
        )
        reprojected_error = np.sqrt(reprojected_variance)
        
    print("Completed.")
    # Return either (reprojected_data) or (reprojected_data, reprojected_error)
    if reprojected_error is not None:
        return reprojected_data, reprojected_error
    else:
        return reprojected_data
    
    
    
    
    
    
    
    
    
    
    
    

    
#########################################################################################################        
def save_fits(
    data: np.ndarray,
    data_header: fits.Header,
    output_path: str,
    overwrite: bool = False
    ) -> None:
    """
    Save a 2D data array and its FITS header to a file.

    Parameters
    ----------
    data : np.ndarray
        2D data array to be saved.
    data_header : fits.Header
        FITS header corresponding to `data`.
    output_path : str
        Output path (file name) to save the FITS file.
    overwrite : bool, optional
        If True, overwrite an existing file at `output_path`. Default is False.

    Returns
    -------
    None
        The function writes the FITS file to disk and does not return anything.
    """
    out_path = Path(output_path)  # Convert to Path object

    try:
        hdu = fits.PrimaryHDU(data, header=data_header)

        # We can pass a Path object directly to `writeto` in recent Astropy versions.
        # If needed (for older versions), you can do: str(out_path)
        hdu.writeto(out_path, overwrite=overwrite)
        print(f"File saved: {out_path}")

    except (OSError, FileExistsError) as e:
        # If the file already exists and overwrite=False, raise an error
        if out_path.exists() and not overwrite:
            raise FileExistsError(
                f"File '{out_path}' already exists and cannot be overwritten. "
                "Set `overwrite=True` if you want to overwrite it."
            ) from e
        else:
            # Catch any other OSError
            raise OSError(
                f"An error occurred while trying to write the file to '{out_path}'."
            ) from e  














#########################################################################################################
# v250428 - Now it can work with units, too! 
def subtract_background( 
    data: np.ndarray | Quantity,
    background_value: float | Quantity
) -> np.ndarray | Quantity:
    """
    Subtract a constant background level from a 2D flux map, with full astropy.units support.

    Parameters
    ----------
    data : np.ndarray or astropy.units.Quantity
        A 2D flux map. If a Quantity, it must have a unit.
    background_value : float or astropy.units.Quantity
        The constant background level. If a Quantity, it must have a unit.

    Returns
    -------
    np.ndarray or astropy.units.Quantity
        A new 2D array (or Quantity) with `background_value` subtracted from each pixel.

    Raises
    ------
    TypeError
        If exactly one of `data` or `background_value` has a unit.
    ValueError
        If `data` is not 2D, or if both have units but are not compatible.
    """
    # Detect which inputs are Quantities
    data_is_qty = isinstance(data, Quantity)
    bg_is_qty   = isinstance(background_value, Quantity)

    # Case 1: neither has a unit → plain subtraction
    if not data_is_qty and not bg_is_qty:
        if not isinstance(data, np.ndarray):
            raise TypeError(f"'data' must be a NumPy array, got {type(data)}.")
        if data.ndim != 2:
            raise ValueError(f"'data' must be 2D, got {data.ndim}D.")
        return data - background_value

    # Case 2: one has a unit, the other doesn’t → error
    if data_is_qty and not bg_is_qty:
        raise TypeError(
            f"data has unit {data.unit!r}, but background_value is dimensionless. "
            "Both must be dimensionless or share the same unit."
        )
    if not data_is_qty and bg_is_qty:
        raise TypeError(
            f"background_value has unit {background_value.unit!r}, but data is dimensionless. "
            "Both must be dimensionless or share the same unit."
        )

    # Case 3: both have units → convert & subtract
    # check data dimensionality
    if data.value.ndim != 2:
        raise ValueError(f"'data' must be 2D, got {data.value.ndim}D.")
    # try converting background to data's unit
    try:
        bg_converted = background_value.to(data.unit)
    except UnitConversionError:
        raise ValueError(
            f"Cannot convert background unit {background_value.unit!r} "
            f"to data unit {data.unit!r}."
        )

    return data - bg_converted














#########################################################################################################
def radial_profile_from_SED(mass_map, temperature_map, beta_map=None, header=None, 
                            distance_mpc=None, gc_ra_deg=None, gc_dec_deg=None, xlim=None,
                            plot_style: str | None = None,   
                            save_fig: bool = False, output_path: str | None = None, overwrite: bool = False):
    """
    Plots the radial profiles from SED maps of a galaxy.
    
    Parameters
    ----------
      mass_map : np.ndarray
          Dust mass map (e.g., in M$_\odot$/pc$^2$).
      temperature_map : np.ndarray
          Dust temperature map (in Kelvin).
      beta_map (2D array, optional): np.ndarray, optional
          Dust beta map. If not provided, only mass and temperature will be plotted.
      header : 
          Header containing WCS information.
      distance : float
          Distance to the galaxy in Mpc.
      gc_ra_deg : float
          RA of the galaxy center in degrees.
      gc_dec_deg : float
          Dec of the galaxy center in degrees.
      xlim : List of two floats
          [xlim1, xlim2]
      plot_style : str, optional
          Path to a matplotlib style file (e.g., 'my-style.mplstyle'). If provided,
          `plt.style.use(plot_style)` will be applied.
      save : bool, optional
          If True, saves the final figure to `output_path`. Default is False.
      output_path : str, optional
          The file path where the figure will be saved. Default path is '~Desktop/fig.png'.
      overwrite : bool, optional
          If True, overwrite any existing file at `output_path`. Default is False.
    Returns
    -------
    """
    # Check required inputs
    if distance_mpc is None:
        raise Exception("Need distance in Mpc to the galaxy!")
    if header is None:
        raise Exception("Need header!")
    if gc_ra_deg is None or gc_dec_deg is None:
        raise Exception("Need center coordinates of the galaxy in deg!")
        
    # Create a WCS object from the header
    wcs = WCS(header)

    # Create a grid of pixel coordinates
    ny, nx = mass_map.shape  # image dimensions
    y_indices, x_indices = np.indices((ny, nx))

    # Convert pixel coordinates to (RA, DEC)
    ra_map, dec_map = wcs.all_pix2world(x_indices, y_indices, 0)

    # Compute the angular separation from the galaxy center for each pixel
    galaxy_center  = SkyCoord(ra=gc_ra_deg*u.deg, dec=gc_dec_deg*u.deg, frame='icrs')
    pixel_coords   = SkyCoord(ra=ra_map*u.deg,    dec=dec_map*u.deg,    frame='icrs')
    separation     = galaxy_center.separation(pixel_coords)
    separation_deg = separation.deg
    separation_rad = np.deg2rad(separation_deg)
    
    # Convert angular separation to physical radius in kpc.
    conversion_factor = distance_mpc * 1e3 # (kpc)
    radius_kpc = separation_rad * conversion_factor
    
    # Apply a custom style file if provided
    if plot_style is not None:
        style_path = Path(plot_style)
        if style_path.exists():
            plt.style.use(style_path)
        else:
            raise FileNotFoundError(f"Style file '{style_path}' does not exist.")

    # Choose number of subplots based on beta_map availability
    if beta_map is None:
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(8, 12))
    else:
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(8, 12))

    # Plot 1: Radius vs Mass (log y-axis)
    axs[0].scatter(radius_kpc.flatten(), mass_map.flatten(), s=1, alpha=0.5, c="navy")
    axs[0].set_ylabel("$\sum_{dust}$ (M$_\odot$/pc$^2$)")
    axs[0].set_yscale('log')

    # Plot 2: Radius vs Temperature
    axs[1].scatter(radius_kpc.flatten(), temperature_map.flatten(), s=1, alpha=0.5, c="navy")
    axs[1].set_ylabel("T$_{dust}$ (K)")
    
    if beta_map is not None:
        # Plot 3: Radius vs Beta
        axs[2].scatter(radius_kpc.flatten(), beta_map.flatten(), s=1, alpha=0.5, c="navy")
        axs[2].set_ylabel(r"$\beta$")
        axs[2].set_xlabel("Radius (kpc)")
    else:
        axs[1].set_xlabel("Radius (kpc)")
        
    if xlim is not None:
        for ax_i in axs:
            ax_i.set_xlim(xlim[0], xlim[1])

    plt.subplots_adjust(hspace=0)
    plt.show()
    
    if save_fig:
        from mu.plotting import save_fig
        save_fig(fig=fig, output_path=output_path, overwrite=overwrite)
    return radius_kpc
    
    
    
    
    
    
    
    
    
#########################################################################################################
def cutout_data(data, header, center_ra, center_dec, width_deg, height_deg,
                quick_plot=False):
    """
    Create a cutout (sub-image) around the specified center RA/Dec, using
    a rectangular area given in degrees (width x height).

    Parameters
    ----------
    data : 2D numpy array
        The FITS data array to be cut out.
    header : FITS header
        The header corresponding to the FITS data; must contain valid WCS info.
    center_ra : float
        Right Ascension of the desired center (in degrees).
    center_dec : float
        Declination of the desired center (in degrees).
    width_deg : float
        Desired width of the sub-image (in degrees).
    height_deg : float
        Desired height of the sub-image (in degrees).
    quick_plot : bool, optional
        If True, displays the original image (with rectangle) and the cutout.

    Returns
    -------
    cut_data : 2D numpy array
        The 2D data array of the cutout.
    cut_header : FITS header
        A header for the cutout image, updated with the appropriate WCS.
    """    
    # Create a WCS object from the header
    wcs_in = WCS(header)

    # Define the central sky coordinate
    sky_coord = SkyCoord(center_ra, center_dec, unit='deg')

    # Define the size of the cutout region in (height, width)
    cutout_size = (height_deg, width_deg) * u.deg

    # Create the cutout
    cutout = Cutout2D(data, sky_coord, cutout_size, wcs=wcs_in)

    # Extract the cutout data array
    cut_data = cutout.data

    # Generate a new header with updated WCS
    cut_header = cutout.wcs.to_header()

    # following three lines are updated on v250510
    cut_header["NAXIS"]  = 2
    cut_header["NAXIS1"] = cut_data.shape[1]   # x‑size
    cut_header["NAXIS2"] = cut_data.shape[0]   # y‑size

    # Optional quick plot
    if quick_plot:
        # Get the pixel slices corresponding to the cutout
        yslice, xslice = cutout.slices_original
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # --- Left subplot: original data with rectangle ---
        ax1 = axes[0]
        ax1.imshow(data, origin='lower', cmap='gray', vmin=np.nanpercentile(data, 5), vmax=np.nanpercentile(data, 95))
        ax1.set_title("Input Data")

        # Draw a rectangle to show the cutout region
        rect = Rectangle(
            (xslice.start, yslice.start),
            xslice.stop - xslice.start,
            yslice.stop - yslice.start,
            edgecolor='red',
            facecolor='none',
            linewidth=1.5
        )
        ax1.add_patch(rect)

        # --- Right subplot: cutout data ---
        ax2 = axes[1]
        ax2.imshow(cut_data, origin='lower', cmap='gray', vmin=np.nanpercentile(data, 5), vmax=np.nanpercentile(data, 95))
        ax2.set_title("Cutout Region")

        plt.tight_layout()
        plt.show()

    return cut_data, cut_header

    
    
    
    
    
    
    
    
    
#########################################################################################################
def elliptical_annulus_photometry(
    data: np.ndarray,
    header: fits.Header,
    center_ra: u.Quantity,
    center_dec: u.Quantity,
    inner_semi_major: u.Quantity,
    inner_semi_minor: u.Quantity,
    outer_semi_major: u.Quantity,
    outer_semi_minor: u.Quantity,
    position_angle: u.Quantity,
    # ---- background annulus ----
    bkg_inner_semi_major: u.Quantity,
    bkg_inner_semi_minor: u.Quantity,
    bkg_outer_semi_major: u.Quantity,
    bkg_outer_semi_minor: u.Quantity,
    # ----------------------------
    method: str = "mean",
    quick_plot: bool = False,
    save_fname: str | None = None,
) -> tuple[float, float, float, float]:
    """
    Returns
    -------
    mean_val : float           weighted mean flux in science annulus
    mean_err : float           σ_bkg / √N_eff
    bkg_std  : float           standard deviation in background annulus
    bkg_mean : float           weighted mean flux in background annulus
    """
    from photutils.aperture import SkyEllipticalAperture
    
    # --- 1. Basic Checks 
    if not isinstance(data, np.ndarray):
        raise TypeError("'data' must be a NumPy array")
    if data.ndim != 2:
        raise ValueError("'data' must be 2-D")
    if not isinstance(header, fits.Header):
        raise TypeError("'header' must be a fits.Header")
    method = method.lower()
    if method not in {"mean", "median"}:
        raise NotImplementedError("method must be 'mean' or 'median'")

    # --- 2. Helper: Force Degrees
    def _ensure_deg(q, name):
        if isinstance(q, (int, float)):
            return q * u.deg
        if hasattr(q, "unit"):
            return q.to(u.deg)
        raise TypeError(f"{name} must be number or Quantity")

    # --- 3. Coordinates
    wcs   = WCS(header)
    galaxy_centre = SkyCoord(_ensure_deg(center_ra, "center_ra"),
                             _ensure_deg(center_dec, "center_dec"))

    # --- 4. Science Apertures
    ap_out = SkyEllipticalAperture(galaxy_centre, _ensure_deg(outer_semi_major, "a_out"),
                                   _ensure_deg(outer_semi_minor, "b_out"),
                                   _ensure_deg(position_angle, "theta"))
    ap_in  = SkyEllipticalAperture(galaxy_centre, _ensure_deg(inner_semi_major, "a_in"),
                                   _ensure_deg(inner_semi_minor, "b_in"),
                                   _ensure_deg(position_angle, "theta"))
    
    # ----- 4a. Fractional coverage images for the outer and inner ellipses
    cov_out = ap_out.to_pixel(wcs).to_mask("exact").to_image(data.shape)
    cov_in  = ap_in.to_pixel(wcs).to_mask("exact").to_image(data.shape)

    # ----- 4b. Pixel centre must be inside outer and not inside inner ellipse
    mask_annulus = (cov_out > 0) & ~(cov_in > 0)

    # --- 5. Background Apertures
    ap_bkg_out = SkyEllipticalAperture(galaxy_centre, _ensure_deg(bkg_outer_semi_major, "bkg_a_out"),
                                       _ensure_deg(bkg_outer_semi_minor, "bkg_b_out"),
                                       _ensure_deg(position_angle, "theta"))
    ap_bkg_in  = SkyEllipticalAperture(galaxy_centre, _ensure_deg(bkg_inner_semi_major, "bkg_a_in"),
                                       _ensure_deg(bkg_inner_semi_minor, "bkg_b_in"),
                                       _ensure_deg(position_angle, "theta"))
    
    # ----- 5a. Fractional coverage images for the outer and inner ellipses
    cov_bkg_out = ap_bkg_out.to_pixel(wcs).to_mask("exact").to_image(data.shape)
    cov_bkg_in  = ap_bkg_in.to_pixel(wcs).to_mask("exact").to_image(data.shape)

    # ----- 5b. Pixel centre must be inside outer and not inside inner ellipse
    mask_bkg = (cov_bkg_out > 0) & ~(cov_bkg_in > 0)


    # --- 6. Science Flux (weighted mean on the ring)
    sci_ok   = mask_annulus & np.isfinite(data)
    weights  = cov_out[sci_ok]
    sci_vals = data[sci_ok]

    n_eff    = weights.sum()
    if n_eff == 0:
        raise RuntimeError("Science annulus contains no valid pixels")

    if method == "mean":
        # Weighted mean
        stat_val = (weights * sci_vals).sum() / n_eff
    elif method == "median":
        # Weighted median
        def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
            """
            Weighted median of *values* with non-negative *weights*.

            Returns the smallest x such that Σ w_i (v_i <= x) ≥ 0.5 Σ w_i.
            """
            srt = np.argsort(values)
            v   = values[srt]
            w   = weights[srt]

            cdf = np.cumsum(w)
            cut = 0.5 * w.sum()
            return v[np.searchsorted(cdf, cut)]
        
        stat_val = _weighted_median(sci_vals, weights)


    # --- 7. Background Stats
    bkg_ok    = mask_bkg & np.isfinite(data)
    bkg_w     = cov_bkg_out[bkg_ok]
    bkg_vals  = data[bkg_ok]
    if bkg_vals.size == 0:
        raise RuntimeError("Background annulus contains no valid pixels")

    if method == "mean":
        # Weighted mean
        bkg_val  = (bkg_w * bkg_vals).sum() / bkg_w.sum()
    elif method == "median":
        # Weighted median
        bkg_val = _weighted_median(bkg_vals, bkg_w)

    bkg_std  = np.sqrt( ((bkg_w * (bkg_vals - bkg_val) **2).sum()) / bkg_w.sum() )
    stat_err = bkg_std / np.sqrt(n_eff)
    print(f"{method.capitalize()} flux = {stat_val:.6e} ± {stat_err:.6e}")
    print(f"Background {method} = {bkg_val:.6e}\n")
    print(f"Background σ = {bkg_std:.6e}")


    # --- 8. Quick Plot
    if quick_plot:
        import matplotlib.pyplot as plt
        vmin, vmax = np.nanpercentile(data, [1, 99])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

        ap_out.to_pixel(wcs).plot(ax=ax, color="red")
        ap_in.to_pixel(wcs).plot(ax=ax, color="red")
        ap_bkg_out.to_pixel(wcs).plot(ax=ax, color="cyan", ls=":")
        ap_bkg_in.to_pixel(wcs).plot(ax=ax, color="cyan", ls=":")
        plt.show()

    # --- 9. Save the fits if requested
    if save_fname is not None:
        masked_data = np.full_like(data, np.nan, dtype=data.dtype)
        masked_data[mask_annulus] = data[mask_annulus]      # keep science pixels only

        fits.writeto(save_fname, masked_data, header, overwrite=True)
        print(f"wrote annulus mask ->  {save_fname}")
        
    return stat_val, stat_err, bkg_std, bkg_val
