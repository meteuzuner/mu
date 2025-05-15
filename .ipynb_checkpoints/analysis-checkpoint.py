from typing import Optional, Tuple

def read_fits(data_path: str,
              data_ext: int = 0, 
              error_path: Optional[str] = None,
              error_ext: int = 0, 
              SNR_cut: bool = False, 
              SNR_threshold: Optional[float] = None,
              min_value: Optional[float] = None,
              max_value: Optional[float] = None,
              min_err_value: Optional[float] = None,
              max_err_value: Optional[float] = None
             ) -> Tuple["np.ndarray", "fits.Header", Optional["np.ndarray"], Optional["fits.Header"]]:
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
    import numpy as np
    import astropy.io.fits as fits
    from pathlib import Path
    
    # ------------------ 1. Check data file and read data ------------------
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data, header = fits.getdata(data_path, ext=data_ext, header=True)
    
    # ------------------ 2. Check error file and read error ------------------
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
    
    # ------------------ 3. Apply min_value / max_value masking ------------------
    # --- 3a. Mask by min_value / max_value (data + error) ---
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
        
    # --- 3b. Mask by min_err_value / max_err_value (error + data) ---
    #     Only meaningful if we actually have an error file
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

        

    # ------------------ 4. Return the results ------------------ 

    # Return either (data, header, error, error_header) or (data, header)
    if error is not None:
        return data, header, error, error_header
    else:
        return data, header














#########################################################################################################
def convolve_fits(data: "np.ndarray",
                  kernel_data: "np.ndarray",
                  error: Optional["np.ndarray"] = None,
                 ) -> Tuple["np.ndarray", Optional["np.ndarray"]]:
    
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

    import numpy as np
    from astropy.convolution import convolve_fft
    
    # --- Check data ---
    if not isinstance(data, np.ndarray):
        raise TypeError(f"'data' is not a NumPy array. Got {type(data)}.")
    if data.ndim != 2:
        raise ValueError(f"'data' must be 2D. Got {data.ndim}D.")

    # --- Check kernel file ---
    if not isinstance(kernel_data, np.ndarray):
        raise TypeError(f"'kernel_data' is not a NumPy array. Got {type(kernel_data)}.")
    if kernel_data.ndim != 2:
        raise ValueError(f"'kernel_data' must be 2D. Got {kernel_data.ndim}D.")
    
    # --- Check normalization of kernel data ---
    kernel_sum = np.nansum(kernel_data)
    if not np.isclose(kernel_sum, 1.0, atol=1e-3):
        raise ValueError(f"Kernel sum is {kernel_sum:.4g}, which is not close to 1. "
                         "Ensure the kernel is normalized.")  
    
    
    
    # ------------------ Convolve data ------------------
    print("Convolving data with the provided kernel ...")
    convolved_data = convolve_fft(
        data,
        kernel_data,
        allow_huge=True,
        normalize_kernel=False,  # kernel is already normalized
        boundary="fill",
        fill_value=0.0,
        nan_treatment="fill",
        preserve_nan=True,
    )

    
    
    
    # ------------------ Convolve Error (optional) ------------------
    convolved_error = None  # Default if no error map is provided.
    # --- Check error map ---
    if error is not None:
        if not isinstance(error, np.ndarray):
            raise TypeError(f"'error' is not a NumPy array. Got {type(error)}.")
        if error.ndim != 2:
            raise ValueError(f"'error' must be 2D. Got {error.ndim}D.") 
    
        # --- Convolve error map ---
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
def reproject_fits(data: "np.ndarray",
                   data_header: "fits.Header",
                   target_header: "fits.Header",
                   error: Optional["np.ndarray"] = None,
                  ) -> Tuple["np.ndarray", Optional["np.ndarray"]]:
    
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

    import numpy as np
    from reproject import reproject_adaptive
    from astropy.io import fits
    
    # --- Check data ---
    if not isinstance(data, np.ndarray):
        raise TypeError(f"'data' is not a NumPy array. Got {type(data)}.")
    if data.ndim != 2:
        raise ValueError(f"'data' must be 2D. Got {data.ndim}D.")

    # --- Check headers ---
    if not isinstance(data_header, fits.Header):
        raise TypeError(f"'data_header' is not a fits.Header object. Got {type(data_header)}.")
    if not isinstance(target_header, fits.Header):
        raise TypeError(f"'target_header' is not a fits.Header object. Got {type(target_header)}.")
    
    
    
    
    # ------------------ Reproject data ------------------
    print("Reprojecting data to the provided target WCS...\n")
    reprojected_data, _ = reproject_adaptive(
        input_data=(data, data_header),
        output_projection=target_header,
        conserve_flux=True
    )
    
    
    
    # ------------------ Reproject error (optional) ------------------
    reprojected_error = None  # Default if no error map is provided.
    # --- Check error (if provided) ---
    if error is not None:
        if not isinstance(error, np.ndarray):
            raise TypeError(f"'error' is not a NumPy array. Got {type(error)}.")
        if error.ndim != 2:
            raise ValueError(f"'error' must be 2D. Got {error.ndim}D.") 
    
        # --- Reproject the error map ---
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
    data: "np.ndarray",
    data_header: "fits.Header",
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
    
    from pathlib import Path
    from astropy.io import fits

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
def subtract_background(
    data: "np.ndarray",
    background_value: float
) -> "np.ndarray":
    """
    Subtract a constant background level from a 2D flux map.

    Parameters
    ----------
    data : np.ndarray
        A 2D flux map (e.g., in units of intensity, brightness, etc.).
    background_value : float
        The constant background level to be subtracted from every pixel.

    Returns
    -------
    result : np.ndarray
        A new 2D array with `background_value` subtracted from each pixel.
    """
    import numpy as np

    # 1. Basic checks
    if not isinstance(data, np.ndarray):
        raise TypeError(f"'data' must be a NumPy array. Got {type(data)}.")
    if data.ndim != 2:
        raise ValueError(f"'data' must be 2D. Got {data.ndim}D.")

    # 2. Subtract background
    result = data - background_value

    return result