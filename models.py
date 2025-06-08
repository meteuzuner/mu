import astropy.units as u
import numpy as np
import astropy.constants as const
import matplotlib.pyplot as plt

def modified_blackbody(
    nu_or_lambda: u.Quantity,
    Md: u.Quantity,
    T: u.Quantity,
    beta: float,
    D: u.Quantity,
    kappa0: u.Quantity = 0.192 * (u.m**2 / u.kg),
    lambda0: u.Quantity = 350 * u.um,
    quick_plot: bool = False
) -> u.Quantity:
    """
    Compute the modified blackbody flux at frequency(ies) nu for dust 
    mass Md at distance D, with temperature T and dust emissivity index beta.

    The function converts all inputs to SI units internally, performs the 
    calculation, and finally returns flux in Jy arcsec^-2.

    Parameters
    ----------
    nu_or_lambda : Quantity
        Spectral axis for the calculation, either as a frequency (e.g. 300 GHz) 
        or wavelength (e.g. 350 um). Can be a single value or an array.
    Md : Quantity
        Dust mass, with units of mass (e.g. kg).
    T : Quantity
        Dust temperature, with units of K.
    beta : float
        Dust emissivity index (dimensionless).
    D : Quantity
        Distance to the source, with units of length (e.g. m or pc).
    kappa0 : Quantity, optional
        Dust absorption coefficient (opacity) at reference wavelength lambda0.
        Default is 0.192 m^2/kg. Must have units of (area / mass).
    lambda0 : Quantity, optional
        Reference wavelength for kappa0, default 350 µm.
        Must be a length. Example: 350 * u.um
        Internally converted to a reference frequency nu0 = c / lambda0.
    quick_plot : bool, optional
        If True, produce a quick log-log plot of F_nu vs frequency.

    Returns
    -------
    F_nu_jy_as2 : Quantity
        The modified blackbody flux at frequencies nu, 
        in Jy arcsec^-2. 
        Shape matches the input `nu`.

    Notes
    -----
    1) Internally, the function:
       - Converts all parameters to SI.
       - Computes the Planck function B(ν,T).
       - Scales it by [kappa(ν) * Md / D^2].
       - Converts from (W m^-2 Hz^-1 sr^-1) to Jy sr^-1, then to Jy arcsec^-2.

    2) The dust opacity is assumed to scale as (ν / ν0)^β.
    3) If you want flux per steradian (Jy sr^-1) instead of per arcsec², you
       can remove the final sr→arcsec² conversion step.
    """
    # ------------------------------
    # 1) Convert everything to SI units
    # ------------------------------
    try:
        nu_si = nu_or_lambda.to(u.Hz, equivalencies=u.spectral()).value
    except u.UnitConversionError as e:
        raise ValueError(
            f"Could not interpret 'nu_or_lambda'={nu_or_lambda} as a valid spectral unit. "
            "Please provide something like X * u.Hz or Y * u.um, etc."
        ) from e
    
    T_si  = T.to(u.K).value
    Md_si = Md.to(u.kg).value
    D_si  = D.to(u.m).value
    kappa0_si = kappa0.to(u.m**2 / u.kg).value  # opacity in m^2/kg
    lambda0_m = lambda0.to(u.m).value
    nu0_si = const.c.value / lambda0_m  # c in m/s => freq in Hz

    # Physical constants in SI
    h     = const.h.value      # J s
    kB    = const.k_B.value    # J/K
    c_si  = const.c.value      # m/s
    M_sun = const.M_sun.value  # kg

    # ------------------------------
    # 2) Compute Planck function B_nu
    #    B_nu = 2h nu^3 / c^2 * 1/[exp(h nu / kB T) - 1]
    # ------------------------------
    exponent = (h * nu_si) / (kB * T_si)
    B_nu_si  = (2.0 * h * nu_si**3 / c_si**2) / (np.exp(exponent) - 1.0) # in SI unit: W m^-2 sr^-1 Hz^-1)

    # ------------------------------
    # 3) kappa(nu) = kappa0 * (nu/nu0)^beta
    #    Then multiply by (Md / D^2)
    # ------------------------------
    kappa_nu = kappa0_si * (nu_si / nu0_si)**beta # m^2/kg
    factor   = (kappa_nu * Md_si) / (D_si**2)     # dimensionless

    F_nu_si = B_nu_si * factor # W m^-2 Hz^-1
    
    F_nu_jy  = F_nu_si * 1e26 * u.Jy 
    
    # -------------------------------------
    # Optionally quick-plot (log-log scale)
    # -------------------------------------
    if quick_plot:
        wavelength_vals = c_si/nu_si * 1e6 # in µm
        plt.figure(figsize=(7, 5))
        plt.loglog(wavelength_vals, F_nu_jy * 1000, 
                   label=fr"M={(Md_si/M_sun):.1e} M$_\odot$, T={T_si:.1f} K, $\beta$={beta:.1f}", 
                   color="red", lw=2)
        plt.xlabel(r"$\lambda$ (µm)")
        plt.ylabel(r"F (mJy)")
        plt.title("Modified Blackbody")
        plt.grid(True, which="both", linestyle="--", alpha=0.25)
        plt.legend()
        plt.show()

    return F_nu_jy

