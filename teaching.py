import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const

def modified_blackbody():
    """
    Teaching/demo function: Plots five subplots, each showing how changing one
    parameter (Mass, Temperature, Beta, Distance, Kappa) affects the shape of 
    a modified blackbody SED (flux density in Jy vs. wavelength in micrometers).

    No inputs, no returns -- just shows a figure with five subplots.
    """
    #
    # --- Compute the modified blackbody ---
    #
    def compute_modBB(wavelength, T, beta, Md, D, kappa0, lambda0):
        """
        Compute the modified blackbody flux density in Jy for given inputs.
        """
        # Speed of light
        c_si = const.c.to("m/s").value

        # Convert wavelength array to frequency
        freq_si = c_si / wavelength.to(u.m).value

        # Convert reference wavelength -> reference frequency
        nu0_si = c_si / lambda0.to(u.m).value

        # Convert other parameters to SI
        T_si  = T.to(u.K).value
        Md_si = Md.to(u.kg).value # total dust mass!
        D_si  = D.to(u.m).value
        kappa0_si = kappa0.to(u.m**2 / u.kg).value

        # Constants
        h  = const.h.value     # Planck's constant (J.s)
        kB = const.k_B.value   # Boltzmann constant (J/K)

        # Planck function B(ν) in SI => W m^-2 sr^-1 Hz^-1
        exponent = (h * freq_si) / (kB * T_si)
        B_nu_si = (2.0 * h * freq_si**3 / c_si**2) / (np.exp(exponent) - 1.0)

        # kappa(ν) = kappa0 * (ν/ν0)^beta
        kappa_nu = kappa0_si * (freq_si / nu0_si)**beta

        # Multiply by (Md / D^2)
        factor = (kappa_nu * Md_si) / (D_si**2)

        # Flux in SI => W m^-2 Hz^-1
        F_nu_si = B_nu_si * factor

        # Convert to Jy(1 Jy = 1e-26 W m^-2 Hz^-1)
        F_nu_jy = F_nu_si * 1e26  # => Jy

        return F_nu_jy

    #
    # --- Set up wavelength range (20 to 1000 microns) ---
    #
    wgrid = np.linspace(20, 1000, 500) * u.um

    # Common constants / conversions
    Msun = const.M_sun
    Mpc  = 1 * u.Mpc  

    #
    # --- Prepare the figure with 5 subplots (3 top, 2 bottom) ---
    #
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))#,sharex=True, sharey=True)
    # Hide the last unused axis at axes[1,2].
    axes[1,2].axis('off')

    # Color cycle
    color_cycle = ["red", "orange", "green", "purple", "navy"]

    #
    # 1) MASS subplot: 5 mass values = [10**1.5, 10**2.0, 10**2.5, 10**3.0, 10**3.5] Msun
    #
    ax = axes[0, 0]
    masses = [10**3.5, 10**3.0, 10**2.5, 10**2.0, 10**1.5]  # Msun
    for i, mval in enumerate(masses):
        F_jy = compute_modBB(
            wavelength=wgrid,
            T=20*u.K,
            beta=2.0,
            Md=mval*Msun,
            D=3*Mpc,
            kappa0=0.192*(u.m**2/u.kg),
            lambda0=350*u.um
        )
        ax.plot(wgrid.value, F_jy*1000, color=color_cycle[i])

    ax.set_xlabel("Wavelength [μm]")
    ax.set_ylabel("Flux Density [mJy]")
    ax.set_title("Effect of Dust Mass")
    labels=[r"$10^{1.5} \, M_{\odot}$",
            r"$10^{2.0} \, M_{\odot}$",
            r"$10^{2.5} \, M_{\odot}$",
            r"$10^{3.0} \, M_{\odot}$",
            r"$10^{3.5} \, M_{\odot}$"]
    ax.legend(labels)
    ax.grid(True, linestyle="--", alpha=0.5)

    #
    # 2) TEMPERATURE subplot: [10, 15, 20, 25, 30] K
    #
    ax = axes[0, 1]
    temps = [30, 25, 20, 15, 10]  # K
    for i, tval in enumerate(temps):
        F_jy = compute_modBB(
            wavelength=wgrid,
            T=tval*u.K,
            beta=2.0,
            Md=(10**2.5)*Msun,
            D=3*Mpc,
            kappa0=0.192*(u.m**2/u.kg),
            lambda0=350*u.um
        )
        ax.plot(wgrid.value, F_jy*1000, color=color_cycle[i],
                label=fr"{tval} K")

    ax.set_xlabel("Wavelength [μm]")
    ax.set_ylabel("Flux Density [mJy]")
    ax.set_title("Effect of Temperature")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    #
    # 3) BETA subplot: [1, 1.5, 2, 2.5, 3]
    #
    ax = axes[0, 2]
    betas = [3.0, 2.5, 2.0, 1.5, 1.0]
    for i, bval in enumerate(betas):
        F_jy = compute_modBB(
            wavelength=wgrid,
            T=20*u.K,
            beta=bval,
            Md=(10**2.5)*Msun,
            D=3*Mpc,
            kappa0=0.192*(u.m**2/u.kg),
            lambda0=350*u.um
        )
        ax.plot(wgrid.value, F_jy*1000, color=color_cycle[i],
                label=fr"$\beta$={bval}")

    ax.set_xlabel("Wavelength [μm]")
    ax.set_ylabel("Flux Density [mJy]")
    ax.set_title("Effect of Emissivity Index (β)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    #
    # 4) DISTANCE subplot: [1, 2, 3, 4, 5] Mpc
    #
    ax = axes[1, 0]
    distances = [5, 4, 3, 2, 1]  # Mpc
    for i, dval in enumerate(distances):
        F_jy = compute_modBB(
            wavelength=wgrid,
            T=20*u.K,
            beta=2.0,
            Md=(10**2.5)*Msun,
            D=dval*Mpc,
            kappa0=0.192*(u.m**2/u.kg),
            lambda0=350*u.um
        )
        ax.plot(wgrid.value, F_jy*1000, color=color_cycle[i],
                label=fr"{dval} Mpc")

    ax.set_xlabel("Wavelength [μm]")
    ax.set_ylabel("Flux Density [mJy]")
    ax.set_title("Effect of Distance")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    #
    # 5) KAPPA subplot: 
    #    kappa values = [0.051, 0.07, 0.192, 4.5, 6.4] m^2/kg 
    #    reference wavelengths = [500, 850, 350, 100, 250] μm
    #
    ax = axes[1, 1]
    kappas      = [0.051, 0.07, 0.192, 4.5, 6.4]
    lambda0_arr = [500,   850,  350,   100, 250]  # µm
    for i, (kap, lam0) in enumerate(zip(kappas, lambda0_arr)):
        F_jy = compute_modBB(
            wavelength=wgrid,
            T=20*u.K,
            beta=2.0,
            Md=(10**2.5)*Msun,
            D=3*Mpc,
            kappa0=kap*(u.m**2/u.kg),
            lambda0=lam0*u.um
        )
        ax.plot(wgrid.value, F_jy*1000, color=color_cycle[i],
                label=fr"kappa={kap}, λ0={lam0} μm")
        
    ax.set_xlabel("Wavelength [μm]")
    ax.set_ylabel("Flux Density [mJy]")
    ax.set_title("Effect of Kappa")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    #
    # --- Show ---
    #
    plt.tight_layout()
    plt.show()
