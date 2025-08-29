import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.integrate import quad
import emcee
import corner

# --- Configuration ---
RADIUS_COL = 'R_kpc'
VELOCITY_COL = 'Vobs_kms'
ERROR_VEL_COL = 'errV_kms' # Needed for robust MCMC
GAS_VEL_COL = 'Vgas_kms'
DISK_VEL_COL = 'Vdisk_kms'
BULGE_VEL_COL = 'Vbul_kms'
BARYONIC_MASS_PROXY_COL = 'log_baryonic_mass'

# Constants
G_CONST = 4.30091e-6  # kpc (km/s)^2 / M_sun
C_KMS = 299792.458 # Speed of light in km/s

# --- Model Definitions ---

def ebc_model(r, A, alpha):
    """The EBC (PRC) model: V = A * r^alpha"""
    return A * r**alpha

def nfw_velocity_halo(r, V200, c):
    """Calculates the velocity contribution from an NFW halo ONLY."""
    H0 = 70
    R200 = V200 / (10 * H0 / 1000)
    x = r / R200
    with np.errstate(divide='ignore', invalid='ignore'):
        v_sq = V200**2 * (np.log(1 + c * x) - (c * x) / (1 + c * x)) / (x * (np.log(1 + c) - c / (1 + c)))
    return np.sqrt(np.nan_to_num(v_sq))

def combined_nfw_baryons_model(r, V200, c, v_gas, v_disk, v_bulge):
    """Full rotation curve model including NFW halo and baryonic components."""
    v_halo_sq = nfw_velocity_halo(r, V200, c)**2
    return np.sqrt(v_halo_sq + v_gas**2 + v_disk**2 + v_bulge**2)

# --- MCMC Likelihood and Prior Functions ---

# For EBC Model
def log_likelihood_ebc(theta, r, v_obs, v_err):
    A, alpha = theta
    model_v = ebc_model(r, A, alpha)
    sigma2 = v_err**2
    return -0.5 * np.sum((v_obs - model_v)**2 / sigma2 + np.log(2 * np.pi * sigma2))

def log_prior_ebc(theta):
    A, alpha = theta
    if 1.0 < A < 500.0 and -0.5 < alpha < 1.5:
        return 0.0
    return -np.inf

def log_probability_ebc(theta, r, v_obs, v_err):
    lp = log_prior_ebc(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_ebc(theta, r, v_obs, v_err)

# For NFW Model
def log_likelihood_nfw(theta, r, v_obs, v_err, v_gas, v_disk, v_bulge):
    V200, c = theta
    model_v = combined_nfw_baryons_model(r, V200, c, v_gas, v_disk, v_bulge)
    sigma2 = v_err**2
    return -0.5 * np.sum((v_obs - model_v)**2 / sigma2 + np.log(2 * np.pi * sigma2))

def log_prior_nfw(theta):
    V200, c = theta
    if 10.0 < V200 < 500.0 and 0.1 < c < 50.0:
        return 0.0
    return -np.inf

def log_probability_nfw(theta, r, v_obs, v_err, v_gas, v_disk, v_bulge):
    lp = log_prior_nfw(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_nfw(theta, r, v_obs, v_err, v_gas, v_disk, v_bulge)


# --- Lensing Calculation Functions ---

def ebc_potential_derivative(r, A, alpha):
    """d(Phi)/dr for the EBC model."""
    return (ebc_model(r, A, alpha)**2) / r

def nfw_full_potential_derivative(r, V200, c, r_data, v_gas_data, v_disk_data, v_bulge_data):
    """d(Phi)/dr for the combined NFW+Baryons model."""
    # Interpolate baryonic components to the requested radius `r`
    v_g_sq = np.interp(r, r_data, v_gas_data**2)
    v_d_sq = np.interp(r, r_data, v_disk_data**2)
    v_b_sq = np.interp(r, r_data, v_bulge_data**2)
    v_halo_sq = nfw_velocity_halo(r, V200, c)**2
    return (v_halo_sq + v_g_sq + v_d_sq + v_b_sq) / r

def calculate_deflection_angle(potential_deriv_func, b, args_for_deriv):
    """Generic function to calculate lensing deflection angle."""
    integrand = lambda r, b_val: potential_deriv_func(r, *args_for_deriv) / np.sqrt(r**2 - b_val**2)
    try:
        # The integral is from b to infinity. We use a large multiple of b as a proxy.
        integral, _ = quad(integrand, b, 500 * b, args=(b,), limit=100)
        alpha_radians = (2 / C_KMS**2) * integral
        return alpha_radians * (180 / np.pi) * 3600 # Convert to arcseconds
    except Exception:
        return np.nan


# ==============================================================================
# PART 1: INITIAL FIT FOR ALL GALAXIES
# ==============================================================================

def perform_initial_fitting(data_dir, output_csv):
    print("\n--- Running Part 1: Initial EBC model fitting for all galaxies ---")
    if os.path.exists(output_csv):
        print(f"  '{output_csv}' already exists. Skipping fitting.")
        return pd.read_csv(output_csv)

    galaxy_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    results = []
    for filename in galaxy_files:
        try:
            filepath = os.path.join(data_dir, filename)
            galaxy_data = pd.read_csv(filepath, comment='#')
            if not all(col in galaxy_data.columns for col in [RADIUS_COL, VELOCITY_COL]):
                continue
            
            baryons_v_sq = galaxy_data.get(GAS_VEL_COL, 0)**2 + galaxy_data.get(DISK_VEL_COL, 0)**2 + galaxy_data.get(BULGE_VEL_COL, 0)**2
            last_r = galaxy_data[RADIUS_COL].iloc[-1]
            total_baryonic_mass = (baryons_v_sq.iloc[-1] * last_r) / G_CONST
            if total_baryonic_mass <= 0: continue
            log_mass = np.log10(total_baryonic_mass)

            popt, _ = curve_fit(ebc_model, galaxy_data[RADIUS_COL], galaxy_data[VELOCITY_COL], p0=[50, 0.5])
            results.append({
                'galaxy_id': filename,
                BARYONIC_MASS_PROXY_COL: log_mass,
                'parameter_A': popt[0], 'parameter_alpha': popt[1],
                'v_obs_flat': galaxy_data[VELOCITY_COL].iloc[-1]
            })
        except Exception as e:
            print(f"Skipping {filename} due to an error: {e}")
            
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\nFitting complete. Results for {len(results_df)} galaxies saved to '{output_csv}'.")
    return results_df


# ==============================================================================
# PART 2 & 3: BARYONIC CORRELATION AND TULLY-FISHER TESTS
# ==============================================================================

def run_broad_tests(results_df, output_dir):
    print("\n--- Running Parts 2 & 3: Baryonic Mass and Tully-Fisher Tests ---")
    # This combines the plotting for the first two tests.
    
    # --- Baryonic Correlation ---
    fig1, axes1 = plt.subplots(1, 2, figsize=(18, 7))
    fig1.suptitle('Test 1: Physical Basis of EBC Parameters', fontsize=20, y=1.03)
    clean_df = results_df.dropna()
    x_mass = clean_df[BARYONIC_MASS_PROXY_COL]
    
    y_A = clean_df['parameter_A']
    axes1[0].scatter(x_mass, y_A, alpha=0.7, edgecolors='k', label='Fitted Galaxy Data')
    m_A, b_A = np.polyfit(x_mass, y_A, 1)
    r2_A = np.corrcoef(x_mass, y_A)[0, 1]**2
    axes1[0].plot(x_mass, m_A * x_mass + b_A, 'r--', label=f'Linear Fit (R² = {r2_A:.4f})')
    axes1[0].set_xlabel('Log10 (Baryonic Mass Proxy)', fontsize=12)
    axes1[0].set_ylabel('EBC Parameter A', fontsize=12)
    axes1[0].set_title('Parameter A vs. Galaxy Baryonic Mass', fontsize=16)
    axes1[0].legend()

    y_alpha = clean_df['parameter_alpha']
    axes1[1].scatter(x_mass, y_alpha, alpha=0.7, edgecolors='k')
    m_alpha, b_alpha = np.polyfit(x_mass, y_alpha, 1)
    r2_alpha = np.corrcoef(x_mass, y_alpha)[0, 1]**2
    axes1[1].plot(x_mass, m_alpha * x_mass + b_alpha, 'r--', label=f'Linear Fit (R² = {r2_alpha:.4f})')
    axes1[1].set_xlabel('Log10 (Baryonic Mass Proxy)', fontsize=12)
    axes1[1].set_ylabel('EBC Parameter alpha (α)', fontsize=12)
    axes1[1].set_title('Parameter α vs. Galaxy Baryonic Mass', fontsize=16)
    axes1[1].legend()
    
    plot_path1 = os.path.join(output_dir, "baryonic_mass_correlation.png")
    fig1.savefig(plot_path1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"  Successfully saved Baryonic Correlation plot to {plot_path1}")

    # --- Tully-Fisher ---
    last_radii = [pd.read_csv(os.path.join('rc_parsed', gid), comment='#')[RADIUS_COL].iloc[-1] for gid in clean_df['galaxy_id']]
    clean_df['last_r'] = last_radii
    clean_df['v_ebc_flat'] = ebc_model(clean_df['last_r'], clean_df['parameter_A'], clean_df['parameter_alpha'])
    
    fig2, axes2 = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    fig2.suptitle('Test 2: Reproducing the Tully-Fisher Relation', fontsize=22, y=1.03)

    x_obs = np.log10(clean_df['v_obs_flat'])
    y_obs = clean_df[BARYONIC_MASS_PROXY_COL]
    axes2[0].scatter(x_obs, y_obs, alpha=0.6, edgecolors='k', label='Observed Data')
    m_obs, b_obs = np.polyfit(x_obs, y_obs, 1)
    r2_obs = np.corrcoef(x_obs, y_obs)[0, 1]**2
    axes2[0].plot(x_obs, m_obs * x_obs + b_obs, 'r--', label=f'Fit (R² = {r2_obs:.4f})')
    axes2[0].set_xlabel('Log10 (Observed V_flat / km/s)', fontsize=14)
    axes2[0].set_ylabel('Log10 (Baryonic Mass / M_sun)', fontsize=14)
    axes2[0].set_title('Observed Tully-Fisher Relation', fontsize=18)
    axes2[0].legend()
    axes2[0].set_xlim(left=1.1)

    x_ebc = np.log10(clean_df['v_ebc_flat'])
    axes2[1].scatter(x_ebc, y_obs, alpha=0.6, edgecolors='k', color='green', label='EBC Model Predictions')
    m_ebc, b_ebc = np.polyfit(x_ebc, y_obs, 1)
    r2_ebc = np.corrcoef(x_ebc, y_obs)[0, 1]**2
    axes2[1].plot(x_ebc, m_ebc*x_ebc + b_ebc, 'r--', label=f'Fit (R² = {r2_ebc:.4f})')
    axes2[1].set_xlabel('Log10 (EBC Model V_flat / km/s)', fontsize=14)
    axes2[1].set_title('EBC Model Tully-Fisher Relation', fontsize=18)
    axes2[1].legend()
    axes2[1].set_xlim(left=1.1)
    
    plot_path2 = os.path.join(output_dir, "tully_fisher_relation.png")
    fig2.savefig(plot_path2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print(f"  Successfully saved Tully-Fisher plot to {plot_path2}")


# ==============================================================================
# PART 4: MCMC GRAVITATIONAL LENSING TEST
# ==============================================================================

def test_lensing_mcmc(galaxies_to_test, data_dir, output_dir):
    print("\n--- Running Part 4: Referee-Proof MCMC Lensing Test ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    mcmc_plot_dir = os.path.join(output_dir, 'mcmc_details')
    os.makedirs(mcmc_plot_dir, exist_ok=True)

    for galaxy_id in galaxies_to_test:
        print(f"\n--- Starting Full MCMC Lensing Analysis for: {galaxy_id} ---")
        file_path = os.path.join(data_dir, galaxy_id + '.csv')
        galaxy_data = pd.read_csv(file_path, comment='#', skipinitialspace=True)
        if ERROR_VEL_COL not in galaxy_data.columns:
            galaxy_data[ERROR_VEL_COL] = 0.1 * galaxy_data[VELOCITY_COL] # Assign 10% error
        
        r_data = galaxy_data[RADIUS_COL].values
        v_data = galaxy_data[VELOCITY_COL].values
        verr_data = galaxy_data[ERROR_VEL_COL].values
        vgas_data = galaxy_data[GAS_VEL_COL].values
        vdisk_data = galaxy_data[DISK_VEL_COL].values
        vbulge_data = galaxy_data[BULGE_VEL_COL].values

        # --- EBC MCMC ---
        print("  Running MCMC for EBC model...")
        pos_ebc = np.array([100.0, 0.5]) + 1e-4 * np.random.randn(32, 2)
        nwalkers, ndim = pos_ebc.shape
        sampler_ebc = emcee.EnsembleSampler(nwalkers, ndim, log_probability_ebc, args=(r_data, v_data, verr_data))
        sampler_ebc.run_mcmc(pos_ebc, 5000, progress=True)
        flat_samples_ebc = sampler_ebc.get_chain(discard=1000, thin=15, flat=True)

        # --- NFW MCMC ---
        print("  Running MCMC for NFW model...")
        pos_nfw = np.array([150.0, 10.0]) + 1e-1 * np.random.randn(32, 2)
        nwalkers, ndim = pos_nfw.shape
        sampler_nfw = emcee.EnsembleSampler(nwalkers, ndim, log_probability_nfw, args=(r_data, v_data, verr_data, vgas_data, vdisk_data, vbulge_data))
        sampler_nfw.run_mcmc(pos_nfw, 5000, progress=True)
        flat_samples_nfw = sampler_nfw.get_chain(discard=1000, thin=15, flat=True)

        # --- Corner Plots ---
        corner.corner(flat_samples_ebc, labels=["A", "alpha"], title=f"EBC: {galaxy_id}")
        plt.savefig(os.path.join(mcmc_plot_dir, f'corner_ebc_{galaxy_id}.png'))
        plt.close()
        
        corner.corner(flat_samples_nfw, labels=["V200", "c"], title=f"NFW: {galaxy_id}")
        plt.savefig(os.path.join(mcmc_plot_dir, f'corner_nfw_{galaxy_id}.png'))
        plt.close()

        # --- Lensing Calculation ---
        print("  Calculating lensing predictions with error bands...")
        impact_parameters = np.linspace(max(r_data.min(), 0.1), r_data.max(), 50)
        
        # EBC Lensing Bands
        ebc_samples = flat_samples_ebc[np.random.randint(len(flat_samples_ebc), size=100)]
        preds_ebc = np.array([calculate_deflection_angle(ebc_potential_derivative, b, (A, alpha)) for b in impact_parameters for A, alpha in ebc_samples]).reshape(50, 100)
        ebc_med, ebc_low_1, ebc_high_1, ebc_low_2, ebc_high_2 = np.nanpercentile(preds_ebc, [50, 16, 84, 2.5, 97.5], axis=1)

        # NFW Lensing Bands
        nfw_samples = flat_samples_nfw[np.random.randint(len(flat_samples_nfw), size=100)]
        args_nfw_data = (r_data, vgas_data, vdisk_data, vbulge_data)
        preds_nfw = np.array([calculate_deflection_angle(nfw_full_potential_derivative, b, (V200, c) + args_nfw_data) for b in impact_parameters for V200, c in nfw_samples]).reshape(50, 100)
        nfw_med, nfw_low_1, nfw_high_1, nfw_low_2, nfw_high_2 = np.nanpercentile(preds_nfw, [50, 16, 84, 2.5, 97.5], axis=1)
        
        # --- Plot Final Comparison ---
        plt.figure(figsize=(12, 8))
        plt.plot(impact_parameters, ebc_med, 'b-', label='EBC Model (Median)', lw=2.5)
        plt.fill_between(impact_parameters, ebc_low_1, ebc_high_1, color='blue', alpha=0.3, label='EBC 68% C.I.')
        plt.fill_between(impact_parameters, ebc_low_2, ebc_high_2, color='blue', alpha=0.15, label='EBC 95% C.I.')
        
        plt.plot(impact_parameters, nfw_med, 'r--', label='NFW + Baryons (Median)', lw=2.5)
        plt.fill_between(impact_parameters, nfw_low_1, nfw_high_1, color='red', alpha=0.3, label='NFW 68% C.I.')
        plt.fill_between(impact_parameters, nfw_low_2, nfw_high_2, color='red', alpha=0.15, label='NFW 95% C.I.')

        plt.xlabel('Impact Parameter, b (kpc)', fontsize=14)
        plt.ylabel('Deflection Angle, α (arcseconds)', fontsize=14)
        plt.title(f'Referee-Proof Lensing Predictions for {galaxy_id}', fontsize=18, weight='bold')
        plt.legend(fontsize=12)
        plt.yscale('log')
        plt.grid(True, which="both", ls="--")
        
        plot_filename = os.path.join(output_dir, f'lensing_mcmc_{galaxy_id}.png')
        plt.savefig(plot_filename)
        plt.close()
        print(f"  Successfully saved final plot to {plot_filename}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    base_dir = os.getcwd()
    data_dir = os.path.join(base_dir, 'rc_parsed')
    
    results_dir = os.path.join(base_dir, 'analysis_results')
    data_output_dir = os.path.join(results_dir, 'data')
    plots_output_dir = os.path.join(results_dir, 'plots')
    os.makedirs(data_output_dir, exist_ok=True)
    os.makedirs(plots_output_dir, exist_ok=True)

    fit_results_csv_path = os.path.join(data_output_dir, 'ebc_fit_results.csv')
    
    # Run all parts of the analysis
    fit_results_df = perform_initial_fitting(data_dir, fit_results_csv_path)
    
    if fit_results_df is not None and not fit_results_df.empty:
        run_broad_tests(fit_results_df, plots_output_dir)
    
    galaxies_for_lensing_test = ['NGC3198_rotmod', 'DDO154_rotmod']
    test_lensing_mcmc(galaxies_for_lensing_test, data_dir, plots_output_dir)

    print("\n--- All analyses complete. All results are in the 'analysis_results' folder. ---")


