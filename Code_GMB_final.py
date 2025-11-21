import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import re

# === Helper function for filename sanitization ===
def sanitize_filename(title: str) -> str:
    """Convert a plot title into a safe filename."""
    clean = re.sub(r'[^a-zA-Z0-9_\-]+', '_', title)  # keep only safe chars
    return clean.strip('_') + ".pdf"


# === Fixed general parameters ===
p = 0.5
r = 0.02

# === Helper function for 3D plotting ===
def plot_surface(X, Y, Z, title, zlabel, xlabel, ylabel, cmap='viridis'):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cmap, edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel, labelpad=7.5)
    ax.zaxis.label.set_rotation(90) # newly added to rotate z-label
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.tight_layout()
    
    filename = sanitize_filename(title)
    plt.savefig(filename, format='pdf', bbox_inches='tight', transparent=True)
    print(f"✅ Saved: {filename}")
    plt.close()
    #plt.show()

def plot_combined_surface(X, Y, Z1, Z2, xlabel, ylabel, zlabel, title, label1, label2):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z1, cmap='Blues', alpha=0.7, edgecolor='none')
    ax.plot_surface(X, Y, Z2, cmap='Reds', alpha=0.7, edgecolor='none')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.view_init(elev=30, azim=135)
    legend_lines = [
        Line2D([0], [0], color='blue', lw=4, label=label1),
        Line2D([0], [0], color='red', lw=4, label=label2)
    ]
    ax.legend(handles=legend_lines, loc='upper left')
    plt.tight_layout()
    
    filename = sanitize_filename(title)
    plt.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0.8, transparent=True)
    print(f"✅ Saved: {filename}")
    plt.close()
    #plt.show()

# CASE 1: Varying mu_low, mu_high (fixed sigma)
sigma_low = 0.15
sigma_high = 0.25
mu_lows = np.linspace(0.04, 0.08, 30)
mu_highs = np.linspace(0.08, 0.12, 30)
MU_LOW, MU_HIGH = np.meshgrid(mu_lows, mu_highs)

def evaluate_mu_case():
    BETA_AVG = np.zeros_like(MU_LOW)
    BETA_WORST = np.zeros_like(MU_LOW)
    G_ROB = np.zeros_like(MU_LOW)
    C_ROB = np.zeros_like(MU_LOW)
    R_C = np.zeros_like(MU_LOW)
    R_G = np.zeros_like(MU_LOW)

    for i in range(MU_LOW.shape[0]):
        for j in range(MU_LOW.shape[1]):
            mu_low = MU_LOW[i, j]
            mu_high = MU_HIGH[i, j]
            mu_mid = 0.5 * (mu_low + mu_high)
            sigma_mid = 0.5 * (sigma_low + sigma_high)

            if np.isclose(mu_mid, r, atol=1e-8):
                beta_avg = 0.0
            elif mu_mid > r:
                beta_avg = (mu_mid - r) / ((1 - p) * sigma_mid**2)
            else:
                beta_avg = (mu_mid - r) / ((1 - p) * sigma_mid**2)
                
            
            if mu_low < r:
                beta_worst = (mu_high - r) / ((1 - p) * sigma_high**2)
            elif mu_low < r < mu_high:
                beta_worst = 0.0
            else:
                beta_worst = (mu_low - r) / ((1 - p) * sigma_high**2)
            mu_worst = mu_low if beta_worst >= 0 else mu_high
            sigma_worst = sigma_high

            U_avg = p * r + p * (mu_mid - r) * beta_avg - 0.5 * p * (1-p) * sigma_mid**2 * beta_avg**2
            U_worst = p * r + p * (mu_worst - r) * beta_worst - 0.5 * p * (1-p) * sigma_worst**2 * beta_worst**2
            rob_cost = U_worst - U_avg
            rel_growth_loss = rob_cost / U_avg if U_avg != 0 else 0
            delta_beta = beta_worst - beta_avg

            BETA_AVG[i, j] = beta_avg
            BETA_WORST[i, j] = beta_worst
            G_ROB[i, j] = delta_beta
            C_ROB[i, j] = rob_cost
            R_C[i, j] = rel_growth_loss
            R_G[i, j] = delta_beta / abs(beta_avg)

    xlabel = r'$\mu_{\mathrm{low}}$'
    ylabel = r'$\mu_{\mathrm{high}}$'
    plot_surface(MU_LOW, MU_HIGH, BETA_AVG, r'Average-Case Optimal Leverage (variable $\mu$)', r'$\beta^*_{\mathrm{avg}}$', xlabel, ylabel, 'viridis')
    plot_surface(MU_LOW, MU_HIGH, BETA_WORST, r'Worst-Case Optimal Leverage (variable $\mu$)', r'$\beta^*_{\mathrm{worst}}$', xlabel, ylabel, 'plasma')
    plot_combined_surface(MU_LOW, MU_HIGH, BETA_AVG, BETA_WORST,
                          xlabel, ylabel, r'$\beta^*$', r'Overlay: $\beta_{avg}$ vs $\beta_{worst}$ (variable $\mu$)',
                          r'$\beta^*_{\mathrm{avg}}$', r'$\beta^*_{\mathrm{worst}}$')
    plot_surface(MU_LOW, MU_HIGH, G_ROB, r'Robustness Gap $\Delta \beta$ (variable $\mu$)', r'$\Delta \beta$', xlabel, ylabel, 'coolwarm')
    plot_surface(MU_LOW, MU_HIGH, C_ROB, r'Cost of Robustness (variable $\mu$)', r'$C_{\mathrm{rob}}$', xlabel, ylabel, 'cividis')
    plot_surface(MU_LOW, MU_HIGH, R_C, r'Relative Growth Loss (variable $\mu$)', r'RC', xlabel, ylabel, 'inferno')
    plot_surface(MU_LOW, MU_HIGH, R_G, r'Relative Leverage Gap (variable $\mu$)', r'RG', xlabel, ylabel, 'seismic')

# CASE 2: Varying sigma_low, sigma_high (fixed mu)
mu_low_fixed = 0.06
mu_high_fixed = 0.10
mu_mid_fixed = 0.5 * (mu_low_fixed + mu_high_fixed)
sigma_lows = np.linspace(0.10, 0.20, 30)
sigma_highs = np.linspace(0.20, 0.30, 30)
SIGMA_LOW, SIGMA_HIGH = np.meshgrid(sigma_lows, sigma_highs)

def evaluate_sigma_case():
    BETA_AVG = np.zeros_like(SIGMA_LOW)
    BETA_WORST = np.zeros_like(SIGMA_LOW)
    G_ROB = np.zeros_like(SIGMA_LOW)
    C_ROB = np.zeros_like(SIGMA_LOW)
    R_C = np.zeros_like(SIGMA_LOW)
    R_G = np.zeros_like(SIGMA_LOW)

    for i in range(SIGMA_LOW.shape[0]):
        for j in range(SIGMA_LOW.shape[1]):
            sigma_low = SIGMA_LOW[i, j]
            sigma_high = SIGMA_HIGH[i, j]
            sigma_mid = 0.5 * (sigma_low + sigma_high)

            if np.isclose(mu_mid_fixed, r, atol=1e-8):
                beta_avg = 0.0
            elif mu_mid_fixed > r:
                beta_avg = (mu_mid_fixed - r) / ((1 - p) * sigma_mid**2)
            else:
                beta_avg = (mu_mid_fixed - r) / ((1 - p) * sigma_mid**2)
                
                
            if mu_low_fixed < r:
                beta_worst = (mu_high_fixed - r) / ((1 - p) * sigma_high**2)
            elif mu_low_fixed < r < mu_high_fixed:
                beta_worst = 0.0
            else:
                beta_worst = (mu_low_fixed - r) / ((1 - p) * sigma_high**2)
            
            mu_worst = mu_low_fixed if beta_avg >= 0 else mu_high_fixed
            sigma_worst = sigma_high
            
            U_avg = p * r + p * (mu_mid_fixed - r) * beta_avg - 0.5 * p * (1-p) * sigma_mid**2 * beta_avg**2
            U_worst = p * r + p * (mu_worst - r) * beta_worst - 0.5 * p * (1-p) * sigma_worst**2 * beta_worst**2
            rob_cost = U_worst - U_avg
            rel_growth_loss = rob_cost / U_avg if U_avg != 0 else 0
            delta_beta = beta_worst - beta_avg

            BETA_AVG[i, j] = beta_avg
            BETA_WORST[i, j] = beta_worst
            G_ROB[i, j] = delta_beta
            C_ROB[i, j] = rob_cost
            R_C[i, j] = rel_growth_loss
            R_G[i, j] = delta_beta / abs(beta_avg)

    xlabel = r'$\sigma_{\mathrm{low}}$'
    ylabel = r'$\sigma_{\mathrm{high}}$'
    plot_surface(SIGMA_LOW, SIGMA_HIGH, BETA_AVG, r'Average-Case Optimal Leverage (variable $\sigma$)', r'$\beta^*_{\mathrm{avg}}$', xlabel, ylabel, 'viridis')
    plot_surface(SIGMA_LOW, SIGMA_HIGH, BETA_WORST, r'Worst-Case Optimal Leverage (variable $\sigma$)', r'$\beta^*_{\mathrm{worst}}$', xlabel, ylabel, 'plasma')
    plot_combined_surface(SIGMA_LOW, SIGMA_HIGH, BETA_AVG, BETA_WORST,
                          xlabel, ylabel, r'$\beta^*$', r'Overlay: $\beta_{avg}$ vs $\beta_{worst}$ (variable $\sigma$)',
                          r'$\beta^*_{\mathrm{avg}}$', r'$\beta^*_{\mathrm{worst}}$')
    plot_surface(SIGMA_LOW, SIGMA_HIGH, G_ROB, r'Robustness Gap $\Delta \beta$ (variable $\sigma$)', r'$\Delta \beta$', xlabel, ylabel, 'coolwarm')
    plot_surface(SIGMA_LOW, SIGMA_HIGH, C_ROB, r'Cost of Robustness (variable $\sigma$)', r'$C_{\mathrm{rob}}$', xlabel, ylabel, 'cividis')
    plot_surface(SIGMA_LOW, SIGMA_HIGH, R_C, r'Relative Growth Loss (variable $\sigma$)', r'RC', xlabel, ylabel, 'inferno')
    plot_surface(MU_LOW, MU_HIGH, R_G, r'Relative Leverage Gap (variable $\sigma$)', r'RG', xlabel, ylabel, 'seismic')


# Plot for utility functions for different betas
def plot_utility_different_betas(mu_low, mu_high, sigma_low, sigma_high, case_type):
    beta_space = np.linspace(-5, 5, 1000)
    utilities = []
    
    for beta_val in beta_space:
        if case_type == 'mid':
            mu_star = (mu_low + mu_high ) / 2
            sigma_star =  (sigma_low + sigma_high) / 2
        elif case_type == 'worst':
            if beta_val < 0:
                mu_star = mu_high
            elif beta_val >= 0:
                mu_star = mu_low
            sigma_star = sigma_high
        else:
            raise ValueError("case_type must be either 'mid' or 'worst'")
        utility = p * r + p * (mu_star - r) * beta_val - 0.5 * p * (1-p) * sigma_star**2 * beta_val**2
        utilities.append(utility)
        
    ix = np.argmax(utilities)
    beta_star = beta_space[ix]
    U_star = utilities[ix]
        
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.plot(beta_space, utilities, label='Utility Function', color='steelblue', lw=2)
    plt.scatter(beta_star, U_star, color='red', zorder=5)
    plt.axhline(0, color='gray', lw=1, ls='--')
    plt.axvline(0, color='gray', lw=1, ls='--')

    title = fr'Long-Term Growth Rate for Different β Values ({case_type.capitalize()} Case)'
    plt.title(title)
    plt.xlabel(r'$\beta$')
    plt.ylabel('Long-term growth rate')
    plt.legend(loc='upper left')
    plt.grid(alpha=0.3)

    
    # annotate the optimum
    plt.annotate(fr'$\beta^*\!\approx {beta_star:.2f}$'+'\n'+fr'$U\!\approx {U_star:.3f}$',
             xy=(beta_star, U_star), xycoords='data',
             xytext=(0, -30), textcoords='offset points',
             ha='center', va='top',
             arrowprops=dict(arrowstyle='->', lw=1.2, color='black'),
             fontsize=11)
    
    interval_text = (
        fr"$\mu \in [{mu_low:.2f}, {mu_high:.2f}]$" + '\n' +
        fr"$\sigma \in [{sigma_low:.2f}, {sigma_high:.2f}]$"
    )

    plt.text(0.98, 0.02, interval_text, transform=plt.gca().transAxes,
            fontsize=11, va='bottom', ha='right',
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85, boxstyle='round,pad=0.3'))


    plt.tight_layout()
    
    filename = sanitize_filename(title)
    plt.savefig(filename, format='pdf', bbox_inches='tight', transparent=True)
    print(f"✅ Saved: {filename}")
    plt.close()
    #plt.show()



plot_utility_different_betas(0.04, 0.08, 0.15, 0.25, case_type='worst')
plot_utility_different_betas(0.04, 0.08, 0.15, 0.25, case_type='mid')

# === Run both cases ===
evaluate_mu_case()
evaluate_sigma_case()


print("All evaluations completed.")