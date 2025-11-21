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

precision_opt_beta = 300
precision_max_eta = 600

p = 0.5
r = 0.015
mu = [0.05,  0.08]
mu.append((mu[0] + mu[1]) / 2)
rho = [-0.93, -0.75]
rho.append((rho[0] + rho[1]) / 2)
b = [0.1, 0.2]
b.append((b[0] + b[1]) / 2)
a = [3.0, 10.0]
a.append((a[0] + a[1]) / 2)
sigma = [0.82, 0.93]
sigma.append((sigma[0] + sigma[1]) / 2)
params = {name: eval(name) for name in ['p', 'r', 'sigma', 'mu', 'rho', 'a', 'b']}

# === eta function definitions ===
def eta(beta_val, sigma_val, rho_val, a_val):
    term1 = a_val - p * beta_val * rho_val * sigma_val
    return (1 / sigma_val**2) * (np.sqrt(term1**2 + p * (1 - p) * beta_val**2 * sigma_val**2) - term1)

def find_maximizer_eta(beta_val, sigmas_val, rho_val, a_val):
    sigma_space = np.linspace(sigmas_val[0], sigmas_val[1], precision_max_eta)
    etas = np.array([eta(beta_val, sigma_val, rho_val, a_val) for sigma_val in sigma_space])
    maximizer = np.argmax(etas)
    sigma_worst = sigma_space[maximizer]
    return sigma_worst

# === utility function definitions ===
def utility_worst_case(beta_val, mu_val, sigma_val, rho_val, a_val, b_val):
    if beta_val < 0:
        mu_star = mu_val[1]
        rho_star = rho_val[0]
    else:
        mu_star = mu_val[0]
        rho_star = rho_val[1]
    a_star = a_val[0]
    b_star = b_val[1]
    
    sigma_star = find_maximizer_eta(beta_val, sigma_val, rho_star, a_star)
    
    term1 = p * (beta_val * mu_star + (1 - beta_val) * r)
    penalty = -b_star * eta(beta_val, sigma_star, rho_star, a_star)
    return term1 + penalty

# === Optimization over beta ===
def find_opt_beta_worst_case(mu_val, sigma_val, rho_val, a_val, b_val):
    beta_vals = np.linspace(-5, 5, precision_opt_beta)
    utilities = np.array([utility_worst_case(beta_val, mu_val, sigma_val, rho_val, a_val, b_val) for beta_val in beta_vals])
    idx = np.argmax(utilities)
    return beta_vals[idx], utilities[idx], utilities


def utility_mid_case(beta_val, mu_val, sigma_val, rho_val, a_val, b_val):
    mu_star = mu_val[2]
    rho_star = rho_val[2]
    a_star = a_val[2]
    b_star = b_val[2]
    sigma_star = sigma_val[2]
    
    term1 = p * (beta_val * mu_star + (1 - beta_val) * r)
    penalty = -b_star * eta(beta_val, sigma_star, rho_star, a_star)
    return term1 + penalty

# === Optimization over beta ===
def find_opt_beta_mid_case(mu_val, sigma_val, rho_val, a_val, b_val):
    beta_vals = np.linspace(-5, 5, precision_opt_beta)
    utilities = np.array([utility_mid_case(beta_val, mu_val, sigma_val, rho_val, a_val, b_val) for beta_val in beta_vals])
    idx = np.argmax(utilities)
    return beta_vals[idx], utilities[idx], utilities


def evaluate_case(parameter_name, parameter_lower_bound, parameter_higher_bound):
    
    parameter_lows = np.linspace(parameter_lower_bound, (parameter_lower_bound + parameter_higher_bound)/2, 30)
    parameter_highs = np.linspace((parameter_lower_bound + parameter_higher_bound)/2, parameter_higher_bound, 30)
    PARAMETER_LOW, PARAMETER_HIGH = np.meshgrid(parameter_lows, parameter_highs)
    
    BETA_AVG = np.zeros_like(PARAMETER_LOW)
    BETA_WORST = np.zeros_like(PARAMETER_LOW)
    G_ROB = np.zeros_like(PARAMETER_LOW)
    C_ROB = np.zeros_like(PARAMETER_LOW)
    R_C = np.zeros_like(PARAMETER_LOW)
    R_G = np.zeros_like(PARAMETER_LOW)

    for i in range(PARAMETER_LOW.shape[0]):
        for j in range(PARAMETER_LOW.shape[1]):
                                       
            params[parameter_name][0] = PARAMETER_LOW[i,j]
            params[parameter_name][1] = PARAMETER_HIGH[i,j]
            params[parameter_name][2] = (PARAMETER_LOW[i,j] + PARAMETER_HIGH[i,j]) / 2
            
            
            #beta_avg, U_avg = find_opt_beta(mu_val = params['mu'][2], sigma_val = [params['sigma'][2], params['sigma'][2]], rho_val = params['rho'][2], a_val = params['a'][2], b_val = params['b'][1])[:2]
            #beta_worst, U_worst = find_opt_beta(mu_val = params['mu'][1], sigma_val = params['sigma'][:2], rho_val = params['rho'][0], a_val = params['a'][0], b_val = params['b'][1])[:2]
            beta_avg, U_avg = find_opt_beta_mid_case(mu_val = params['mu'], sigma_val = params['sigma'], rho_val = params['rho'], a_val = params['a'], b_val = params['b'])[:2]
            beta_worst, U_worst = find_opt_beta_worst_case(mu_val = params['mu'], sigma_val = params['sigma'], rho_val = params['rho'], a_val = params['a'], b_val = params['b'])[:2]


            rob_cost = U_worst - U_avg
            rel_growth_loss = rob_cost / U_avg if U_avg != 0 else 0
            delta_beta = beta_worst - beta_avg

            BETA_AVG[i, j] = beta_avg
            BETA_WORST[i, j] = beta_worst
            G_ROB[i, j] = delta_beta
            C_ROB[i, j] = rob_cost
            R_C[i, j] = rel_growth_loss
            R_G[i, j] = delta_beta / abs(beta_avg)

    
    
    latex_names = {'mu': r'\mu','sigma': r'\sigma','rho': r'\rho','a': r'a','b': r'b', 'r': r'r'}
    param_tex = latex_names.get(parameter_name, parameter_name)

    xlabel = fr'${param_tex}_{{\mathrm{{low}}}}$'
    ylabel = fr'${param_tex}_{{\mathrm{{high}}}}$'
        
    plot_surface(PARAMETER_LOW, PARAMETER_HIGH, BETA_AVG, fr'Average-Case Optimal Leverage (variable ${param_tex}$)', r'$\beta^*_{\mathrm{avg}}$', xlabel, ylabel, 'viridis')
    plot_surface(PARAMETER_LOW, PARAMETER_HIGH, BETA_WORST, fr'Worst-Case Optimal Leverage (variable ${param_tex}$)', r'$\beta^*_{\mathrm{worst}}$', xlabel, ylabel, 'plasma')
    plot_combined_surface(PARAMETER_LOW, PARAMETER_HIGH, BETA_AVG, BETA_WORST,
                          xlabel, ylabel, r'$\beta^*$', r'Overlay: $\beta_{avg}$ vs $\beta_{worst}$' f'(variable ${param_tex}$)',
                          r'$\beta^*_{\mathrm{avg}}$', r'$\beta^*_{\mathrm{worst}}$')
    plot_surface(PARAMETER_LOW, PARAMETER_HIGH, G_ROB, fr'Robustness Gap $\Delta \beta$ (variable ${param_tex}$)', r'$\Delta \beta$', xlabel, ylabel, 'coolwarm')
    plot_surface(PARAMETER_LOW, PARAMETER_HIGH, C_ROB, fr'Cost of Robustness (variable ${param_tex}$)', r'$C_{\mathrm{rob}}$', xlabel, ylabel, 'cividis')
    plot_surface(PARAMETER_LOW, PARAMETER_HIGH, R_C, fr'Relative Growth Loss (variable ${param_tex}$)', r'RC', xlabel, ylabel, 'inferno')
    plot_surface(PARAMETER_LOW, PARAMETER_HIGH, R_G, fr'Relative Leverage Gap (variable ${param_tex}$)', r'RG', xlabel, ylabel, 'seismic')





def plot_utility_different_betas(parameter, case_type):
    beta_space = np.linspace(-5, 5, precision_opt_beta)
    if case_type == 'mid':
        beta_star, U_star, utilities = find_opt_beta_mid_case(mu_val = params['mu'], sigma_val = params['sigma'], rho_val = params['rho'], a_val = params['a'], b_val = params['b'])
    elif case_type == 'worst':
        beta_star, U_star, utilities = find_opt_beta_worst_case(mu_val = params['mu'], sigma_val = params['sigma'], rho_val = params['rho'], a_val = params['a'], b_val = params['b'])
    else:
        raise ValueError("case_type must be either 'mid' or 'worst'")
    

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

    # === Parameter intervals text box (bottom-right, smaller) ===   
    latex_names = {'mu': r'\mu','sigma': r'\sigma','rho': r'\rho','a': r'a','b': r'b', 'p': r'p', 'r': r'r'}
    interval_text = ""
    for parameter_name, values in parameter.items():
        param_tex = latex_names.get(parameter_name, parameter_name)
        if isinstance(values, float):
            interval_text += fr'${param_tex} = {values:.2f}$' + '\n'
        elif isinstance(values, (list, np.ndarray)) and len(values) == 3:
            interval_text += fr'${param_tex} \in [{values[0]:.2f}, {values[1]:.2f}]$' + '\n'
        
    
    plt.text(0.82, 0.02, interval_text, transform=plt.gca().transAxes,
            fontsize=11, va='bottom', ha='left',
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85, boxstyle='round,pad=0.3'))

    
    # annotate the optimum
    plt.annotate(fr'$\beta^*\!\approx {beta_star:.2f}$'+'\n'+fr'$U\!\approx {U_star:.3f}$',
             xy=(beta_star, U_star), xycoords='data',
             xytext=(0, -30), textcoords='offset points',
             ha='center', va='top',
             arrowprops=dict(arrowstyle='->', lw=1.2, color='black'),
             fontsize=11)

    plt.tight_layout()
    
    
    filename = sanitize_filename(title)
    plt.savefig(filename, format='pdf', bbox_inches='tight', transparent=True)
    print(f"✅ Saved: {filename}")
    plt.close()
    #plt.show()


plot_utility_different_betas(params,  case_type='worst')
plot_utility_different_betas(params,  case_type='mid')



# === Run all cases ===
evaluate_case(parameter_name='mu', parameter_lower_bound=0.01, parameter_higher_bound=0.1)
evaluate_case(parameter_name='sigma', parameter_lower_bound=0.5, parameter_higher_bound=1.2)
evaluate_case(parameter_name='rho', parameter_lower_bound=-0.8, parameter_higher_bound=0.8)
evaluate_case(parameter_name='b', parameter_lower_bound=0.01, parameter_higher_bound=0.8)
evaluate_case(parameter_name='a', parameter_lower_bound=2, parameter_higher_bound=12)

print("All evaluations completed.")