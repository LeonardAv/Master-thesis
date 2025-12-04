import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from scipy.optimize import shgo
from scipy.optimize import minimize
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
    
    plt.show()

precision_opt_beta = 200
precision_max_eta = 600

p = 0.5
mu = [0.06,  0.1]
mu.append((mu[0] + mu[1]) / 2)
zeta = [0.08,  0.25]
zeta.append((zeta[0] + zeta[1]) / 2)
rho = [-0.9, -0.5]
rho.append((rho[0] + rho[1]) / 2)
b = [0.06, 0.1]
b.append((b[0] + b[1]) / 2)
a = [6.0, 9.0]
a.append((a[0] + a[1]) / 2)
sigma = [0.2, 0.5]
sigma.append((sigma[0] + sigma[1]) / 2)


##try new values
zeta = 0.35
zeta = [0.1,  0.8]
zeta.append((zeta[0] + zeta[1]) / 2)
a = [11.0, 17.0]
a.append((a[0] + a[1]) / 2)

params = {name: eval(name) for name in ['p', 'mu', 'zeta', 'rho', 'b', 'a', 'sigma']}


# === Lambda function ===
def lambda_function(zeta_val, rho_val, b_val, a_val, sigma_val, beta_val):
    x = -0.5 * (p * (beta_val - 1) * sigma_val / a_val)**2 + p**2 * beta_val * (beta_val - 1) * zeta_val * rho_val * sigma_val / a_val + (p * (beta_val - 1) * b_val) / a_val
    return x

def minimizer2D(Array1, Array2):
    intial_guess = [np.mean(Array1), np.mean(Array2)]
    bounds = [(Array1[0], Array1[1]), (Array2[0], Array2[1])]
    
def utility_function(mu_val, zeta_val, rho_val, b_val, a_val, sigma_val, beta_val):
    # Differntiate between all the cases
    
    if beta_val >= 1 and rho_val[1] > 0:
    # \beta >= 1 & \bar(\rho) > 0
        mu_star = mu_val[0]
        zeta_star = zeta_val[1]
        rho_star = rho_val[1]
        b_star = b_val[1]
        #res = shgo_safe(lambda x: -lambda_function(zeta_star, rho_star, b_star, x[0], x[1], beta_val), bounds=[(a_val[0], a_val[1]), (sigma_val[0], sigma_val[1])], iters=3)
        x0 = [np.mean(a_val), np.mean(sigma_val)]  # initial guess inside bounds
        bounds_minimize = [(a_val[0], a_val[1]), (sigma_val[0], sigma_val[1])]
        res = minimize(
            lambda x: lambda_function(zeta_star, rho_star, b_star, x[0], x[1], beta_val),
            x0,
            method='L-BFGS-B',     # supports bounds
            bounds=bounds_minimize,
            options={'ftol': 1e-9}
        )
        
        a_star, sigma_star = map(float, res.x)
        lambda_value = lambda_function(zeta_star, rho_star, b_star, a_star, sigma_star, beta_val)
    elif beta_val >= 1 and rho_val[1] < 0:
    # \beta >= 1 & \bar(\rho) < 0 
        mu_star = mu_val[0]
        rho_star = rho_val[1]
        b_star = b_val[1]
        sigma_star = sigma_val[0]
        #res = shgo_safe(lambda x: -lambda_function(x[0], rho_star, b_star, x[1], sigma_star, beta_val), bounds=[(zeta_val[0], zeta_val[1]), (a_val[0], a_val[1])], iters=3)
        x0 = [np.mean(zeta_val), np.mean(a_val)]  # initial guess inside bounds
        bounds_minimize = [(zeta_val[0], zeta_val[1]), (a_val[0], a_val[1])]
        res = minimize(
            lambda x: lambda_function(x[0], rho_star, b_star, x[1], sigma_star, beta_val),
            x0,
            method='L-BFGS-B',     # supports bounds
            bounds=bounds_minimize,
            options={'ftol': 1e-9}
        )
        
        zeta_star, a_star = map(float, res.x)
        lambda_value = lambda_function(zeta_star, rho_star, b_star, a_star, sigma_star, beta_val)
    elif 0 <= beta_val < 1 and rho_val[0] < 0:
    # 0 <= \beta < 1 & \underline(\rho) < 0 
        mu_star = mu_val[0]
        zeta_star = zeta_val[1]
        rho_star = rho_val[0]
        b_star = b_val[1]
        #res = shgo_safe(lambda x: -lambda_function(zeta_star, rho_star, b_star, x[0], x[1], beta_val), bounds=[(a_val[0], a_val[1]), (sigma_val[0], sigma_val[1])], iters=3)
        x0 = [np.mean(a_val), np.mean(sigma_val)]  # initial guess inside bounds
        bounds_minimize = [(a_val[0], a_val[1]), (sigma_val[0], sigma_val[1])]
        res = minimize(
            lambda x: lambda_function(zeta_star, rho_star, b_star, x[0], x[1], beta_val),
            x0,
            method='L-BFGS-B',     # supports bounds
            bounds=bounds_minimize,
            options={'ftol': 1e-9}
        )
        
        a_star, sigma_star = map(float, res.x)
        lambda_value = lambda_function(zeta_star, rho_star, b_star, a_star, sigma_star, beta_val)
    elif 0 <= beta_val < 1 and rho_val[0] > 0:
    # 0 <= \beta < 1 & \underline(\rho) > 0
        mu_star = mu_val[0]
        rho_star = rho_val[0]
        b_star = b_val[1]
        a_star = a_val[1]
        sigma_star = sigma_val[0]
        #res = shgo_safe(lambda x: -lambda_function(x[0], rho_star, b_star, a_star, sigma_star, beta_val), bounds=[(zeta_val[0], zeta_val[1])], iters=3)
        x0 = [np.mean(zeta_val)]  # initial guess inside bounds
        bounds_minimize = [(zeta_val[0], zeta_val[1])]
        res = minimize(
            lambda x: lambda_function(x[0], rho_star, b_star, a_star, sigma_star, beta_val),
            x0,
            method='L-BFGS-B',     # supports bounds
            bounds=bounds_minimize,
            options={'ftol': 1e-9}
        )
        
        zeta_star = float(res.x[0])
        lambda_value = lambda_function(zeta_star, rho_star, b_star, a_star, sigma_star, beta_val)
    elif beta_val < 0 and rho_val[1] > 0:
    # \beta < 0 & \bar(\rho) > 0
        mu_star = mu_val[1]
        rho_star = rho_val[1]
        b_star = b_val[0]
        zeta_star = zeta_val[1]
        #res = shgo_safe(lambda x: -lambda_function(zeta_star, rho_star, b_star, x[0], x[1], beta_val), bounds=[(a_val[0], a_val[1]), (sigma_val[0], sigma_val[1])], iters=3)
        x0 = [np.mean(a_val), np.mean(sigma_val)]  # initial guess inside bounds
        bounds_minimize = [(a_val[0], a_val[1]), (sigma_val[0], sigma_val[1])]
        res = minimize(
            lambda x: lambda_function(zeta_star, rho_star, b_star, x[0], x[1], beta_val),
            x0,
            method='L-BFGS-B',     # supports bounds
            bounds=bounds_minimize,
            options={'ftol': 1e-9}
        )
        
        a_star, sigma_star = map(float, res.x)
        lambda_value = lambda_function(zeta_star, rho_star, b_star, a_star, sigma_star, beta_val)
    else:
    # \beta < 0 & \bar(\rho) < 0
        mu_star = mu_val[1]
        rho_star = rho_val[1]
        b_star = b_val[0]
        a_star = a_val[1]
        sigma_star = sigma_val[0]
        #res = shgo_safe(lambda x: -lambda_function(x[0], rho_star, b_star, a_star, sigma_star, beta_val), bounds=[(zeta_val[0], zeta_val[1])], iters=3)
        x0 = [np.mean(zeta_val)]  # initial guess inside bounds
        bounds_minimize = [(zeta_val[0], zeta_val[1])]
        res = minimize(
            lambda x: lambda_function(x[0], rho_star, b_star, a_star, sigma_star, beta_val),
            x0,
            method='L-BFGS-B',     # supports bounds
            bounds=bounds_minimize,
            options={'ftol': 1e-9}
        )
        
        zeta_star = float(res.x[0])
        lambda_value = lambda_function(zeta_star, rho_star, b_star, a_star, sigma_star, beta_val)
    
    # Calculate the utility
    util = p * beta_val * mu_star - 0.5 * p * (1-p) * beta_val**2 * zeta_star**2 - lambda_value    
    
    return util


def maximize_over_beta(parameter):
    beta_space = np.linspace(-5, 5, precision_opt_beta)
    utilities = np.array([utility_function(params['mu'], params['zeta'], params['rho'], params['b'], params['a'], params['sigma'], beta_val) for beta_val in beta_space])
    idx = np.argmax(utilities)
    return beta_space[idx], utilities[idx], utilities


def utility_mid_case(mu_val, zeta_val, rho_val, b_val, a_val, sigma_val, beta_val):
    # Use mid values for all parameters
    mu_star = mu_val[2]
    zeta_star = zeta_val[2]
    rho_star = rho_val[2]
    b_star = b_val[2]
    a_star = a_val[2]
    sigma_star = sigma_val[2]
    
    lambda_value = lambda_function(zeta_star, rho_star, b_star, a_star, sigma_star, beta_val)
    
    util_mid_case = p * beta_val * mu_star - 0.5 * p * (1-p) * beta_val**2 * zeta_star**2 - lambda_value    
    
    return util_mid_case
    
def maximize_over_beta_mid_case(parameter):
    beta_space = np.linspace(-5, 5, precision_opt_beta)
    utilities = np.array([utility_mid_case(parameter['mu'], parameter['zeta'], parameter['rho'], parameter['b'], parameter['a'], parameter['sigma'], beta_val) for beta_val in beta_space])
    idx = np.argmax(utilities)
    return beta_space[idx], utilities[idx], utilities



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

            
            beta_avg, U_avg = maximize_over_beta_mid_case(params)[:2]
            beta_worst, U_worst = maximize_over_beta(params)[:2]

            rob_cost = U_worst - U_avg
            rel_growth_loss = rob_cost / U_avg if U_avg != 0 else 0
            delta_beta = beta_worst - beta_avg

            BETA_AVG[i, j] = beta_avg
            BETA_WORST[i, j] = beta_worst
            G_ROB[i, j] = delta_beta
            C_ROB[i, j] = rob_cost
            R_C[i, j] = rel_growth_loss
            R_G[i, j] = delta_beta / abs(beta_avg)

    
    
    latex_names = {'mu': r'\mu','sigma': r'\sigma','zeta': r'\zeta','rho': r'\rho','a': r'a','b': r'b', 'r': r'r'}
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
    plot_surface(PARAMETER_LOW, PARAMETER_HIGH, R_C, fr'Relative Growth Loss (variable ${param_tex}$)', r'$RC$', xlabel, ylabel, 'inferno')
    plot_surface(PARAMETER_LOW, PARAMETER_HIGH, R_G, fr'Relative Leverage Gap (variable ${param_tex}$)', r'$RG$', xlabel, ylabel, 'seismic')



def plot_utility_different_betas(parameter, case_type):
    beta_space = np.linspace(-5, 5, precision_opt_beta)
    if case_type == 'mid':
        beta_star, U_star, utilities = maximize_over_beta_mid_case(parameter)
    elif case_type == 'worst':
        beta_star, U_star, utilities = maximize_over_beta(parameter)
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
    latex_names = {'mu': r'\mu','sigma': r'\sigma','zeta': r'\zeta', 'rho': r'\rho','a': r'a','b': r'b'}
    interval_text = ""
    for parameter_name, values in parameter.items():
        param_tex = latex_names.get(parameter_name, parameter_name)
        if isinstance(values, float):
            interval_text += fr'${param_tex} = {values:.2f}$' + '\n'
        elif isinstance(values, (list, np.ndarray)) and len(values) == 3:
            interval_text += fr'${param_tex} \in [{values[0]:.2f}, {values[1]:.2f}]$' + '\n'
        # Remove  last newline character
    interval_text = interval_text.strip()
        
    
    plt.text(0.82, 0.02, interval_text, transform=plt.gca().transAxes,
            fontsize=11, va='bottom', ha='left',
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.85, boxstyle='round,pad=0.3'))


    
    # annotate the optimum
    plt.annotate(fr'$\beta^*\!\approx {beta_star:.2f}$'+'\n'+fr'$U\!\approx {U_star:.3f}$',
             xy=(beta_star, U_star), xycoords='data',
             xytext=(0, -30), textcoords='offset points',   # ↓ straight down
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
evaluate_case(parameter_name='mu', parameter_lower_bound=0.02, parameter_higher_bound=0.1)
evaluate_case(parameter_name='sigma', parameter_lower_bound=0.1, parameter_higher_bound=0.8)
evaluate_case(parameter_name='rho', parameter_lower_bound=-1, parameter_higher_bound=1)
evaluate_case(parameter_name='b', parameter_lower_bound=0.01, parameter_higher_bound=0.8)
evaluate_case(parameter_name='zeta', parameter_lower_bound=0.04, parameter_higher_bound=0.8)
evaluate_case(parameter_name='a', parameter_lower_bound=2, parameter_higher_bound=12)


print("All evaluations completed.")