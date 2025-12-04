import numpy as np
import matplotlib.pyplot as plt
'''
p = 0.5 # power utility exponent
r = 0.015 # risk-free rate
mu = 0.065
mu_k = 0.9
xi = 0.15
rho = -0.84
b = 0.15
a = 6.5
sigma = 0.875


def eta(beta_val, sigma_val, rho_val, a_val):
    term1 = a_val - p * beta_val * rho_val * sigma_val
    return (1 / sigma_val**2) * (np.sqrt(term1**2 + p * (1 - p) * beta_val**2 * sigma_val**2) - term1)


def util(beta_val):
    utility = p * (beta_val * mu + (1-beta_val)*r) + xi * (-1+(1+beta_val * (mu_k - 1))**p) - b * eta(beta_val, sigma, rho, a)
    return utility

def maximize_utility(beta_space):
    utilities = np.array([util(beta_val) for beta_val in beta_space])
    idx = np.argmax(utilities)
    return beta_space[idx], utilities[idx], utilities

beta_space = np.linspace(-5, 5, 10000)
beta_opt, util_opt, utilities = maximize_utility(beta_space)

print(f"Optimal beta: {beta_opt}, Utility: {util_opt}")


# Plotting the utility function
plt.figure(figsize=(10, 6))
plt.plot(beta_space, utilities, label='Utility Function', color='steelblue', lw=2)
plt.scatter(beta_opt, util_opt, color='blue', zorder=5)
plt.annotate(fr'$\beta^*\!\approx {beta_opt:.2f}$'+'\n'+fr'$U\!\approx {util_opt:.3f}$',
            xy=(beta_opt, util_opt), xycoords='data',
            xytext=(30, -30), textcoords='offset points',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
            fontsize=12, color='blue')
plt.axhline(0, color='gray', lw=1, ls='--')
plt.title('Utility Function vs. Beta')
plt.xlabel('Beta')
plt.ylabel('Utility')
plt.legend()
plt.grid()
plt.show()
'''



#----------------------------
# Next part
#mu = 0.8
p = 0.5
mu = 0.6
#zeta = 0.165
zeta = 0.35
rho = -0.7
#b = 0.8
b = 6
#a = 7.5
a = 14
sigma = 0.35


def func_lambda(beta_val):
    lam = -1/2 * (p * (beta_val-1) * sigma/a )**2 + p**2 * beta_val * (beta_val-1)*zeta*rho*sigma/a + p * (beta_val-1)*b / a
    return lam


def util2(beta_val):
    utility = p * beta_val * mu - 1/2 * p * (1-p) * beta_val**2 * zeta**2 - func_lambda(beta_val)
    return utility

def maximize_utility2(beta_space):
    utilities = np.array([util2(beta_val) for beta_val in beta_space])
    idx = np.argmax(utilities)
    return beta_space[idx], utilities[idx], utilities

beta_space = np.linspace(-5, 5, 10000)
beta_opt, util_opt, utilities = maximize_utility2(beta_space)

print(f"Optimal beta: {beta_opt}, Utility: {util_opt}")


# Plotting the utility function
plt.figure(figsize=(10, 6))
plt.plot(beta_space, utilities, label='Utility Function', color='steelblue', lw=2)
plt.scatter(beta_opt, util_opt, color='blue', zorder=5)
plt.annotate(fr'$\beta^*\!\approx {beta_opt:.2f}$'+'\n'+fr'$U\!\approx {util_opt:.3f}$',
            xy=(beta_opt, util_opt), xycoords='data',
            xytext=(30, -30), textcoords='offset points',
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
            fontsize=12, color='blue')
plt.axhline(0, color='gray', lw=1, ls='--')
plt.title('Utility Function vs. Beta')
plt.xlabel('Beta')
plt.ylabel('Utility')
plt.legend()
plt.grid()
plt.show()