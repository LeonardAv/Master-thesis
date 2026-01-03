import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

r  = 0.02
p = 0.5
mu = [0.04, 0.08]
sigma = [0.15, 0.25]

mu_k = [0.80, 0.98]
xi = [2.0, 9.0]


mu_k_0 = [1.0, 1.0]
xi_0 = [0.0, 0.0]

precision_opt_beta = 10000

def mus_star(mu_val, beta_val):
    if beta_val < 0:
        return mu_val[1]
    else:
        return mu_val[0]
    
    
def xi_star_calc(xi_val, mu_k_val, beta_val):
    mu_k_star = mus_star(mu_k_val, beta_val)
    inside = -1 + (1 + beta_val * (mu_k_star - 1)) ** p
    if inside < 0:
        return xi_val[1]
    else:
        return xi_val[0]

def minimizeLambda(mu_val, sigma_val, mu_k_val, xi_val, beta_val):
    mu_star = mus_star(mu_val, beta_val)
    mu_k_star = mus_star(mu_k_val, beta_val)
    sigma_star = sigma_val[1]
    xi_star = xi_star_calc(xi_val, mu_k_val, beta_val)
    return p*r + p * (mu_star - r) * beta_val - 0.5 * p * (1 - p) * sigma_star**2 * beta_val**2 + xi_star * (-1 + (1 + beta_val * (mu_k_star - 1)) ** p)



beta_space = np.linspace(-5, 5, precision_opt_beta)

def maximize_over_beta(mu_val, mu_k_val, sigma_val, xi_val):
    utilities = np.array([minimizeLambda(mu_val, sigma_val, mu_k_val, xi_val, beta_val) for beta_val in beta_space])
    idx = np.argmax(utilities)
    return beta_space[idx], utilities[idx], utilities

beta_opt, util_opt, utilities = maximize_over_beta(mu, mu_k, sigma, xi)
beta_opt_no_jump, util_opt_no_jump, utilities_no_jump = maximize_over_beta(mu, mu_k_0, sigma, xi_0)

print(f"Optimal beta (with jumps): {beta_opt}, Utility: {util_opt}")
print(f"Optimal beta (no jumps): {beta_opt_no_jump}, Utility: {util_opt_no_jump}")

plt.figure(figsize=(10, 6))
plt.plot(beta_space, utilities, label='Utility Function (Jump Process)', color='steelblue', lw=2)
plt.scatter(beta_opt, util_opt, color='blue', zorder=5)
plt.annotate(fr'$\beta^*\!\approx {beta_opt:.2f}$'+'\n'+fr'$U\!\approx {util_opt:.3f}$',
            xy=(beta_opt, util_opt), xycoords='data',
            xytext=(-30, -15), textcoords='offset points',
            ha='center', va='top',
            arrowprops=dict(arrowstyle='->', lw=1.2, color='black'),
            fontsize=7)
plt.axhline(0, color='gray', lw=1, ls='--')
plt.axvline(0, color='gray', lw=1, ls='--')
plt.title('Long-term growth rate (Merton Jump Diffusion Model)')
plt.xlabel(r'$\beta$')
plt.ylabel('Long-term growth rate')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
#plt.show()
filename = 'Long-Term_Growth_Rate_for_Different_β_Values_Merton_Jump_Model.pdf'
plt.savefig(filename, format='pdf', bbox_inches='tight', transparent=True)
print(f"✅ Saved: {filename}")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(beta_space, utilities, label='Utility Function (Jump Process)', color='steelblue', lw=2)
plt.plot(beta_space, utilities_no_jump, label='Utility Function (No Jumps Process)', color='green', lw=2)
plt.scatter(beta_opt, util_opt, color='blue', zorder=5)
plt.scatter(beta_opt_no_jump, util_opt_no_jump, color='darkgreen', zorder=5)
plt.annotate(fr'$\beta^*\!\approx {beta_opt:.2f}$'+'\n'+fr'$U\!\approx {util_opt:.3f}$',
            xy=(beta_opt, util_opt), xycoords='data',
            xytext=(-30, -15), textcoords='offset points',
            ha='center', va='top',
            arrowprops=dict(arrowstyle='->', lw=1.2, color='black'),
            fontsize=7)
plt.annotate(fr'$\beta^*\!\approx {beta_opt_no_jump:.2f}$'+'\n'+fr'$U\!\approx {util_opt_no_jump:.3f}$',
            xy=(beta_opt_no_jump, util_opt_no_jump), xycoords='data',
            xytext=(40, -10), textcoords='offset points',
            ha='center', va='top',
            arrowprops=dict(arrowstyle='->', lw=1.2, color='black'),
            fontsize=7)
plt.axhline(0, color='gray', lw=1, ls='--')
plt.axvline(0, color='gray', lw=1, ls='--')
plt.title('Long-term growth rate (Merton Jump Diffusion Model)')
plt.xlabel(r'$\beta$')
plt.ylabel('Long-term growth rate')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
#plt.show()
filename = 'Long-Term_Growth_Rate_for_Different_β_Values_Merton_Jump.pdf'
plt.savefig(filename, format='pdf', bbox_inches='tight', transparent=True)
print(f"✅ Saved: {filename}")
plt.close()



plt.figure(figsize=(10, 6))
plt.plot(beta_space, utilities, label='Utility Function (Jump Process)', color='steelblue', lw=2)
plt.plot(beta_space, utilities_no_jump, label='Utility Function (No Jumps Process)', color='green', lw=2)
plt.axhline(0, color='gray', lw=1, ls='--')
plt.axvline(0, color='gray', lw=1, ls='--')
plt.xlim(-0.45, 0.3)
plt.ylim(-0.03, 0.015)
plt.title('Long-term growth rate (Merton Jump Diffusion Model)')
plt.xlabel(r'$\beta$')
plt.ylabel('Long-term growth rate')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
#plt.show()
filename = 'Long-Term_Growth_Rate_for_Different_β_Values_Merton_Jump_Zoom.pdf'
plt.savefig(filename, format='pdf', bbox_inches='tight', transparent=True)
print(f"✅ Saved: {filename}")
plt.close()



diff_utilities = utilities - utilities_no_jump

plt.figure(figsize=(10, 6))
plt.plot(beta_space, diff_utilities, label='Utility Function', color='steelblue', lw=2)
plt.axhline(0, color='gray', lw=1, ls='--')
plt.axvline(0, color='gray', lw=1, ls='--')
plt.title('Difference in long-term growth rate between Merton Jump Diffusion and GBM')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\Delta$ Long-term growth rate')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
#plt.show()
filename = 'Long-Term_Growth_Rate_Difference_Merton_Jump_GBM.pdf'
plt.savefig(filename, format='pdf', bbox_inches='tight', transparent=True)
print(f"✅ Saved: {filename}")
plt.close()