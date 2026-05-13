import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from scipy.stats import beta, norm

app = Flask(__name__)

DATA_SCENARIOS = {
    'sparse': {'heads': 3, 'trials': 3, 'label': 'Small Dataset (3 Heads, 3 Trials)'},
    'balanced': {'heads': 14, 'trials': 20, 'label': 'Moderate Dataset (14 Heads, 20 Trials)'},
    'large': {'heads': 720, 'trials': 1000, 'label': 'Large Dataset (720 Heads, 1000 Trials)'}
}

PRIOR_TYPES = {
    'uniform': {'alpha': 1, 'beta': 1, 'label': 'Uniform / Uninformative (Beta(1,1))'},
    'biased_fair': {'alpha': 10, 'beta': 10, 'label': 'Strong Fair Coin Belief (Beta(10,10))'},
    'biased_heads': {'alpha': 8, 'beta': 2, 'label': 'Strong Heads Bias Belief (Beta(8,2))'}
}

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_data = request.form.get('data_scenario', 'sparse')
    selected_prior = request.form.get('prior_type', 'biased_fair')
    
    n_heads = DATA_SCENARIOS[selected_data]['heads']
    n_trials = DATA_SCENARIOS[selected_data]['trials']
    n_tails = n_trials - n_heads
    
    alpha_p = PRIOR_TYPES[selected_prior]['alpha']
    beta_p = PRIOR_TYPES[selected_prior]['beta']
    
    theta_range = np.linspace(0, 1, 500)
    
    # 1. MLE Point & Asymptotic Normal Variance Calculation
    theta_mle = n_heads / n_trials
    
    # Variance of proportion = (p * (1-p)) / n
    # Boundary correction applied for edge scenarios (e.g., 3/3 heads) to avoid 0-variance collapse
    p_adjusted = max(0.01, min(0.99, theta_mle))
    mle_variance = (p_adjusted * (1 - p_adjusted)) / n_trials
    mle_std_error = np.sqrt(mle_variance)
    
    # Frequentist distribution for the parameter estimate
    mle_pdf = norm.pdf(theta_range, loc=theta_mle, scale=mle_std_error)
    
    # 2. Bayesian Calculation
    alpha_post = alpha_p + n_heads
    beta_post = beta_p + n_tails
    
    if alpha_post > 1 and beta_post > 1:
        theta_map = (alpha_post - 1) / (alpha_post + beta_post - 2)
    else:
        theta_map = theta_mle
        
    prior_pdf = beta.pdf(theta_range, alpha_p, beta_p)
    posterior_pdf = beta.pdf(theta_range, alpha_post, beta_post)
    
    # 3. Create Unified Unified Matplotlib Plot
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    # Plot curves
    ax.plot(theta_range, prior_pdf, color='#ff7f0e', linestyle=':', linewidth=2, label='Bayesian Prior (Beta)')
    ax.plot(theta_range, mle_pdf, color='#1f77b4', linewidth=2.5, label='MLE Sampling Distribution (Normal)')
    ax.plot(theta_range, posterior_pdf, color='#2ca02c', linewidth=2.5, label='Bayesian Posterior (Beta)')
    
    # Add vertical estimate indicator markers
    ax.axvline(x=theta_mle, color='#1f77b4', linestyle='--', linewidth=1.5, label=f'MLE Estimate ({theta_mle:.3f})')
    ax.axvline(x=theta_map, color='#2ca02c', linestyle='--', linewidth=1.5, label=f'MAP Estimate ({theta_map:.3f})')
    
    ax.set_title("Unified Framework Comparison: Probability Densities of θ", fontsize=12, fontweight='bold')
    ax.set_xlabel("Success Parameter (θ)")
    ax.set_ylabel("Probability Density")
    ax.set_xlim(0, 1)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='#e0e0e0')
    
    # Save chart to memory buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=110)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    
    return render_template('index.html', 
                           scenarios=DATA_SCENARIOS, 
                           priors=PRIOR_TYPES,
                           selected_data=selected_data,
                           selected_prior=selected_prior,
                           theta_mle=round(theta_mle, 4),
                           theta_map=round(theta_map, 4),
                           mle_se=round(mle_std_error, 4),
                           combined_img=img_base64)

if __name__ == '__main__':
    app.run(debug=True)
