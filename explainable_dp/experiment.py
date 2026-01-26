import pandas as pd
import numpy as np

# Using the Adult Dataset from UCI
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 
           'occupation', 'relationship', 'race', 'sex', 'cap_gain', 'cap_loss', 
           'hours', 'country', 'income']
df = pd.read_csv('adult_clean.csv', names=columns)
data = df['age'].values

# Parameters
epsilon = 1.0
lower_bound = 0
upper_bound = 100

# M(X) - The DP Mean
def get_dp_mean(data, eps, low, high):
    # Clipping threshold c is applied 
    clipped_data = np.clip(data, low, high)
    true_mean = np.mean(clipped_data)
    
    # Sensitivity for mean is (high - low) / n 
    sensitivity = (high - low) / len(data)
  
    # Adding Laplace noise 
    noise = np.random.laplace(0, sensitivity / eps)
    return true_mean + noise, sensitivity

# m_x = DP-sanitized output M(X)
m_x, sensitivity = get_dp_mean(data, epsilon, lower_bound, upper_bound)

# Define Candidate Trace T based on the Table 1 fields
trace_candidates = [
    {"id": 0, "name": "Clipping Bounds", "value": f"[{lower_bound}, {upper_bound}]", "type": "data-aware"},
    {"id": 1, "name": "Mechanism Type", "value": "Laplace", "type": "post-processing"},
    {"id": 2, "name": "Privacy Budget", "value": epsilon, "type": "post-processing"},
    {"id": 3, "name": "Sensitivity", "value": sensitivity, "type": "data-aware"}
]

def calculate_preprocessing(data, trace_candidates):
    W = [] # Privacy Costs (Log-Odds Shifts) 
    U = [] # Utility

    for field in trace_candidates:
        # Privacy Cost calculation (Log-Odds Shift) 
        if field['type'] == "post-processing":
            W.append(0.00) 
        elif field['name'] == "Clipping Bounds":
            # Data-aware provenance varies with dataset and adds risk
            W.append(0.02) 
        elif field['name'] == "Sensitivity":
            # Max log-odds shift caused by observing trace beyond DP output
            W.append(0.08)

        # Utility calculation 
        if field['name'] == "Clipping Bounds":
            U.append(0.85) 
        elif field['name'] == "Mechanism Type":
            U.append(0.40)
        elif field['name'] == "Privacy Budget":
            U.append(0.50)
        elif field['name'] == "Sensitivity":
            U.append(0.95)

    return W, U

W, U = calculate_preprocessing(data, trace_candidates)

# Optimization (0-1 Knapsack) to select fields for tau* 
# Objective: Maximize sum of x_i * u_i subject to sum of x_i * w_i <= epsilon_total 
def solve_knapsack(weights, values, capacity):
    n = len(values)
    cap = int(capacity * 100)
    wt = [int(w * 100) for w in weights]
    val = [int(v * 100) for v in values]
    dp = np.zeros((n + 1, cap + 1))
    
    for i in range(1, n + 1):
        for w in range(cap + 1):
            if wt[i-1] <= w:
                dp[i][w] = max(val[i-1] + dp[i-1][w-wt[i-1]], dp[i-1][w])
            else:
                dp[i][w] = dp[i-1][w]
                
    selected_indices = []
    res = dp[n][cap]
    w = cap
    for i in range(n, 0, -1):
        if res <= 0: break
        if res != dp[i-1][w]:
            selected_indices.append(i-1)
            res -= val[i-1]
            w -= wt[i-1]
    return selected_indices

# Ablation Study Execution

def run_evaluation_study(m_x, candidates, weights, values, epsilon_total):
    print("="*60)
    print("Ablation Study")
    print("="*60)

    # Level 1: Without Provenance 
    print(f"\n[Level 1: Baseline Output (No Provenance)]")
    print(f"Released: M(X) = {m_x:.4f}")
    print("Privacy Leakage: 0.00 (Standard DP Guarantee)")

    # Level 2: Post-Processing Only 
    print(f"\n[Level 2: Post-Processing Only]")
    print(f"Released: M(X) = {m_x:.4f}")
    for i, field in enumerate(candidates):
        if weights[i] == 0:
            print(f" + REVEALED (Free): {field['name']} = {field['value']}")
    print("Privacy Leakage: 0.00 (Post-processing property)")

    # Level 3: Optimized Provenance (Algorithm 1) 
    print(f"\n[Level 3: Optimized Provenance")
    print(f"Released: M(X) = {m_x:.4f}")
    selected_idx = solve_knapsack(weights, values, epsilon_total)
    
    total_utility = 0
    total_cost = 0
    for idx in selected_idx:
        print(f" + REVEALED: {candidates[idx]['name']} = {candidates[idx]['value']}")
        total_utility += values[idx]
        total_cost += weights[idx]
    
    print(f"Summary: Total Utility = {total_utility:.2f} | Total Privacy Cost = {total_cost:.2f}/{epsilon_total}")

epsilon_limit = 0.10
run_evaluation_study(m_x, trace_candidates, W, U, epsilon_limit)