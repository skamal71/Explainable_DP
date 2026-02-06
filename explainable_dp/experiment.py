import pandas as pd
import numpy as np

# Using the Adult Dataset from UCI
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 
           'occupation', 'relationship', 'race', 'sex', 'cap_gain', 'cap_loss', 
           'hours', 'country', 'income']
df = pd.read_csv('adult_clean.csv', names=columns)
data = df['age'].values
print(data)

def get_adaptive_bounds(data, epsilon_budget):
    """
    Calculates data-aware clipping bounds.
    Consumes 'epsilon_budget' to privately estimate min and max.
    """
    true_min = np.min(data)
    true_max = np.max(data)
    
    # For 'Age', changing one person can strictly only change the min/max 
    # within the theoretical universe of ages (e.g., 0 to 120). 
    # For simplicity, we assume global sensitivity = 1.
    sensitivity = 1.0 
    
    eps_per_stat = epsilon_budget / 2.0
    scale = sensitivity / eps_per_stat
    
    noisy_min = true_min + np.random.laplace(0, scale)
    noisy_max = true_max + np.random.laplace(0, scale)
    
    # Post-processing sanity check
    if noisy_min > noisy_max:
        noisy_min, noisy_max = noisy_max, noisy_min
        
    return noisy_min, noisy_max

# Parameters
epsilon_query = 0.90   # Budget for the Mean Calculation
epsilon_bounds = 0.10  # Budget SPECIFICALLY for calculating bounds (Data-Aware)
adap_low, adap_high = get_adaptive_bounds(data, epsilon_bounds)

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
m_x, sensitivity = get_dp_mean(data, epsilon_query, adap_low, adap_high)

# Define Candidate Trace T based on the Table 1 fields
trace_candidates = [
    {
        "id": 0, "name": "Adaptive Clipping Bounds", "value": f"[{adap_low:.1f}, {adap_high:.1f}]", 
        "type": "data-aware",
        "actual_cost": epsilon_bounds,
        "utility": 0.85
    },
    {
        "id": 1, 
        "name": "Mechanism Type", 
        "value": "Laplace", 
        "type": "post-processing",
        "actual_cost": 0.00,
        "utility": 0.40
    },
    {
        "id": 2, 
        "name": "Privacy Budget", 
        "value": epsilon_query, 
        "type": "post-processing",
        "actual_cost": 0.00,
        "utility": 0.50
    },
    {
        "id": 3, 
        "name": "Sensitivity", 
        "value": f"{sensitivity:.5f}", 
        "type": "data-aware",
        "actual_cost": 0.05, # Hypothetical cost if we added noise to sensitivity
        "utility": 0.95
    }
]

def calculate_preprocessing(trace_candidates):
    W = [] 
    U = [] 

    for field in trace_candidates:
        # We now pull the cost directly from the field definition
        # This respects the Sequential Composition logic
        W.append(field['actual_cost'])
        U.append(field['utility'])

    return W, U

W, U = calculate_preprocessing(trace_candidates)

def solve_knapsack(weights, values, capacity):
    # standard DP implementation
    n = len(values)
    cap = int(capacity * 1000) # Higher precision for small epsilons
    wt = [int(w * 1000) for w in weights]
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

def run_evaluation_study(m_x, candidates, weights, values, epsilon_limit):
    print("\n" + "="*60)
    print(f"OPTIMIZATION STUDY (Limit: {epsilon_limit})")
    print("="*60)
    
    selected_idx = solve_knapsack(weights, values, epsilon_limit)
    
    total_cost = 0
    print(f"Main Output: {m_x:.4f}\n")
    print("Selected Provenance Trace:")
    
    for idx in selected_idx:
        cand = candidates[idx]
        print(f"{cand['name']:<25} | Val: {cand['value']} | Cost: {cand['actual_cost']}")
        total_cost += weights[idx]
        
    print(f"\nTotal Provenance Cost: {total_cost:.3f} / {epsilon_limit}")
    
    # Check if Adaptive Bounds were rejected due to high cost
    if 0 not in selected_idx: # ID 0 is Clipping Bounds
        print("\nNOTE: Adaptive Bounds were computed but REJECTED by Knapsack.")
        print("      To the analyst, these bounds remain HIDDEN to save budget.")

# Run with a tight budget
run_evaluation_study(m_x, trace_candidates, W, U, epsilon_limit=0.12)