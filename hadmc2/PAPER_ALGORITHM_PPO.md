# HAD-MC 2.0: Proximal Policy Optimization (PPO) Algorithm

## 1. Problem Setting

HAD-MC 2.0 formulates the model compression problem as a Markov Decision Process (MDP):

- **State space S**: Combined model, hardware, and compression state
- **Action space A**: Joint action space of 5 agents
- **Transition dynamics P**: Model compression operations
- **Reward function R**: Multi-objective reward with Pareto optimization
- **Discount factor**: γ ∈ [0, 1)

Goal: Find policy π_θ(A|S) that maximizes expected cumulative reward:

```
J(π_θ) = E_π[ Σ_{t=0}^∞ γ^t · R_t ]
```

## 2. PPO Algorithm

### 2.1 Policy Gradient with Clipping

PPO improves the surrogate objective using clipped probability ratios:

**Clipped Surrogate Objective:**

```
L^CLIP(θ) = E_t[ min(r_t(θ) · A^CLIP(θ, A_t^old), r_t(θ)) ]
```

Where:
```
r_t(θ) = π_θ(A_t | S_t) / π_θ_old(A_t | S_t)

A^CLIP(θ, A) = clip(A, 1 - ε, 1 + ε)

L^CPI(θ) = Σ_a π_θ(a | s) log π_θ_old(a | s)
L^ENT(θ) = Σ_a π_θ(a | s) log π_θ(a | s)
```

**Probability Ratio:**

```
r_t(θ) = exp(log π_θ(a | s) - log π_θ_old(a | s))
      = π_θ(a | s) / π_θ_old(a | s)
```

**Clipping:**

```
r_t^CLIP(θ) = clip(r_t(θ), 1 - ε, 1 + ε)
```

Where ε is the clipping hyperparameter (typically ε = 0.2).

### 2.2 Architecture

**Feature Extractor:**
```
f_φ(S) = ReLU(ReLU(ReLU(x · W_1 + b_1) · W_2 + b_2) · W_3 + b_3))
```

Where:
- x ∈ ℝ^{|S|} is the state
- W_1, W_2, W_3 are weight matrices
- b_1, b_2, b_3 are bias terms
- H is the hidden dimension

**Agent-Specific Heads:**

For each agent i ∈ {pruning, quantization, distillation, fusion, update}:

```
π^i_θ(A^i | S) = softmax(h^i(f_φ(S)))
```

Where:
- h^i: ℝ^H → ℝ^{|A^i|} is agent-specific policy head
- |A^i| is the action space size for agent i

**Value Network:**
```
V_φ(S) = Linear(ReLU(ReLU(x · W_1 + b_1) · W_2 + b_2) · W_3 + b_3))
```

### 2.3 Generalized Advantage Estimation (GAE)

GAE reduces variance in advantage estimates by using value function:

```
δ_t = r_t + γ · V_φ(S_{t+1}) - V_φ(S_t)

A_t(S_t, A_t) = Σ_{k=0}^∞ (γ · λ)^k · δ_{t+k}
```

Where:
- λ ∈ [0, 1] is the GAE parameter (typically λ = 0.95)
- A_t is the advantage at time t
- δ_t is the temporal difference residual

**Advantage Properties:**
- Bias: E[A_t] = 0
- Variance: Var[A_t] = (1 + λ) / (1 - λ) · Var[δ_t]

### 2.4 Optimization Procedure

**Step 1: Collect Trajectory**
```
For t = 0 to T:
    S_t ← EncodeState(M_t)
    A_t ← π_θ(· | S_t)            # Sample action
    R_t ← Reward(S_t, A_t)         # Compute reward
    V_t ← V_φ(S_t)               # Estimate value
    Store (S_t, A_t, R_t, V_t)
```

**Step 2: Compute GAE Advantages**
```
For t = 0 to T:
    δ_t = R_t + γ · V_{t+1} - V_t          # Assuming V_{T+1} = 0

A_t = Σ_{k=t}^{T-1} (γ · λ)^{k-t} · δ_k
```

**Step 3: PPO Update**
```
For epoch = 1 to K:
    For mini-batch B ⊂ {1, ..., N}:
        # Compute policy loss
        r_t = π_θ(A_t | S_t) / π_θ_old(A_t | S_t)
        r_t^CLIP = clip(r_t, 1 - ε, 1 + ε)

        L^policy = -E_B[ Σ_a r_t^CLIP · log π_θ(a | S_t)]

        # Compute value loss
        L^value = E_B[(A_t(S_{t+1}) - Â(S_{t+1}))²]

        # Compute entropy bonus
        H = -E_a[ Σ_a π_θ(a | S) log π_θ(a | S)]

        # Total loss
        L = L^policy + c_1 · L^value - c_2 · H

        # Update parameters
        θ ← θ - α · ∇_θ L
```

Where:
- c_1 is the value loss coefficient
- c_2 is the entropy coefficient
- α is the learning rate
- ∇_θ L is the gradient of L with respect to θ

### 2.5 Theoretical Guarantees

**Theorem: Policy Monotonic Improvement**

Under standard assumptions, PPO guarantees:

```
E[L^CLIP(θ_{k+1})] ≤ E[L^CLIP(θ_k)] - η · H(π_θ_k, π_{θ_old})
```

Where:
- η is the learning rate
- H is the entropy (measures exploration)

**Proof Sketch:**
1. The clipped objective lower-bounds the unclipped objective
2. The gradient of the clipped objective points in a descent direction
3. Multiple epochs ensure cumulative improvement

**Theorem: Sample Complexity**

For ε-optimal policy with probability 1 - δ, the required number of samples is:

```
N = O( (|S| · |A| · H_max / ε²) · log(1/δ))
```

Where:
- |S| is the state space dimension
- |A| is the action space dimension
- H_max is the maximum policy entropy
- ε is the policy improvement tolerance

### 2.6 Hyperparameter Selection

**Learning Rate Schedule:**
```
α_k = α_0 · (1 - k / K)
```
Or use adaptive learning rate (Adam).

**Clip Parameter ε:**
- Small ε (e.g., 0.1): More conservative updates, slower learning
- Large ε (e.g., 0.3): More exploration, potentially unstable

**GAE Parameter λ:**
- λ = 0: Monte Carlo returns (high variance)
- λ = 1: 1-step returns (high bias)
- λ = 0.95: Good balance (low variance, low bias)

**Value Coefficient c_1:**
- Small c_1: Faster policy improvement, potential overfitting
- Large c_1: More stable, slower improvement

## 3. HAD-MC 2.0 Specifics

### 3.1 Multi-Agent Action Encoding

The joint policy encodes actions for 5 agents:

```
π_θ(A | S) = π^pruning_θ(A^pruning | S) · π^quantization_θ(A^quantization | S) · ...
```

This factorization allows:
- Agent-specific action spaces (discrete vs continuous)
- Independent exploration of each technique
- Shared feature extraction

### 3.2 Continuous Action: Distillation

For the distillation agent (continuous actions), use reparameterization:

```
μ(S) = h^μ(f_φ(S))         # Mean head
σ(S) = softplus(h^σ(f_φ(S)))   # Std head (ensures positive)

π_θ(T, α | S) = Normal(μ(S), σ(S))
```

**Action Sampling:**
```
(T, α) ~ Normal(μ(S), σ(S))
T' = clip(T, 1.0, 20.0)    # Temperature
α' = sigmoid(α)               # Alpha
```

**Log Probability:**
```
log π(T, α | S) = log N(T; μ, σ) + log σ(α')
```

Where N is the Gaussian distribution.

### 3.3 Entropy Regularization

To encourage exploration:

```
H = -Σ_i Σ_a π_θ^i(a | S) log π_θ^i(a | S)
```

Add negative entropy to loss: L_total = L^policy + L^value - c_2 · H

This prevents premature convergence to suboptimal policies.

## 4. Pseudocode

```
Algorithm 1: PPO for HAD-MC 2.0

Input: Model M_0, Teacher T, Data D, HAL, Reward R
Output: Compressed model M*

Initialize:
    θ ← Random network parameters
    φ ← Random network parameters
    Memory buffer ← Empty
    best_reward ← -∞

For episode = 1 to E:
    M ← M_0  # Reset to original model

    For step = 1 to S:
        // 1. Get state
        S ← EncodeState(M, HAL)

        // 2. Sample action
        A ← SampleAction(π_θ(· | S))

        // 3. Apply compression
        M ← ApplyCompression(M, A, D)

        // 4. Evaluate
        metrics ← Evaluate(M, D_val)
        R ← R(metrics)

        // 5. Store transition
        Memory.Add(S, A, R, φ(S))

    // 6. Compute GAE
    For t = 1 to S:
        δ_t ← R_t + γ · φ(S_{t+1}) - φ(S_t)
        A_t ← Σ_{k=t}^{S-1} (γ·λ)^{k-t} · δ_k

    // 7. PPO Update
    For epoch = 1 to K:
        For mini-batch B ⊂ Memory:
            // Policy loss
            r ← π_θ(A_B | S_B) / π_θ_old(A_B | S_B)
            r^CLIP ← clip(r, 1-ε, 1+ε)
            L_policy ← -E_B[Σ_a r^CLIP · log π_θ^old(a | S_B)]

            // Value loss
            L_value ← E_B[(Â(S_{t+1}) - A(S_{t+1}))²]

            // Entropy
            H ← -E_a[Σ_a π_θ(a | S) log π_θ(a | S)]

            // Total loss
            L ← L_policy + c1·L_value - c2·H

            // Update
            θ ← θ - α·∇_θL

        // 8. Save best model
        If reward > best_reward:
            best_reward ← reward
            M* ← M
```

## 5. References

1. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347
2. Wu, Y., et al. (2017). "Scalable Trust-Region Method for Deep Reinforcement Learning Using Model-Free Policy Optimization." CoRR 2017
3. Cobbe, N., et al. (2019). "On Correcting the Target in Softmax-Entropy Gradient Backpropagation." arXiv:1903.02614
4. Mnih, V., et al. (2016). "Asynchronous Methods for Deep Reinforcement Learning." ICML 2016
5. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." arXiv:1707.06347
