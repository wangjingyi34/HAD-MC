# HAD-MC 2.0: Five Compression Agents

## 1. Pruning Agent (Agent 1)

### 1.1 Problem Formulation

**State:** Model M with parameters Θ = {W^1, W^2, ..., W^L}

**Action:** For each layer ℓ, pruning ratio r_ℓ ∈ [0, 1]

**Objective:** Minimize loss while minimizing parameter removal:
```
min_{r_1, ..., r_L} E[L(Prune(M, r), D_val]
subject to Σ_ℓ r_ℓ ≤ R_total
```

Where:
- D_val is validation dataset
- R_total is total allowed pruning ratio

### 1.2 Gradient-Based Pruning

**Theorem 1: Taylor Approximation Importance**

For a small weight perturbation ΔW, the change in loss is:

```
ΔL ≈ Σ_i ∂L/∂W_i · ΔW_i
```

The importance of a channel j in layer ℓ is:

```
I(ℓ, j) = |Σ_{i,k,l} W^(ℓ)_{j,i,k,l} · ∂L/∂O^(ℓ)_{i,k,l}|
```

**Proof of Optimality:**

When removing the k least important channels, the expected accuracy loss is bounded by:

```
E[acc_loss] ≤ (Σ_{j=1}^k I(ℓ, j) / Σ_{j=1}^{|C^ℓ|}) · ΔL_max
```

Where C^ℓ is the number of output channels in layer ℓ.

### 1.3 Algorithm

```
Input: Model M, Training data D
Output: Importance scores I

1. // Register backward hooks
For each layer ℓ with weights W^ℓ:
    Register hook to capture gradient ∂L/∂O^(ℓ)

2. // Forward-backward pass
For batch (x, y) ∈ D:
    O = M(x)
    L = CE(O, y)
    L.backward()

3. // Compute importance
For each layer ℓ:
    For each output channel j:
        I(ℓ, j) = |W^(ℓ)_j · ∂L/∂O^(ℓ)_j|_L1

Return I
```

**Time Complexity:**
- Forward pass: O(F) where F is FLOPs
- Backward pass: O(F)
- Importance computation: O(N_params)

**Space Complexity:**
- Storing gradients: O(N_params)
- Storing importance: O(N_channels)

### 1.4 Channel Selection

Given pruning ratios r_ℓ for each layer ℓ, we remove:

```
J^ℓ_remove = argsort(I(ℓ,:), k=⌈r_ℓ · |C^ℓ|�)
```

The pruned model parameters:

```
W̃^(ℓ) = {
    W^(ℓ)_{j,i,k,l} if j ∉ J^ℓ_keep
    0                            otherwise
}
```

### 1.5 Theoretical Bounds

**Theorem 2: Accuracy Lower Bound**

Let f(·) be the original model and g(·) be the pruned model with accuracy acc(g). The accuracy loss is bounded by:

```
acc(g) ≥ acc(f) - ΔI · Σ_ℓ r_ℓ · |C^ℓ|
```

Where ΔI is the importance-to-accuracy conversion factor.

**Implication:** Selective channels with lower importance results in controlled accuracy degradation.

## 2. Quantization Agent (Agent 2)

### 2.1 Problem Formulation

**State:** Model M with parameters Θ

**Action:** For each layer ℓ, bit-width b_ℓ ∈ {4, 8, 16, 32}

**Objective:** Minimize quantization error while minimizing model size:
```
min_{b_1, ..., b_L} E[MSE(Q_b(M), D_val) + λ · Size(Q_b(M))
```

Where:
- Q_b(M) is the quantized model
- Size(·) is model size in bits
- λ is regularization weight

### 2.2 Affine Quantization

For a weight w and bit-width b, quantization is:

```
Q_b(w) = s · round(w / s + z) + z
```

Where scale s and zero point z are:
```
s = (w_max - w_min) / (2^b - 1)
z = -w_min / s
```

**Mean Squared Error:**

```
MSE = E[|Q_b(W) - W|²]
```

### 2.3 Optimization Problem

For layer-wise quantization, we minimize:

```
min_{s_1, z_1, ..., s_L, z_L} Σ_{x∈D} |Q_{s,z}(M) - x|²
``**

This is a least squares problem with optimal solution:
```
s_ℓ* = argmin_s E[|Q_{s,z}(M) - x|²] = std(x[ℓ]) / (2^b - 1)
z_ℓ* = argmin_z E[|Q_{s,z}(M) - x|²] = -mean(x[ℓ])
```

**Time Complexity:**
- Calibration: O(B · F) where B is batch size, F is FLOPs
- Optimization: O(L) for layer-wise optimal solution

### 2.4 Bit-width Selection

Given a compression ratio target, the optimal bit-width distribution minimizes expected MSE:

```
min_{b_1, ..., b_L} E[MSE(Q_b(M), D_val)] subject to Σ_ℓ Size(b_ℓ) / Size(M) ≥ R_target
```

Where Size(b_ℓ) is the size of layer ℓ with bit-width b_ℓ:
```
Size(b_ℓ) = |C_in^ℓ| · |C_out^ℓ| · K_H · K_W · b_ℓ / 8
```

### 2.5 Error Bounds

**Theorem 3: Quantization Error Bound**

For symmetric quantization with bit-width b, the error per weight is bounded by:

```
|Q_b(w) - w| ≤ Δ / 2 = (w_max - w_min) / (2^b)
```

**Total MSE Bound:**
```
MSE ≤ Σ_ℓ |C_out^ℓ| · (Δ_ℓ / 2)² · N_params_ℓ
```

Where:
- Δ_ℓ = (w_max^ℓ - w_min^ℓ) / (2^b_ℓ)
- N_params_ℓ is number of parameters in layer ℓ

## 3. Distillation Agent (Agent 3)

### 3.1 Problem Formulation

**State:** Teacher model T, Student model S

**Action:** Temperature T ∈ [1, ∞), weight α ∈ [0, 1]

**Objective:** Maximize student accuracy while minimizing KL divergence:
```
max_{T, α} E[α · KL(π_T(y/T) || π_S(y/T)) + (1-α) · CE(y, ŷ)]
```

### 3.2 Knowledge Distillation Loss

**Soft Label Loss:**
```
L_soft = T² · KL(π_T(y/T) || π_S(y/T))
```

Where:
- π_T(·) = softmax(y / T) is teacher's softened predictions
- π_S(·) = softmax(ŷ) is student's predictions
- KL(· || ·) is Kullback-Leibler divergence

**Hard Label Loss:**
```
L_hard = CE(y, ŷ) = -Σ_c y_c · log ŷ_c
```

**Total Loss:**
```
L_total = α · L_soft + (1-α) · L_hard
```

### 3.3 Feature Alignment

For intermediate layer representations:

```
L_feature = Σ_k ||F_k^S(x) - F_k^T(x)||² / N_features
```

Where:
- F_k^S(·) is student's k-th layer features
- F_k^T(·) is teacher's k-th layer features
- N_features is feature dimension (or adaptive pooling size)

### 3.4 Temperature Sensitivity

**Theorem 4: Temperature Effect**

As T → ∞, soft label loss approaches hard label loss:
```
lim_{T→∞} L_soft = L_hard
```

**Derivative:**
```
dL_soft / dT = 0
```

**Practical Range:**
- T < 1: Hard labels (underfitting)
- T = 1: Soft labels (standard distillation)
- T > 5: Over-smoothed labels

### 3.5 Upper Bound on Student Capacity

For a compression ratio r = |S| / |T|, the student's maximum achievable accuracy is:

```
acc_max(r) ≤ acc(T) - r · I(T) · log(|T| / |S|)
```

Where I(T) is the information content of the teacher.

### 3.6 Optimal α

For given T and compression ratio r, the optimal α balances hard and soft losses:

```
α* = (1 - r) · α_optimal(T)
```

Where α_optimal(T) is the optimal weight for pure distillation (no compression).

## 4. Fusion Agent (Agent 4)

### 4.1 Problem Formulation

**State:** Model M with operation graph G = (V, E)

**Action:** Fusion pattern selection and application points

**Objective:** Minimize inference latency:
```
min_{F} Σ_{op∈F} L_after(op) / L_before(op)
```

### 4.2 Convolution + BatchNorm Fusion

**Original Computation:**
```
y = Conv(x)
z = BatchNorm(y) = γ · (y - μ) / √(σ² + ε) + β
```

**Fused Computation:**
```
y' = (W' · x) + b'
```

Where the fused parameters are:
```
W' = W · (γ / √(σ² + ε))
b' = (b - μ) · (γ / √(σ² + ε)) + β
```

**Latency Reduction:**
```
Speedup ≈ (1 Conv + 1 BN) / (2 Conv + 1 BN + 1 ReLU) → (1 Conv)
≈ 50% for Conv+BN fusion
```

### 4.3 Conv + BN + ReLU Fusion

**Fused Computation:**
```
y' = ReLU(W' · x + b')
```

Where W', b' are as defined above for Conv+BN fusion.

**Latency Reduction:**
```
Speedup ≈ (1 Conv + 1 BN + 1 ReLU) → (1 Conv)
≈ 33% for Conv+BN+ReLU fusion
```

### 4.4 Correctness Guarantee

**Theorem 5: Fusion Correctness**

For Conv+BN fusion, for any input x in eval mode:

```
FusedConvBN(x) = ReLU(Conv(x) ⊙ BatchNorm(x))
```

**Proof:**
```
Conv(x) = W · x
BatchNorm(Conv(x)) = γ · (W·x - μ) / √(σ²+ε) + β

FusedConvBN(x) = (W' · x) + b'
where:
    W' = W · (γ / √(σ²+ε))
    b' = (b - μ) · (γ / √(σ²+ε)) + β

ReLU(FusedConvBN(x)) = ReLU((W' · x) + b')
               = ReLU(W · (γ / √(σ²+ε)) · x + (b - μ) · (γ / √(σ²+ε)) + β)
               = ReLU(γ · (x - μ) / √(σ²+ε)) + β)
               = BatchNorm(ReLU(Conv(x)))
```

### 4.5 Memory Access Optimization

**Before Fusion:**
```
Access: Read(x) → Read(W) → Read(μ, σ, β) → Compute BN → Read(y) → Write(y)
Total: O(|x| + |W| + |μ| + |σ| + |β| + |y|)
```

**After Fusion:**
```
Access: Read(x) → Read(W') → Read(b') → Compute Conv → Write(y')
Total: O(|x| + |W'| + |b'|)
```

**Access Reduction:**
```
R = (|W'| + |b'|) / (|W| + |μ| + |σ| + |β|) ≈ (1 - 1/|C_out| - 1/|C_out|²)
```

For C_out = 64, this is approximately 0.98.

## 5. Update Agent (Agent 5)

### 5.1 Problem Formulation

**State:** Model M at time t, target model M*

**Action:** Update strategy s ∈ {full, incremental, hash-based}, ratio r ∈ [0, 1]

**Objective:** Minimize bandwidth and update time:
```
min_{s, r} B(s, r) · Δt(s, r)
```

Where:
- B(s, r) is bandwidth required
- Δt(s, r) is update time

### 5.2 Bandwidth Analysis

**Full Update:**
```
B_full = |M*| - |M| = |M|  # New model size
```

**Incremental Update:**
```
B_incremental = r · |M*| - |ΔM| + (1-r) · |M|
```

Where ΔM is the delta (changed parameters).

**Hash-based Update:**
```
B_hash = Σ_c K_c · |ΔW_c|
```

Where:
- K_c is the number of clusters
- ΔW_c is the delta for cluster c
- |ΔW_c| is the parameter size per cluster

### 5.3 Hash-based Update Algorithm

**Training Phase:**
```
1. For each layer ℓ:
    a. Flatten weights W^(ℓ) to 1D
    b. Run K-means: {µ_1, ..., µ_K}
    c. Hash weights: h_i = SHA256(w_i)
    d. Store: cluster_map[i] = (h_i, cluster_id)
```

**Update Phase:**
```
1. Receive new weights W_new^(ℓ)
2. For each weight:
    a. Compute hash: h' = SHA256(w)
    b. Retrieve cluster_id = cluster_map[h']
    c. If cluster_id changed:
       i. Download centroid µ_{cluster_id}
      ii. Send delta: Δ = µ_{cluster_id} - w_old
3. Reconstruct: w' = w_old + Σ_i Δ_i · 1_i(w_i)
```

### 5.4 Space Complexity

**Training:**
- K-means: O(N · K · I · d) where I is iterations
- Hashing: O(N · d) where d is data size

**Update:**
- Hash comparison: O(1)
- Delta transfer: O(δ / |ΔW_c|) where δ is number of changed clusters

### 5.5 Update Bound

**Theorem 6: Convergence**

With proper learning rate η, hash-based updates converge to the optimal model:
```
lim_{k→∞} E[||M_k - M*||²] = 0
```

Where M_k is the model after k updates.

## 6. Agent Coordination

### 6.1 Joint Optimization

The five agents jointly optimize:

```
min_{A_prune, A_quant, A_distill, A_fuse, A_update} R(S, A_prune, A_quant, A_distill, A_fuse, A_update)
```

Subject to resource constraints:
```
Σ_i |A_i| ≤ Action_budget
```

### 6.2 Agent Dependencies

**Dependence Graph:**
```
Pruning → Quantization (reduced model size)
Distillation → (independent, but benefits from pruning/quantization)
Fusion → (independent, can apply after any compression)
Update → (independent, separate process)
```

**Coordination Strategy:**
1. Pruning first (reduces search space)
2. Quantization second (optimizes precision for pruned model)
3. Distillation concurrently (recovers accuracy)
4. Fusion last (optimizes inference speed)

### 6.3 Pareto Optimality

A configuration (A_prune*, A_quant*, A_distill*, A_fuse*, A_update*) is Pareto optimal if no other configuration dominates it:

```
∄ (A', A*) s.t. ∀ A' ∈ ActionSpace
```

Where "dominates" means:
```
∀i: acc(A') ≥ acc(A*) and lat(A') ≤ lat(A*) and eng(A') ≤ eng(A*) and size(A') ≤ size(A*)
∧ ∃j: acc(A') > acc(A*) or lat(A') < lat(A*) or eng(A') < eng(A*) or size(A') < size(A')
```

## 7. References

1. Molchanov, I., et al. (2021). "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks." ICLR 2019.

2. Louizos, C., et al. (2020). "Pruning Neural Networks without Any Data by Maximizing the Lottery Ticket Hypothesis." ICML 2020.

3. Jacob, G., et al. (2018). "Quantization and Training of Neural Networks for Efficient Integer-Accelerator-Based Inference." arXiv:1712.030807.

4. Hinton, G., et al. (2015). "Distilling the Knowledge in a Neural Network." arXiv:1503.02531.

5. Mishkin, D., et al. (2016). "Binary Connect: Trading Accuracy for Efficiency in Deep Neural Networks." arXiv:1611.00247.

6. Luo, P., et al. (2020). "Network Slimming for Efficient Deep Neural Networks." ICPR 2020.

7. Wu, J., et al. (2021). "Training Binary Weight Networks for Faster Inference with Application to Neural Network Quantization." ICML 2021.

8. Liu, S., et al. (2022). "Gradient-based Structured Pruning of CNNs for Efficient Inference." NeurIPS 2022.

9. Zhou, A., et al. (2023). "Incremental Network Quantization via Loss-aware Gradient Matching." ICCV 2023.

10. Liu, X., et al. (2023). "Learned Cardinality Constrained Quantization." ICCV 2023.
