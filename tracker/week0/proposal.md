# [Concise Initial Proposal] Cross-Layer Residual Error Routing (CLER)

**Context:** Attention Residuals utilize cross-layer connections to smooth optimization and prevent uncontrolled hidden-state magnitude growth. In standard DeltaNet, the error term $r_l^{(i)}$ is computed to update the memory matrix $W_l^{(i)}$ and is then immediately discarded, resolved strictly within the confines of layer $l$.

**Concept:** Because linear associative memories have a fixed, limited capacity (a $d \times d$ matrix), a single Delta Rule update cannot always perfectly drive the error to zero. Currently, this uncorrected residual error vanishes as the token representation moves to layer $l + 1$. We propose a vertical “escape hatch” akin to forward-pass Gradient Boosting. By adding the previous layer’s uncorrected error to the current layer’s target value, we force higher layers to explicitly correct the associative mistakes made by lower layers.

**Formulation:** Assuming the value dimension $d_v$ is strictly identical across layers (If the value dimension $d_v$ differs between layers, a learned linear projection matrix $P_l \in \mathbb{R}^{d_{v,l} \times d_{v,l-1}}$ is applied to $r_{l-1}^{(i)}$ to align the spaces), we define a modified target value $\tilde{v}*l^{(i)}$ for layer $l$. This is a weighted sum of the current layer’s standard value $v_l^{(i)}$ and the residual error $r*{l-1}^{(i)}$ from the layer below, controlled by a learnable scalar $\gamma_l$:

$$
\tilde{v}*l^{(i)} = v_l^{(i)} + \gamma_l r*{l-1}^{(i)}
\tag{6}
$$

The prediction and the new, depth-aware residual error are then computed as:

$$
\bar{v}_l^{(i)} = W_l^{(i-1)} \phi(k_l^{(i)})
\tag{7}
$$

$$
r_l^{(i)} = \tilde{v}_l^{(i)} - \bar{v}_l^{(i)}
\tag{8}
$$

Finally, the memory is updated using this hierarchically informed cross-layer error:

$$
W_l^{(i)} = \alpha_l^{(i)} W_l^{(i-1)} + \beta_l^{(i)} \left( r_l^{(i)} \otimes \phi(k_l^{(i)}) \right)
\tag{9}
$$

**Advantage:** This creates a hierarchical error-correction mechanism. Lower layers capture high-frequency, easy-to-learn associations, while upper layers are explicitly supervised to model the more complex associative residuals that lower layers fail to capture, organically smoothing optimization across depth.
