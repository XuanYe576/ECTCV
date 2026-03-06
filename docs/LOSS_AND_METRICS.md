# Loss and metrics

- **L1:** IoU-style; **L2 (SLS):** alpha = (min + (Δ/2)²) / (max + (Δ/2)²), Δ = A_p − A_t.
- **L4:** alpha uses variance 全局 abs: var = |Δ|/2, then min/(max + 2*var).
- **LLoss:** centroid = sum(x * mass) / sum(mass), not mean over pixels (fixed).
- **Pd/FA:** computed with 255*sigmoid and bin 5 (thresh 0.5) in validation.
