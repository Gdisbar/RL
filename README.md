# RL

[RLHF - DPO](https://www.youtube.com/watch?app=desktop&v=aI8cyr-gH6M)

[Trading with RL - PPO](https://www.youtube.com/watch?v=oW4hgB1vIoY)

# PPO logging 🧭

| Category                        | Metric                                           | Meaning                                            | Healthy Range / Watch For     |
| ------------------------------- | ------------------------------------------------ | -------------------------------------------------- | ----------------------------- |
| 🎯 **Reward**                   | `env/reward_mean`                                | Avg. reward (performance)                          | ↑ steadily over training      |
|                                 | `env/reward_std`                                 | Reward variability                                 | Moderate; not exploding       |
|                                 | `env/reward_dist`                                | Reward distribution                                | Balanced; no long tails       |
| 🔀 **KL Divergence**            | `objective/kl`                                   | Divergence old↔new policy                          | Small positive (≈ 0.01–0.1)   |
|                                 | `objective/kl_coef`                              | KL penalty coefficient                             | Increases if KL too high      |
|                                 | `ppo/mean_non_score_reward`                      | Mean KL penalty (–KL loss)                         | Should not dominate rewards   |
|                                 | **Watch:**                                       | If KL ↑ → over-optimization; KL ↓ → under-training |                               |
| 🎲 **Entropy**                  | `objective/entropy`                              | Policy randomness                                  | High early → ↓ gradually      |
|                                 | `ppo/policy/entropy`                             | Same (softmax-based)                               | Tracks exploration level      |
| ⚖️ **Policy Stability**         | `ppo/policy/clipfrac`                            | Fraction of clipped ratios                         | < 0.2 ideally                 |
|                                 | `ppo/policy/approxkl`                            | Approx. KL (fast estimator)                        | < 0.01–0.05                   |
|                                 | `ppo/policy/ratio`                               | π_new / π_old                                      | ≈ 1; >200 = unstable          |
|                                 | `ppo/policy/advantages_mean`                     | Avg. GAE advantage                                 | Centered near 0               |
|                                 | `ppo/policy/advantages`                          | Advantage distribution                             | Balanced (not skewed)         |
| 📈 **Value Function**           | `ppo/val/mean`, `ppo/val/var`                    | Predicted values                                   | Stable variance               |
|                                 | `ppo/val/error`                                  | Value MSE vs. returns                              | Should ↓ smoothly             |
|                                 | `ppo/val/var_explained`                          | % variance explained by critic                     | > 0.8 = good                  |
|                                 | `ppo/val/clipfrac`                               | Clipped value updates                              | Low (too high → overfit)      |
|                                 | `ppo/returns/mean`, `ppo/returns/var`            | TD(λ) returns                                      | Check scaling consistency     |
| ⚙️ **Losses**                   | `ppo/loss/policy`                                | Actor loss                                         | Stable, small oscillations ok |
|                                 | `ppo/loss/value`                                 | Critic loss                                        | No NaNs or big spikes         |
|                                 | `ppo/loss/total`                                 | Combined objective                                 | Gradual ↓ trend               |
| ✏️ **Token Stats** *(for RLHF)* | `tokens/queries_len_mean` / `responses_len_mean` | Avg. token length                                  | Monitor for drift             |
|                                 | `tokens/responses_len_std`                       | Variability                                        | Not exploding                 |
| 📊 **Log-Probs**                | `objective/logprobs`, `objective/ref_logprobs`   | Log-probs of actions vs. ref                       | Inspect for shifts            |

# Crucial Values to Remember 🚨

| Metric                | Ideal Behavior             | Problem If                |
| --------------------- | -------------------------- | ------------------------- |
| `objective/kl`        | Small positive (~0.01–0.1) | Too high → instability    |
| `ppo/policy/ratio`    | ≈ 1                        | > 200 → over-optimization |
| `ppo/policy/clipfrac` | < 0.2                      | High → aggressive updates |
| `ppo/loss/value`      | Stable                     | NaN/spikes → bad critic   |
| `env/reward_mean`     | Increasing                 | Plateau → stagnation      |

# Core RL Concepts 🧠

| Term                                       | Definition                                                                                 |
| ------------------------------------------ | ------------------------------------------------------------------------------------------ |
| **Policy (π)**                             | Function mapping states to action probabilities.                                           |
| **Value Function (V(s))**                  | Expected cumulative reward from a state under π.                                           |
| **Advantage (A(s,a))**                     | How much better action *a* is than average at state *s*.                                   |
| **Return (Gₜ)**                            | Discounted sum of future rewards: Σ γⁿ rₜ₊ₙ                                                |
| **Entropy**                                | Randomness in policy — promotes exploration.                                               |
| **KL Divergence**                          | Measures how much new policy diverges from old policy.                                     |
| **Clipping (PPO)**                         | Restricts how much the policy ratio can deviate from 1, to avoid destructive updates.      |
| **GAE (Generalized Advantage Estimation)** | Smoothed advantage estimator balancing bias–variance.                                      |
| **Explained Variance**                     | Fraction of variance in returns explained by the critic — measures value function quality. |

🧩 TL;DR — What to Remember

Reward mean ↑ → Learning works.

KL small & positive → Stable policy updates.

Entropy ↓ slowly → Controlled exploration to exploitation.

Clipfrac < 0.2, ratio ≈ 1 → PPO updates stable.

Value loss steady, explained variance high → good critic.
