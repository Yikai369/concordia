# Particle Filter for Belief Tracking in Impression Management

## Overview

The impression management study uses a **1-D particle filter** to track an actor's belief about the audience's hidden evaluation state. The particle filter enables the actor to maintain a probabilistic estimate of how the audience perceives their performance, updating this belief as new observations (audience responses) become available.

## Purpose in the Study

In the impression management conversation system:

- **Actor (Interviewee)**: Tries to maximize perceived competence
- **Audience (Interviewer)**: Has a true hidden evaluation state I_t ∈ [0,1] that the actor cannot directly observe
- **Challenge**: The actor must infer the audience's evaluation from their verbal and non-verbal responses
- **Solution**: Use a particle filter to maintain a belief distribution over possible evaluation states

The particle filter allows the actor to:
1. Track uncertainty about the audience's evaluation
2. Update beliefs probabilistically based on noisy observations
3. Handle non-linear and non-Gaussian belief dynamics
4. Provide a principled framework for belief updating

## Mathematical Formulation

### State Space

The hidden state is a scalar value in the bounded interval [0,1]:
- **I_t**: True evaluation state at turn t (hidden, known only to audience)
- **I_hat_t**: Actor's estimated belief about I_t (computed from particle filter)

### Process Model (State Transition)

The particle filter assumes the evaluation state evolves according to a random walk with Gaussian noise:

```
I_{t+1} = I_t + ε_process
```

where ε_process ~ N(0, σ_process²) and the result is clamped to [0,1].

**Default parameters**:
- `process_sigma = 0.03`: Standard deviation of process noise
- This models the assumption that the audience's evaluation may drift slightly over time

### Observation Model (Measurement)

The actor receives noisy observations (measurements) of the hidden state through the audience's responses. The measurement is extracted via an LLM that interprets the audience's dialogue and body language:

```
z_t = I_t + ε_obs
```

where ε_obs ~ N(0, σ_obs²) and z_t ∈ [0,1].

**Default parameters**:
- `obs_sigma = 0.06`: Standard deviation of observation noise (used in likelihood computation)
- Note: The actual measurement extraction uses `obs_sigma = 0.03` in the update step

### Likelihood Function

Given a measurement z_t and a particle x_i, the likelihood is:

```
p(z_t | x_i) = (1 / (σ_obs * √(2π))) * exp(-0.5 * ((z_t - x_i) / σ_obs)²)
```

For computational efficiency, we use the unnormalized form:

```
w_i ∝ exp(-0.5 * ((z_t - x_i) / σ_obs)²)
```

## Algorithm

The particle filter maintains:
- **N particles**: {x₁, x₂, ..., x_N} where each x_i ∈ [0,1]
- **N weights**: {w₁, w₂, ..., w_N} where Σw_i = 1

### Step 1: Initialization

On the first turn, particles are initialized:

```python
# Default: Sample from Gaussian around 0.5 (weakly informed prior)
particles = [clamp(0.5 + N(0, 0.15)) for _ in range(N)]
weights = [1/N] * N  # Uniform weights
```

**Parameters**:
- `num_particles = 200`: Number of particles (default)
- Initial distribution: N(0.5, 0.15²) clamped to [0,1]

### Step 2: Prediction (Process Update)

Before incorporating new observations, particles are diffused with process noise:

```python
particles_pred = [clamp(x_i + N(0, σ_process²)) for x_i in particles]
```

This step:
- Models uncertainty about how the state may have changed
- Spreads particles to account for possible state evolution
- Maintains the same weights (no observation yet)

### Step 3: Measurement Extraction

The actor uses an LLM to extract a measurement from the audience's response:

```python
# LLM prompt: "Estimate audience's internal evaluation from their response [0,1]"
measurement = extract_numeric_from_llm_response(audience_response)
measurement = clamp(measurement, 0.0, 1.0)  # Ensure in [0,1]
```

The measurement is a noisy observation of the true state I_t.

### Step 4: Update (Measurement Update)

Particles are weighted by their likelihood given the measurement:

```python
# Compute unnormalized weights
weights = [exp(-0.5 * ((measurement - x_i) / σ_obs)²) for x_i in particles_pred]

# Normalize
weights = weights / sum(weights)
```

If all weights collapse to zero (numerical issue), fall back to uniform weights.

### Step 5: Effective Sample Size (ESS) and Resampling

**Effective Sample Size** measures particle diversity:

```
ESS = 1 / Σ(w_i²)
```

- **ESS = N**: All particles have equal weight (maximum diversity)
- **ESS << N**: Few particles have most of the weight (particle degeneracy)

**Resampling threshold**: If ESS < 0.5 * N, resample particles.

**Systematic Resampling Algorithm**:
1. Generate N positions uniformly spaced in [0,1] with random offset
2. Map positions to cumulative weight distribution
3. Select particles based on positions
4. Reset weights to uniform (1/N)

This ensures:
- Particles with high weight are more likely to be selected
- Particle diversity is restored
- Computational efficiency (O(N) complexity)

### Step 6: Posterior Mean (Belief Estimate)

The actor's belief I_hat is the weighted mean of particles:

```python
I_hat = Σ(x_i * w_i)  # Weighted mean
```

If weights are uniform (after resampling):
```python
I_hat = mean(particles)  # Simple mean
```

## Implementation Details

### Class Structure

```python
class ParticleFilter:
    def __init__(self, num_particles=200, process_sigma=0.03,
                 obs_sigma=0.08, rng=None):
        # Initialize filter parameters

    def initialize(self, particles=None):
        # Initialize particles and weights

    def predict(self, particles):
        # Apply process noise (random walk)

    def update(self, particles, observation):
        # Weight particles by observation likelihood
        # Resample if ESS < threshold
        # Returns: (particles, weights, ess, resampled)

    def _systematic_resample(self, weights):
        # Systematic resampling algorithm
```

### Integration with Actor

The particle filter is used in `actor_update_particles()`:

```python
def actor_update_particles(self, turn, listener_utt, goal_description):
    # 1. Initialize or load PF state
    if not self.memory.pf_particles:
        particles, weights = pf_model.initialize()
    else:
        particles = self.memory.pf_particles
        weights = self.memory.pf_weights

    # 2. Predict step
    prior_mean = mean(particles)
    particles_pred = pf_model.predict(particles)

    # 3. Extract measurement from audience response
    measurement = extract_measurement_via_llm(listener_utt)

    # 4. Update step (weight + resample)
    particles_upd, weights_upd, ess, resampled = pf_model.update(
        particles_pred, measurement
    )

    # 5. Compute posterior mean
    I_hat = weighted_mean(particles_upd, weights_upd)

    # 6. Store state
    self.memory.pf_particles = particles_upd
    self.memory.pf_weights = weights_upd
    self.memory.pf_history.append({
        "turn": turn,
        "prior_mean": prior_mean,
        "I_hat": I_hat,
        "ess": ess,
        "resampled": resampled,
        "measurement": measurement
    })

    return I_hat, ess
```

## Parameters and Tuning

### Key Parameters

| Parameter | Default | Description | Effect of Increase |
|-----------|---------|-------------|-------------------|
| `num_particles` | 200 | Number of particles | More accurate but slower |
| `process_sigma` | 0.03 | Process noise std dev | More belief drift between turns |
| `obs_sigma` | 0.08 | Observation noise std dev | Less trust in measurements |

### Parameter Effects

**num_particles**:
- **Low (50-100)**: Faster computation, less accurate, more variance
- **Medium (200)**: Good balance (default)
- **High (500-1000)**: More accurate, slower, diminishing returns

**process_sigma**:
- **Low (0.01)**: Belief changes slowly, assumes stable evaluation
- **Medium (0.03)**: Moderate drift (default)
- **High (0.05-0.1)**: Rapid belief changes, assumes volatile evaluation

**obs_sigma**:
- **Low (0.03-0.05)**: High trust in measurements, sharp likelihood
- **Medium (0.08)**: Moderate trust (default for likelihood)
- **High (0.1-0.2)**: Low trust, diffuse likelihood, slower updates

**Note**: The implementation uses `obs_sigma = 0.03` in the actual update step (line 510), which is more conservative than the default 0.08.

## Prediction Error Computation

The prediction error (PE) is computed as the change in belief:

```python
PE_t = |I_hat_{t-1} - I_hat_t|
```

This measures:
- **How much the actor's belief changed** after observing the audience's response
- **Not the error from true state** (which is unknown to the actor)

A large PE indicates:
- The audience's response was surprising given previous belief
- The actor's understanding of the situation changed significantly
- Potential need for behavioral adaptation

## Effective Sample Size (ESS)

ESS is a diagnostic metric for particle filter health:

```
ESS = 1 / Σ(w_i²)
```

### Interpretation

- **ESS ≈ N**: Healthy filter, good particle diversity
- **ESS < 0.5 * N**: Particle degeneracy, resampling triggered
- **ESS << N**: Severe degeneracy, most weight on few particles

### Why Resample?

When ESS is low:
- Most particles have negligible weight
- Belief estimate relies on very few particles
- Filter loses ability to represent uncertainty
- Resampling restores diversity

### Resampling Strategy

**Systematic Resampling**:
- Deterministic spacing with random offset
- O(N) complexity
- Low variance compared to multinomial resampling
- Preserves particle diversity better than simple resampling

## State Persistence

The particle filter state is stored in `IMPEMemoryComponent`:

```python
# Stored state
pf_particles: List[float]  # Current particles
pf_weights: List[float]   # Current weights
pf_history: List[Dict]     # History of updates
```

**pf_history** entries contain:
- `turn`: Turn number
- `prior_mean`: Mean before update
- `I_hat`: Posterior mean (belief estimate)
- `ess`: Effective sample size
- `resampled`: Whether resampling occurred
- `measurement`: Extracted measurement value

This allows:
- **Checkpointing**: Save/restore filter state
- **Analysis**: Track belief evolution over time
- **Debugging**: Inspect filter behavior

## Use Cases and Applications

### In Impression Management Study

1. **Belief Tracking**: Actor maintains probabilistic belief about audience evaluation
2. **Adaptive Behavior**: Actor adjusts actions based on I_hat
3. **Uncertainty Quantification**: ESS provides measure of confidence
4. **Learning Signal**: PE computed from belief changes drives reflection

### General Applications

Particle filters are useful when:
- State space is non-linear or non-Gaussian
- Observations are noisy
- Uncertainty quantification is important
- Real-time inference is needed
- State space is bounded or constrained

## Limitations and Considerations

### Assumptions

1. **Gaussian Noise**: Process and observation models assume Gaussian noise
   - May not capture all uncertainty types
   - Works well for bounded [0,1] state space

2. **Random Walk Process**: Assumes state evolves as random walk
   - May not capture structured state dynamics
   - Appropriate for slowly changing evaluations

3. **LLM Measurement Extraction**: Measurement quality depends on LLM
   - LLM may misinterpret audience responses
   - Measurement noise may be non-Gaussian

### Computational Considerations

- **O(N) per update**: Linear in number of particles
- **Resampling overhead**: Systematic resampling is efficient
- **Memory**: Store N particles and weights per agent
- **Scalability**: Works well for 1D state, may need more particles for higher dimensions

### Alternatives

For comparison, other approaches:
- **Kalman Filter**: Assumes linear dynamics and Gaussian noise (not suitable for bounded [0,1])
- **Extended Kalman Filter**: Handles non-linear dynamics but still Gaussian assumptions
- **Unscented Kalman Filter**: Better for non-linear but more complex
- **Grid-based Methods**: Discretize state space (loses continuous representation)

## Example Usage

```python
# Initialize filter
pf = ParticleFilter(num_particles=200, process_sigma=0.03, obs_sigma=0.08)

# Initialize particles
particles, weights = pf.initialize()

# Each turn:
# 1. Predict
particles_pred = pf.predict(particles)

# 2. Get measurement (from LLM)
measurement = 0.75  # Extracted from audience response

# 3. Update
particles_upd, weights_upd, ess, resampled = pf.update(
    particles_pred, measurement
)

# 4. Compute belief
I_hat = sum(p * w for p, w in zip(particles_upd, weights_upd))

print(f"Belief: {I_hat:.2f}, ESS: {ess:.1f}, Resampled: {resampled}")
```

## References and Further Reading

### Particle Filter Theory

- **Sequential Monte Carlo Methods**: General framework for particle filters
- **Bootstrap Particle Filter**: Original algorithm by Gordon et al. (1993)
- **Systematic Resampling**: Low-variance resampling strategy

### Applications

- **Robotics**: Localization and SLAM
- **Tracking**: Object tracking in computer vision
- **Finance**: State estimation in financial models
- **Neuroscience**: Neural state estimation

### Related Concepts

- **Bayesian Filtering**: General framework (Kalman filter, particle filter)
- **State Estimation**: Inferring hidden states from observations
- **Probabilistic Programming**: Implementing filters in probabilistic frameworks

## Summary

The particle filter in the impression management study provides a principled method for the actor to:
1. **Track beliefs** about the audience's hidden evaluation state
2. **Update probabilistically** based on noisy observations
3. **Quantify uncertainty** through ESS and particle diversity
4. **Adapt behavior** based on belief estimates (I_hat) and prediction errors

The implementation uses a simple 1D particle filter with Gaussian process and observation models, systematic resampling, and ESS-based resampling triggers. This provides a robust foundation for belief tracking in the conversational system.
