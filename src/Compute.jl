# --- PRIVATE METHODS BELOW HERE -------------------------------------------------------------------------------------- #
"""
    _attention_weights(X, s, β; logit_bias = nothing) -> Vector{Float64}

Compute the modern-Hopfield / Stochastic-Attention probability vector
`p = softmax(β · X' · s + logit_bias)`. Centralizing the softmax here lets
every recall, sampling, and masked-sampling routine share exactly the same
logit construction.

The single `logit_bias` knob covers the two conditioning regimes used in
this practicum:
- **Hard mask** (Task 3a): `logit_bias[j] = 0.0` for in-class memories,
  `-Inf` for out-of-class memories. The exponential then exactly zeros the
  out-of-class weights — same trick GPT uses for causal masking.
- **Soft bias** (Task 3b, Varner 2026, arXiv:2603.20115): `logit_bias[j] = b`
  for in-class memories with `b` finite, `0.0` for out-of-class memories.
  The relative attention weight on the in-class subset is steered by `b`
  but no pattern is hard-excluded.

`logit_bias = nothing` recovers the unconditional sampler.
"""
function _attention_weights(X::AbstractMatrix{<:Real}, s::AbstractVector{<:Real},
    β::Real;
    logit_bias::Union{Nothing, AbstractVector{<:Real}} = nothing)::Vector{Float64}

    logits = β .* (transpose(X) * s)
    if logit_bias !== nothing
        @assert length(logit_bias) == length(logits) "logit_bias length must match number of stored patterns"
        logits = logits .+ logit_bias
    end
    return NNlib.softmax(logits)
end

"""
    _hard_mask_bias(mask::AbstractVector{Bool}) -> Vector{Float64}

Convert a boolean keep-mask into a `logit_bias` vector that hard-excludes
every pattern with `mask[j] == false` (assigns `-Inf` to its logit).
"""
function _hard_mask_bias(mask::AbstractVector{Bool})::Vector{Float64}
    return [m ? 0.0 : -Inf for m in mask]
end

"""
    _soft_bias_vector(in_subset::AbstractVector{Bool}, bias::Real) -> Vector{Float64}

Convert an in-subset indicator into a `logit_bias` vector that adds `bias`
to in-subset logits and `0.0` to out-of-subset logits. Combined with a
non-`-Inf` softmax, this is the multiplicity-ratio conditioning of
Varner (2026, arXiv:2603.20115).
"""
function _soft_bias_vector(in_subset::AbstractVector{Bool}, bias::Real)::Vector{Float64}
    return [m ? Float64(bias) : 0.0 for m in in_subset]
end
# --- PRIVATE METHODS ABOVE HERE -------------------------------------------------------------------------------------- #


# --- PUBLIC METHODS BELOW HERE --------------------------------------------------------------------------------------- #
"""
    modern_hopfield_recall(model::MyModernHopfieldNetworkModel, sₒ::Vector{Float64};
        maxiterations::Int = 1000, ϵ::Float64 = 1e-10) -> (s, frames, probabilities)

One-shot retrieval from a modern Hopfield memory by iterating the closed-form
softmax-attention update

    p = softmax(β · X' · s);    s ← X · p

until either `‖p_t − p_{t-1}‖² ≤ ϵ` or we hit `maxiterations`.

This is the exact retrieval rule of Ramsauer et al. (2020), and — in the
β → ∞ limit — recovers the H-Mem `W_assoc · k` read step that students saw
in L15a. Sweeping β at fixed `X, sₒ` is therefore the bridge between the
Hebbian and modern-Hopfield views of the same memory mechanism.

### Returns
- `s::Vector{Float64}`: converged state.
- `frames::Dict{Int, Vector{Float64}}`: state at each iteration, keyed from 0.
- `probabilities::Dict{Int, Vector{Float64}}`: attention weights at each
  iteration, keyed from 0.
"""
function modern_hopfield_recall(model::MyModernHopfieldNetworkModel,
    sₒ::Vector{Float64};
    maxiterations::Int = 1000, ϵ::Float64 = 1e-10)

    X = model.X
    β = model.β

    frames        = Dict{Int, Vector{Float64}}()
    probabilities = Dict{Int, Vector{Float64}}()

    s = copy(sₒ)
    frames[0]        = copy(s)
    probabilities[0] = _attention_weights(X, s, β; logit_bias = nothing)

    iter = 1
    Δ = Inf
    while true
        p = _attention_weights(X, s, β; logit_bias = nothing)
        s = X * p
        frames[iter]        = copy(s)
        probabilities[iter] = p

        if iter > 1
            Δ = norm(probabilities[iter] - probabilities[iter - 1])^2
        end
        if iter >= maxiterations || Δ ≤ ϵ
            break
        end
        iter += 1
    end
    return s, frames, probabilities
end

"""
    sa_initial_states(model::MyStochasticAttentionModel, n_samples::Int;
        rng::AbstractRNG = Random.GLOBAL_RNG,
        hard_mask::Union{Nothing, AbstractVector{Bool}} = nothing) -> Matrix{Float64}

Build the warm-start matrix `S₀` that `stochastic_attention_sample` would use
internally if no `sₒ` is supplied: each column is a randomly chosen *visible*
stored pattern plus a small Gaussian kick. Exposing this so the notebook can
display *both* the starting image of a chain and the final sample side by
side, rather than just the final.

If `hard_mask` is supplied, the warm starts are drawn only from the columns
where `hard_mask[j] == true`.
"""
function sa_initial_states(model::MyStochasticAttentionModel, n_samples::Int;
    rng::AbstractRNG = Random.GLOBAL_RNG,
    hard_mask::Union{Nothing, AbstractVector{Bool}} = nothing)::Matrix{Float64}

    X = model.X
    d, N = size(X)
    allowed = hard_mask === nothing ? collect(1:N) : findall(hard_mask)
    @assert !isempty(allowed) "hard_mask excludes every stored pattern; nothing to sample from"
    S0 = zeros(Float64, d, n_samples)
    for k in 1:n_samples
        j = rand(rng, allowed)
        S0[:, k] = X[:, j] .+ 0.05 .* randn(rng, d)
    end
    return S0
end

"""
    stochastic_attention_sample(model::MyStochasticAttentionModel, n_samples::Int;
        n_steps::Int = 200,
        sₒ::Union{Nothing, Matrix{Float64}} = nothing,
        rng::AbstractRNG = Random.GLOBAL_RNG,
        hard_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
        soft_bias::Union{Nothing, AbstractVector{<:Real}} = nothing) -> Matrix{Float64}

Draw `n_samples` novel patterns from the modern-Hopfield Boltzmann distribution
defined by the model's stored memories `X` and inverse temperature `β`, by
running `n_steps` of Langevin dynamics on the SA energy:

    s_{t+1} = (1 − η) · s_t + η · X · softmax(β · X' · s_t + logit_bias)
              + noise_scale · ξ_t,    ξ_t ~ 𝒩(0, I)

Setting `η = step_size = 1.0` and `noise_scale = 0.0` reduces to one
modern-Hopfield update per step; small `η` with non-zero noise is the
training-free generative sampler of Varner (2026, arXiv:2603.14717).

### Conditioning (used by Task 3)
- `hard_mask::Vector{Bool}` of length `size(X, 2)` excludes every memory
  with `hard_mask[j] == false` by setting its softmax logit to `-Inf`.
  This is the GPT-style causal-mask trick applied to a Hopfield memory.
- `soft_bias::Vector{<:Real}` of length `size(X, 2)` adds an entry-wise
  scalar bias to each memory's softmax logit. Combine with a per-class
  indicator (`bias_value` for in-class, `0.0` for out-of-class) to recover
  the multiplicity-ratio conditioning of Varner (2026, arXiv:2603.20115).
- The two forms compose: `hard_mask` and `soft_bias` are summed before the
  softmax. Pass only one or the other in this practicum.

### Initialization
If `sₒ` is supplied (shape `(d, n_samples)`), it is the initial state for
each chain. Otherwise chains start from random combinations of *visible*
stored patterns plus light Gaussian noise — a cheap warm start that lives
in the data manifold from step zero.

### Returns
- `Matrix{Float64}` of shape `(d, n_samples)`: one generated pattern per column.
"""
function stochastic_attention_sample(model::MyStochasticAttentionModel,
    n_samples::Int;
    n_steps::Int = 200,
    sₒ::Union{Nothing, Matrix{Float64}} = nothing,
    rng::AbstractRNG = Random.GLOBAL_RNG,
    hard_mask::Union{Nothing, AbstractVector{Bool}} = nothing,
    soft_bias::Union{Nothing, AbstractVector{<:Real}} = nothing)::Matrix{Float64}

    X = model.X
    d, N = size(X)
    β  = model.β
    η  = model.step_size
    σ  = model.noise_scale

    # Build a single combined logit_bias vector: hard mask sets -Inf where
    # excluded; soft bias adds a finite shift. Summing them lets a student
    # do either one, both, or neither without changing the inner loop.
    logit_bias = nothing
    if hard_mask !== nothing || soft_bias !== nothing
        lb = zeros(Float64, N)
        if hard_mask !== nothing
            @assert length(hard_mask) == N "hard_mask length must match number of stored patterns"
            lb = lb .+ _hard_mask_bias(hard_mask)
        end
        if soft_bias !== nothing
            @assert length(soft_bias) == N "soft_bias length must match number of stored patterns"
            lb = lb .+ Float64.(soft_bias)
        end
        logit_bias = lb
    end

    # Warm start: each chain begins as a random combination of the *visible*
    # patterns (after applying the hard mask, if any) plus a small Gaussian
    # kick. Starting on the data manifold avoids a long burn-in.
    S = if sₒ === nothing
        sa_initial_states(model, n_samples; rng = rng, hard_mask = hard_mask)
    else
        @assert size(sₒ) == (d, n_samples) "sₒ must have shape (d, n_samples)"
        copy(sₒ)
    end

    # One Langevin step per chain per outer iteration.
    for _ in 1:n_steps, k in 1:n_samples
        s = view(S, :, k)
        p = _attention_weights(X, s, β; logit_bias = logit_bias)
        # (1 − η) s + η · X p is the η-blended modern-Hopfield update; the
        # noise term is what turns a deterministic recall into a sampler.
        S[:, k] = (1 - η) .* s .+ η .* (X * p) .+ σ .* randn(rng, d)
    end

    # Final noise-free read-out: one modern-Hopfield update with no noise
    # projects each chain back onto the data manifold (a convex combination
    # of stored patterns) so the sample is visually clean. This is the
    # standard MAP / denoising step used at the end of score-based and
    # Langevin samplers; the noisy chain above is what defines the
    # distribution we're sampling from, this is just how we read it out.
    for k in 1:n_samples
        s = view(S, :, k)
        p = _attention_weights(X, s, β; logit_bias = logit_bias)
        S[:, k] = X * p
    end
    return S
end

"""
    classify_by_nearest_mean(s::AbstractVector{<:Real},
        class_means::Dict{Int, Vector{Float64}}) -> Int

Return the class label whose mean pattern is closest to `s` in Euclidean
distance. Used as a deterministic, training-free reference classifier in
Task 3 to score whether a class-conditional sample landed in the right
class — the alternative would be training an MLP, which is overkill for
the practicum and would add a heavy dependency.
"""
function classify_by_nearest_mean(s::AbstractVector{<:Real},
    class_means::Dict{Int, Vector{Float64}})::Int
    best_class = first(keys(class_means))
    best_dist = Inf
    for (c, μ) in class_means
        d = norm(s .- μ)
        if d < best_dist
            best_dist = d
            best_class = c
        end
    end
    return best_class
end

"""
    build_class_means(X::AbstractMatrix{<:Real}, y::AbstractVector{<:Integer})
        -> Dict{Int, Vector{Float64}}

Build the per-class mean of the stored patterns: one centroid per unique label.
"""
function build_class_means(X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Integer})::Dict{Int, Vector{Float64}}

    means = Dict{Int, Vector{Float64}}()
    for c in unique(y)
        cols = findall(==(c), y)
        means[c] = vec(mean(X[:, cols]; dims = 2))
    end
    return means
end
# --- PUBLIC METHODS ABOVE HERE --------------------------------------------------------------------------------------- #
