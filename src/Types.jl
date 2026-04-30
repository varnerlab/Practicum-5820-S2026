abstract type AbstractHopfieldNetworkModel end
abstract type AbstractStochasticAttentionModel end

"""
    MyModernHopfieldNetworkModel <: AbstractHopfieldNetworkModel

A modern Hopfield network in the Ramsauer et al. (2020) sense: stored memories
sit on the columns of a single matrix `X`, and recall is one closed-form
softmax-attention update parameterized by inverse temperature `β`.

This is the same associative-memory mechanism as the spiking H-Mem from L15a/b
viewed in the rate / high-`β` limit — the columns of `X` play the role of
stored value patterns, and the softmax replaces the Hebbian `W_assoc * k`
read step.

### Fields
- `X::Matrix{Float64}`: stored memory matrix; column `j` is memory pattern `j`.
- `β::Float64`: inverse temperature for the softmax-attention update.
"""
mutable struct MyModernHopfieldNetworkModel <: AbstractHopfieldNetworkModel
    X::Matrix{Float64}
    β::Float64

    MyModernHopfieldNetworkModel() = new()
end

"""
    MyStochasticAttentionModel <: AbstractStochasticAttentionModel

A Stochastic Attention (SA) sampler from Varner (2026, arXiv:2603.14717): the
modern Hopfield energy is reinterpreted as a Boltzmann distribution and we
draw novel samples by Langevin dynamics whose score function is the closed-form
modern Hopfield update. Training-free, no GPU; cost is linear in the number
of stored patterns.

The labels vector `y` is kept alongside `X` so that Task 3 can build a
class-conditional mask without re-loading the dataset.

### Fields
- `X::Matrix{Float64}`: stored memory matrix (columns are patterns).
- `y::Vector{Int}`: class label of each stored pattern, `length(y) == size(X, 2)`.
- `β::Float64`: inverse temperature for the SA energy / softmax.
- `step_size::Float64`: Langevin step size `η`.
- `noise_scale::Float64`: standard deviation of the per-step Gaussian noise
  injected by Langevin dynamics; `0.0` reduces to gradient descent on the energy.
"""
mutable struct MyStochasticAttentionModel <: AbstractStochasticAttentionModel
    X::Matrix{Float64}
    y::Vector{Int}
    β::Float64
    step_size::Float64
    noise_scale::Float64

    MyStochasticAttentionModel() = new()
end
