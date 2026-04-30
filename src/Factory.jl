# -- PUBLIC METHODS BELOW HERE ---------------------------------------------------------------------------------------- #
"""
    build(modeltype::Type{MyModernHopfieldNetworkModel}, data::NamedTuple) -> MyModernHopfieldNetworkModel

Factory for a modern Hopfield network model. The named tuple should contain:

- `memories::Matrix{Float64}`: memory matrix with one pattern per column.
- `β::Float64`: inverse temperature.
"""
function build(modeltype::Type{MyModernHopfieldNetworkModel},
    data::NamedTuple)::MyModernHopfieldNetworkModel

    model = modeltype()
    model.X = data.memories
    model.β = data.β
    return model
end

"""
    build(modeltype::Type{MyStochasticAttentionModel}, data::NamedTuple) -> MyStochasticAttentionModel

Factory for a Stochastic Attention sampler (Varner, 2026). The named tuple
should contain:

- `memories::Matrix{Float64}`: memory matrix with one pattern per column.
- `labels::Vector{Int}`: class label for each column of `memories`.
- `β::Float64`: inverse temperature.
- `step_size::Float64`: Langevin step size `η` (typical: `1e-2`).
- `noise_scale::Float64`: per-step Gaussian noise scale (typical: `1e-2`;
  set to `0.0` for deterministic gradient descent on the energy).
"""
function build(modeltype::Type{MyStochasticAttentionModel},
    data::NamedTuple)::MyStochasticAttentionModel

    @assert size(data.memories, 2) == length(data.labels) "labels must align with memory columns"

    model = modeltype()
    model.X           = data.memories
    model.y           = data.labels
    model.β           = data.β
    model.step_size   = data.step_size
    model.noise_scale = data.noise_scale
    return model
end
# --- PUBLIC METHODS ABOVE HERE --------------------------------------------------------------------------------------- #
