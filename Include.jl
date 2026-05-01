# setup paths -
const _ROOT = @__DIR__
const _PATH_TO_DATA = joinpath(_ROOT, "data");
const _PATH_TO_SRC = joinpath(_ROOT, "src");
const _PATH_TO_FIGS = joinpath(_ROOT, "figs");
const _PATH_TO_MNIST = joinpath(_PATH_TO_DATA, "mnist");
const _PATH_TO_FACES = joinpath(_PATH_TO_DATA, "olivetti");

!isdir(_PATH_TO_DATA) && mkpath(_PATH_TO_DATA);
!isdir(_PATH_TO_FIGS) && mkpath(_PATH_TO_FIGS);

# flag to check if the include file was called -
const _DID_INCLUDE_FILE_GET_CALLED = true;

using Pkg;
Pkg.activate(_ROOT);
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false)
    Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# load external packages -
using Statistics
using StatsBase
using LinearAlgebra
using Random
using CSV
using DataFrames
using PrettyTables
using Plots
using Colors
using FileIO
using ImageIO
using Images
using NNlib
using Test

# set the random seed for reproducibility -
Random.seed!(42);

# load local source files -
include(joinpath(_PATH_TO_SRC, "Types.jl"));
include(joinpath(_PATH_TO_SRC, "Factory.jl"));
include(joinpath(_PATH_TO_SRC, "Compute.jl"));
include(joinpath(_PATH_TO_SRC, "Files.jl"));
