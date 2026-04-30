"""
    list_mnist_files(class::Int) -> Vector{String}

Return the absolute paths of every `.jpg` MNIST image under
`data/mnist/<class>/`, sorted by filename. The on-disk subset is balanced
across the ten classes; see the project README for provenance.
"""
function list_mnist_files(class::Int)::Vector{String}
    @assert 0 <= class <= 9 "class must lie in 0:9"
    dir = joinpath(_PATH_TO_MNIST, string(class))
    @assert isdir(dir) "directory $(dir) does not exist; did you run Include.jl from the practicum root?"
    files = filter(f -> endswith(f, ".jpg"), readdir(dir))
    return sort([joinpath(dir, f) for f in files])
end

"""
    load_mnist_image(class::Int, idx::Int = 1) -> Matrix{Float64}

Load the `idx`-th MNIST image of the given class as a `28 × 28` matrix of
grayscale values in `[0, 1]`.
"""
function load_mnist_image(class::Int, idx::Int = 1)::Matrix{Float64}
    files = list_mnist_files(class)
    @assert 1 <= idx <= length(files) "idx out of range; class $(class) has $(length(files)) images"
    img = Float64.(Gray.(load(files[idx])))
    return img
end

"""
    load_mnist_subset(images_per_class::Int = 30) -> (X, y)

Load up to `images_per_class` images from each of the 10 MNIST classes,
flatten each `28 × 28` image into a length-784 vector, and stack them as
columns of `X`. The companion vector `y` records the class label of each
column.

### Returns
- `X::Matrix{Float64}` of shape `(784, 10 * images_per_class)`: stored
  memory matrix in the same layout the modern-Hopfield / SA factories expect.
- `y::Vector{Int}` of length `10 * images_per_class`: class label of each column.
"""
function load_mnist_subset(images_per_class::Int = 30)::Tuple{Matrix{Float64}, Vector{Int}}
    @assert images_per_class >= 1
    columns = Vector{Vector{Float64}}()
    labels  = Int[]
    for c in 0:9
        files = list_mnist_files(c)
        n = min(images_per_class, length(files))
        for k in 1:n
            img = Float64.(Gray.(load(files[k])))
            push!(columns, vec(img))
            push!(labels, c)
        end
    end
    X = reduce(hcat, columns)
    return X, labels
end

"""
    image_grid(samples::AbstractMatrix{<:Real}, ncols::Int = 10;
        rows::Int = 28, cols::Int = 28, title::String = "") -> Plots.Plot

Lay out the columns of `samples` (each a flattened image of length
`rows * cols`) into a single grid figure with `ncols` columns. Used for
all sample-grid visualizations in the notebook.
"""
function image_grid(samples::AbstractMatrix{<:Real}, ncols::Int = 10;
    rows::Int = 28, cols::Int = 28, title::String = "")
    n = size(samples, 2)
    nrows = cld(n, ncols)
    plots = Vector{Any}(undef, nrows * ncols)
    for k in 1:(nrows * ncols)
        if k <= n
            img = clamp.(reshape(samples[:, k], rows, cols), 0.0, 1.0)
            plots[k] = heatmap(img;
                color = :grays, axis = false, ticks = false, colorbar = false,
                yflip = true, aspect_ratio = :equal)
        else
            plots[k] = plot(framestyle = :none)
        end
    end
    return plot(plots...; layout = (nrows, ncols), size = (60 * ncols, 60 * nrows),
        plot_title = title)
end
