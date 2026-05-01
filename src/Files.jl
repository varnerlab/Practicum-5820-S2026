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
    load_olivetti_subset(images_per_subject::Int = 10;
        n_subjects::Int = 40) -> (X, y)

Load up to `images_per_subject` face images per subject from the Olivetti
faces dataset (40 subjects × 10 images, each 64 × 64 pixels in `[0, 1]`).
Each image is flattened to a length-4096 column of `X`. The companion
vector `y` is the integer subject id of each column.

The on-disk layout is `data/olivetti/faces.csv` (400 × 4096 row-per-image)
and `data/olivetti/targets.csv` (400-row subject ids), inherited from
the L10d lab.

### Returns
- `X::Matrix{Float64}` of shape `(4096, n_subjects * images_per_subject)`:
  one face per column, in the same layout the modern-Hopfield / SA factories expect.
- `y::Vector{Int}` of length `n_subjects * images_per_subject`: subject id of each column.
"""
function load_olivetti_subset(images_per_subject::Int = 10;
    n_subjects::Int = 40)::Tuple{Matrix{Float64}, Vector{Int}}

    @assert 1 <= images_per_subject <= 10
    @assert 1 <= n_subjects <= 40

    faces_path   = joinpath(_PATH_TO_FACES, "faces.csv")
    targets_path = joinpath(_PATH_TO_FACES, "targets.csv")
    @assert isfile(faces_path)   "missing $(faces_path); copy from L10d data dir"
    @assert isfile(targets_path) "missing $(targets_path); copy from L10d data dir"

    faces_df   = CSV.read(faces_path,   DataFrame; header = false)
    targets_df = CSV.read(targets_path, DataFrame; header = false)
    X_full = Float64.(Matrix(faces_df))      # 400 × 4096
    y_full = Int.(targets_df[!, 1])          # 400-vector of subject ids in 0:39

    # Pick the first `images_per_subject` shots of each of the first `n_subjects` people.
    cols = Vector{Vector{Float64}}()
    labels = Int[]
    for s in 0:(n_subjects - 1)
        rows = findall(==(s), y_full)
        n = min(images_per_subject, length(rows))
        for r in rows[1:n]
            push!(cols, vec(X_full[r, :]))
            push!(labels, s)
        end
    end
    X = reduce(hcat, cols)
    return X, labels
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
        rows::Int = 28, cols::Int = 28, title::String = "",
        tile_titles::Union{Nothing, AbstractVector{<:AbstractString}} = nothing,
        tile_px::Int = 110) -> Plots.Plot

Lay out the columns of `samples` (each a flattened image of length
`rows * cols`) into a single grid figure with `ncols` columns. Matches the
L10d-lab face-grid aesthetic: each tile is a `Gray.(...)` heatmap with
`aspect_ratio = :equal`, axes / ticks / colorbar off, and a small per-tile
title if `tile_titles` is supplied. An optional outer `title` runs across
the top of the whole grid.
"""
function image_grid(samples::AbstractMatrix{<:Real}, ncols::Int = 10;
    rows::Int = 64, cols::Int = 64, title::String = "",
    tile_titles::Union{Nothing, AbstractVector{<:AbstractString}} = nothing,
    tile_px::Int = 110,
    outer_title_fontsize::Int = 10,
    tile_title_fontsize::Int = 8)

    n = size(samples, 2)
    nrows = cld(n, ncols)
    plots = Vector{Any}(undef, nrows * ncols)
    for k in 1:(nrows * ncols)
        if k <= n
            mat = clamp.(reshape(samples[:, k], rows, cols), 0.0, 1.0)
            img = Gray.(mat)
            t = (tile_titles === nothing || k > length(tile_titles)) ? "" : tile_titles[k]
            plots[k] = heatmap(img;
                color = :grays, axis = false, ticks = false, colorbar = false,
                aspect_ratio = :equal,
                title = t, titlefontsize = tile_title_fontsize)
        else
            plots[k] = plot(framestyle = :none)
        end
    end

    has_title = !isempty(title)
    has_tile_titles = tile_titles !== nothing
    title_px = has_title ? (has_tile_titles ? 80 : 50) : 0
    top_mm  = has_title ? (has_tile_titles ? 10 : 6) : 0
    return plot(plots...;
        layout = (nrows, ncols),
        size = (tile_px * ncols, tile_px * nrows + title_px),
        top_margin    = top_mm * Plots.PlotMeasures.mm,
        bottom_margin = 2Plots.PlotMeasures.mm,
        left_margin   = 2Plots.PlotMeasures.mm,
        right_margin  = 2Plots.PlotMeasures.mm,
        plot_title = title,
        plot_titlefontsize = outer_title_fontsize)
end
