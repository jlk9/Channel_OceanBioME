# plot_common.jl
#
# Generic Makie plotting machinery for the reentrant-channel runs. Deliberately
# knows NOTHING about this channel — no grid, no wall, no field names, no units.
# It takes 2-D arrays plus coordinate vectors and lays them out on a grid of
# heat maps. The channel-specific part (reading a run directory, building land
# masks, deriving vorticity/buoyancy/κ-ratio, choosing which figures to make)
# lives in plot_channel.jl.
#
# This replaces the three near-identical hardcoded layout functions in the old
# makie_abernathey.jl (`plot_variables_four_panels_2x2`,
# `plot_variables_two_panels_1x2`, `plot_variables_two_panels_2x1`), each of
# which carried its own copy of the limit logic with per-panel positional
# arguments named p1_…, p2_…, p3_…, p4_…. There is now ONE function,
# `plot_variables_rxc`, that takes an arbitrary number of panels.
#
#####
##### The panel contract
#####
#
# A panel is a NamedTuple. Only `data`, `x` and `y` are required; everything
# else has a default:
#
#   data       2-D array, size (length(x), length(y))
#   x, y       coordinate vectors, in whatever units you want on the axes
#              (plot_channel.jl converts horizontal distances to km first)
#   title      axis title                                        (default "")
#   cb         colorbar label, normally the units                (default "")
#   cmap       colormap                                     (default :balance)
#   symmetric  true  → limits centered on zero (diverging fields: velocities,
#                      sensitivities, anomalies)
#              false → limits span the data (sequential fields: T, S, κ)
#                                                              (default true)
#   robust     false → plain extrema; a number in (0,1) → that central quantile
#              range; nothing → ROBUST_QUANTILE[] if symmetric, extrema if not
#                                                           (default nothing)
#   clims      explicit (lo, hi), overriding symmetric/robust  (default nothing)
#   mask       Bool array, same size as data; true cells are drawn in
#              NAN_COLOR[] and excluded from the limit computation
#                                                              (default nothing)
#   contour    true → overlay black contour lines on the heat map
#                                                                (default false)
#   mark       :heatmap or :contourf, overriding PLOT_MARK[]   (default nothing)
#   xlabel,
#   ylabel     axis labels                                       (default "")
#   aspect     axis width/height ratio, overriding the figure-level `aspect`
#
# Layout follows the container: a Matrix of panels plots at exactly its own
# shape, so `[pT pS; pU pV]` is a 2×2 figure and the source reads like the
# picture. A Vector is filled row-major into `layout = (nrows, ncols)`, or into
# an automatic near-square grid when `layout` is omitted. Empty cells are left
# blank, so a 5-panel figure in a 2×3 grid is fine.
#
#####
##### Switchable defaults
#####
#
# The `Ref` constants below are knobs meant to be flipped from the driver or the
# REPL without editing this file. Two exist because of known Makie friction:
#
#   PLOT_MARK[]        :heatmap draws one true cell per grid cell — no
#                      interpolation, no invented structure. If detail renders
#                      poorly, `PLOT_MARK[] = :contourf` restores the old
#                      script's 60-level filled-contour look for every figure.
#
#   LIMIT_PADDING[]    the old script inflated every limit (`2 * min(0, …)`,
#                      then `.* 0.8`) because Makie could render the extreme
#                      colors badly. That fudge is off by default (1.0). If
#                      minima/maxima come out wrong, set this to e.g. 1.25 to
#                      pad the color range back out without touching the data.

using Statistics
using Printf
using GLMakie

# Render offscreen: `save` works without a window appearing. Comment out to get
# interactive windows back when running from the REPL.
GLMakie.activate!(; visible = false)

const PLOT_MARK       = Ref{Symbol}(:heatmap)   # :heatmap | :contourf
const LIMIT_PADDING   = Ref(1.0)                # >1 pads the color range outward
const ROBUST_QUANTILE = Ref(0.99)               # default central quantile for diverging fields
const NAN_COLOR       = Ref{Any}(:gray50)       # masked / non-finite cells
const NLEVELS         = Ref(60)                 # filled-contour levels when mark = :contourf
const PANEL_SIZE      = Ref((640, 540))         # px per panel when the aspect is not numeric
const ANNOTATE_CONST  = Ref(true)               # append "≡ value" to a constant panel's title
const CONSTANT_RTOL   = Ref(1e-9)               # relative spread below which a field counts as constant
const TITLESIZE       = Ref(22)
const LABELSIZE       = Ref(18)
const TICKLABELSIZE   = Ref(15)

# Panel geometry used when the figure declares a numeric `aspect` (axis
# width/height). The old script fixed the Figure size and let each
# aspect-constrained axis float inside an oversized cell, which is where its
# figures got their broad white margins and their colorbars that ran taller than
# the plot beside them. Here the axis width is the primitive: the row height
# follows from it and the aspect, and `rowsize!(…, Aspect(…))` pins the cell to
# the axis box so the colorbar ends up exactly as tall as the heat map.
const AXIS_WIDTH      = Ref(430)                # px, the axis box width
const CB_ALLOWANCE    = Ref(150)                # px reserved for a colorbar column
const AXIS_CHROME     = Ref(110)                # px for title + tick labels + axis label

#####
##### Color limits
#####

# Last-resort guard against lo == hi, which Makie rejects. `constancy` below
# catches the ordinary constant-field case first and centers a range on the
# value; this covers whatever slips past it.
function _widen(lo, hi)
    lo < hi && return (Float64(lo), Float64(hi))
    pad = max(abs(Float64(lo)), 1.0) * 1e-6
    return (Float64(lo) - pad, Float64(hi) + pad)
end

_pad_limits(lo, hi) = begin
    f = LIMIT_PADDING[]
    f == 1 && return (lo, hi)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    (mid - f * half, mid + f * half)
end

# Unmasked, finite values of a panel — the sample every limit decision is made
# from. Masked (dry) cells and NaNs never participate.
function panel_values(data, mask)
    raw = mask === nothing ? vec(data) : vec(data[.!mask])
    return Float64[v for v in raw if isfinite(v)]
end

# Is this field constant to within float noise? Returns (isconstant, value, spread).
#
# An EXACT test is not enough here: S is 35 g/kg by construction but comes back
# spread over ~1e-11 by round-off, which is enough to survive `lo == hi`, give
# the panel a color range 12 orders of magnitude narrower than its own values,
# and render pure numerical noise as though it were structure. The test is
# therefore relative to the field's own magnitude, which leaves genuinely small
# fields (sensitivities of order 1e-12) alone because their spread is comparable
# to their scale.
function constancy(vals)
    isempty(vals) && return (true, NaN, 0.0)
    lo, hi = extrema(vals)
    scale  = max(abs(lo), abs(hi))
    tol    = CONSTANT_RTOL[] * (scale > 0 ? scale : 1.0)
    return (hi - lo <= tol, (lo + hi) / 2, hi - lo)
end

"""
    color_limits(data; mask, symmetric, robust, clims)

Color range for one panel, computed over unmasked, finite cells only.

Sensitivity fields in this project are strongly heavy-tailed — a handful of
cells near the wall or the outcrop carry values orders of magnitude above the
rest — so a plain symmetric ±max leaves every panel looking empty. The default
for a diverging field is therefore the `ROBUST_QUANTILE[]` quantile of |value|,
which keeps the bulk of the field visible and saturates the outliers. Sequential
fields (T, S, κ) default to plain extrema, where the tails are the signal.
"""
function color_limits(data; mask = nothing, symmetric::Bool = true,
                      robust = nothing, clims = nothing)

    clims === nothing || return _pad_limits(Float64(clims[1]), Float64(clims[2]))

    vals = panel_values(data, mask)
    isempty(vals) && return (-1.0, 1.0)

    # A constant field gets a range centered on its value, so it renders as one
    # flat color instead of a magnified picture of its round-off.
    isconst, v, _ = constancy(vals)
    if isconst
        pad = max(abs(v), 1.0) * 1e-3
        return (v - pad, v + pad)
    end

    q = robust === nothing ? (symmetric ? ROBUST_QUANTILE[] : 1.0) :
        robust === false   ? 1.0 :
        robust === true    ? ROBUST_QUANTILE[] : Float64(robust)

    if symmetric
        a = abs.(vals)
        m = q >= 1 ? maximum(a) : quantile(a, q)
        m > 0 || (m = maximum(a))
        return _pad_limits(_widen(-m, m)...)
    else
        if q >= 1
            lo, hi = extrema(vals)
        else
            t = (1 - q) / 2
            lo, hi = quantile(vals, t), quantile(vals, 1 - t)
        end
        return _pad_limits(_widen(lo, hi)...)
    end
end

"""
    mask_nan(data, mask)

Copy of `data` as Float64 with masked cells set to NaN, so Makie paints them in
`NAN_COLOR[]`. Returns `data` untouched when there is no mask.
"""
function mask_nan(data, mask)
    mask === nothing && return data
    size(mask) == size(data) ||
        error("mask size $(size(mask)) does not match data size $(size(data))")
    z = Array{Float64}(data)
    z[mask] .= NaN
    return z
end

"""
    constant_note(data, mask) -> String

`"  ≡ 35"` when the field is constant, `"  ≈ 35 (spread 8.2e-12)"` when it is
constant only to within round-off, and `""` otherwise.

Several fields in this setup are genuinely uniform — S is constant by
construction, e is identically 0, and the GM/Redi κ is a constant. Without the note, such a panel is a flat wash under
a colorbar with no tick labels, which is easy to misread as a rendering failure
rather than as a statement about the field.
"""
function constant_note(data, mask)
    vals = panel_values(data, mask)
    isempty(vals) && return "  (no finite cells)"
    isconst, v, spread = constancy(vals)
    isconst || return ""
    return spread == 0 ? (@sprintf "  ≡ %.4g" v) :
                         (@sprintf "  ≈ %.4g (spread %.2g)" v spread)
end

#####
##### Layout
#####

# Normalize whatever the caller passed into a (nrows, ncols) Matrix whose empty
# cells hold `nothing`.
function _panel_grid(panels, layout)
    if panels isa AbstractMatrix
        nrows, ncols = size(panels)
        grid = Matrix{Any}(nothing, nrows, ncols)
        for r in 1:nrows, c in 1:ncols
            grid[r, c] = panels[r, c]
        end
        return grid
    end

    v = collect(panels)
    n = length(v)
    n > 0 || error("plot_variables_rxc: no panels given")

    if layout === nothing
        ncols = ceil(Int, sqrt(n))
        nrows = ceil(Int, n / ncols)
    else
        nrows, ncols = layout
        nrows * ncols >= n ||
            error("layout $((nrows, ncols)) has room for $(nrows * ncols) panels but $n were given")
    end

    grid = Matrix{Any}(nothing, nrows, ncols)
    for (i, p) in enumerate(v)          # row-major fill
        grid[fld(i - 1, ncols) + 1, mod(i - 1, ncols) + 1] = p
    end
    return grid
end

#####
##### The figure
#####

"""
    plot_variables_rxc(panels; filename, layout = nothing, aspect = nothing,
                       supertitle = nothing, mark = PLOT_MARK[],
                       panel_size = PANEL_SIZE[])

Draw `panels` (see the panel contract at the top of this file) as a grid of heat
maps with one colorbar each, save to `filename`, and return the Figure.

`panels` may be a Matrix — whose shape is taken as the figure layout — or a
Vector, filled row-major into `layout` or into an automatic near-square grid.
Intermediate directories of `filename` are created.
"""
function plot_variables_rxc(panels;
                            filename,
                            layout     = nothing,
                            aspect     = nothing,
                            supertitle = nothing,
                            mark       = PLOT_MARK[],
                            panel_size = nothing)

    grid = _panel_grid(panels, layout)
    nrows, ncols = size(grid)

    # A numeric figure-level aspect lets the panel size be derived rather than
    # guessed, which is what keeps the margins tight at any aspect ratio.
    if panel_size === nothing
        panel_size = aspect isa Real ?
            (AXIS_WIDTH[] + CB_ALLOWANCE[],
             round(Int, AXIS_WIDTH[] / aspect) + AXIS_CHROME[]) :
            PANEL_SIZE[]
    end

    fig = Figure(size = (ncols * panel_size[1], nrows * panel_size[2]))

    for r in 1:nrows, c in 1:ncols
        p = grid[r, c]
        p === nothing && continue

        symmetric = get(p, :symmetric, true)
        msk       = get(p, :mask, nothing)
        lims      = color_limits(p.data;
                                 mask      = msk,
                                 symmetric = symmetric,
                                 robust    = get(p, :robust, nothing),
                                 clims     = get(p, :clims, nothing))
        z    = mask_nan(p.data, msk)
        cmap = get(p, :cmap, :balance)

        ttl = get(p, :title, "")
        ANNOTATE_CONST[] && (ttl *= constant_note(p.data, msk))

        # Axis in an odd column, its colorbar in the even column beside it.
        axkw = (; title          = ttl,
                  xlabel         = get(p, :xlabel, ""),
                  ylabel         = get(p, :ylabel, ""),
                  titlesize      = TITLESIZE[],
                  xlabelsize     = LABELSIZE[],
                  ylabelsize     = LABELSIZE[],
                  xticklabelsize = TICKLABELSIZE[],
                  yticklabelsize = TICKLABELSIZE[])

        pa = get(p, :aspect, aspect)
        ax = pa === nothing ? Axis(fig[r, 2c-1]; axkw...) :
                              Axis(fig[r, 2c-1]; aspect = pa, axkw...)

        pmark = something(get(p, :mark, nothing), mark)
        hm = if pmark === :heatmap
            heatmap!(ax, p.x, p.y, z;
                     colormap = cmap, colorrange = lims, nan_color = NAN_COLOR[])
        elseif pmark === :contourf
            contourf!(ax, p.x, p.y, z;
                      colormap   = cmap,
                      levels     = collect(range(lims[1], lims[2], length = NLEVELS[])),
                      nan_color  = NAN_COLOR[],
                      extendlow  = :auto,
                      extendhigh = :auto)
        else
            error("unknown mark $pmark; expected :heatmap or :contourf")
        end

        if get(p, :contour, false)
            contour!(ax, p.x, p.y, z;
                     levels = collect(range(lims[1], lims[2], length = 12)),
                     color = (:black, 0.45), linewidth = 0.8)
        end

        Colorbar(fig[r, 2c], hm;
                 label         = get(p, :cb, ""),
                 labelsize     = LABELSIZE[],
                 ticklabelsize = TICKLABELSIZE[])
    end

    # Tie each row's height to the axis column's width through the shared aspect.
    # Axis decorations are protrusions rather than cell content, so the cell then
    # IS the heat map, and the colorbar beside it inherits exactly that height.
    if aspect isa Real && all(p -> p === nothing || get(p, :aspect, nothing) === nothing, grid)
        for r in 1:nrows
            rowsize!(fig.layout, r, Aspect(1, 1 / aspect))
        end
    end

    supertitle === nothing ||
        Label(fig[0, :], supertitle; fontsize = TITLESIZE[] + 4, font = :bold)

    mkpath(dirname(filename))
    save(filename, fig)
    return fig
end
