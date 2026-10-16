# plot_channel.jl
#
# Diagnostic figures for a reentrant-channel run directory. Point it at a run
# directory (the one abernathey_channel.jl wrote `data_init.jld2` and
# `data_final.jld2` into) and it writes a set of PNGs into a `plots/`
# subdirectory of it.
#
#     julia --project scripts/plot_channel.jl runs/<config name>/
#
# or from the REPL, which is the faster way to iterate on a single figure:
#
#     julia> include("scripts/plot_channel.jl")
#     julia> make_plots("runs/<config name>/")
#
# The generic layout engine lives in plot_common.jl (`plot_variables_rxc`); this
# file holds everything that is specific to THIS model: reading the save files,
# rebuilding coordinates and the land mask, deriving vorticity/buoyancy, and
# choosing the figures.
#
#####
##### Why this needs no Oceananigans
#####
#
# The old script rebuilt a full ImmersedBoundaryGrid just to get node positions
# and the land mask, which meant carrying Oceananigans (and, through the shared
# environment, the whole Reactant/Enzyme stack) for a plotting job. It no longer
# has to: `save_init!` records `z_faces`, `Lx`, `Ly`, `Lz`, `Nx/Ny/Nz` and the
# materialized `bottom_height`, which is everything the coordinates and the mask
# need. So this file depends only on JLD2 + Makie and loads in seconds, the same
# bargain compute_eke.jl makes.
#
# The mask convention is GridFittedBottom's default (CenterImmersedCondition):
# a cell is dry when its CENTER lies at or below the (already snapped) bottom
# height, `zc[k] < bh[i,j]`. Verified against a saved run: this reproduces the
# cells where `T_final` and `u` are identically zero exactly, on both the center
# and the x-face (periodic-wrap OR) locations.
#
# One trap in the save files: `clock_time`, `clock_iteration` and `clock_last_dt`
# are Reactant `ConcretePJRTNumber`s. Reading them outside a Reactant session
# produces `JLD2.ReconstructedMutable` values and a wall of warnings, so this
# file reads a NAMED LIST of keys rather than slurping `keys(f)` the way
# compute_eke.jl does.
#
#####
##### What gets plotted
#####
#
# Horizontal (x–y) maps at a few depths, given in METRES and resolved to the
# nearest cell center, so the same call works at any Nz instead of hardcoding
# k = 31/14/4 the way the old script did; and zonal-mean (y–z) sections, which
# are the natural decomposition for an x-periodic channel and the same reduction
# the stratification term of the objective is built on.
#
# Figures that depend on the GM/Redi closure (κ_skew, κ_symmetric and their
# adjoints) are emitted only when those keys are present, so a run without
# GM/Redi produces the state and sensitivity figures and simply skips the κ
# ones with a note. Likewise the biogeochemical-tracer figures: every tracer in
# the run's `tracer_names` other than T, S and e gets its x–y maps, zonal-mean
# section and sensitivity ∂J/∂(tracer), read from <name>_final / d<name>.
#
# Sensitivity panels are divided by the cell thickness Δz by default
# (`per_thickness = true`). ∂J/∂T is a per-CELL derivative, so on this stretched
# grid the raw field partly just reports that deep cells are 20× thicker than
# surface ones; dividing by Δz makes depths comparable. The title of a
# normalized panel carries a `÷ Δz` so the two can never be confused.
#
# Colorbar labels are written "J per K", "J per (m/s)" and so on: J is the
# composite objective of the run that produced the file (transport +
# stratification + tracer terms), not a transport in Sv, and the labels avoid
# claiming otherwise.

using JLD2
using Printf
using Statistics

@isdefined(plot_variables_rxc) || include(joinpath(@__DIR__, "plot_common.jl"))
# ALPHA_DEFAULT / G_DEFAULT, the model's own constants, as fallbacks for files
# without EOS metadata (pure Julia, include-guarded).
include(joinpath(@__DIR__, "channel_geometry.jl"))

#####
##### Reading
#####

# Read only these. Anything absent is simply missing from the returned Dict, and
# the figure that wanted it is skipped. `clock_*` is deliberately NOT here.
const INIT_KEYS = ["Nx", "Ny", "Nz", "z_faces", "Lx", "Ly", "Lz", "alpha", "g",
                   "bottom_height", "T_init", "S_init", "e_init", "ssh",
                   "u_wind_stress", "v_wind_stress", "T_flux",
                   "kappa_skew", "kappa_symmetric",
                   "tracer_names", "biogeochemistry",
                   "run_type", "Nspinup", "Ntimesteps", "advection", "git_commit"]

const FINAL_KEYS = ["Nx", "Ny", "Nz",
                    "T_final", "S_final", "e_final", "ssh", "u", "v", "w",
                    "dT", "dS", "du", "dv", "dssh",
                    "du_wind_stress", "dv_wind_stress", "dT_flux",
                    "kappa_skew", "kappa_symmetric",
                    "dkappa_skew", "dkappa_symmetric",
                    "zonal_transport_and_buoyancy",
                    "run_type", "Nspinup", "Ntimesteps", "advection"]

# The tracers beyond the physical ones (biogeochemical tracers), from the run's
# `tracer_names` (none for files without it), and the keys they are saved under.
const PHYSICAL_TRACERS = ("T", "S", "e")
other_tracers(init) = [String(n) for n in get(init, "tracer_names", String[]) if !(String(n) in PHYSICAL_TRACERS)]
other_tracer_keys(names; init) = init ? [n * "_init" for n in names] :
                                        vcat([n * "_final" for n in names], ["d" * n for n in names])

function read_keys(path, wanted)
    isfile(path) || error("no such file: $path")
    return jldopen(path, "r") do f
        present = keys(f)
        Dict{String,Any}(k => read(f, k) for k in wanted if k in present)
    end
end

#####
##### Coordinates and land mask
#####

"""
    channel_plot_grid(init) -> NamedTuple

Coordinates and dry-cell masks derived from a run's `data_init.jld2` contents.

Horizontal coordinates are returned in km (`*_km`, what the axes use) since
every horizontal panel plots in km; `z` stays in metres. Masks are `true` on DRY
cells and are provided at each staggered location that gets plotted: centers
(T, S, e, κ, ssh), x-faces (u), y-faces (v) and the x–y-face corner (vorticity).
A face is dry when EITHER cell it separates is dry, which is what makes the
masked region match where Oceananigans leaves the velocity identically zero.
"""
function channel_plot_grid(init)
    Nx, Ny, Nz = Int(init["Nx"]), Int(init["Ny"]), Int(init["Nz"])
    zf = Vector{Float64}(init["z_faces"])
    Lx = Float64(get(init, "Lx", 1000e3))
    Ly = Float64(get(init, "Ly", 2000e3))
    Lz = Float64(get(init, "Lz", -zf[1]))

    zc = 0.5 .* (zf[1:end-1] .+ zf[2:end])
    Δz = diff(zf)
    Δx, Δy = Lx / Nx, Ly / Ny

    xc = [(i - 0.5) * Δx for i in 1:Nx]
    xf = [(i - 1.0) * Δx for i in 1:Nx]        # periodic x: Nx faces, not Nx+1
    yc = [(j - 0.5) * Δy for j in 1:Ny]
    yf = [(j - 1.0) * Δy for j in 1:Ny+1]

    bh = Float64.(init["bottom_height"][:, :, 1])

    dry_c = [zc[k] < bh[i, j] for i in 1:Nx, j in 1:Ny, k in 1:Nz]

    dry_u = similar(dry_c)
    for i in 1:Nx
        im = i == 1 ? Nx : i - 1               # periodic wrap at the seam
        dry_u[i, :, :] .= dry_c[i, :, :] .| dry_c[im, :, :]
    end

    dry_v = falses(Nx, Ny + 1, Nz)
    dry_v[:, 2:Ny, :] .= dry_c[:, 1:Ny-1, :] .| dry_c[:, 2:Ny, :]
    dry_v[:, 1, :]      .= dry_c[:, 1, :]      # bounded y: walls at j = 1, Ny+1
    dry_v[:, Ny+1, :]   .= dry_c[:, Ny, :]

    dry_ζ = falses(Nx, Ny + 1, Nz)             # (Face, Face, Center)
    for i in 1:Nx
        im = i == 1 ? Nx : i - 1
        dry_ζ[i, 2:Ny, :] .= dry_c[i, 2:Ny, :] .| dry_c[im, 2:Ny, :] .|
                             dry_c[i, 1:Ny-1, :] .| dry_c[im, 1:Ny-1, :]
    end
    dry_ζ[:, 1, :]    .= true                  # no vorticity on the outer walls
    dry_ζ[:, Ny+1, :] .= true

    return (; Nx, Ny, Nz, Lx, Ly, Lz, Δx, Δy, Δz, zf, zc,
              xc, xf, yc, yf,
              xc_km = xc ./ 1e3, xf_km = xf ./ 1e3,
              yc_km = yc ./ 1e3, yf_km = yf ./ 1e3,
              dry_c, dry_u, dry_v, dry_ζ)
end

# Coordinates + mask for a staggered location.
function location(g, loc::Symbol)
    loc === :c && return (g.xc_km, g.yc_km, g.dry_c)
    loc === :u && return (g.xf_km, g.yc_km, g.dry_u)
    loc === :v && return (g.xc_km, g.yf_km, g.dry_v)
    loc === :ζ && return (g.xf_km, g.yf_km, g.dry_ζ)
    error("unknown location $loc; expected :c, :u, :v or :ζ")
end

"""
    depth_indices(g, depths) -> Vector{Int}

k-indices of the cell centers nearest the requested depths (metres, negative
down), deepest-first reversed to surface-first so figures read top-down.
"""
depth_indices(g, depths) = [argmin(abs.(g.zc .- d)) for d in depths]

#####
##### Derived fields
#####

"""
    vorticity(u, v, g)

Relative vorticity ζ = ∂v/∂x − ∂u/∂y at (Face, Face, Center), size
(Nx, Ny+1, Nz), in s⁻¹. x differences wrap (periodic); the outer y walls, where
one of the two u cells does not exist, are left NaN and render as land.

A plain uniform-Δ difference of the saved interior fields — adequate for looking
at the eddy field, not a substitute for Oceananigans' staggered operators.
"""
function vorticity(u, v, g)
    Nx, Ny, Nz = g.Nx, g.Ny, g.Nz
    ζ = fill(NaN, Nx, Ny + 1, Nz)
    for k in 1:Nz, j in 2:Ny, i in 1:Nx
        im = i == 1 ? Nx : i - 1
        ζ[i, j, k] = (v[i, j, k] - v[im, j, k]) / g.Δx -
                     (u[i, j, k] - u[i, j-1, k]) / g.Δy
    end
    return ζ
end

# Linear-EOS buoyancy, temperature-only: the same b = α g T that the
# stratification term of the objective is built on (S is uniform without the
# spice anomaly).
buoyancy(T, α, g0) = (α * g0) .* T

# Divide a 3-D field by cell thickness — see the header note on why sensitivity
# panels default to this.
per_dz(a, g) = a ./ reshape(g.Δz, 1, 1, :)

"""
    zonal_mean(a, dry) -> Matrix (Ny, Nz)

Wet-masked zonal average: the mean over WET cells in x, not over all Nx, so the
ridge and the wall dilute nothing. Columns with no wet cell are NaN and render
as land. This is the same reduction the objective's stratification term uses.
"""
function zonal_mean(a, dry)
    nx, ny, nz = size(a)
    out = fill(NaN, ny, nz)
    for k in 1:nz, j in 1:ny
        s, n = 0.0, 0
        for i in 1:nx
            if !dry[i, j, k] && isfinite(a[i, j, k])
                s += a[i, j, k]
                n += 1
            end
        end
        n > 0 && (out[j, k] = s / n)
    end
    return out
end

#####
##### Panel builders
#####

# Colorbar label for a sensitivity: J is the run's composite objective, so the
# honest unit is "objective per unit of the thing differentiated".
per_unit(unit; dz = false) = dz ? "J per $unit per m" : "J per $unit"

# Axis width/height ratios, declared per FIGURE rather than per panel so
# plot_common.jl can derive the panel size from them (see AXIS_WIDTH there).
const XY_ASPECT      = 0.5    # x 0–1000 km by y 0–2000 km: the true data aspect
const SECTION_ASPECT = 2.6    # y 0–2000 km by z ~2.2 km: data aspect is useless here

"""
    xy_panel(a, g, k; loc, title, cb, ...) -> panel NamedTuple

Horizontal slice of a 3-D field at k-index `k`, with the coordinates and land
mask of staggered location `loc`. Extra keyword arguments are passed straight
through to the panel (and override these defaults).
"""
function xy_panel(a, g, k; loc::Symbol = :c, kw...)
    x, y, dry = location(g, loc)
    return (; data = a[:, :, k], x, y, mask = dry[:, :, k],
              xlabel = "x (km)", ylabel = "y (km)", kw...)
end

"""
    section_panel(a, g; loc, ...) -> panel NamedTuple

Wet-masked zonal-mean (y–z) section of a 3-D field.
"""
function section_panel(a, g; loc::Symbol = :c, kw...)
    _, y, dry = location(g, loc)
    return (; data = zonal_mean(a, dry), x = y, y = g.zc,
              xlabel = "y (km)", ylabel = "z (m)", kw...)
end

# Title for an x–y panel, reporting the depth actually resolved rather than the
# one that was asked for.
zlabel(g, k) = @sprintf("z = %.0f m", g.zc[k])

#####
##### The figure set
#####

"""
    make_plots(dir; depths = [-15, -500, -1500], per_thickness = true,
               plotdir = joinpath(dir, "plots"))

Read the run in `dir` and write its diagnostic figures to `plotdir`.

`depths` are in metres (negative down) and are snapped to the nearest cell
center. `per_thickness` divides sensitivity fields by Δz (see the header).
Figures whose inputs are missing from the save files are skipped with a note,
so runs without GM/Redi or biogeochemistry work without special-casing.
"""
function make_plots(dir::AbstractString;
                    depths        = [-15.0, -500.0, -1500.0],
                    per_thickness = true,
                    plotdir       = joinpath(dir, "plots"))

    init  = read_keys(joinpath(dir, "data_init.jld2"),  INIT_KEYS)
    bgc   = other_tracers(init)
    isempty(bgc) || merge!(init, read_keys(joinpath(dir, "data_init.jld2"), other_tracer_keys(bgc; init = true)))
    final = read_keys(joinpath(dir, "data_final.jld2"), vcat(FINAL_KEYS, other_tracer_keys(bgc; init = false)))

    g  = channel_plot_grid(init)
    ks = depth_indices(g, depths)
    mkpath(plotdir)
    out(name) = joinpath(plotdir, name)

    α    = Float64(get(init, "alpha", ALPHA_DEFAULT))
    g0   = Float64(get(init, "g", G_DEFAULT))
    Jsuf = per_thickness                       # shorthand used below
    dzt  = Jsuf ? " ÷ Δz" : ""
    nrm  = Jsuf ? (a -> per_dz(a, g)) : identity

    @info @sprintf("%s: run_type = %s, %d×%d×%d, Nspinup = %s, Ntimesteps = %s, advection = %s",
                   dir,
                   get(final, "run_type", get(init, "run_type", "?")),
                   g.Nx, g.Ny, g.Nz,
                   get(init, "Nspinup", "?"), get(init, "Ntimesteps", "?"),
                   get(init, "advection", "?"))
    haskey(final, "zonal_transport_and_buoyancy") &&
        @info @sprintf("  objective J = %.6e", final["zonal_transport_and_buoyancy"])
    @info "  depths $(round.(depths, digits=0)) m → k = $ks (z = $(round.(g.zc[ks], digits=1)) m)"

    ##### 1. Tracers ########################################################
    if haskey(final, "T_final") && haskey(final, "S_final")
        panels = Any[]
        for k in ks
            push!(panels, xy_panel(final["T_final"], g, k;
                                   title = "T($(zlabel(g, k)))", cb = "°C",
                                   cmap = :thermal, symmetric = false))
            push!(panels, xy_panel(final["S_final"], g, k;
                                   title = "S($(zlabel(g, k)))", cb = "g/kg",
                                   cmap = :haline, symmetric = false))
            haskey(final, "e_final") &&
                push!(panels, xy_panel(final["e_final"], g, k;
                                       title = "e($(zlabel(g, k)))", cb = "m² s⁻²",
                                       cmap = :tempo, symmetric = false))
        end
        ncols = haskey(final, "e_final") ? 3 : 2
        plot_variables_rxc(panels; layout = (length(ks), ncols),
                           aspect = XY_ASPECT, filename = out("tracers_xy.png"))
        @info "  wrote tracers_xy.png"
    end

    ##### 2. Velocities and vorticity #######################################
    if haskey(final, "u") && haskey(final, "v")
        ζ = vorticity(final["u"], final["v"], g)
        panels = Any[]
        for k in ks
            push!(panels, xy_panel(final["u"], g, k; loc = :u,
                                   title = "u($(zlabel(g, k)))", cb = "m s⁻¹"))
            push!(panels, xy_panel(final["v"], g, k; loc = :v,
                                   title = "v($(zlabel(g, k)))", cb = "m s⁻¹"))
            push!(panels, xy_panel(ζ, g, k; loc = :ζ,
                                   title = "ζ($(zlabel(g, k)))", cb = "s⁻¹",
                                   cmap = :curl))
        end
        plot_variables_rxc(panels; layout = (length(ks), 3),
                           aspect = XY_ASPECT, filename = out("velocities_xy.png"))
        @info "  wrote velocities_xy.png"
    end

    ##### 3. Drift over the AD window #######################################
    if haskey(final, "T_final") && haskey(init, "T_init")
        ΔT = final["T_final"] .- init["T_init"]
        ΔS = haskey(final, "S_final") && haskey(init, "S_init") ?
             final["S_final"] .- init["S_init"] : nothing
        panels = Any[]
        for k in ks
            push!(panels, xy_panel(ΔT, g, k;
                                   title = "T − T₀($(zlabel(g, k)))", cb = "°C"))
            ΔS === nothing ||
                push!(panels, xy_panel(ΔS, g, k;
                                       title = "S − S₀($(zlabel(g, k)))", cb = "g/kg"))
        end
        plot_variables_rxc(panels; layout = (length(ks), ΔS === nothing ? 1 : 2),
                           aspect = XY_ASPECT, filename = out("drift_xy.png"))
        @info "  wrote drift_xy.png"
    end

    ##### 4. Surface: SSH and the forcing ###################################
    if haskey(final, "ssh")
        panels = Any[]
        push!(panels, xy_panel(final["ssh"], g, 1; title = "SSH", cb = "m"))
        haskey(init, "u_wind_stress") &&
            push!(panels, (; xy_panel(init["u_wind_stress"], g, 1; loc = :u,
                                      title = "τₓ", cb = "m² s⁻²")...,
                             mask = g.dry_u[:, :, g.Nz]))
        haskey(init, "v_wind_stress") &&
            push!(panels, (; xy_panel(init["v_wind_stress"], g, 1; loc = :v,
                                      title = "τᵧ", cb = "m² s⁻²")...,
                             mask = g.dry_v[:, :, g.Nz]))
        haskey(init, "T_flux") &&
            push!(panels, (; xy_panel(init["T_flux"], g, 1;
                                      title = "surface T flux", cb = "K m s⁻¹")...,
                             mask = g.dry_c[:, :, g.Nz]))
        panels[1] = (; panels[1]..., mask = g.dry_c[:, :, g.Nz])
        plot_variables_rxc(panels; aspect = XY_ASPECT, filename = out("surface_xy.png"))
        @info "  wrote surface_xy.png"
    end

    ##### 5. GM/Redi coefficients (conditional) #############################
    if haskey(final, "kappa_skew") && haskey(final, "kappa_symmetric")
        κs, κy = final["kappa_skew"], final["kappa_symmetric"]
        panels = Any[]
        for k in ks
            push!(panels, xy_panel(κs, g, k; title = "κ_skew($(zlabel(g, k)))",
                                   cb = "m² s⁻¹", cmap = :viridis, symmetric = false))
            push!(panels, xy_panel(κy, g, k; title = "κ_symmetric($(zlabel(g, k)))",
                                   cb = "m² s⁻¹", cmap = :viridis, symmetric = false))
        end
        plot_variables_rxc(panels; layout = (length(ks), 2),
                           aspect = XY_ASPECT, filename = out("kappa_xy.png"))
        @info "  wrote kappa_xy.png"
    else
        @info "  no kappa_skew/kappa_symmetric in this run — skipping the κ figures"
    end

    ##### 6. State sensitivities ############################################
    if haskey(final, "dT")
        panels = Any[]
        for k in ks
            push!(panels, xy_panel(nrm(final["dT"]), g, k;
                                   title = "∂J/∂T$dzt($(zlabel(g, k)))",
                                   cb = per_unit("K"; dz = Jsuf)))
            haskey(final, "dS") &&
                push!(panels, xy_panel(nrm(final["dS"]), g, k;
                                       title = "∂J/∂S$dzt($(zlabel(g, k)))",
                                       cb = per_unit("(g/kg)"; dz = Jsuf)))
            haskey(final, "du") &&
                push!(panels, xy_panel(nrm(final["du"]), g, k; loc = :u,
                                       title = "∂J/∂u$dzt($(zlabel(g, k)))",
                                       cb = per_unit("(m/s)"; dz = Jsuf)))
            haskey(final, "dv") &&
                push!(panels, xy_panel(nrm(final["dv"]), g, k; loc = :v,
                                       title = "∂J/∂v$dzt($(zlabel(g, k)))",
                                       cb = per_unit("(m/s)"; dz = Jsuf)))
        end
        ncols = 1 + haskey(final, "dS") + haskey(final, "du") + haskey(final, "dv")
        plot_variables_rxc(panels; layout = (length(ks), ncols),
                           aspect = XY_ASPECT, filename = out("sensitivity_state_xy.png"))
        @info "  wrote sensitivity_state_xy.png"
    end

    ##### 7. κ sensitivities (conditional) ##################################
    if haskey(final, "dkappa_skew") && haskey(final, "dkappa_symmetric")
        panels = Any[]
        for k in ks
            push!(panels, xy_panel(nrm(final["dkappa_skew"]), g, k;
                                   title = "∂J/∂κ_skew$dzt($(zlabel(g, k)))",
                                   cb = per_unit("(m²/s)"; dz = Jsuf)))
            push!(panels, xy_panel(nrm(final["dkappa_symmetric"]), g, k;
                                   title = "∂J/∂κ_sym$dzt($(zlabel(g, k)))",
                                   cb = per_unit("(m²/s)"; dz = Jsuf)))
        end
        plot_variables_rxc(panels; layout = (length(ks), 2),
                           aspect = XY_ASPECT, filename = out("sensitivity_kappa_xy.png"))
        @info "  wrote sensitivity_kappa_xy.png"
    else
        @info "  no dkappa_skew/dkappa_symmetric — skipping the κ sensitivity figure"
    end

    ##### 8. Forcing sensitivities ##########################################
    begin
        panels = Any[]
        haskey(final, "dssh") &&
            push!(panels, (; xy_panel(final["dssh"], g, 1;
                                      title = "∂J/∂η", cb = per_unit("m"))...,
                             mask = g.dry_c[:, :, g.Nz]))
        haskey(final, "du_wind_stress") &&
            push!(panels, (; xy_panel(final["du_wind_stress"], g, 1; loc = :u,
                                      title = "∂J/∂τₓ", cb = per_unit("(m²/s²)"))...,
                             mask = g.dry_u[:, :, g.Nz]))
        haskey(final, "dv_wind_stress") &&
            push!(panels, (; xy_panel(final["dv_wind_stress"], g, 1; loc = :v,
                                      title = "∂J/∂τᵧ", cb = per_unit("(m²/s²)"))...,
                             mask = g.dry_v[:, :, g.Nz]))
        haskey(final, "dT_flux") &&
            push!(panels, (; xy_panel(final["dT_flux"], g, 1;
                                      title = "∂J/∂(T flux)",
                                      cb = per_unit("(K m/s)"))...,
                             mask = g.dry_c[:, :, g.Nz]))
        if !isempty(panels)
            plot_variables_rxc(panels; aspect = XY_ASPECT, filename = out("sensitivity_forcing_xy.png"))
            @info "  wrote sensitivity_forcing_xy.png"
        end
    end

    ##### 9. Zonal-mean sections: state #####################################
    if haskey(final, "T_final")
        panels = Any[]
        push!(panels, section_panel(final["T_final"], g;
                                    title = "⟨T⟩ₓ(y, z)", cb = "°C",
                                    cmap = :thermal, symmetric = false, contour = true))
        haskey(final, "S_final") &&
            push!(panels, section_panel(final["S_final"], g;
                                        title = "⟨S⟩ₓ(y, z)", cb = "g/kg",
                                        cmap = :haline, symmetric = false))
        haskey(final, "u") &&
            push!(panels, section_panel(final["u"], g; loc = :u,
                                        title = "⟨u⟩ₓ(y, z)", cb = "m s⁻¹"))
        push!(panels, section_panel(buoyancy(final["T_final"], α, g0), g;
                                    title = "⟨b⟩ₓ(y, z) = ⟨αgT⟩ₓ", cb = "m s⁻²",
                                    cmap = :thermal, symmetric = false, contour = true))
        plot_variables_rxc(panels; layout = (2, 2),
                           aspect = SECTION_ASPECT, filename = out("sections_state.png"))
        @info "  wrote sections_state.png"
    end

    ##### 10. Zonal-mean sections: κ (conditional) ##########################
    if haskey(final, "kappa_skew") && haskey(final, "kappa_symmetric")
        κs, κy = final["kappa_skew"], final["kappa_symmetric"]
        panels = Any[section_panel(κs, g; title = "⟨κ_skew⟩ₓ(y, z)", cb = "m² s⁻¹",
                                   cmap = :viridis, symmetric = false),
                     section_panel(κy, g; title = "⟨κ_symmetric⟩ₓ(y, z)", cb = "m² s⁻¹",
                                   cmap = :viridis, symmetric = false)]
        plot_variables_rxc(panels; layout = (1, 2),
                           aspect = SECTION_ASPECT, filename = out("sections_kappa.png"))
        @info "  wrote sections_kappa.png"
    end

    ##### 11. Zonal-mean sections: sensitivities ############################
    if haskey(final, "dT")
        panels = Any[section_panel(nrm(final["dT"]), g;
                                   title = "⟨∂J/∂T$dzt⟩ₓ(y, z)",
                                   cb = per_unit("K"; dz = Jsuf))]
        haskey(final, "dS") &&
            push!(panels, section_panel(nrm(final["dS"]), g;
                                        title = "⟨∂J/∂S$dzt⟩ₓ(y, z)",
                                        cb = per_unit("(g/kg)"; dz = Jsuf)))
        haskey(final, "du") &&
            push!(panels, section_panel(nrm(final["du"]), g; loc = :u,
                                        title = "⟨∂J/∂u$dzt⟩ₓ(y, z)",
                                        cb = per_unit("(m/s)"; dz = Jsuf)))
        haskey(final, "dkappa_skew") ?
            push!(panels, section_panel(nrm(final["dkappa_skew"]), g;
                                        title = "⟨∂J/∂κ_skew$dzt⟩ₓ(y, z)",
                                        cb = per_unit("(m²/s)"; dz = Jsuf))) :
            haskey(final, "dv") &&
                push!(panels, section_panel(nrm(final["dv"]), g; loc = :v,
                                            title = "⟨∂J/∂v$dzt⟩ₓ(y, z)",
                                            cb = per_unit("(m/s)"; dz = Jsuf)))
        plot_variables_rxc(panels; layout = (2, 2),
                           aspect = SECTION_ASPECT, filename = out("sections_sensitivity.png"))
        @info "  wrote sections_sensitivity.png"
    end

    ##### 12. Biogeochemical tracers (conditional) ##########################
    # One column per tracer in `bgc` (from tracer_names): the end state at the
    # chosen depths, its zonal-mean section, and ∂J/∂(initial tracer). Units
    # are the biogeochemistry's own, so the labels name none.
    present = [n for n in bgc if haskey(final, n * "_final")]
    if isempty(present)
        isempty(bgc) || @info "  tracer_names lists $bgc, but data_final.jld2 holds none of them — skipping"
    else
        panels = Any[]
        for k in ks, n in present
            push!(panels, xy_panel(final[n * "_final"], g, k;
                                   title = "$n($(zlabel(g, k)))", cb = n,
                                   cmap = :viridis, symmetric = false))
        end
        plot_variables_rxc(panels; layout = (length(ks), length(present)),
                           aspect = XY_ASPECT, filename = out("bgc_tracers_xy.png"))
        @info "  wrote bgc_tracers_xy.png"

        panels = Any[section_panel(final[n * "_final"], g; title = "⟨$n⟩ₓ(y, z)", cb = n,
                                   cmap = :viridis, symmetric = false) for n in present]
        plot_variables_rxc(panels; layout = (1, length(present)),
                           aspect = SECTION_ASPECT, filename = out("sections_bgc.png"))
        @info "  wrote sections_bgc.png"

        sens = [n for n in present if haskey(final, "d" * n)]
        if !isempty(sens)
            panels = Any[]
            for k in ks, n in sens
                push!(panels, xy_panel(nrm(final["d" * n]), g, k;
                                       title = "∂J/∂$n$dzt($(zlabel(g, k)))",
                                       cb = per_unit(n; dz = Jsuf)))
            end
            plot_variables_rxc(panels; layout = (length(ks), length(sens)),
                               aspect = XY_ASPECT, filename = out("sensitivity_bgc_xy.png"))
            @info "  wrote sensitivity_bgc_xy.png"

            panels = Any[section_panel(nrm(final["d" * n]), g; title = "⟨∂J/∂$n$dzt⟩ₓ(y, z)",
                                       cb = per_unit(n; dz = Jsuf)) for n in sens]
            plot_variables_rxc(panels; layout = (1, length(sens)),
                               aspect = SECTION_ASPECT, filename = out("sections_bgc_sensitivity.png"))
            @info "  wrote sections_bgc_sensitivity.png"
        end
    end

    @info "  → $plotdir"
    return plotdir
end

# `julia --project scripts/plot_channel.jl <dir> [more...]` as well as `include`.
if abspath(PROGRAM_FILE) == @__FILE__
    isempty(ARGS) &&
        error("usage: julia --project scripts/plot_channel.jl <run_directory> [more...]")
    for d in ARGS
        make_plots(d)
    end
end
