# channel_geometry.jl
#
# Dependency-free channel geometry shared by channel_common.jl (the model build)
# and the host-side tools (plot_channel.jl). Extracting it here gives ONE source
# of truth for the domain constants, the stretched vertical grid, the immersed
# wall/ridge shape, and the GridFittedBottom snapping — so a change to any of
# them can no longer silently disagree between the files that use them.
#
# Deliberately pure Julia: no Oceananigans / Reactant / Enzyme / Units. That is
# what lets the plotting tools include it without pulling in the AD stack, and
# it lets channel_common.jl include it before any of those packages are loaded.
#
# All lengths are SI metres. Every function is parameterized by the domain
# extents (Lx, Ly, Lz) rather than closing over globals, so the same geometry
# serves a run at any resolution or domain, e.g. a saved run whose grid differs
# from the one channel_common.jl hardcodes.
#
# The whole file is wrapped in an include guard so that a second include in the
# same session (e.g. the driver and plot_channel.jl in one REPL) is a no-op.

if !@isdefined(_CHANNEL_GEOMETRY_LOADED)
    _CHANNEL_GEOMETRY_LOADED = true

    #####
    ##### Canonical channel constants (SI)
    #####
    # channel_common.jl aliases these to its short names (Lx, Ly, α, β_S, g);
    # host-side tools use them as FALLBACKS for metadata a data file omits.

    const LX_DEFAULT    = 1000e3    # [m]    zonal domain length
    const LY_DEFAULT    = 2000e3    # [m]    meridional domain length
    const ALPHA_DEFAULT = 1.67e-4 #2e-4      # [K⁻¹]  thermal expansion coefficient
    const BETA_S_DEFAULT = 7.8e-4   # [psu⁻¹] haline contraction coefficient (Oceananigans' default)
    const G_DEFAULT     = 9.8061    # [m/s²] gravitational acceleration

    """
        spice_amplitude(forcing_parameters) -> Float64

    The density-compensated spice anomaly amplitude (`ΔS_spice`, psu) a run was
    made with, read from its saved `forcing_parameters`. Files written before the
    anomaly existed (or with no `forcing_parameters`) had none, so they return 0.
    Here rather than in channel_common.jl so dependency-free tools can use it too.
    """
    spice_amplitude(fp) = (fp !== nothing && hasproperty(fp, :ΔS_spice)) ? Float64(fp.ΔS_spice) : 0.0

    """
        reference_salinity(forcing_parameters) -> Float64

    The reference salinity `S₀` (psu) a run was made with: its initial value and
    S-sponge target. Files written before `S₀` was a parameter return 35, the
    uniform salinity every one of those runs started from.
    """
    reference_salinity(fp) = (fp !== nothing && hasproperty(fp, :S₀)) ? Float64(fp.S₀) : 35.0

    """
        linear_buoyancy(T, S′; α, β_S, g)

    Linear-EOS buoyancy `g (α T − β_S S′)` from temperature `T` and the salinity
    ANOMALY `S′ = S − S₀`. The anomaly keeps the large constant `g β_S S₀`
    (≈ 0.27 m/s²) out of b. This is the one definition of b for every place that
    builds it, so they cannot drift apart. It broadcasts, so the same function serves host
    Arrays and traced device arrays. Written as `(α g)·T − (β_S g)·S′` so that
    `S′ ≡ 0` reproduces the old temperature-only `(α g)·T` bit for bit.
    """
    linear_buoyancy(T, S′; α, β_S, g) = (α * g) .* T .- (β_S * g) .* S′

    #####
    ##### Vertical grid
    #####

    """
        channel_z_faces(Nz) -> Vector{Float64}

    Stretched vertical face positions, `Nz+1` long, increasing from `-Lz` to `0`.

    NOTE the trap baked into this formula: `Δz_center = 10·1.104^(Nz−k)` makes
    `Lz = Σ Δz` a function of `Nz` (2184 m at Nz=32, but ~54 km at Nz=64). Changing
    the vertical resolution therefore changes the DOMAIN, not just its
    discretization — runs to be compared must share `Nz` until that formula is
    rewritten to hold `Lz` fixed.
    """
    function channel_z_faces(Nz::Integer)
        Δz_center = [10 * 1.104^(Nz - k) for k in 1:Nz]
        Lz = sum(Δz_center)
        z_faces = vcat(-Lz, -Lz .+ cumsum(Δz_center))
        z_faces[Nz+1] = 0.0
        return z_faces
    end

    cell_centers(faces) = 0.5 .* (faces[1:end-1] .+ faces[2:end])

    #####
    ##### Immersed topography
    #####

    """
        wall_function(x, y, Lx, Ly, Lz)

    The immersed wall: a full-depth 60 km meridional barrier at 470–530 km, open
    in the 400–1000 km latitude gap. Returns a bottom height (`+1` inside the
    barrier, `-Lz` elsewhere) suitable for `GridFittedBottom`. This is the
    topography the model actually runs with (`make_grid` sets it via `set!`).
    """
    function wall_function(x, y, Lx, Ly, Lz)
        zonal = (x > 470e3) && (x < 530e3)
        gap   = (y < 400e3) || (y > 1000e3)
        return (Lz + 1) * zonal * gap - Lz
    end

    """
        ridge_function(x, y, Lx, Ly, Lz)

    Smooth Gaussian ridge alternative to `wall_function`: a partial-height
    meridional ridge tapered open across the 1/6–1/2 latitude band. Not currently
    wired into the model — `make_grid` uses `wall_function` — but kept as the
    sibling topography option. The `1e9` length scale below was `1e6 * kilometers`
    in the Oceananigans-Units original.
    """
    function ridge_function(x, y, Lx, Ly, Lz)
        zonal = (Lz + 3000) * exp(-(x - Lx/2)^2 / 1e9)
        gap   = 1 - 0.5 * (tanh((y - Ly/6)/1e5) - tanh((y - Ly/2)/1e5))
        return zonal * gap - Lz
    end

    """
        numerical_bottom_height(Nx, Ny, z_faces; Lx, Ly) -> Matrix (Nx, Ny)

    Reproduce Oceananigans' `GridFittedBottom` materialization: the analytic
    bottom height is snapped UP to the top face of the highest cell whose CENTER
    lies at or below it (`CenterImmersedCondition`, the default). Matching this
    snapping exactly is what makes a mask derived on the host agree cell-for-cell
    with the one the model actually runs with.
    """
    function numerical_bottom_height(Nx, Ny, z_faces; Lx = LX_DEFAULT, Ly = LY_DEFAULT)
        Nz = length(z_faces) - 1
        Lz = -z_faces[1]
        zc = cell_centers(z_faces)
        xc = (collect(1:Nx) .- 0.5) .* (Lx / Nx)
        yc = (collect(1:Ny) .- 0.5) .* (Ly / Ny)

        bh = fill(z_faces[1], Nx, Ny)
        for j in 1:Ny, i in 1:Nx
            zb = wall_function(xc[i], yc[j], Lx, Ly, Lz)
            for k in 1:Nz
                zc[k] ≤ zb && (bh[i, j] = z_faces[k+1])
            end
        end
        return bh
    end

end # include guard
