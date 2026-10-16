# ad_checks.jl
#
# Verification of the AD gradient, run by abernathey_channel.jl ([checks] in
# the run config). Both work host-side, between compiled calls, on the compiled
# forward and reverse functions the run already has:
#
#   * The DETERMINISM PROBE (determinism_probe): J must be a pure function of its
#     inputs, because every window restores its starting state (ModelState),
#     forcing and clock at its top. It evaluates J(p₀), then an AD call at a
#     perturbed point p₁, then J(p₀) again, and requires the two J(p₀) to be
#     BITWISE equal: same compiled function, same inputs, so any difference is
#     state leaking from one window into the next. Typical culprits: a stateful
#     closure missing from closure_state_fields, or a biogeochemistry that steps
#     fields ModelState does not carry (e.g. sediments). The FD check depends on
#     it, since every FD evaluation is a separate window.
#
#   * The FD CHECK (run_fd_check): along a direction v in the space of the inputs
#     below, the AD directional derivative ⟨∇J, v⟩ (from the shadows of the run's
#     one AD call) against central differences (J(p₀ + εv) − J(p₀ − εv)) / 2ε,
#     for ε = FD_EPSILONS. Two kinds of direction:
#       - random directions over ALL the inputs at once ([checks]
#         fd_random_directions): the whole gradient;
#       - one direction per named input ([checks] fd_components): which pathway a
#         failure is in.
#     HOW TO READ IT: look for a relative error that keeps FALLING as ε shrinks,
#     down to a best value, then rises again as round-off takes over (the FD
#     error grows like (round-off in J)/ε). Agreement at a single ε, with the
#     error jumping around elsewhere, also happens by chance across a
#     discontinuity in J (CATKE has several), so it is not a pass. The summary
#     line flags whether the error falls monotonically to its best value.
#     Centered advection is required: WENO's Float32 smoothness indicators put a
#     ~1e-7 relative noise floor on J that swamps the FD arm.
#
# THE INPUTS CHECKED ("FD components"): every input of J whose AD gradient the
# run computes and that can be changed host-side between compiled calls:
#   * the window's initial state (ModelState): every tracer by its name (T, S,
#     CATKE's e, biogeochemical tracers), u, v, ssh (η), U, V. Not w (diagnosed
#     from continuity at the restore, so its gradient is zero), the closure state
#     or the clock time;
#   * the forcing (ModelInputs): u_wind_stress, v_wind_stress, temp_flux;
#   * fields inside the model: kappa_skew and kappa_symmetric (with GM/Redi), and
#     kappa_z, the background vertical diffusivity field (shared by all tracers).
# Plain numbers inside the model (κh, νh, a biogeochemistry's parameters stored
# as Float64 struct fields) are baked into the compiled graph: they have no
# gradient and changing them between calls does nothing. To check such a
# parameter, make it a Field or device array that the model reads.
#
# WHERE a component is perturbed: its wet entries only. Dry cells and faces
# (inside the wall, below the bottom, the walls' normal velocity) are masked or
# ignored by the model, so FD and AD both see nothing there; perturbing them
# would only dilute the direction.
#
# HOW BIG: each component's perturbation is Gaussian noise of RMS = its scale
# (fd_scale: the RMS of its wet values, or FD_SCALE_OVERRIDES), so ε is a
# relative perturbation size for every component, and one ε sweep fits all.
#
# Needs channel_common.jl included first (to_host, find_closure, z_faces, Nz,
# parameters).

using Random: Xoshiro, randn
using Printf
using Statistics: mean

if !@isdefined(_AD_CHECKS_LOADED)
    _AD_CHECKS_LOADED = true

    # The ε sweep of the FD check.
    const FD_EPSILONS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)

    # Perturbation scales [units of the input] for inputs whose wet RMS is a
    # poor scale: S is uniform at S₀ = 35 psu without the spice anomaly (an RMS
    # perturbation would be a 35·ε psu buoyancy kick), and the meridional wind
    # stress is zero.
    const FD_SCALE_OVERRIDES = Dict("S" => 0.1, "v_wind_stress" => parameters.τ)

    # The determinism probe's perturbation size (relative, like ε) and seed.
    const PROBE_DELTA = 1e-2
    const PROBE_SEED  = 2026

    #####
    ##### Perturbable inputs
    #####

    """
        FDComponent(name, primal, write!, gradient, mask)

    One input of J for the AD checks: `primal()` returns a host copy of its
    current value (interior), `write!(a)` writes a host array into it,
    `gradient()` returns a host copy of ∂J/∂(it) from the run's AD call (its
    shadow), and `mask` marks the wet entries that are perturbed.
    """
    struct FDComponent
        name     :: String
        primal   :: Function
        write!   :: Function
        gradient :: Function
        mask     :: BitArray{3}
    end

    # Wet masks at cell centers (c), x-faces (u) and y-faces (v), from the
    # materialized bottom height: a cell is dry when its center lies below the
    # bottom (GridFittedBottom's rule, as plot_channel.jl rebuilds it); a face
    # is dry when either cell beside it is; the y walls' faces are dry.
    function fd_wet_masks(model)
        bh = to_host(bottom_height_field(model.grid))[:, :, 1]
        zc = (z_faces[1:end-1] .+ z_faces[2:end]) ./ 2
        nx, ny = size(bh)
        c = BitArray([zc[k] ≥ bh[i, j] for i in 1:nx, j in 1:ny, k in 1:Nz])
        u = c .& circshift(c, (1, 0, 0))                   # face i: cells i−1 (periodic) and i
        v = falses(nx, ny + 1, Nz)
        v[:, 2:ny, :] .= c[:, 1:ny-1, :] .& c[:, 2:ny, :]
        return (; c, u, v)
    end

    # The wet mask for `field`'s interior, by its location. Surface fields (one
    # level: η, U, V, the fluxes and stresses) take the top layer's mask.
    function fd_mask(masks, field)
        ℓx, ℓy, _ = Oceananigans.location(field)
        m = ℓx === Face ? masks.u : ℓy === Face ? masks.v : masks.c
        n = size(interior(field))
        n[3] == 1 && Nz > 1 && (m = m[:, :, Nz:Nz])
        size(m) == n || error("FD check: no wet mask of size $n for a field at $(Oceananigans.location(field))")
        return m
    end

    #####
    ##### NaN derivatives on dry entries
    #####
    # DISCLAIMER: on the CPU architecture, Enzyme returns NaN derivatives on DRY
    # (immersed) entries of the velocity shadows: du, dv, dU, dV in the wall
    # (2026-10-16, 10-step spinup / 6-step window: every dry u-face and the dry
    # interior v-faces; wet entries finite and equal to Reactant's to ~1e-16).
    # Reactant gives finite but NONZERO, equally meaningless values there (dU up
    # to 1.3e-2 in the wall, as large as anywhere wet), which are left as they
    # are: read dry entries of Reactant shadows as junk too. The likely source is a 0·∞ in the
    # reverse pass of an Oceananigans kernel that divides by a zero water-column
    # depth and hides the result with ifelse, which set_strong_zero does not
    # catch on the CPU path (not confirmed). A velocity in a wall is held at
    # zero, so its derivative carries no information; but a NaN poisons every
    # sum it enters (the FD check's directional derivatives, plots), so
    # zero_dry_nans! sets those entries to 0 right after the run's AD call.
    # Saved dry-entry derivatives are therefore 0 by construction, not computed.
    # A NaN on a WET entry is never zeroed: it is an error.

    # Zero the NaNs of one shadow `a` (host-side) on the dry entries of `wet`;
    # error on a wet NaN. `write!(h)` writes the cleaned host copy back. Returns
    # the number of entries zeroed.
    function zero_dry_nans_of!(name, h, wet, write!)
        nans = isnan.(h)
        any(nans) || return 0
        n_wet = count(nans .& wet)
        n_wet == 0 || error("AD: ∂J/∂$name has $n_wet NaN entries on WET points: a broken gradient, not a dry-cell artifact")
        h[nans] .= 0
        write!(h)
        return count(nans)
    end

    """
        zero_dry_nans!(model, dmodel, dinputs, dstate) -> String

    Set to 0 every NaN of the run's shadows that lies on a dry (immersed) entry,
    erroring on any NaN on a wet entry (see the DISCLAIMER above). Covers every
    shadow that is saved or read by the FD check: the ModelState shadow (every
    tracer, u, v, w, η, U, V), the forcing shadows, and the κz / GM/Redi κ
    shadows. Host-side, between compiled calls; a shadow without NaNs is not
    written. Returns a report ("" if nothing was zeroed), also logged.
    """
    function zero_dry_nans!(model, dmodel, dinputs, dstate)
        masks = fd_wet_masks(model)
        # z-faces of w: wet between two wet cells, and at the top of a wet
        # column; the bottom face (k = 1) and faces on the immersed bottom are dry.
        c = masks.c
        w_wet = falses(size(c, 1), size(c, 2), Nz + 1)
        w_wet[:, :, 2:Nz] .= c[:, :, 1:Nz-1] .& c[:, :, 2:Nz]
        w_wet[:, :, Nz+1] .= c[:, :, Nz]

        zeroed = Pair{String, Int}[]
        record!(name, n) = n > 0 && push!(zeroed, name => n)

        # ModelState shadow leaves (bare arrays), with the model field that gives
        # each one's location.
        fs = model.free_surface
        leaves = [[(String(n), dstate.tracers[n], fd_mask(masks, model.tracers[n])) for n in keys(dstate.tracers)];
                  [("u",   dstate.u, fd_mask(masks, model.velocities.u)),
                   ("v",   dstate.v, fd_mask(masks, model.velocities.v)),
                   ("w",   dstate.w, w_wet),
                   ("ssh", dstate.η, fd_mask(masks, fs.displacement)),
                   ("U",   dstate.U, fd_mask(masks, fs.barotropic_velocities.U)),
                   ("V",   dstate.V, fd_mask(masks, fs.barotropic_velocities.V))]]
        for (name, da, wet) in leaves
            record!(name, zero_dry_nans_of!(name, Array(da), wet, h -> copyto!(da, h)))
        end

        # Field shadows: the forcing, κz (one Field shared by every tracer), GM/Redi κ.
        fields = Tuple{String, Any}[(String(k), getfield(dinputs, k)) for k in (:u_wind_stress, :v_wind_stress, :temp_flux)]
        vsc = vertical_scalar_closure(dmodel)
        vsc !== nothing && vsc.κ.T isa Field && push!(fields, ("kappa_z", vsc.κ.T))
        dissd = find_closure(dmodel, IsopycnalSkewSymmetricDiffusivity)
        dissd !== nothing && push!(fields, ("kappa_skew", dissd.κ_skew), ("kappa_symmetric", dissd.κ_symmetric))
        for (name, df) in fields
            record!(name, zero_dry_nans_of!(name, to_host(df), fd_mask(masks, df), h -> write_interior_host!(df, h)))
        end

        isempty(zeroed) && return ""
        report = "zeroed NaN derivatives on dry entries: " * join(["d$k $n" for (k, n) in zeroed], ", ")
        @warn "AD: $report (see the DISCLAIMER in ad_checks.jl; wet entries have no NaN)"
        return report
    end

    # Write `a` (interior-sized host array) into a Field's interior, host-side
    # between compiled calls. Through a host copy of the parent, because the
    # parent is a device array that takes whole host arrays (copyto!), while an
    # interior view of it does not. The halos keep their values: each compiled
    # run refills the halos it reads (fill_state_halos!, fill_gmredi_κ_halos!,
    # the boundary-condition set!s).
    function write_interior_host!(field, a)
        p = convert(Array, parent(field))
        offsets = map(ax -> 1 - first(ax), axes(field.data))   # interior index i ↦ parent index i + offset
        p[map((n, o) -> (1:n) .+ o, size(a), offsets)...] .= a
        copyto!(parent(field), p)
        return nothing
    end

    # A ModelState leaf (bare device array) and its shadow. Read with Array(·),
    # which always copies: on the CPU the leaf is already an Array, and
    # convert(Array, ·) would return the live array itself, so the saved p₀
    # would change with every perturbation and restore! would restore nothing.
    state_component(name, a, da, mask) =
        FDComponent(name, () -> Array(a), h -> (copyto!(a, h); nothing), () -> Array(da), mask)

    # A Field and its shadow Field.
    field_component(name, f, df, mask) =
        FDComponent(name, () -> to_host(f), h -> write_interior_host!(f, h), () -> to_host(df), mask)

    """
        fd_components(model, inputs, state, dmodel, dinputs, dstate) -> Vector{FDComponent}

    Every input the AD checks can perturb (see the header), with the shadows of
    the run's AD call (`dmodel`, `dinputs`, `dstate`, the same objects that
    call fills). Cheap: closures and masks only, so it can run before the
    compile to validate [checks] fd_components.
    """
    function fd_components(model, inputs, state, dmodel, dinputs, dstate)
        masks = fd_wet_masks(model)
        comps = FDComponent[]
        for name in keys(state.tracers)
            push!(comps, state_component(String(name), state.tracers[name], dstate.tracers[name],
                                         fd_mask(masks, model.tracers[name])))
        end
        fs = model.free_surface
        for (name, a, da, f) in (("u",   state.u, dstate.u, model.velocities.u),
                                 ("v",   state.v, dstate.v, model.velocities.v),
                                 ("ssh", state.η, dstate.η, fs.displacement),
                                 ("U",   state.U, dstate.U, fs.barotropic_velocities.U),
                                 ("V",   state.V, dstate.V, fs.barotropic_velocities.V))
            push!(comps, state_component(name, a, da, fd_mask(masks, f)))
        end
        for k in (:u_wind_stress, :v_wind_stress, :temp_flux)
            f = getfield(inputs, k)
            push!(comps, field_component(String(k), f, getfield(dinputs, k), fd_mask(masks, f)))
        end
        issd, dissd = find_closure(model, IsopycnalSkewSymmetricDiffusivity), find_closure(dmodel, IsopycnalSkewSymmetricDiffusivity)
        if issd !== nothing
            push!(comps, field_component("kappa_skew", issd.κ_skew, dissd.κ_skew, fd_mask(masks, issd.κ_skew)))
            push!(comps, field_component("kappa_symmetric", issd.κ_symmetric, dissd.κ_symmetric, fd_mask(masks, issd.κ_symmetric)))
        end
        vsc, dvsc = vertical_scalar_closure(model), vertical_scalar_closure(dmodel)
        if vsc !== nothing && vsc.κ.T isa Field
            # One field shared by every tracer (build_model), and its one shadow.
            push!(comps, field_component("kappa_z", vsc.κ.T, dvsc.κ.T, fd_mask(masks, vsc.κ.T)))
        end
        names = [c.name for c in comps]
        allunique(names) || error("FD check: duplicate input names $names (a tracer named like another input?)")
        return comps
    end

    # Check [checks] fd_components against the available inputs. Before the
    # compile, so a typo fails in seconds. Returns the selected components.
    function select_fd_components(comps, names)
        available = [c.name for c in comps]
        unknown = setdiff(names, available)
        isempty(unknown) ||
            error("[checks] fd_components: unknown $(unknown); this model's inputs are $(available)")
        return [comps[findfirst(==(n), available)] for n in names]
    end

    #####
    ##### Directions and perturbations
    #####

    # The perturbation size of a component (see the header).
    function fd_scale(c::FDComponent)
        haskey(FD_SCALE_OVERRIDES, c.name) && return FD_SCALE_OVERRIDES[c.name]
        s = sqrt(mean(abs2, c.primal()[c.mask]))
        s > 0 || (@warn "FD check: $(c.name) is zero on its wet entries; perturbing it at scale 1"; return 1.0)
        return s
    end

    # A random direction over `comps`, one host array per component: Gaussian
    # noise on its wet entries, scaled to RMS = fd_scale, zero elsewhere.
    function fd_direction(comps, seed)
        rng = Xoshiro(seed)
        return map(comps) do c
            r = zeros(Float64, size(c.mask))
            r[c.mask] .= randn(rng, count(c.mask))
            r .*= fd_scale(c) / sqrt(mean(abs2, r[c.mask]))
        end
    end

    # Set every component to base + δ·direction.
    function perturb!(comps, base, dirs, δ)
        for (c, p₀, v) in zip(comps, base, dirs)
            c.write!(p₀ .+ δ .* v)
        end
        return nothing
    end

    # Put every component back to `base` and verify that it is bitwise there.
    function restore!(comps, base)
        for (c, p₀) in zip(comps, base)
            c.write!(p₀)
            isequal(c.primal(), p₀) || error("AD checks: restoring $(c.name) did not reproduce its value bitwise")
        end
        return nothing
    end

    # ⟨∇J, v⟩ from the AD shadows.
    ad_directional(comps, dirs) = sum(sum(c.gradient() .* v) for (c, v) in zip(comps, dirs))

    #####
    ##### Determinism probe
    #####

    """
        determinism_probe(J, J_with_gradient, comps) -> J₀

    `J()` evaluates the compiled forward function at the current inputs;
    `J_with_gradient()` makes an AD call there with FRESH shadows (not the run's)
    and returns its primal J. Perturbs every component in `comps` by
    PROBE_DELTA along a fixed random direction for the AD call, and requires
    J(p₀) before and after it to be bitwise equal. Leaves every input bitwise
    as it found it. Errors on failure.
    """
    function determinism_probe(J, J_with_gradient, comps)
        base = [c.primal() for c in comps]
        dirs = fd_direction(comps, PROBE_SEED)

        J₀ = J()
        J₁ = try
            perturb!(comps, base, dirs, PROBE_DELTA)
            J_with_gradient()
        finally
            restore!(comps, base)
        end
        J₀′ = J()

        J₁ != J₀ || @warn "determinism probe: J(p₁) == J(p₀), so the window in between did not " *
                          "differ from the probed one and the probe proves little"
        J₀′ === J₀ || error("""
            DETERMINISM PROBE FAILED: J(p₀) = $J₀ before and $J₀′ after a window at other
            inputs (relative difference $(abs(J₀′ - J₀) / abs(J₀))). Something carries over
            from one window to the next that the window does not restore, so J is not a pure
            function of its inputs. Look for state missing from ModelState: a stateful closure
            (closure_state_fields in channel_common.jl), or fields a biogeochemistry steps.""")
        @info @sprintf("Determinism probe passed: J(p₀) = %.16e, bitwise identical after an AD call at other inputs (J(p₁) = %.16e)",
                       J₀, J₁)
        return J₀
    end

    #####
    ##### FD check
    #####

    # One direction: the AD directional derivative and the FD sweep. Leaves every
    # input bitwise as it found it.
    function fd_check_direction(J, comps, dirs; epsilons = FD_EPSILONS)
        base = [c.primal() for c in comps]
        ad = ad_directional(comps, dirs)
        n = length(epsilons)
        fd, Jp, Jm = zeros(n), zeros(n), zeros(n)
        try
            for (i, ε) in enumerate(epsilons)
                perturb!(comps, base, dirs, +ε); Jp[i] = J()
                perturb!(comps, base, dirs, -ε); Jm[i] = J()
                fd[i] = (Jp[i] - Jm[i]) / 2ε
                rel = abs(fd[i] - ad) / max(abs(fd[i]), abs(ad), eps(Float64))
                @info @sprintf("    eps = %8.1e   FD = %+.12e   AD = %+.12e   rel. err = %.3e", ε, fd[i], ad, rel)
                @info @sprintf("        (J+ = %+.16e, J- = %+.16e)", Jp[i], Jm[i])
            end
        finally
            restore!(comps, base)
        end
        rel_err = abs.(fd .- ad) ./ max.(abs.(fd), abs(ad), eps(Float64))
        return (; ad, fd, rel_err, J_plus = Jp, J_minus = Jm)
    end

    """
        run_fd_check(J, comps, checks) -> NamedTuple

    The FD check of the run's AD gradient: `checks.fd_random_directions` random
    directions over all of `comps`, then one direction confined to each input in
    `checks.fd_components`. Direction seeds are drawn from `checks.fd_seed`, so
    the check is reproducible; each printed seed reruns its direction. Returns
    the results table (see save_fd_check).
    """
    function run_fd_check(J, comps, checks; epsilons = FD_EPSILONS)
        selected = select_fd_components(comps, checks.fd_components)
        cases = vcat([("random (all inputs)", comps) for _ in 1:checks.fd_random_directions],
                     [(c.name, [c]) for c in selected])
        seeds = rand(Xoshiro(checks.fd_seed), 1:typemax(Int32), length(cases))

        @info "FD check: $(length(cases)) directions × $(length(epsilons)) ε × 2 forward windows"
        for c in comps
            g = c.gradient()
            @info @sprintf("    %-16s scale %.3e   max |∂J/∂·| on wet entries %.3e   (%d wet entries)",
                           c.name, fd_scale(c), maximum(abs, g[c.mask]; init = 0.0), count(c.mask))
        end

        results = map(zip(cases, seeds)) do ((label, cs), seed)
            @info "FD check: direction $label, seed $seed"
            (; label, seed, fd_check_direction(J, cs, fd_direction(cs, seed); epsilons)...)
        end

        @info "FD check summary (best rel. err; 'falls' = the error falls monotonically from the largest ε to the best one):"
        for r in results
            i = argmin(r.rel_err)
            falls = i > 1 && issorted(r.rel_err[1:i]; rev = true)
            @info @sprintf("    %-22s seed %10d   AD %+.6e   best rel. err %.2e at eps = %.0e   %s",
                           r.label, r.seed, r.ad, r.rel_err[i], epsilons[i], falls ? "falls" : "DOES NOT FALL: read the table")
        end
        return (; epsilons = collect(epsilons), results)
    end

    # The FD check's table, with the run's config and provenance, as
    # fd_check.jld2 in the run directory.
    function save_fd_check(filename, fd)
        rs = fd.results
        jldsave(filename;
                fd_epsilons     = fd.epsilons,
                fd_labels       = [r.label for r in rs],
                fd_seeds        = [r.seed for r in rs],
                fd_ad           = [r.ad for r in rs],
                fd_fd           = reduce(hcat, [r.fd for r in rs]),        # (ε, direction)
                fd_rel_err      = reduce(hcat, [r.rel_err for r in rs]),
                fd_J_plus       = reduce(hcat, [r.J_plus for r in rs]),
                fd_J_minus      = reduce(hcat, [r.J_minus for r in rs]),
                fd_scale_overrides = FD_SCALE_OVERRIDES,
                config_toml     = RUN_CONFIG.toml,
                run_provenance()...)
        return nothing
    end

end # include guard
