# abernathey_channel.jl
#
# Reentrant-channel model (Oceananigans, HydrostaticFreeSurfaceModel on a
# Reactant architecture) with reverse-mode AD (Enzyme) through the compiled
# time stepping: a platform for testing automatic differentiation of the model
# and of what is added to it (e.g. OceanBioME biogeochemistry).
#
# What a run does:
#   1. Build the model from the run config (run_config.jl): grid, closures
#      (scalar mixing or CATKE, optional constant-κ GM/Redi), forcing, and the
#      optional biogeochemistry (see `biogeochemistry` below).
#   2. Spin up for Nspinup steps from the analytic initial condition (or load a
#      saved spun-up state), and capture the state: the window's initial
#      condition (ModelState).
#   3. Run the estimate window (Ntimesteps steps) and differentiate the
#      objective J (estimate_tracer_error) with respect to the model (fields
#      inside it, e.g. κ), the forcing / BCs (ModelInputs) and the window's
#      initial state (ModelState): Duplicated arguments, whose shadows are
#      saved as sensitivity fields.
#   4. Verify the gradient ([checks], ad_checks.jl): the determinism probe, the
#      forward-mode check and the finite-difference check.
#
# Output (runs/<config name>/): data_init.jld2 (the window's initial
# state, forcing, metadata), data_final.jld2 (the end state, J and every
# sensitivity), forward_check.jld2 (forward vs reverse mode, with [checks]
# forward_check), fd_check.jld2 (the FD table, with [checks] fd_check), and
# config.toml. Plot with scripts/plot_channel.jl.
#
#   export XLA_FLAGS="--xla_backend_extra_options=xla_cpu_use_multi_output_fusion"
#   julia -O0 --project scripts/abernathey_channel.jl configs/<name>.toml
#

# THE RUN CONFIG (run_config.jl): this script's one argument is a config file
# (configs/<name>.toml), parsed and validated here, before anything is
# built. Every per-run setting comes from it; channel_common.jl binds the
# experiment ones ([spice], [kappa], the κ halo and vertical-mixing switches)
# from RUN_CONFIG too. Output goes to runs/<name>/.
include(joinpath(@__DIR__, "run_config.jl"))
const RUN_CONFIG = load_run_config(config_path_from_args(ARGS, relpath(@__FILE__)))
@info "Run config $(RUN_CONFIG.path) → $(RUN_CONFIG.dir)"

const Ntimesteps = RUN_CONFIG.run.Ntimesteps   # timesteps in the estimate window (the AD'ed part)
const Nspinup    = RUN_CONFIG.run.Nspinup      # timesteps of the spinup before it

graph_directory = RUN_CONFIG.dir

# Centered(order=2) advection is Float64 end-to-end and smooth. WENO computes its
# smoothness-indicator divisions in Float32 (see ConvertingDivision{Float32} in the
# model type), which puts a ~1e-7 relative noise floor on the primal and destroys
# the FD arm of a finite-difference comparison.
const use_smooth_advection = RUN_CONFIG.model.advection == "centered"

# GM/Redi ([model] gmredi): the IsopycnalSkewSymmetricDiffusivity closure with
# constant κ_skew / κ_symmetric ([kappa]) in the spinup and the window alike.
# Its κ fields are inputs of the differentiated function, so ∂J/∂κ is saved and
# the FD check can perturb them; it also stirs every tracer (biogeochemical
# ones included) along isopycnals, another AD pathway.
const use_gmredi = RUN_CONFIG.model.gmredi

# SPUN-UP STATE FROM FILE ([run] spunup_state). "": spin up for Nspinup steps. A
# path (relative to the working directory) to an earlier run's data_init.jld2:
# skip the spinup and its compile, and start the window from the exact state
# that run's window started from. A data_final.jld2 works too: the window then
# starts where that run's window ENDED. Nspinup must equal the steps the loaded
# state has run (the source's Nspinup, + its Ntimesteps for a data_final.jld2);
# a mismatch is an error, guarding against a stale path. The source must carry
# the same tracers as this run.
const spunup_state_path = RUN_CONFIG.run.spunup_state
const load_spunup_state = !isempty(spunup_state_path)

# [checks]: the determinism probe, the forward-mode check and the FD gradient
# check (ad_checks.jl, which says how to read them).
const run_determinism_probe  = RUN_CONFIG.checks.determinism_probe
const run_forward_mode_check = RUN_CONFIG.checks.forward_check
const run_fd_gradient_check  = RUN_CONFIG.checks.fd_check

# Horizontal resolution ([model] Nx, Ny). Nz (vertical resolution) is a
# campaign-wide constant in channel_common.jl.
const Nx = RUN_CONFIG.model.Nx
const Ny = RUN_CONFIG.model.Ny

const x_midpoint = Int(Nx / 2) + 1

# Shared constants, `parameters`, `ModelInputs` / `ModelState`, the grid / initial-
# and boundary-condition builders, and the spinup / window stepping:
include(joinpath(@__DIR__, "channel_common.jl"))

# The startup consistency checks of a loaded window state (validate_window_ic).
include(joinpath(@__DIR__, "window_ic_checks.jl"))

# The determinism probe and the FD gradient check.
include(joinpath(@__DIR__, "ad_checks.jl"))

#####
##### Spinup and window (channel_common.jl: spinup!, run_window!)
#####
# Both phases run with FixedKappa: κ as build_model wrote it, with its halos
# filled at the start of each run.
spinup_reentrant_channel_model!(model, inputs, state, Nspinup) =
    spinup!(model, inputs, state, Nspinup, FixedKappa())

run_reentrant_channel_model!(model, inputs, state, Ntimesteps) =
    run_window!(model, inputs, state, Ntimesteps, FixedKappa())

# THE OBJECTIVE ([objective] name): estimate_tracer_error runs the window and
# returns window_objective(OBJECTIVE, ...), resolved while tracing, so each
# compiled graph holds exactly one objective. Each method below says what it is.
const OBJECTIVE = Val(Symbol(RUN_CONFIG.objective.name))

# "default": midline zonal transport PLUS a stratification term PLUS a
# biogeochemical-tracer term.
#
#   J = J_transport + w_strat · J_strat [+ w_tracers · J_tracers]
#
# J_transport is the midline zonal transport (Sv). J_strat is built from the
# zonally-averaged buoyancy b̄(y,z) = ⟨α g T⟩ₓ — the stratification / isopycnal
# structure — reduced to a scalar via its squared magnitude, so ∂J/∂b̄ is smooth
# and Enzyme-friendly. It is computed from the same traced T-slicing / broadcast
# idiom used elsewhere, so it traces cleanly under Reactant.
#
# J_tracers is the same reduction for every OTHER tracer (all but T, S and
# CATKE's e: the ones a biogeochemistry adds), averaged over the (y, z) section
# so that its size does not grow with the grid. Without such tracers it is not
# compiled in at all (the tracer names are known while tracing), so J is exactly
# the two-term objective.
#
# J is a sensitivity functional, not a misfit: what matters for AD verification
# is that it is smooth and depends on every pathway. To make a term a misfit,
# subtract a reference field, e.g. replace `b_zonal` with
# `(b_zonal .- b_zonal_target)`, where b_zonal_target is a precomputed
# (1, Ny, Nz) device array carried in like the other inputs (a plain number
# would be baked in at compile time).
const w_strat   = 1.0
const w_tracers = 1.0

# The tracers J_tracers sees: everything but the physical ones.
other_tracer_names(model) = filter(n -> !(n in (:T, :S, :e)), keys(model.tracers))

function estimate_tracer_error(model, inputs, state, x_midpoint, Ntimesteps)
    run_reentrant_channel_model!(model, inputs, state, Ntimesteps)
    return window_objective(OBJECTIVE, model, inputs, x_midpoint)
end

function window_objective(::Val{:default}, model, inputs, x_midpoint)
    Nx, Ny, Nz = size(model.grid)

    # Fields are read through interior(...), never by range-indexing the Field:
    # since Oceananigans 0.117 `field[i, 1:Ny, 1:Nz]` drops the halo offset under
    # Reactant and reads the parent array (halo cells included), which made J
    # wrong and, through stale halos, different from one window to the next.

    # Term 1 — midline zonal transport (Sv):
    zonal_transport = (interior(model.velocities.u)[x_midpoint, :, :] .* model.grid.Δyᵃᶜᵃ) .* inputs.Δz
    J_transport = sum(zonal_transport) / 1e6

    # Term 2 — zonally-averaged stratification b̄(y,z) = ⟨α g T⟩ₓ, reduced to a
    # scalar via its squared magnitude:
    T = interior(model.tracers.T)
    b = (α * g) .* T                       # linear-EOS buoyancy (thermal part) while S is uniform; with the
                                           # spice anomaly on it is b + g β_S S′, so this sensitivity
                                           # objective then also sees spice (and hence κ_symmetric)
    b_zonal = sum(b, dims = 1) ./ Nx       # (1, Ny, Nz) zonal average
    J_strat = sum(b_zonal .^ 2)

    J = J_transport + w_strat * J_strat

    # Term 3 — the other tracers' zonal means, the same reduction, averaged over
    # the section. One term per tracer, resolved while tracing.
    names = other_tracer_names(model)
    if !isempty(names)
        J_tracers = sum(names) do name
            c_zonal = sum(interior(model.tracers[name]), dims = 1) ./ Nx
            sum(c_zonal .^ 2) / (Ny * Nz)
        end
        J = J + w_tracers * J_tracers
    end

    return J
end

# "surface_pco2" (stage 1 of the OceanBioME roadmap): the mean pCO₂ [μatm] over
# the wet top-layer cells at the end of the window, from DIC, Alk, T and S
# through OceanBioME's CarbonChemistry (surface_pCO₂!, channel_common.jl), so
# ∂J/∂(initial DIC, Alk, T, S, flow, forcing) runs through the H⁺ solve. The
# pCO₂ is written into the model's auxiliary field (objective_auxiliary_fields),
# zero on dry cells; the wet count is a constant of the grid (n_wet_surface).
# The grid is uniform in x and y, so the plain mean is the area mean.
function window_objective(::Val{:surface_pco2}, model, inputs, x_midpoint)
    pCO₂ = model.auxiliary_fields.pCO₂
    surface_pCO₂!(pCO₂, model)
    return sum(interior(pCO₂)) / n_wet_surface
end

# "co2_uptake" (stage 2): the mean air→sea CO₂ flux [mmol C m⁻² day⁻¹,
# positive into the ocean] over the wet top-layer cells at the end of the
# window. It is the flux DIC's gas-exchange boundary condition applies
# (surface_CO₂_uptake!, channel_common.jl, calls that condition's own
# function), from the surface DIC, Alk, T, S through the H⁺ solve; during the
# window the same flux has been changing DIC. Written into an auxiliary field
# like pCO₂, zero on dry cells.
function window_objective(::Val{:co2_uptake}, model, inputs, x_midpoint)
    uptake = model.auxiliary_fields.CO₂_uptake
    surface_CO₂_uptake!(uptake, model)
    return sum(interior(uptake)) / n_wet_surface
end

# The diagnostic Fields an objective writes inside the compiled window, handed to
# build_model as auxiliary_fields (so they are part of the model and its shadow).
objective_auxiliary_fields(::Val{:default}, grid) = NamedTuple()
objective_auxiliary_fields(::Val{:surface_pco2}, grid) = (; pCO₂ = Field{Center, Center, Nothing}(grid))
objective_auxiliary_fields(::Val{:co2_uptake}, grid) = (; CO₂_uptake = Field{Center, Center, Nothing}(grid))

function differentiate_tracer_error(model, inputs, state, dmodel, dinputs, dstate, x_midpoint, Ntimesteps)

    dedν = autodiff(set_strong_zero(Enzyme.ReverseWithPrimal),
                    estimate_tracer_error, Active,
                    Duplicated(model, dmodel),
                    Duplicated(inputs, dinputs),  # ← fixed forcing/BCs + Δz (∂J/∂forcing)
                    Duplicated(state, dstate),    # ← window-initial state (∂J/∂(tracers,u,v,w,η,U,V))
                    Const(x_midpoint), Const(Ntimesteps))

    return dedν
end

# FORWARD MODE (the forward-mode check, ad_checks.jl): the same arguments as
# differentiate_tracer_error, keep them in lockstep. The shadows are TANGENTS
# here: seeded with a direction v on the inputs (tstate, tinputs, tmodel's κ
# fields), they return J̇ = ∂J/∂p · v. Returns (J̇, J), derivative first.
function jvp_tracer_error(model, inputs, state, tmodel, tinputs, tstate, x_midpoint, Ntimesteps)

    return autodiff(set_strong_zero(Enzyme.ForwardWithPrimal),
                    estimate_tracer_error, Duplicated,
                    Duplicated(model, tmodel),
                    Duplicated(inputs, tinputs),
                    Duplicated(state, tstate),
                    Const(x_midpoint), Const(Ntimesteps))
end

#####
##### Actually creating our model and using these functions to run it:
#####

# Architecture ([model] architecture; see use_reactant in channel_common.jl)
architecture = use_reactant ? ReactantState() : CPU()

# Timestep size:
Δt₀ = RUN_CONFIG.model.dt   # [s], [model] dt

# Make the grid:
grid = make_grid(architecture, Nx, Ny, Nz, z_faces)

# BIOGEOCHEMISTRY ([model] biogeochemistry; build_biogeochemistry in
# channel_common.jl): `nothing`, or an Oceananigans AbstractBiogeochemistry built
# on `grid`. Its tracers are added to the model, carried in ModelState, restored
# at every window, saved (<name>_init, <name>_final, d<name>), seen by J
# (J_tracers) and perturbed by the FD check, with no further edit. Their initial
# values are set on the model right after build_model and copied into the
# analytic initial state below. Its parameters are plain numbers baked into the
# compiled graph, so they get no gradient (ad_checks.jl).
biogeochemistry = build_biogeochemistry(grid)

model         = build_model(grid, Δt₀, parameters, Nx, Ny; gmredi = use_gmredi, biogeochemistry,
                            auxiliary_fields = objective_auxiliary_fields(OBJECTIVE, grid))
set_biogeochemical_initial_conditions!(model)
T_flux        = T_flux_init(model.grid, parameters)
u_wind_stress = u_wind_stress_init(model.grid, parameters)
v_wind_stress = v_wind_stress_init(model.grid, parameters)
Tᵢ, Sᵢ        = temperature_salinity_init(model.grid, parameters)
Δzᶜ            = on_architecture(architecture, Δz) #zspacings(model.grid, Center()))

report_biogeochemical_state(model)

# The number of wet top-layer cells (the surface objectives' mean), from the
# same wet mask the AD checks use; a constant of the grid, baked into the
# compiled graph. check_surface_objective verifies it against the kernel's own
# dry test after the first evaluation.
const n_wet_surface = count(fd_wet_masks(model).c[:, :, Nz])
@info "Objective: $(RUN_CONFIG.objective.name)" *
      (OBJECTIVE isa Union{Val{:surface_pco2}, Val{:co2_uptake}} ? " (mean over $n_wet_surface wet surface cells)" : "")

# A surface objective's mean must be over the cells its kernel computed: the
# field's nonzero entries must be exactly the wet mask's top layer.
check_surface_objective(::Val, model) = nothing
check_surface_objective(::Val{:surface_pco2}, model) =
    check_surface_field(model.auxiliary_fields.pCO₂, "surface pCO₂", "μatm", model)
check_surface_objective(::Val{:co2_uptake}, model) =
    check_surface_field(model.auxiliary_fields.CO₂_uptake, "air→sea CO₂ flux", "mmol C m⁻² day⁻¹", model)

function check_surface_field(field, what, units, model)
    p = to_host(field)[:, :, 1]
    wet = fd_wet_masks(model).c[:, :, Nz]
    (p .!= 0) == wet ||
        error("$what: the kernel's wet cells ($(count(!=(0), p))) differ from the wet mask's ($(count(wet)))")
    @info @sprintf("%s at the end of the window: %.4g to %.4g %s (mean %.6g) over %d wet cells",
                   what, minimum(p[wet]), maximum(p[wet]), units, sum(p[wet]) / count(wet), count(wet))
    return nothing
end

# Fixed experiment configuration (never mutated after this point):
# The air–sea CO₂ exchange's controls (nothing without co2_gas_exchange):
# uniform u₁₀ and xCO₂ surface Fields, copied into the boundary condition at
# the start of every run.
model_inputs  = ModelInputs(u_wind_stress, v_wind_stress, T_flux,
                            co2_exchange_input(co2_exchange_wind_speed, model.grid),
                            co2_exchange_input(co2_air_mole_fraction, model.grid), Δzᶜ)

# Two prognostic snapshots, allocated now because @compile bakes in argument
# shapes and types: the analytic pre-spinup IC, and the post-spinup state that
# is the initial condition of the estimate window.
initial_state = allocate_model_state(model)
spunup_state  = allocate_model_state(model)

# The analytic IC: T and S from temperature_salinity_init; every other tracer
# (CATKE's e, biogeochemical tracers) as the model holds it after build_model.
for name in keys(initial_state.tracers)
    source = name === :T ? Tᵢ : name === :S ? Sᵢ : model.tracers[name]
    copyto!(initial_state.tracers[name], to_host(source))
end
capture_closure_state!(initial_state, model)   # the freshly built model's closure state (if any)

@info "Built $model."

# Shadows. model and model_inputs are built in ONE make_zero call so the primal
# aliasing between them (model_inputs' Fields share model.grid) is mirrored in
# the shadows; spunup_state holds bare arrays and shares nothing, so it is built
# separately. Enzyme ACCUMULATES (+=) into shadows, so they must be zero before
# an AD call: the run makes exactly one with these (the probe uses its own).
dmodel, dmodel_inputs = Enzyme.make_zero((model, model_inputs))
dspunup_state         = Enzyme.make_zero(spunup_state)

# The inputs the AD checks can perturb, and the [checks] directions_components
# names validated against them now, before the compile.
fd_inputs = fd_components(model, model_inputs, spunup_state, dmodel, dmodel_inputs, dspunup_state)
@info "Inputs the FD and forward-mode checks can perturb ([checks] directions_components): $(join([c.name for c in fd_inputs], ", "))"
(run_fd_gradient_check || run_forward_mode_check) && select_fd_components(fd_inputs, RUN_CONFIG.checks.directions_components)

# WINDOW IC FROM FILE, checked (validate_window_ic, window_ic_checks.jl: the source's
# config, the steps it has run, the code constants) and loaded BEFORE @compile, so
# a mismatch fails in seconds rather than after the multi-minute compile. Safe:
# the compiled functions take spunup_state as an argument, @compile bakes in its
# arrays' shapes and types but not their values, and the load only copies into
# those arrays. Without a file, the spinup after the compile fills it.
if load_spunup_state
    tic = time()
    consistency_report = validate_window_ic(spunup_state_path, RUN_CONFIG)
    spunup_source = load_spunup_state!(spunup_state, spunup_state_path)
    load_toc = time() - tic
    @show load_toc
end

# The r-prefixed functions are what the rest of the run calls: under Reactant
# the compiled graphs, on the CPU the plain functions themselves (Julia and
# Enzyme then compile on the first call, so that call is slow and compile_toc
# is ~0).
tic = time()
if use_reactant
    @info "Compiling the model run... (spinup, forward window, and its reverse-mode derivative)"
    # The spinup is only compiled when it will run (see spunup_state_path).
    if !load_spunup_state
        rspinup_reentrant_channel_model! = @compile raise_first=true raise=true sync=true  spinup_reentrant_channel_model!(model, model_inputs, initial_state, Nspinup)
    end
    restimate_tracer_error = @compile raise_first=true raise=true sync=true estimate_tracer_error(model, model_inputs, spunup_state, x_midpoint, Ntimesteps)
    rdifferentiate_tracer_error = @compile raise_first=true raise=true sync=true  differentiate_tracer_error(model, model_inputs, spunup_state,
                                                                                                             dmodel, dmodel_inputs, dspunup_state, x_midpoint, Ntimesteps)
    # The forward-mode function only when the forward-mode check will run (one
    # more compile). The run's shadows serve as the tangents' shapes and types;
    # @compile does not run it, so they stay zero.
    if run_forward_mode_check
        rjvp_tracer_error = @compile raise_first=true raise=true sync=true  jvp_tracer_error(model, model_inputs, spunup_state,
                                                                                             dmodel, dmodel_inputs, dspunup_state, x_midpoint, Ntimesteps)
    end
else
    @info "CPU architecture: no @compile; the functions are called directly."
    rspinup_reentrant_channel_model! = spinup_reentrant_channel_model!
    restimate_tracer_error           = estimate_tracer_error
    rdifferentiate_tracer_error      = differentiate_tracer_error
    rjvp_tracer_error                = jvp_tracer_error
end
compile_toc = time() - tic

@show compile_toc

@info "Running the simulation..."

filename = graph_directory * "data_init.jld2"

# (A state loaded from file was loaded before the compile above.)
if !load_spunup_state
    tic = time()
    @info "Spinup the model for $Nspinup timesteps, then snapshot the state:"
    rspinup_reentrant_channel_model!(model, model_inputs, initial_state, Nspinup)

    # Capture the full prognostic state — this is the initial condition of the
    # estimate window, so the window starts from a state that is explicit rather
    # than implicit in whatever the model object happens to hold.
    capture_model_state!(spunup_state, model)
    spunup_source = nothing
    spinup_toc = time() - tic
    @show spinup_toc
end

# The window's starting state must be finite, with non-negative biogeochemical
# tracers (check_window_state, channel_common.jl): otherwise stop here, before
# anything is saved or differentiated.
check_window_state(spunup_state; origin = load_spunup_state ? spunup_state_path : "the $Nspinup-step spinup")

# Per-run metadata (the values that live in this script, not channel_common.jl).
# Δt₀ is the configured timestep; the actual value used is also saved as
# clock_last_dt by common_save_metadata. The κ values are in common_save_metadata.
run_metadata = (; run_type = "channel_ad",
                  architecture = RUN_CONFIG.model.architecture,
                  Ntimesteps, Nspinup,
                  dt = Δt₀,
                  advection = use_smooth_advection ? "centered" : "weno",
                  use_gmredi,
                  # Where the window IC came from ("" = spun up here) and how long
                  # the run that produced it spun up (-1 if unrecorded).
                  spunup_state_path,
                  consistency_report = load_spunup_state ? consistency_report : "",
                  spinup_source_Nspinup = spunup_source === nothing ? Nspinup :
                                          coalesce(spunup_source.Nspinup, -1))

# The resolved config, for humans (config.toml in the run directory); the output
# files carry it as config_toml.
write_run_config(RUN_CONFIG)

# state = spunup_state: the window's full starting state goes into
# data_init.jld2, so a later run can start from this file (spunup_state_path).
save_init!(filename, model, dmodel, u_wind_stress, v_wind_stress, T_flux;
           extra = merge(use_gmredi ? merge(run_metadata, gmredi_save_fields(model, dmodel)) : run_metadata,
                         co2_exchange_entries(model_inputs)),
           state = spunup_state)

# The objective at the current inputs, from the compiled forward function, and
# an AD call there with fresh shadows (the probe's; the run's shadows stay zero
# for the AD call below).
evaluate_J() = convert(Float64, restimate_tracer_error(model, model_inputs, spunup_state, x_midpoint, Ntimesteps))

function evaluate_J_with_fresh_gradient()
    dm, di = Enzyme.make_zero((model, model_inputs))
    ds     = Enzyme.make_zero(spunup_state)
    out = rdifferentiate_tracer_error(model, model_inputs, spunup_state, dm, di, ds, x_midpoint, Ntimesteps)
    return convert(Float64, out[2])
end

# DETERMINISM PROBE: after save_init! and before the run's AD call, so neither
# the saved state nor the run's shadows see it (it leaves every input bitwise as
# it found it).
J_probe = run_determinism_probe ? determinism_probe(evaluate_J, evaluate_J_with_fresh_gradient, fd_inputs) : nothing

@info "Computing the AD gradient (shadows are accumulated in-place):"
dedν = rdifferentiate_tracer_error(model, model_inputs, spunup_state,
                                   dmodel, dmodel_inputs, dspunup_state, x_midpoint, Ntimesteps)

@show dedν

# The model now holds the end of the window, including the objective's
# diagnostic fields: for a surface objective, check the mean was over the right cells.
check_surface_objective(OBJECTIVE, model)

# NaN derivatives on dry (immersed) entries — CPU Enzyme gives them in the
# velocity shadows' wall entries — are set to 0 here, before anything saves or
# reads the shadows; a NaN on a wet entry is an error. See the DISCLAIMER in
# ad_checks.jl. The report is saved as dry_nans_zeroed ("" = none).
dry_nan_report = zero_dry_nans!(model, dmodel, dmodel_inputs, dspunup_state)

# The forward and reverse functions are separately compiled graphs, so their J
# agree to round-off, not bitwise.
J_probe === nothing ||
    @info @sprintf("J: AD call %.16e, forward call %.16e (relative difference %.1e)",
                   dedν[2], J_probe, abs(dedν[2] - J_probe) / abs(J_probe))

# With GM/Redi on, data_final.jld2 also carries the ISSD κ fields + shadows
# (gmredi_save_fields, channel_common.jl). The window never rewrites κ, so
# dkappa_* are ∂J/∂κ accumulated over the window.
filename = graph_directory * "data_final.jld2"
save_final!(filename, model, dmodel, dmodel_inputs, dspunup_state, dedν;
            extra = merge(use_gmredi ? merge(run_metadata, gmredi_save_fields(model, dmodel)) : run_metadata,
                          co2_exchange_entries(dmodel_inputs; prefix = "d"),
                          (; dry_nans_zeroed = dry_nan_report)))

# One forward-mode call at the current inputs with FRESH tangent shadows (the
# run's reverse shadows stay as saved), seeded by `seed!` on the components
# built on them. Returns (J̇, J).
function evaluate_jvp(seed!)
    tm, ti = Enzyme.make_zero((model, model_inputs))   # one call, as for dmodel: mirrors the aliasing
    ts     = Enzyme.make_zero(spunup_state)
    seed!(fd_components(model, model_inputs, spunup_state, tm, ti, ts))
    out = rjvp_tracer_error(model, model_inputs, spunup_state, tm, ti, ts, x_midpoint, Ntimesteps)
    return convert(Float64, out[1]), convert(Float64, out[2])
end

# FORWARD-MODE CHECK of the gradient just saved (ad_checks.jl), along the FD
# check's directions; before the FD check, which costs ~10× more per direction.
# After save_final!: its windows move the model, never the saved files' content.
if run_forward_mode_check
    tic = time()
    forward_results = run_forward_check(evaluate_jvp, evaluate_J, fd_inputs, RUN_CONFIG.checks)
    save_forward_check(graph_directory * "forward_check.jld2", forward_results)
    forward_toc = time() - tic
    @show forward_toc
end

# FD CHECK of the gradient just saved (ad_checks.jl). After save_final!: its
# evaluations move the model, never the saved files' content.
if run_fd_gradient_check
    tic = time()
    fd_results = run_fd_check(evaluate_J, fd_inputs, RUN_CONFIG.checks)
    save_fd_check(graph_directory * "fd_check.jld2", fd_results)
    fd_toc = time() - tic
    @show fd_toc
end
