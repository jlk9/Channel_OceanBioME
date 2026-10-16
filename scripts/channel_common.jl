# channel_common.jl
#
# Shared definitions for the reentrant-channel driver abernathey_channel.jl:
# everything about the model that is not a per-run choice of the driver.
#
# Settings come in two places:
#   * The RUN CONFIG (run_config.jl): one TOML file per run, parsed by the driver
#     BEFORE it includes this file, as the global `RUN_CONFIG`. The run-config
#     block below binds this file's experiment and numerics constants from it
#     (the spice anomaly, the GM/Redi κ, the κ halo and vertical-mixing
#     switches).
#   * Campaign-wide constants: the rest of this file. Domain, Coriolis, EOS, the
#     forcing magnitudes in `parameters`, the scalar closure numerics and the
#     free-surface substeps. Identified by the git commit every output file
#     records, and checked against the source of a loaded window state at
#     startup (window_ic_checks.jl).

# Also here: the stretched vertical grid; the `parameters` NamedTuple (forcing /
# BC / sponge parameters); the grid, model and initial-/boundary-condition
# builders; the `ModelInputs` / `ModelState` bundles with state capture, restore
# and loading; the spinup / window stepping (start_run!, run_steps!); and the
# JLD2 save helpers with run provenance.
#
# What stays in the driver: the objective (`estimate_tracer_error` /
# `differentiate_tracer_error`) and the order of the run.
#
# Included as plain source (not a module), so these `const` globals are spliced
# into the driver's scope exactly as if written inline; the traced functions
# close over them. The driver `include`s this after its per-run settings:
# functions here read Nx, Ny and use_smooth_advection as globals. The float type
# (Oceananigans.defaults.FloatType = Float64) is set below, before anything is
# built.

using Printf
using Statistics
using Dates

using Oceananigans
using Oceananigans.Units
using Oceananigans.OutputReaders: FieldTimeSeries
using Oceananigans.Grids: xnode, ynode, znode
using Oceananigans.TurbulenceClosures: CATKEVerticalDiffusivity, HorizontalFormulation, IsopycnalSkewSymmetricDiffusivity
using Oceananigans.TimeSteppers: update_state!
using Oceananigans.BoundaryConditions: fill_halo_regions!

using SeawaterPolynomials

# Imported, not `using`: its many exports would share the driver's namespace.
import OceanBioME

# Dependency for Reactant + KernelAbstractions, even on CPU-only machines:
using CUDA

using Reactant
using Oceananigans.Architectures: ReactantState, on_architecture
#Reactant.set_default_backend("cpu")

using ReactantCore: Binomial

using Enzyme

Enzyme.API.strictAliasing!(false)

using FileIO, JLD2

import Pkg   # stdlib; only for the package versions in the save provenance

Oceananigans.defaults.FloatType = Float64

#Reactant.set_default_backend("cpu")

# Dependency-free channel geometry, shared with the host-side tools (plotting):
# the canonical domain/physical constants (LX_DEFAULT, …), the stretched
# vertical grid (`channel_z_faces`), and the immersed topography
# (`wall_function`, `ridge_function`, `numerical_bottom_height`). One source of
# truth so the files cannot silently disagree on the grid or wall shape. It is
# pure Julia, so including it here adds no dependency.
include(joinpath(@__DIR__, "channel_geometry.jl"))

#####
##### From the RUN CONFIG (run_config.jl): never set here
#####
# The driver parses its config into `RUN_CONFIG` before including this file.
# Change these in the config file, never here.

# The architecture ([model] architecture): Reactant (runs compiled with
# @compile, the window Binomial-checkpointed) or plain CPU (the same functions
# called directly; @trace loops run as ordinary Julia loops, so the window is
# NOT checkpointed and Enzyme tapes every step). The code paths differ only
# where an array is allocated (on_architecture) and the clock is restored
# (restore_clock!), and in the driver's compile block.
const use_reactant = RUN_CONFIG.model.architecture == "reactant"
use_reactant && @info "To specify the Reactant backend uncomment line 'Reactant.set_default_backend(\"cpu\")' in channel_common.jl"

# The biogeochemistry ([model] biogeochemistry): "none" or "npzd". Built by
# build_biogeochemistry, its initial conditions set by
# set_biogeochemical_initial_conditions! (see "Biogeochemistry" below).
const biogeochemistry_choice = RUN_CONFIG.model.biogeochemistry

# Salinity and the density-compensated spice anomaly S′(y, z) ([spice]): see
# spice_anomaly for the design. They enter `parameters` below (saved as
# forcing_parameters).
const S₀       = RUN_CONFIG.spice.S0   # reference salinity: initial value and S-sponge target [psu]
const ΔS_spice = RUN_CONFIG.spice.dS   # anomaly amplitude [psu]; 0 = off
const y_spice  = RUN_CONFIG.spice.y    # anomaly centre, meridional [m]
const L_spice  = RUN_CONFIG.spice.L    # anomaly Gaussian width σ, meridional [m]
const z_spice  = RUN_CONFIG.spice.z    # anomaly centre, vertical [m]
const H_spice  = RUN_CONFIG.spice.H    # anomaly Gaussian width σ, vertical [m]; Inf = depth-independent

# GM/Redi κ [m²/s] ([kappa] skew / symmetric): the build-time values of the two
# ISSD κ fields, constant in space and time. Nothing rewrites them (both phases
# run with FixedKappa), so the spinup and the window share them. Saved as
# kappa_skew_constant / kappa_symmetric_constant, and the fields themselves as
# kappa_skew / kappa_symmetric (gmredi_save_fields). Scalars: build_model hands
# them to set!, which also accepts a function or an array, should a spatially
# varying κ ever be wanted. NaN for a run without GM/Redi, which has no [kappa]
# and never reads them.
const κ_skew_constant      = RUN_CONFIG.kappa === nothing ? NaN : RUN_CONFIG.kappa.skew
const κ_symmetric_constant = RUN_CONFIG.kappa === nothing ? NaN : RUN_CONFIG.kappa.symmetric

# GM/Redi κ HALOS ([model] fill_kappa_halos). ISSD reads its κ fields as if they
# sat at (Center, Center, Face): its horizontal fluxes average κ over
# (i−1, i) × (k−1, k) (Oceananigans abstract_scalar_diffusivity_closure.jl,
# ℑxzᶠᵃᶠ / ℑyzᵃᶠᶠ), so the bottom layer reads the k = 0 halo and the periodic
# seam the i = 0 halo. Both `set!` forms used on κ fill the interior only
# (Fields/set!.jl), and nothing else fills these halos, so before 2026-10-06 they
# kept their construction value, 0: half κ in the bottom layer (~215 m thick at
# Nz = 32) and across the seam, a quarter in the bottom cell at the seam. true:
# every compiled run fills them after κ is written (fill_gmredi_κ_halos!; x
# periodic, y and z the default zero-gradient condition, i.e. the adjacent
# interior value). false: zero halos, reproducing runs made before the fix.
# Saved as kappa_halos_filled.
const fill_κ_halos = RUN_CONFIG.model.fill_kappa_halos

# VERTICAL MIXING ([model] vertical_mixing). Saved as `vertical_mixing`.
#   "catke": CATKE (CATKEVerticalDiffusivity, Oceananigans' default parameters)
#            plus the constant background κz / νz (campaign-wide constants
#            below). CATKE carries its own prognostic TKE tracer `e` and
#            step-to-step state (closure_state_fields).
#   "scalar_surface_taper": the constant background plus the surface-intensified
#            κz_add below, a cheap stand-in for a mixing scheme; no `e` tracer.
# Under Reactant at 40×80×32 (CPU, -O0 + XLA flag, 2026-10-07), a forward +
# reverse @compile with CATKE took ~5 min (297 s and 302 s; spinup loaded, so
# not compiled). The FD gradient check does not converge with Oceananigans'
# default CATKE over a 1-day window: CATKE has discontinuities (the stability
# function's jump at Ri = 0, the convective branch on the sign of N², the TKE
# floor), so J is not smooth. Scalar mixing passes it cleanly.
const use_catke = RUN_CONFIG.model.vertical_mixing == "catke"

#####
##### CAMPAIGN-WIDE CONSTANTS: from here down to the free-surface substeps
#####
# Fixed in code, not per run, and recorded in every output file
# (code_constants). A run that loads its window state from another run's file
# checks them against that file at startup (window_ic_checks.jl: warn; α, β_S and g
# abort).

#####
##### Vertical grid
#####

const Nz = 32

# Stretched vertical grid — the formula lives in channel_geometry.jl
# (`channel_z_faces`) so the host-side tools rebuild exactly the same one.
z_faces = channel_z_faces(Nz)

const Lx = LX_DEFAULT     # zonal domain length [m]      (shared: channel_geometry.jl)
const Ly = LY_DEFAULT     # meridional domain length [m] (shared: channel_geometry.jl)
const Lz = -z_faces[1]    # domain depth [m] = Σ Δz_center, set by the stretching formula

Δz = reshape(z_faces[2:end] .- z_faces[1:end-1], 1, :)

# Coriolis variables:
const f = -1e-4
const β = 1e-11

const halo_size = 4 #3 for non-immersed grid

# Other model parameters:
const α = ALPHA_DEFAULT   # [K⁻¹]  thermal expansion coefficient (shared: channel_geometry.jl)
const β_S = BETA_S_DEFAULT  # [psu⁻¹] haline contraction coefficient (shared: channel_geometry.jl);
                            # passed to the EOS explicitly, so the spice anomaly's T shift
                            # (β_S/α)·S′ uses exactly the model's value. (β is Coriolis.)
const g = G_DEFAULT       # [m/s²] gravitational acceleration    (shared: channel_geometry.jl)
const cᵖ = 3994.0   # [J/K]  heat capacity
const ρ = 999.8    # [kg/m³] reference density

parameters = (
    Ly = Ly,
    Lz = Lz,
    Qᵇ = 10 / (ρ * cᵖ) * α * g,            # buoyancy flux magnitude [m² s⁻³]
    Qᵀ = 10 / (ρ * cᵖ),                    # temperature flux magnitude
    y_shutoff = 5 / 6 * Ly,                # shutoff location for buoyancy flux [m]
    τ = 0.2 / ρ,                           # surface kinematic wind stress [m² s⁻²]
    μ = 1 / 30days,                      # bottom drag damping time-scale [s⁻¹]
    ΔB = 8 * α * g,                      # surface vertical buoyancy gradient [s⁻²]
    ΔT = 8,                              # surface vertical temperature gradient
    ΔT_front = 4.0,                      # meridional front amplitude [K] (half-range of tanh)
    L_front = 200kilometers,             # meridional front width [m]
    H = Lz,                              # domain depth [m]
    h = 1000.0,                          # exponential decay scale of stable stratification [m]
    y_sponge = 19 / 20 * Ly,               # southern boundary of sponge layer [m]
    λt = 7.0days,                        # relaxation time scale [s]
    # Salinity and the density-compensated spice anomaly: set in the experiment
    # settings block at the top of this file (see spice_anomaly).
    S₀ = S₀,
    ΔS_spice = ΔS_spice,
    y_spice = y_spice,
    L_spice = L_spice,
    z_spice = z_spice,
    H_spice = H_spice,
)

using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: CATKEMixingLength, CATKEEquation

# The vertical-mixing configuration as recorded in save files. (Files written
# before 2026-10-07, which lack the key, ran the scalar closure with κz_add
# tapered from the BOTTOM, a bug fixed then.)
const vertical_mixing = use_catke ? "catke" : "scalar_surface_taper"

# The CATKE closure. tke_time_step = nothing gives one TKE step per model step;
# a shorter TKE step would add a plain-Julia substep loop inside every traced
# time step. The variants below are the FD-check ablations (default = Oceananigans'
# parameters, minimum_tke = 1e-9, …; A removes the convective/entrainment
# branch; B removes the stability function's jump at Ri = 0 by using the stable
# coefficients on both sides; C = A + B, the smoothest; D moves the TKE floor).
# C is active: it made the 144-step FD check converge, though at 432 steps it
# still showed leftover jumps.
#catke_closure() = CATKEVerticalDiffusivity()

# A:
#catke_closure() = CATKEVerticalDiffusivity(; minimum_convective_buoyancy_flux = 1e-7)

# B:
ml₀, eq₀ = CATKEMixingLength(), CATKEEquation()
ml = CATKEMixingLength(; Cᵘⁿu = ml₀.Cˡᵒu, Cᵘⁿc = ml₀.Cˡᵒc, Cᵘⁿe = ml₀.Cˡᵒe)
eq = CATKEEquation(; CᵘⁿD = eq₀.CˡᵒD)
#catke_closure() = CATKEVerticalDiffusivity(; mixing_length = ml, turbulent_kinetic_energy_equation = eq)

# C:
catke_closure() = CATKEVerticalDiffusivity(; minimum_convective_buoyancy_flux = 1e-7,
                           mixing_length = ml, turbulent_kinetic_energy_equation = eq)

# D: small floor
#catke_closure() = CATKEVerticalDiffusivity(; minimum_tke = 1e-12)

# D: high floor
#catke_closure() = CATKEVerticalDiffusivity(; minimum_tke = 1e-6)

# CATKE's full parameter listing, saved (and checked against a loaded window
# state's source) so that a parameter change cannot slip through as
# "catke" = "catke" ("" in scalar mode).
vertical_mixing_details() = use_catke ? sprint(show, catke_closure()) : ""

# Scalar (non-GM/Redi) closure viscosities / diffusivities, used in build_model.
# Lifted to file scope so they can be recorded verbatim in the save metadata.
# κz and νz are the background vertical mixing in both modes. Without CATKE, κz
# gets an extra, exponentially tapered surface contribution: κz_add at the top
# cell, decaying downward over decay_scale layers (k = Nz is the top cell; the
# stretched grid's top layers are ~10–15 m thick).
const κh = 5e-5          # [m²/s] horizontal diffusivity
const νh = 500           # [m²/s] horizontal viscosity
const κz = 5e-5          # [m²/s] background vertical diffusivity
const νz = 3e-3          # [m²/s] background vertical viscosity
const κz_add = 5e-5      # [m²/s] extra vertical diffusivity at the surface (tapered; scalar mode only)
const decay_scale = 5    # [layers] e-folding scale of the κz surface taper (scalar mode only)

# Number of barotropic substeps in the SplitExplicitFreeSurface. Like the
# viscosities above, a numeric choice recorded verbatim in the save metadata
# rather than a literal buried in build_model.
const free_surface_substeps = 40

#####
##### Grid geometry
#####
# `wall_function` / `ridge_function` live in channel_geometry.jl (included above),
# parameterized by (Lx, Ly, Lz). `make_grid` binds this run's globals into the
# 2-argument form Oceananigans' `set!` expects.

function make_grid(architecture, Nx, Ny, Nz, z_faces)

    underlying_grid = RectilinearGrid(architecture,
        topology = (Periodic, Bounded, Bounded),
        size = (Nx, Ny, Nz),
        halo = (halo_size, halo_size, halo_size),
        x = (0, Lx),
        y = (0, Ly),
        z = z_faces)

    # Make into a ridge array:
    ridge = Field{Center, Center, Nothing}(underlying_grid)
    set!(ridge, (x, y) -> wall_function(x, y, Lx, Ly, Lz))

    grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(ridge))
    return grid
end

#####
##### Model Construction:
#####

# `biogeochemistry`: an Oceananigans AbstractBiogeochemistry (e.g. an OceanBioME
# model built on `grid`), or `nothing`. Its required tracers are declared after
# (:T, :S) (model_tracer_names), and ModelState, the restore and the save files
# carry every tracer of the model, so they need no edit for it. Its boundary conditions
# and forcing come with it. What ModelState does NOT carry is any other state
# it keeps from step to step (e.g. sediment fields): see
# report_biogeochemical_state.
function build_model(grid, Δt₀, parameters, Nx, Ny; gmredi = false, biogeochemistry = nothing)

    temperature_flux_bc = FluxBoundaryCondition(Field{Center, Center, Nothing}(grid))

    u_stress_bc = FluxBoundaryCondition(Field{Face, Center, Nothing}(grid))
    v_stress_bc = FluxBoundaryCondition(Field{Center, Face, Nothing}(grid))

    @inline u_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * p.Lz * model_fields.u[i, j, 1]
    @inline v_drag(i, j, grid, clock, model_fields, p) = @inbounds -p.μ * p.Lz * model_fields.v[i, j, 1]

    u_drag_bc = FluxBoundaryCondition(u_drag, discrete_form = true, parameters = parameters)
    v_drag_bc = FluxBoundaryCondition(v_drag, discrete_form = true, parameters = parameters)

    T_bcs = FieldBoundaryConditions(top = temperature_flux_bc)

    u_bcs = FieldBoundaryConditions(top = u_stress_bc, bottom = u_drag_bc)
    v_bcs = FieldBoundaryConditions(top = v_stress_bc, bottom = v_drag_bc)

    #####
    ##### Coriolis
    #####
    coriolis = BetaPlane(f₀ = f, β = β)

    #####
    ##### Forcing and initial condition
    #####
    @inline initial_temperature(z, p) = p.ΔT * (exp(z / p.h) - exp(-p.Lz / p.h)) / (1 - exp(-p.Lz / p.h))
    @inline mask(y, p)                = max(0.0, y - p.y_sponge) / (Ly - p.y_sponge)

    @inline function temperature_relaxation(i, j, k, grid, clock, model_fields, p)
        timescale = p.λt
        y = ynode(j, grid, Center())
        z = znode(k, grid, Center())
        target_T = initial_temperature(z, p)
        T = @inbounds model_fields.T[i, j, k]
    
        return -1 / timescale * mask(y, p) * (T - target_T)
    end
    
    FT = Forcing(temperature_relaxation, discrete_form = true, parameters = parameters)

    # S sponge: the T sponge's mask and timescale, relaxing S to S₀ (i.e. S′ → 0)
    # while T relaxes to its unchanged target. Then the sponge's buoyancy tendency
    # is -mask/λt · (b - g(α T_target - β_S S₀)) exactly as with uniform S, so
    # the spice anomaly leaves b's evolution unchanged here too. Attached only
    # while the anomaly is on: with S uniform it is a no-op in exact arithmetic,
    # but it would act on S's roundoff drift (~1e-12) and so break bitwise
    # reproduction of runs made before it existed.
    @inline function salinity_relaxation(i, j, k, grid, clock, model_fields, p)
        timescale = p.λt
        y = ynode(j, grid, Center())
        S = @inbounds model_fields.S[i, j, k]

        return -1 / timescale * mask(y, p) * (S - p.S₀)
    end

    FS = Forcing(salinity_relaxation, discrete_form = true, parameters = parameters)
    forcing = spice_on(parameters) ? (T = FT, S = FS) : (T = FT,)

    # closure (moderately elevating scalar visc/diff) — κh, νh, κz, νz, κz_add and
    # decay_scale are file-scope consts (campaign-wide constants, near the top of
    # this file), so they can be recorded in the save metadata.

    # Background vertical diffusivity: κz everywhere, plus (scalar mode only) the
    # surface-intensified κz_add. k = Nz is the TOP cell (z_faces ascend from
    # -Lz), so the taper counts layers down from k = Nz. Before 2026-10-07 it
    # counted from k = 1 and so sat at the bottom.
    # Kept as a Field in both modes, so ∂J/∂κz stays available as a diagnostic.
    κz_field = Field{Center, Center, Center}(grid)
    κz_array = fill(Float64(κz), Nx, Ny, Nz)

    if !use_catke
        for k in 1:Nz
            taper = exp(-(Nz - k) / decay_scale)
            κz_array[:, :, k] .+= κz_add * taper
        end
    end

    set!(κz_field, κz_array)

    horizontal_closure = HorizontalScalarDiffusivity(ν = νh, κ = κh)
    vertical_closure = VerticalScalarDiffusivity(ν = νz, κ = κz_field)

    biharmonic_closure = ScalarBiharmonicDiffusivity(HorizontalFormulation(), Oceananigans.defaults.FloatType;
                                                     ν = 1e11)

    # The GM/Redi κ, as Fields rather than numbers: a number would be baked into
    # the compiled graph, while a Field is an argument of the compiled functions,
    # so ∂J/∂κ lands in its shadow (saved as dkappa_skew / dkappa_symmetric) and
    # the FD check can perturb it. set! fills the interior only; the halos are
    # filled inside every compiled run (FixedKappa, fill_gmredi_κ_halos!).
    κ_skew_field      = Field{Center, Center, Center}(grid)
    κ_symmetric_field = Field{Center, Center, Center}(grid)

    @allowscalar set!(κ_skew_field, κ_skew_constant)
    @allowscalar set!(κ_symmetric_field, κ_symmetric_constant)

    closure = (horizontal_closure, vertical_closure)

    if gmredi
        gmredi_closure = IsopycnalSkewSymmetricDiffusivity(κ_skew=κ_skew_field, κ_symmetric=κ_symmetric_field)
        closure = (closure..., gmredi_closure)
    end

    # CATKE must come FIRST: Oceananigans' time_step_catke_equation! takes
    # first(model.closure). Everything else finds closures by type
    # (gmredi_closure, vertical_scalar_closure), so the order is free otherwise.
    use_catke && (closure = (catke_closure(), closure...))

    # Smooth, fully-Float64 advection for gradient verification; WENO for production.
    advection = use_smooth_advection ? Centered(order=2) : WENO(order=3)

    @info "Building a model..."

    model = HydrostaticFreeSurfaceModel(
        grid;
        free_surface = SplitExplicitFreeSurface(substeps=free_surface_substeps),
        momentum_advection = advection,
        tracer_advection = advection,
        #buoyancy = SeawaterBuoyancy(equation_of_state=SeawaterPolynomials.TEOS10EquationOfState(Oceananigans.defaults.FloatType)),
        buoyancy = SeawaterBuoyancy(equation_of_state=LinearEquationOfState(Oceananigans.defaults.FloatType,
                                                                            thermal_expansion=α, haline_contraction=β_S)),
        coriolis = coriolis,
        closure = closure,
        # CATKE adds its TKE tracer `e` itself (declaring it as well is an error);
        # without CATKE nothing uses an `e`, so none is carried. The
        # biogeochemistry's tracers are declared here (model_tracer_names).
        tracers = model_tracer_names(biogeochemistry),
        biogeochemistry = biogeochemistry,
        boundary_conditions = (T = T_bcs, u = u_bcs, v = v_bcs),
        forcing = forcing
    )

    model.clock.last_Δt = Δt₀

    return model
end

#####
##### Biogeochemistry ([model] biogeochemistry)
#####

# The tracers to declare: (:T, :S), then the biogeochemistry's other required
# tracers in its order. They must be declared: a biogeochemistry that requires
# one of ours (NPZD's growth reads T) refuses a model that declares only some
# of its tracers, rather than adding the rest itself.
model_tracer_names(::Nothing) = (:T, :S)
model_tracer_names(bgc) =
    (:T, :S, filter(n -> !(n in (:T, :S)),
                    Oceananigans.Biogeochemistry.required_biogeochemical_tracers(bgc))...)

# The biogeochemistry for `grid`, or `nothing`. "npzd": OceanBioME's NPZD with
# all its defaults (Kuhn et al. 2015 parameters; two-band light under a constant
# surface PAR of 100 W/m², so no clock dependence; detritus sinking out of an
# open bottom; no sediment, no negative-tracer modifiers). Its parameters are
# plain numbers, baked into a compiled graph: no gradient, no FD perturbation.
build_biogeochemistry(grid) =
    biogeochemistry_choice == "npzd" ? OceanBioME.NPZD(grid) : nothing

# NPZD initial conditions [mmol N/m³], functions of depth only, smooth and
# STRICTLY POSITIVE: the two-band light model raises chlorophyll (from P) to a
# fractional power, which has no real derivative at P = 0 and is NaN below.
# Nutrient-poor surface over a nutrient-rich interior; plankton and detritus
# surface-intensified over small positive backgrounds. Saved and checked as a
# code constant (biogeochemistry_initial_conditions).
const npzd_initial_conditions = (; N_surface = 2.0,  N_deep = 15.0, h_N = 300.0,   # nutrients; h in m
                                   P_background = 0.01, P_surface = 0.5, h_P = 50.0,  # phytoplankton
                                   Z_background = 0.01, Z_surface = 0.2, h_Z = 50.0,  # zooplankton
                                   D_background = 0.01, D_surface = 0.1, h_D = 200.0) # detritus

# Set the biogeochemical tracers' initial values on `model`, right after
# build_model (the driver then copies them into the analytic initial state).
set_biogeochemical_initial_conditions!(model) = set_biogeochemical_initial_conditions!(model, Val(Symbol(biogeochemistry_choice)))
set_biogeochemical_initial_conditions!(model, ::Val{:none}) = nothing

function set_biogeochemical_initial_conditions!(model, ::Val{:npzd})
    c = npzd_initial_conditions
    Nᵢ(x, y, z) = c.N_deep + (c.N_surface - c.N_deep) * exp(z / c.h_N)
    Pᵢ(x, y, z) = c.P_background + c.P_surface * exp(z / c.h_P)
    Zᵢ(x, y, z) = c.Z_background + c.Z_surface * exp(z / c.h_Z)
    Dᵢ(x, y, z) = c.D_background + c.D_surface * exp(z / c.h_D)
    @allowscalar set!(model, N = Nᵢ, P = Pᵢ, Z = Zᵢ, D = Dᵢ)
    return nothing
end

biogeochemistry_initial_conditions() = biogeochemistry_choice == "npzd" ? sprint(show, npzd_initial_conditions) : ""

#####
##### Closure accessors: find a closure by TYPE, not by tuple position
#####
# The order of the closure tuple is not stable: CATKE must come FIRST
# (Oceananigans' time_step_catke_equation! takes first(model.closure)), and the
# biharmonic closure may come back. Lookup by type survives any reordering.
#
# Works identically on a model and on its Enzyme shadow: make_zero preserves
# types, so dmodel.closure has the same element types and the same lookup returns
# the shadow closure, whose κ fields hold the adjoints. Always look up the shadow
# with these functions too, never by position.
#
# Resolved while tracing: plain Julia dispatch on a tuple of concrete types, so the
# compiled graph is identical to the one indexing by position gave.

# All closures in `closures` of type C, as a tuple (inferrable, no allocation).
_closures_of(::Type{C}) where C = ()
_closures_of(::Type{C}, c::C, rest...) where C = (c, _closures_of(C, rest...)...)
_closures_of(::Type{C}, c, rest...) where C     = _closures_of(C, rest...)

# The unique closure of type C in model.closure, or `nothing`. Two of one type is
# an error: the caller could not know which one it got.
function find_closure(model, ::Type{C}) where C
    closures = model.closure isa Tuple ? model.closure : (model.closure,)
    found = _closures_of(C, closures...)
    length(found) > 1 &&
        error("model.closure holds $(length(found)) closures of type $C; expected at most one")
    return isempty(found) ? nothing : only(found)
end

# The GM/Redi (ISSD) closure. Required by its callers (gmredi_κ_host and the κ
# save fields), so its absence is an error rather than `nothing`.
# (Callers that work with or without GM/Redi use find_closure directly.)
function gmredi_closure(model)
    c = find_closure(model, IsopycnalSkewSymmetricDiffusivity)
    c === nothing &&
        error("model.closure has no IsopycnalSkewSymmetricDiffusivity (built with gmredi = false?)")
    return c
end

# The background vertical ScalarDiffusivity, or `nothing` (a CATKE configuration
# may drop it). VerticalScalarDiffusivity is Oceananigans' alias for
# ScalarDiffusivity{TD, VerticalFormulation}; CATKE is a different struct, so it
# can never match here.
vertical_scalar_closure(model) = find_closure(model, VerticalScalarDiffusivity)

# Both ISSD κ fields as host arrays.
function gmredi_κ_host(model)
    issd = gmredi_closure(model)
    return (skew = to_host(issd.κ_skew), symmetric = to_host(issd.κ_symmetric))
end

#####
##### Special initial and boundary conditions
#####

# Temperature flux:
function T_flux_init(grid, p)
    @inline temp_flux_function(x, y) = ifelse(y < p.y_shutoff, p.Qᵀ * cos(3π * y / p.Ly), 0.0)
    temp_flux = Field{Center, Center, Nothing}(grid)
    @allowscalar set!(temp_flux, temp_flux_function)
    return temp_flux
end

# wind stress:
function u_wind_stress_init(grid, p)
    @inline u_stress(x, y) = -p.τ * sin(π * y / p.Ly)
    wind_stress = Field{Face, Center, Nothing}(grid)
    @allowscalar set!(wind_stress, u_stress)
    return wind_stress
end

function v_wind_stress_init(grid, p)
    wind_stress = Field{Center, Face, Nothing}(grid)
    @allowscalar set!(wind_stress, 0)
    return wind_stress
end

# Initial condition: exponential stratification PLUS a surface-intensified
# meridional front. The same at every resolution and with or without GM/Redi,
# so those runs are the SAME experiment (same forcing, same initial state). The
# front tilts the isopycnals so GM/Redi has something to act on from the very
# first timestep:
#   slope ≈ ∂yT / ∂zT ≈ (ΔT_front / 2 L_front) / (ΔT / h) ~ O(10⁻³),
# comfortably below the default FluxTapering max slope (10⁻²), so the fluxes are
# untapered and genuinely κ-proportional.
#
# The optional spice anomaly S′ (spice_anomaly) is added on top: S = S₀ + S′ and
# T = T_old + (β_S/α)·S′, so b = g(αT − β_S S) = g(α T_old − β_S S₀) is unchanged
# (to roundoff) and the dynamics are those of the uniform-S model. With
# ΔS_spice = 0 both fields are exactly the uniform-S IC.
function temperature_salinity_init(grid, p)
    # Adding some noise to temperature field:
    #ε(σ) = σ * randn()
    stratification(z) = p.ΔT * (exp(z / p.h) - exp(-Lz / p.h)) / (1 - exp(-Lz / p.h))
    front(y, z)       = 0.5 * p.ΔT_front * tanh((y - p.Ly / 2) / p.L_front) * exp(z / p.h)
    Tᵢ_function(x, y, z) = stratification(z) + front(y, z) + (β_S / α) * spice_anomaly(y, z, p) #+ ε(1e-8)
    Sᵢ_function(x, y, z) = p.S₀ + spice_anomaly(y, z, p)
    Tᵢ = Field{Center, Center, Center}(grid)
    Sᵢ = Field{Center, Center, Center}(grid)
    @allowscalar set!(Tᵢ, Tᵢ_function)
    @allowscalar set!(Sᵢ, Sᵢ_function) # Initial Salinity
    return Tᵢ, Sᵢ
end

# DENSITY-COMPENSATED ("SPICE") SALINITY ANOMALY S′(y, z) [psu]: a zonally
# uniform Gaussian tongue centred at (y_spice, z_spice) with widths
# (L_spice, H_spice), added to S and — as (β_S/α)·S′ — to T, so it carries no
# buoyancy. Under the linear EOS it is exactly passive: it gives Redi
# (κ_symmetric) something to act on, which uniform S never does (Redi mixes
# along isopycnals, so it cannot change b, and with uniform S its T flux
# cancels), without changing the flow. Its S sponge is built in build_model.
#
# Design (defaults in `parameters`; ΔS_spice = 0 turns it off):
#   * Redi needs along-isopycnal CURVATURE, not just a gradient — a uniform
#     gradient has no Redi tendency at constant κ — hence a tongue, not a front.
#     At κ = 1e3 and σ = 250 km it erodes by ~0.14 %/day, so a short spinup
#     leaves it nearly intact (a 1-year spinup removes roughly 40 %).
#   * Centred on the ACC gap and the T front (y = 800 km, ±2σ ≈ 300–1300 km).
#     It is ~6e-5 of its peak at the sponge (y = 1900 km) and ~0.6 % at the
#     southern wall, so neither the sponge nor the wall reshapes it much.
#   * At intermediate depth (z = -800 m, σ = 300 m ≈ 3 cells), ~3 % at the
#     surface: clear of the surface forcing and the slope-tapered layer, like
#     Antarctic Intermediate Water. Its vertical structure differs from N², so
#     GM acts on it along a second geometry. H_spice = Inf makes it
#     depth-independent, if full-depth coverage is wanted instead.
#   * Suggested ΔS_spice = -0.25 psu (cold–fresh, T′ = (β_S/α)·S′ ≈ -1.17 K).
#     Under the linear EOS the amplitude has no dynamical effect.
@inline spice_anomaly(y, z, p) = p.ΔS_spice * exp(-(y - p.y_spice)^2 / (2 * p.L_spice^2)) *
                                              exp(-(z - p.z_spice)^2 / (2 * p.H_spice^2))

spice_on(p) = p.ΔS_spice != 0

#####
##### Simulation inputs and model state
#####
# These two bundles split the arguments of estimate_tracer_error by LIFETIME:
#
#   ModelInputs — experiment configuration. Built once by the `*_init` functions
#     and fixed for the whole run: the surface wind stress / temperature flux
#     that are copied into the boundary-condition fields, plus the vertical grid
#     spacing the objective integrates with. Nothing here evolves.
#
#   ModelState — the prognostic snapshot. Produced BY running the model (the
#     spinup), restored at the top of every window, and it is exactly what
#     evolves. This mirrors what Oceananigans itself calls
#     `prognostic_state(model)` for a HydrostaticFreeSurfaceModel: tracers
#     (every one the model has: T, S, CATKE's e, biogeochemical tracers),
#     velocities, and the split-explicit free surface (η + barotropic U, V).
#
# Why two structs rather than one: the snapshot is captured and restored as a
# unit, so "what defines a window's initial condition" has exactly one answer.
# It also makes the Enzyme activity choice a single decision for the whole
# snapshot instead of a per-field accident — ModelState is Duplicated so that
# ∂J/∂(initial state) is available as a research diagnostic.
#
# Both are immutable, concretely-typed, parametric structs whose leaves are
# mutable array-likes (Fields / ConcreteRArray). Reactant traces them by walking
# the leaves and Enzyme's Duplicated / make_zero recurse into them exactly as
# they do for `model`, so one call builds each whole shadow tree:
#   dmodel_inputs = Enzyme.make_zero(model_inputs)
# Gradients accumulate into the leaf arrays in place, so immutability is no
# obstacle.
struct ModelInputs{US, VS, TF, DZ}
    u_wind_stress :: US
    v_wind_stress :: VS
    temp_flux     :: TF
    Δz            :: DZ
end

# The prognostic state of a HydrostaticFreeSurfaceModel with a SplitExplicit
# free surface. Held as PLAIN DEVICE ARRAYS (ConcreteRArray), sized to each
# field's interior, rather than as Fields: a Field carries its grid and its
# boundary conditions, which would alias `model`'s own grid/BC objects and put a
# false aliasing relationship into the Enzyme shadow trees. Bare arrays share
# nothing with the model, and writing them into the interior (set_interior!) is
# all a restore needs, with halos refilled by fill_state_halos! and the
# mandatory update_state! that follows a restore.
#
# `tracers` holds EVERY tracer of the model, keyed by name in the order of
# model.tracers: T and S, CATKE's TKE `e` when CATKE is on, and the tracers a
# biogeochemistry adds. Built from model.tracers, so a new tracer needs no edit
# here, in the restore or in the save files (saved as <name>_init /
# <name>_final, its sensitivity as d<name>).
#
# `w` is diagnosed from continuity rather than stepped, and ISSD's ϵ_R₃₃ /
# the tendency fields Gⁿ,G⁻ are recomputed — none of them need to be carried
# here (see set_model_state! and the euler-first-step note at run_steps!).
#
# `closure` is the state the turbulence closures carry from one step to the next
# (see "Closure state" below): `nothing` when no closure carries any (the
# scalar + GM/Redi configuration), else a NamedTuple keyed by closure type name,
# e.g. (CATKEVerticalDiffusivity = (Jᵇ = …, κu = …, …),). Unlike the fields
# above, these are FULL PARENT arrays, halos included, restored by copying the
# parent, so the restore is exact whatever halo-filling the closure's own update
# does.
#
# `time` is the model clock's time [s] at this state, as a 0-d device array.
# set_model_state! restores it, so the clock runs on from the end of the spinup
# into the window instead of restarting at 0. Nothing reads the time yet (the
# forcing is time-invariant), but a time-dependent forcing or biogeochemistry
# (e.g. a seasonal light cycle) will. It is an array, not a plain number, so it
# is an argument of the compiled functions rather than a constant baked in at
# @compile time. set_model_state! copies its VALUE into the clock's own number
# (set_traced!). The clock's tick! mutates that number in place, so sharing one
# object would also advance the saved start time.
struct ModelState{TR, UT, VT, WT, HT, BU, BV, CL, TM}
    tracers :: TR   # NamedTuple of every tracer's interior, keyed as model.tracers
    u :: UT
    v :: VT
    w :: WT
    η :: HT   # free-surface displacement
    U :: BU   # barotropic zonal velocity
    V :: BV   # barotropic meridional velocity
    closure :: CL   # closure state carried across steps, or nothing
    time :: TM      # clock time [s] at this state (0-d device array)
end

#####
##### Closure state: what the turbulence closures carry from one step to the next
#####
# A window's J is a pure function of its inputs (the window's initial state, the
# forcing, κ, …) only if EVERYTHING that carries over from one time step to the
# next is restored at the top of the window. For the closures that means their
# `closure_fields` entries that are not recomputed from the current state before
# use. Example: CATKE's running mean of the surface buoyancy flux Jᵇ, relaxed
# toward the new value each step. An unrestored one leaks the previous window
# (e.g. an FD evaluation) into the next. The determinism probe (ad_checks.jl)
# tests this.
#
# Each closure type is classified EXPLICITLY by `closure_state_fields(closure,
# closure_fields)`: `nothing` if it carries no state, else a NamedTuple of the
# Fields to capture and restore. Oceananigans' own `prognostic_state` is not used:
# it is the checkpointer's host-side, halo-and-BC file format, and its generic
# fallback would count ISSD's derived ϵ_R₃₃ as state. Its per-closure lists are
# the reference when classifying a new closure (e.g. prognostic_state(::
# CATKEClosureFields) in catke_vertical_diffusivity.jl).
#
# A closure that keeps closure_fields but has no method here is an ERROR, at
# allocate_model_state, before @compile: adding a stateful closure (CATKE,
# RiBasedVerticalDiffusivity, …) without classifying it would otherwise silently
# make J depend on the previous window.

# No closure_fields at all (scalar diffusivities, explicit-in-time ISSD).
closure_state_fields(closure, ::Nothing) = nothing

function closure_state_fields(closure, closure_fields)
    error("""
        $(nameof(typeof(closure))) keeps closure_fields $(keys(closure_fields)), and
        channel_common.jl does not know whether any of them carry state from one step
        to the next. Add a closure_state_fields method (see "Closure state" there):
        `nothing` if all are recomputed from the current state before use, else the
        fields to capture and restore. Oceananigans' prognostic_state for that
        closure's fields lists the candidates.""")
end

# ISSD: ϵ_R₃₃ is recomputed from the current tracers in update_state!
# (compute_tapered_R₃₃!), and κ is read live from the κ fields (written once,
# by build_model).
closure_state_fields(::IsopycnalSkewSymmetricDiffusivity, ::Nothing) = nothing
closure_state_fields(::IsopycnalSkewSymmetricDiffusivity, closure_fields) = nothing

# CATKE: the list Oceananigans' prognostic_state(::CATKEClosureFields) checkpoints.
# Jᵇ is a running mean of the surface buoyancy flux, relaxed toward the new value
# each step; previous_velocities feed the shear production; κu/κc/κe are carried
# too. Le is recomputed every TKE step, so it is not state. (The TKE itself, e,
# is a tracer and travels in ModelState.tracers.e.)
closure_state_fields(::CATKEVerticalDiffusivity, cf) =
    (; Jᵇ = cf.Jᵇ, u⁻ = cf.previous_velocities.u, v⁻ = cf.previous_velocities.v,
       κu = cf.κu, κc = cf.κc, κe = cf.κe)

# The closure-state Fields of `model`, keyed by closure type name, or `nothing`.
# Same keys and order as a ModelState's `closure` entry built from this model.
function model_closure_state_fields(model)
    closures = model.closure isa Tuple ? model.closure : (model.closure,)
    cfields  = model.closure isa Tuple ? model.closure_fields : (model.closure_fields,)
    entries  = Pair{Symbol,Any}[]
    for (c, cf) in zip(closures, cfields)
        s = closure_state_fields(c, cf)
        s === nothing || push!(entries, Symbol(nameof(typeof(c))) => s)
    end
    isempty(entries) && return nothing
    allunique(first.(entries)) ||
        error("two stateful closures of one type ($(first.(entries))); ModelState keys them by type name")
    return (; entries...)
end

# Host copies of a closure-state tree, for saving: Fields by their parent,
# device arrays as they are.
closure_state_host(::Nothing) = nothing
closure_state_host(tree::NamedTuple) =
    map(nt -> map(x -> convert(Array, x isa Field ? parent(x) : x), nt), tree)

# Copy the model's closure state into `state.closure` (host-side, between
# compiled calls, like capture_model_state!).
function capture_closure_state!(state, model)
    state.closure === nothing && return state
    fields = model_closure_state_fields(model)
    for name in keys(state.closure), k in keys(state.closure[name])
        copyto!(state.closure[name][k], convert(Array, parent(fields[name][k])))
    end
    return state
end

# Write `closure_state` into the model's closure fields, parents included. Runs
# INSIDE the compiled functions, from set_model_state!. Resolved while tracing:
# with no stateful closure it adds nothing to the graph.
function set_closure_state!(model, closure_state)
    closure_state === nothing && return nothing
    fields = model_closure_state_fields(model)
    for name in keys(closure_state), k in keys(closure_state[name])
        parent(fields[name][k]) .= closure_state[name][k]
    end
    return nothing
end

#####
##### Building / capturing / restoring the state snapshot
#####

# Biogeochemical state that ModelState does NOT carry. OceanBioME-style models
# keep auxiliary fields (e.g. the PAR light field), which Oceananigans
# recomputes in update_state! (update_biogeochemical_state!), and so the
# mandatory update_state! after a restore rebuilds them. But a biogeochemistry
# may also keep fields it STEPS (e.g. sediment tracers), which nothing here
# restores: J would then depend on the previous window. The driver logs the
# auxiliary fields once (this function), as a reminder to check them; the
# determinism probe (ad_checks.jl) catches the stepped kind.
function report_biogeochemical_state(model)
    model.biogeochemistry === nothing && return nothing
    aux = keys(Oceananigans.Biogeochemistry.biogeochemical_auxiliary_fields(model.biogeochemistry))
    @info "Biogeochemistry $(nameof(typeof(model.biogeochemistry))): tracers carried in ModelState: " *
          "$(keys(model.tracers)); auxiliary fields NOT carried (they must be recomputed by " *
          "update_state!, which the determinism probe tests): $(isempty(aux) ? "none" : aux)"
    return nothing
end

# Allocate a zeroed ModelState shaped like `model`'s prognostic fields. Called
# HOST-SIDE, before @compile, because the compiled functions bake in the shapes
# and types of their arguments. Each field's interior size already accounts for
# its location and topology (u is (Nx,Ny,Nz) on periodic x, v is (Nx,Ny+1,Nz),
# w is (Nx,Ny,Nz+1), η is (Nx,Ny,1)), so nothing is hardcoded. Closure state is
# sized to each field's parent (see ModelState). The time starts at 0, the
# start time of a run from the analytic initial condition.
function allocate_model_state(model)
    FT = Oceananigans.defaults.FloatType
    arch = Oceananigans.Architectures.architecture(model.grid)
    zeroed(field) = on_architecture(arch, zeros(FT, size(interior(field))...))
    zeroed_parent(field) = on_architecture(arch, zeros(FT, size(parent(field))...))

    cfields = model_closure_state_fields(model)
    closure = cfields === nothing ? nothing : map(nt -> map(zeroed_parent, nt), cfields)

    return ModelState(map(zeroed, model.tracers),
                      zeroed(model.velocities.u),
                      zeroed(model.velocities.v),
                      zeroed(model.velocities.w),
                      zeroed(model.free_surface.displacement),
                      zeroed(model.free_surface.barotropic_velocities.U),
                      zeroed(model.free_surface.barotropic_velocities.V),
                      closure,
                      on_architecture(arch, fill(zero(Float64))))
end

# Copy `model`'s current prognostic state into `state`, in place. Called
# host-side, ONCE, between compiled calls (after the spinup) — never inside a
# traced region: `to_host` is a device→host transfer and forces a sync. The
# host round-trip is deliberate; `copyto!(::ConcreteRArray, ::Array)` is the
# supported Reactant path back to the device. `to_host` reads through
# `interior`, which is 1-based for every field (η's window shows up in
# `axes(η)`, not in `axes(interior(η))`), so no offset handling is needed here.
function capture_model_state!(state, model)
    for name in keys(state.tracers)
        copyto!(state.tracers[name], to_host(model.tracers[name]))
    end
    copyto!(state.u, to_host(model.velocities.u))
    copyto!(state.v, to_host(model.velocities.v))
    copyto!(state.w, to_host(model.velocities.w))
    copyto!(state.η, to_host(model.free_surface.displacement))
    copyto!(state.U, to_host(model.free_surface.barotropic_velocities.U))
    copyto!(state.V, to_host(model.free_surface.barotropic_velocities.V))
    capture_closure_state!(state, model)
    copyto!(state.time, fill(convert(Float64, model.clock.time)))
    return state
end

#####
##### Starting a window from a saved state (no spinup)
#####
# Where the ModelState fields live in each kind of save file. Each tracer is
# saved under its name plus a suffix (T_init, S_init, e_init, …; T_final, …), and
# the names themselves as `tracer_names`.
const STATE_KEYS_INIT  = (u = "u_init", v = "v_init", w = "w_init", η = "ssh", U = "U_init",  V = "V_init")
const STATE_KEYS_FINAL = (u = "u",      v = "v",      w = "w",      η = "ssh", U = "U_final", V = "V_final")
const TRACER_KEY_SUFFIX = (data_init = "_init", data_final = "_final")
tracer_key(name, kind::Symbol) = string(name, TRACER_KEY_SUFFIX[kind])

# `(T_init = …, S_init = …, …)` for jldsave, from a NamedTuple of host arrays
# keyed by tracer name: `suffix` is "_init" or "_final", or prefix = "d" for
# the sensitivities (dT, dS, …).
tracer_entries(arrays::NamedTuple; prefix = "", suffix = "") =
    NamedTuple{Tuple(Symbol(prefix, name, suffix) for name in keys(arrays))}(values(arrays))

# ...and the closure state (ModelState.closure), written only when there is any.
const CLOSURE_STATE_KEY = (data_init = "closure_state_init", data_final = "closure_state_final")

# ...and the clock time (ModelState.time), a plain Float64 [s].
const STATE_TIME_KEY = (data_init = "time_init", data_final = "time_final")

# The closure state from an open save file into state.closure. The source run and
# this one must agree on which closures carry state: a window that starts without
# the source's closure state (or with state the source never had) is not the
# source's window.
function load_closure_state!(state, f, path, key)
    FT = Oceananigans.defaults.FloatType
    if state.closure === nothing
        key in keys(f) && error("""
            $path carries closure state ($key: $(keys(read(f, key)))), but this run's
            closures carry none: the source run used a different vertical-mixing closure.""")
        return state
    end
    key in keys(f) || error("""
        $path carries no closure state ($key), but this run's closures do
        ($(keys(state.closure))): the source run used a different closure, or predates
        closure-state saving. Spin up here instead (spunup_state_path = "").""")
    saved = read(f, key)
    for name in keys(state.closure)
        hasproperty(saved, name) ||
            error("$path: $key has no state for $name (it has $(keys(saved)))")
        for k in keys(state.closure[name])
            a, dest = getproperty(getproperty(saved, name), k), state.closure[name][k]
            size(a) == size(dest) ||
                error("$path: $key.$name.$k is $(size(a)) but this run expects $(size(dest))")
            copyto!(dest, Array{FT}(a))
        end
    end
    return state
end

"""
    load_spunup_state!(state, path) -> NamedTuple

Fill `state` (a ModelState from allocate_model_state) from an earlier run's
`data_init.jld2` (the state that run's window started from) or its
`data_final.jld2` (the state its window ended in): the prognostic fields, the
closure state and the clock time. Host-side, between compiled calls, with the
same `copyto!` capture_model_state! uses; the saved arrays are exact Float64
copies, so the window starts bitwise where the source's did.

Loading only, with the checks loading itself needs (the same tracers as this
run, every field present and shaped for this grid, the closure state matching
this run's closures, a saved clock time). Whether the source is CONSISTENT with
this run (its config, the code constants, the steps it has run) is checked
before, with the other startup checks (check_window_ic!, window_ic_checks.jl).
Returns the source's provenance for this run's metadata.
"""
function load_spunup_state!(state, path)
    isfile(path) || error("window IC: no such file: $path")
    FT = Oceananigans.defaults.FloatType

    kind, t₀, src_Nspinup = jldopen(path, "r") do f
        ks = keys(f)
        kind = "T_init"  in ks ? "data_init"  :
               "T_final" in ks ? "data_final" :
               error("$path holds no model state (neither T_init nor T_final)")
        table = kind == "data_init" ? STATE_KEYS_INIT : STATE_KEYS_FINAL

        # The source's tracers must be this run's: a missing one cannot be
        # loaded, and an extra one (e.g. a biogeochemical tracer, or CATKE's e,
        # that this run does not have) would be silently dropped. A file without
        # `tracer_names` is checked for this run's tracers only.
        mine = String[String(n) for n in keys(state.tracers)]
        if "tracer_names" in ks
            theirs = String.(read(f, "tracer_names"))
            Set(theirs) == Set(mine) || error("""
                $path holds the tracers $(theirs), but this run has $(mine): the source
                run had a different vertical mixing (CATKE's e) or biogeochemistry.""")
        end
        entries = vcat([(tracer_key(n, Symbol(kind)), state.tracers[n]) for n in keys(state.tracers)],
                       [(table[fld], getfield(state, fld)) for fld in keys(table)])
        absent = [k for (k, _) in entries if !(k in ks)]
        isempty(absent) || error("""
            $path lacks $(join(absent, ", ")): it was written by a run with a different
            configuration (e.g. e_init / e_final by a run without CATKE's TKE).""")
        for (k, dest) in entries
            a = read(f, k)
            size(a) == size(dest) ||
                error("$path: $k is $(size(a)) but this run expects $(size(dest)) — a different grid")
            copyto!(dest, Array{FT}(a))
        end
        load_closure_state!(state, f, path, CLOSURE_STATE_KEY[Symbol(kind)])
        time_key = STATE_TIME_KEY[Symbol(kind)]
        time_key in ks || error("$path has no $time_key (no saved clock time); re-run its source")
        kind, Float64(read(f, time_key)), ("Nspinup" in ks ? Int(read(f, "Nspinup")) : missing)
    end
    copyto!(state.time, fill(t₀))

    @info "Window starts from the saved state in $path ($kind of a run spun up for $src_Nspinup steps" *
          (kind == "data_final" ? ", i.e. where that run's window ENDED" : "") *
          "; clock time t₀ = $t₀ s); no spinup."
    return (; path, kind, Nspinup = src_Nspinup, t₀)
end

# Copy the fixed forcing / boundary-condition fields into the model's BC slots.
# Runs INSIDE the differentiated function (the BC fields are on the tape).
function set_forcing_and_bcs!(model, inputs)
    set!(model.velocities.u.boundary_conditions.top.condition, inputs.u_wind_stress)
    set!(model.velocities.v.boundary_conditions.top.condition, inputs.v_wind_stress)
    set!(model.tracers.T.boundary_conditions.top.condition, inputs.temp_flux)
    return nothing
end

# Write `a` (a bare, interior-sized, 1-based array) into `field`'s interior.
#
# This goes through `interior(field)` rather than `set!` because the
# split-explicit free-surface displacement is a WINDOWED field: it is built as
# `ZFaceField(grid; indices = (:, :, Nz+1))`, so although `size(η)` is (Nx,Ny,1),
# `axes(η)` is (1:Nx, 1:Ny, Nz+1:Nz+1). `set!(η, a)` lowers to `η .= a`, and the
# broadcast then throws
#     DimensionMismatch: destination axes (…, OneTo(1)) are not compatible with
#     source axes (…, 33:33)
# against any 1-based array. `axes(interior(f))` is 1-based for EVERY field, η
# included, so writing through it is uniform and needs no per-field special case.
# Interior-only, exactly like set!; halos are refilled by the mandatory
# update_state! that follows a restore.
set_interior!(field, a) = (interior(field) .= a; nothing)

# Restore the prognostic state — the window's initial condition. Runs INSIDE the
# differentiated function, in the slot the old `set_ics_and_bcs!` tracer set!s
# occupied, so the adjoints of the window-initial state land in the ModelState
# shadow exactly as ∂J/∂Tᵢ used to land in the ModelInputs shadow.
#
# CALLERS MUST FOLLOW THIS WITH update_state!(model). This writes the prognostic
# fields and refills their halos, but every DERIVED quantity is left stale — halos, hydrostatic
# pressure, w, the closure fields (ISSD's ϵ_R₃₃), and critically the momentum
# tendencies Gⁿ, which hydrostatic_ab2_step! consumes BEFORE the next
# update_state! runs. Without the refresh, the first step of a window advances
# the restored state using the *previous* window's momentum tendencies.
# Refill the halos of every field set_model_state! wrote.
#
# This is NOT optional and is NOT covered by the update_state! that follows.
# update_state! fills TRACER halos only — its own comment says "the halos of the
# free-surface variables and the horizontal velocities are filled within the
# time-stepping after the state evolution", i.e. inside ab2_step!, AFTER the
# velocity update. But update_state! also calls compute_momentum_tendencies!
# BEFORE that, so without this, Gⁿ is computed from restored interiors bordered
# by the PREVIOUS window's halos and the window is not reproducible. (Halos are
# a function of the interior plus the boundary conditions, so refilling them
# reconstructs exactly the halo state that was present at capture — which is why
# ModelState can store interiors and need not carry halos.)
function fill_state_halos!(model)
    fill_halo_regions!(model.tracers, model.clock, fields(model))
    fill_halo_regions!(model.velocities, model.clock, fields(model))
    fill_halo_regions!(model.free_surface.displacement)
    fill_halo_regions!(model.free_surface.barotropic_velocities.U)
    fill_halo_regions!(model.free_surface.barotropic_velocities.V)
    return nothing
end

# Fill the halos of every ISSD κ field (see fill_κ_halos). Call INSIDE the compiled
# functions, after κ is written and before the next time step: from FixedKappa's
# start hook, for κ written host-side (build_model, or the FD check's
# perturbations). A new place that writes κ must call it too. A no-op without
# the GM/Redi closure or with fill_κ_halos = false; both are resolved while
# tracing.
function fill_gmredi_κ_halos!(model)
    fill_κ_halos || return nothing
    issd = find_closure(model, IsopycnalSkewSymmetricDiffusivity)
    issd === nothing && return nothing
    fill_halo_regions!(issd.κ_skew)
    fill_halo_regions!(issd.κ_symmetric)
    return nothing
end

# Overwrite a traced number IN PLACE with the value of another (traced code
# only). The idiom Oceananigans' Reactant extension uses for the clock (tick!);
# see set_model_state! for why a fresh assignment is not enough.
set_traced!(x, value) = (x.mlir_data = value.mlir_data; nothing)

# Restore the clock for a run: the time from the state's 0-d array, and the
# iteration. Under Reactant IN PLACE (see set_model_state!); on the CPU the
# clock holds plain numbers and is simply reassigned.
#
# THE CPU ITERATION IS 1, NOT 0, DELIBERATELY. On the CPU, Oceananigans'
# time_step! runs maybe_prepare_first_time_step! when clock.iteration == 0, and
# its reconcile_state! recomputes the barotropic U, V from u, v
# (compute_barotropic_mode!), overwriting the U, V just restored from
# ModelState. J would then not depend on the restored U, V (∂J/∂U = ∂J/∂V = 0)
# and would be a different function from the Reactant one, whose time_step!
# override never calls it. Starting at 1 skips it, so both architectures run
# the same steps from the same restored state. Safe because start_run! already
# does what it is for (update_state! after the restore), build_model sets
# clock.last_Δt, and the first step is forward-Euler by `euler = true`. The
# iteration is a step counter nothing else reads. Re-check after Oceananigans
# upgrades (0.114.0: TimeSteppers.jl, maybe_prepare_first_time_step!).
function restore_clock!(::ReactantState, clock, time)
    set_traced!(clock.time, time[])
    set_traced!(clock.iteration, zero(clock.iteration))
    return nothing
end

function restore_clock!(::CPU, clock, time)
    clock.time = time[]
    clock.iteration = 1
    return nothing
end

function set_model_state!(model, state)
    # The clock first, so the halo fill below (which passes the clock to the
    # boundary conditions) and the κ policy's start hook see the restored time.
    # The time is the state's (continuous from the spinup into the window); the
    # iteration restarts, since it counts the steps of this run (from 0 under
    # Reactant, from 1 on the CPU: see restore_clock!). Neither is read by the
    # current configuration. The clock is not itself part of ModelState.
    #
    # Under Reactant the clock holds Reactant numbers, restored IN PLACE
    # (restore_clock!, via set_traced!), as Oceananigans' Reactant tick! updates it.
    # Assigning a new number (`model.clock.time = …`) computes correctly inside
    # the compiled function, but after a compiled AD call Reactant does not copy
    # the new object's value back to the host clock (after a forward call it
    # does), so the host clock would stay stale, e.g. the time_final saved after
    # the first AD call. Tested 2026-10-08: in place, the end time is copied back
    # after forward and AD calls alike, and the gradient flows through the
    # restored time. The Int iteration is copied back after forward calls only:
    # after an AD call the host keeps the restored 0. Nothing reads it.
    restore_clock!(Oceananigans.Architectures.architecture(model.grid), model.clock, state.time)

    # Every tracer, by name. A Julia loop over the (compile-time) tracer names:
    # one write per tracer, not repeated model work, so it is not an @trace loop.
    for name in keys(state.tracers)
        set_interior!(model.tracers[name], state.tracers[name])
    end
    set_interior!(model.velocities.u, state.u)
    set_interior!(model.velocities.v, state.v)
    set_interior!(model.velocities.w, state.w)
    set_interior!(model.free_surface.displacement, state.η)
    set_interior!(model.free_surface.barotropic_velocities.U, state.U)
    set_interior!(model.free_surface.barotropic_velocities.V, state.V)
    fill_state_halos!(model)
    set_closure_state!(model, state.closure)   # parents, halos included: no fill needed
    return nothing
end

#####
##### Running the model: the spinup and the estimate window
#####
# The driver runs the model through these functions, so the spinup and the
# window step identically by construction.
#
# What may differ between runs goes through a κ POLICY, never through branches
# here: the functions below must not ask which model is running. A new behavior
# is a new policy type with its own methods, and a run that needs a genuinely
# different protocol (output writers, time-averaging, no AD) gets its own loop
# rather than a flag in this one.
#
# A policy decides what happens to the GM/Redi κ at two points of a run:
#   kappa_at_start!(policy, model)   — after the state (and clock) restore,
#                                      before the update_state! that follows it;
#   kappa_after_step!(policy, model) — after every time_step!.
# Each run phase takes its own policy (today both use FixedKappa). Rules for
# policies:
#   * Time: read model.clock.time (traced, and continuous from the spinup into
#     the window: set_model_state! restores it).
#   * Parameters (κ values, scaling coefficients, …) must reach the compiled
#     functions as device arrays. Plain numbers would be baked in at @compile
#     time, so changing them would silently do nothing.
#   * A policy that carries κ from one step to the next (relaxation, running
#     means) needs that κ captured and restored with ModelState, like the
#     closure state. Not built.
#   * ORDERING: a policy that rewrites κ after a step must do so before the
#     next tracer-tendency computation, which reads it. In Oceananigans 0.113.5
#     and 0.114.0 that is compute_tracer_tendencies! inside
#     hydrostatic_ab2_step! (inside time_step!), not update_state!, so a κ
#     written in kappa_after_step! is the next step's. Re-check after
#     Oceananigans upgrades (FixedKappa does not depend on it: it never rewrites
#     κ).
abstract type KappaPolicy end

"""
    FixedKappa()

κ as already written into the ISSD fields host-side: build_model's constants
([kappa] skew / symmetric), or the FD check's perturbations of them. The start
hook fills the κ halos (`set!` writes the interior only, see fill_κ_halos);
nothing changes κ after that.
"""
struct FixedKappa <: KappaPolicy end
kappa_at_start!(::FixedKappa, model)   = fill_gmredi_κ_halos!(model)
kappa_after_step!(::FixedKappa, model) = nothing

# Start a run from `state`: the fixed forcing / BCs, the state (prognostic
# fields, closure state, clock time), the policy's κ, then the derived fields.
# INSIDE the compiled functions; when differentiated, the restore is on the tape,
# so ∂J/∂(initial state) lands in the ModelState shadow.
function start_run!(model, inputs, state, κ::KappaPolicy)
    set_forcing_and_bcs!(model, inputs)
    set_model_state!(model, state)
    kappa_at_start!(κ, model)

    # MANDATORY after set_model_state!. It writes the prognostic fields only;
    # halos, hydrostatic pressure, w, the closure fields and the momentum
    # tendencies Gⁿ are all still derived from the PREVIOUS state, and
    # hydrostatic_ab2_step! consumes Gⁿ before the next update_state! runs.
    update_state!(model)
    return nothing
end

# N time steps under the policy κ. `checkpointed` (a plain Bool, so resolved
# while tracing) selects Binomial checkpointing of the traced loop, for a run
# that is differentiated: the window always, and a spinup if it is ever put on
# the AD tape.
#
# EULER FIRST STEP. A run opens with a forward-Euler step (χ = -0.5), which
# zeroes the G⁻ term inside ab2_step!. G⁻ still holds the tendencies from
# whatever this model object did last, and under Reactant the automatic trigger
# is GONE: the Reactant extension's time_step! drops
# `euler |= (Δt != clock.last_Δt)` (it would make χ a TracedRNumber), so `euler`
# must be passed explicitly, as Oceananigans' own first_time_step! does for QAB2.
# After this step G⁻ has been overwritten from this run's own Gⁿ, which is why
# ModelState carries no G⁻. The first step is outside the @trace loop so the
# traced body stays uniform.
function run_steps!(model, N, κ::KappaPolicy; checkpointed::Bool)
    Δt = model.clock.last_Δt

    time_step!(model, Δt; euler = true)
    kappa_after_step!(κ, model)

    if checkpointed
        Nremaining = N - 1
        b = max(1, Int(floor(log2(max(Nremaining, 2)))))
        @trace mincut = true checkpointing = Binomial(b) track_numbers = false for i = 1:Nremaining
            time_step!(model, Δt)
            kappa_after_step!(κ, model)
        end
    else
        @trace mincut = true track_numbers = false for i = 2:N
            time_step!(model, Δt)
            kappa_after_step!(κ, model)
        end
    end
    return nothing
end

# The spinup: Nspinup steps from `state` (the analytic initial condition), not
# checkpointed, since no driver differentiates it today.
function spinup!(model, inputs, state, Nspinup, κ::KappaPolicy)
    start_run!(model, inputs, state, κ)
    run_steps!(model, Nspinup, κ; checkpointed = false)
    return nothing
end

# The estimate window: Ntimesteps steps from `state` (the spun-up state),
# checkpointed for the reverse pass.
function run_window!(model, inputs, state, Ntimesteps, κ::KappaPolicy)
    start_run!(model, inputs, state, κ)
    run_steps!(model, Ntimesteps, κ; checkpointed = true)
    return nothing
end

#####
##### Output (JLD2)
#####
# When the GM/Redi closure is on, the driver also saves the ISSD (GM/Redi)
# κ_skew / κ_symmetric fields and their shadows (see gmredi_save_fields). Those
# ride in via the optional `extra` NamedTuple, so the shared body only touches
# the background vertical closure (vertical_κ_shadow).
# `save_final!` writes a different file from `save_init!` — pass the two
# filenames at the call sites. `save_final!` additionally writes the ModelState
# shadow (the window-initial-state sensitivities), which is why it takes
# `dmodel_state`.

# field → host Array, for saving.
to_host(field) = convert(Array, interior(field))

# ISSD (GM/Redi) κ fields + shadows, for the `extra` of save_init! / save_final!.
# Recomputed at each save, since the shadows fill during the AD call. Only valid
# when the model was built with gmredi = true (gmredi_closure errors otherwise).
function gmredi_save_fields(model, dmodel)
    κ, dκ = gmredi_κ_host(model), gmredi_κ_host(dmodel)
    return (kappa_skew  = κ.skew,  kappa_symmetric  = κ.symmetric,
            dkappa_skew = dκ.skew, dkappa_symmetric = dκ.symmetric)
end

# Shadow of the background vertical diffusivity, as `<prefix>T` / `<prefix>S`
# entries for jldsave (e.g. dkappaT_init), or nothing without that closure.
# Taken by tracer NAME, not position. All tracers share ONE κz field
# (build_model), and make_zero mirrors that aliasing, so the two entries are the
# same array: the summed sensitivity to the shared κz through every tracer
# (biogeochemical ones included), not T's and S's separately. Both keys are kept
# for existing readers.
function vertical_κ_shadow(dmodel, suffix)
    v = vertical_scalar_closure(dmodel)
    v === nothing && return (;)
    return NamedTuple{(Symbol("dkappaT_", suffix), Symbol("dkappaS_", suffix))}(
        (to_host(v.κ.T), to_host(v.κ.S)))
end

# ModelState leaves are bare device arrays (no halo region, so no interior view);
# they convert with a plain Array(...) rather than to_host's interior() path.
state_to_host(a) = convert(Array, a)

# Best-effort provenance stamped into every save file. Everything here is
# optional: if git or Pkg is unavailable the run still saves ("unknown").
#
# Packages recorded by version, git-tree hash and tracked git revision. The tree
# hash is what identifies Reactant, which is tracked from its GitHub `main` (the
# version number does not change with every commit there).
const PROVENANCE_PACKAGES = ("Oceananigans", "OceanBioME", "Reactant", "Reactant_jll", "Enzyme", "Enzyme_jll",
                             "JLD2")

# Code provenance, captured ONCE, when this file is included: that is the code the
# run executes, even if the scripts are edited while it runs. git runs in the
# directory holding the scripts, not the working directory.
#   git_dirty / git_status — whether tracked files differ from git_commit, and which
#     (`git status --porcelain`, untracked files ignored: run directories are);
#   git_diff               — `git diff HEAD`: the exact edits to tracked files, i.e.
#     the per-run configuration typed into the scripts (and any Manifest change);
#   xla_flags              — XLA_FLAGS, which changes XLA's fusion and so the
#     floating-point evaluation order;
#   julia_version          — checked against a window IC source like xla_flags
#     (window_ic_checks.jl): a different Julia resolves different packages;
#   julia_opt_level        — the -O level (-O0 does not change device arithmetic).
function launch_provenance()
    dir = @__DIR__
    git(args...) = try
        readchomp(pipeline(`git -C $dir $args`; stderr = devnull))
    catch
        nothing
    end
    status = git("status", "--porcelain", "--untracked-files=no")

    packages = Pair{Symbol,String}[]
    deps = try
        Dict(p.name => p for p in values(Pkg.dependencies()))
    catch
        Dict{String,Any}()
    end
    for name in PROVENANCE_PACKAGES
        p   = get(deps, name, nothing)
        key = lowercase(name)
        push!(packages, Symbol(key, "_version")      => (p === nothing || p.version === nothing) ? "unknown" : string(p.version))
        push!(packages, Symbol(key, "_tree_hash")    => (p === nothing || p.tree_hash === nothing) ? "unknown" : string(p.tree_hash))
        push!(packages, Symbol(key, "_git_revision") => (p === nothing || p.git_revision === nothing) ? "" : string(p.git_revision))
    end

    return (; git_commit      = something(git("rev-parse", "HEAD"), "unknown"),
              git_dirty       = status === nothing ? missing : !isempty(status),
              git_status      = something(status, "unknown"),
              git_diff        = something(git("diff", "HEAD"), "unknown"),
              hostname        = gethostname(),
              julia_version   = string(VERSION),
              julia_opt_level = Int(Base.JLOptions().opt_level),
              xla_flags       = get(ENV, "XLA_FLAGS", ""),
              packages...)
end

const LAUNCH_PROVENANCE = launch_provenance()

# The Reactant backend the run computes on ("cpu", "cuda", …), read at save time,
# after the compiled calls have initialized it; "none" for a run that is not on
# Reactant ([model] architecture = "cpu").
reactant_backend() = !use_reactant ? "none" : try
    string(Reactant.XLA.platform_name(Reactant.XLA.default_backend()))
catch
    "unknown"
end

# Recomputed at each save so save_time reflects the actual write.
run_provenance() = (; LAUNCH_PROVENANCE...,
                      reactant_backend = reactant_backend(),
                      save_time        = string(Dates.now()))

# THE CODE-LEVEL CONSTANTS a run depends on, under their save-file keys: the
# ONE list of them. common_save_metadata writes them into every output file,
# and window_ic_checks.jl compares a run's live values with a loaded state's
# source's saved ones, entry by entry. A constant added here (e.g. a
# biogeochemical parameter set) is therefore saved and checked with no second
# edit. Config-level settings are not here: they are compared as config
# sections (run_config.jl). `parameters` is included whole; its spice fields
# come from the config.
code_constants() = (; coriolis_f              = f,
                      coriolis_beta           = β,
                      alpha                   = α,
                      beta_S                  = β_S,
                      g,
                      halo_size,
                      kappa_h                 = κh,
                      nu_h                    = νh,
                      kappa_z                 = κz,
                      nu_z                    = νz,
                      kappa_z_add             = κz_add,
                      decay_scale,
                      free_surface_substeps,
                      vertical_mixing_details = vertical_mixing_details(),   # CATKE's parameters ("" without)
                      biogeochemistry_initial_conditions = biogeochemistry_initial_conditions(),  # "" without
                      forcing_parameters      = parameters)

# Metadata common to init and final files: the code-level constants, a few
# config-derived values recorded for readers, the run's resolved config, the
# live model clock, and run provenance. Recomputed at each save because the
# clock advances between them. Keys are ASCII so downstream string-keyed readers
# (the plot scripts) stay simple, except tracer names, which are the model's.
function common_save_metadata(model)
    return (; code_constants()...,
              kappa_skew_constant      = κ_skew_constant,       # [kappa] skew (NaN without GM/Redi)
              kappa_symmetric_constant = κ_symmetric_constant,  # [kappa] symmetric
              kappa_halos_filled  = fill_κ_halos,     # [model] fill_kappa_halos
              vertical_mixing,                        # [model] vertical_mixing
              # The model's tracers, in order: each is saved as <name>_init /
              # <name>_final (and d<name>), and load_spunup_state! checks them.
              tracer_names        = String[String(n) for n in keys(model.tracers)],
              biogeochemistry     = model.biogeochemistry === nothing ? "" : string(nameof(typeof(model.biogeochemistry))),
              config_toml         = RUN_CONFIG.toml,  # the run's resolved config (run_config.jl)
              clock_time          = model.clock.time,
              clock_iteration     = model.clock.iteration,
              clock_last_dt       = model.clock.last_Δt,
              run_provenance()...)
end

# The window's full starting state as host arrays: from `state` (the
# spunup_state every window restores) when given, else from the model. Taking it
# from `state` is what keeps data_init.jld2 correct when the spinup was skipped
# and the model itself still holds the analytic pre-spinup IC.
function state_host_arrays(state, model)
    state !== nothing &&
        return (tracers = map(state_to_host, state.tracers),
                u = state_to_host(state.u), v = state_to_host(state.v), w = state_to_host(state.w),
                η = state_to_host(state.η), U = state_to_host(state.U), V = state_to_host(state.V),
                closure = closure_state_host(state.closure),
                time = only(state_to_host(state.time)))
    return (tracers = map(to_host, model.tracers),
            u = to_host(model.velocities.u), v = to_host(model.velocities.v), w = to_host(model.velocities.w),
            η = to_host(model.free_surface.displacement),
            U = to_host(model.free_surface.barotropic_velocities.U),
            V = to_host(model.free_surface.barotropic_velocities.V),
            closure = closure_state_host(model_closure_state_fields(model)),
            time = convert(Float64, model.clock.time))
end

# `(key = value,)` for jldsave, or nothing to splat when `value` is nothing: for
# entries only some configurations have (closure state), so files from
# configurations without them simply lack the key.
optional_entry(key, value) = value === nothing ? (;) : NamedTuple{(Symbol(key),)}((value,))
closure_state_entry(key, host) = optional_entry(key, host)

# `state`: pass spunup_state. data_init.jld2 then holds the COMPLETE window
# starting state (the ModelState fields: every tracer as <name>_init, plus
# closure_state_init when a closure carries state, and the clock time
# time_init), so a later run can start its window from this file with
# load_spunup_state! instead of spinning up.
function save_init!(filename, model, dmodel, u_wind_stress, v_wind_stress, T_flux;
                    extra = (;), state = nothing)
    mkpath(dirname(filename))
    s = state_host_arrays(state, model)

    if isa(model.grid, ImmersedBoundaryGrid)
        bottom_height = bottom_height_field(model.grid)
    else
        bottom_height = Field{Center, Center, Nothing}(model.grid)
        set!(bottom_height, -Lz)
    end

    jldsave(filename; Nx, Ny, Nz,
            # Grid / EOS metadata, so downstream tools (plot_channel.jl) can
            # rebuild the exact grid this run used instead of re-deriving it from
            # a duplicated copy of the formulas above.
            z_faces = collect(z_faces), Lx, Ly, Lz,   # (α, β_S, g: code_constants)
            bottom_height = to_host(bottom_height),
            tracer_entries(s.tracers; suffix = "_init")...,   # T_init, S_init, (e_init,) biogeochemical tracers
            ssh    = s.η,
            u_init = s.u,
            v_init = s.v,
            w_init = s.w,
            U_init = s.U,          # barotropic velocities of the split-explicit
            V_init = s.V,          # free surface (not derivable from u, v)
            closure_state_entry(CLOSURE_STATE_KEY.data_init, s.closure)...,
            time_init = s.time,    # clock time [s] at the window's start (ModelState.time)
            u_wind_stress = to_host(u_wind_stress),
            v_wind_stress = to_host(v_wind_stress),
            vertical_κ_shadow(dmodel, "init")...,
            T_flux = to_host(T_flux),
            common_save_metadata(model)...,
            extra...)
    return nothing
end

# `dmodel_state` carries the window-initial-state sensitivities
# ∂J/∂(tracers, u, v, w, η, U, V): the shadow of the Duplicated ModelState.
function save_final!(filename, model, dmodel, dmodel_inputs, dmodel_state, dedν; extra = (;))
    jldsave(filename; Nx, Ny, Nz,
            tracer_entries(map(to_host, model.tracers); suffix = "_final")...,   # T_final, S_final, …
            ssh = to_host(model.free_surface.displacement),
            u = to_host(model.velocities.u),
            v = to_host(model.velocities.v),
            w = to_host(model.velocities.w),
            # With U and V this file also holds a complete state (the window's
            # END), so load_spunup_state! can start a new window from it.
            U_final = to_host(model.free_surface.barotropic_velocities.U),
            V_final = to_host(model.free_surface.barotropic_velocities.V),
            time_final = convert(Float64, model.clock.time),   # clock time [s] at the window's end
            closure_state_entry(CLOSURE_STATE_KEY.data_final,
                                closure_state_host(model_closure_state_fields(model)))...,
            zonal_transport_and_buoyancy = convert(Float64, dedν[2]),
            du_wind_stress = to_host(dmodel_inputs.u_wind_stress),
            dv_wind_stress = to_host(dmodel_inputs.v_wind_stress),
            dT_flux = to_host(dmodel_inputs.temp_flux),
            # Sensitivity of the objective to the window's initial state
            # (dT, dS, (de,) d<biogeochemical tracer>, du, …):
            tracer_entries(map(state_to_host, dmodel_state.tracers); prefix = "d")...,
            du   = state_to_host(dmodel_state.u),
            dv   = state_to_host(dmodel_state.v),
            dw   = state_to_host(dmodel_state.w),
            dssh = state_to_host(dmodel_state.η),
            dU   = state_to_host(dmodel_state.U),
            dV   = state_to_host(dmodel_state.V),
            closure_state_entry("dclosure_state", closure_state_host(dmodel_state.closure))...,
            vertical_κ_shadow(dmodel, "final")...,
            common_save_metadata(model)...,
            extra...)
    return nothing
end
