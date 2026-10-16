# run_config.jl
#
# Run configuration: one TOML file per run, the only place a run's settings are
# written. The driver takes exactly one command-line argument, its config file:
#
#   julia -O0 --project scripts/abernathey_channel.jl configs/<name>.toml
#
# A config holds everything the run uses that is not a
# campaign-wide constant: [model] numerics, [run] step counts and spinup source,
# [spice] the salinity anomaly, [kappa] the constant GM/Redi κ, and [checks]
# the AD verification (determinism probe, finite-difference gradient check).
#
# ONE FACT, ONE PLACE:
#   * Every setting lives in a config file or in the code, never both. The code
#     holds no defaults for config keys: a missing key is an error (start from
#     the files in configs/).
#   * Configs hold independent inputs only. Derived values (x_midpoint, Lz, the
#     run directory) are computed in code.
#   * Unknown keys are errors, so a typo cannot pass as an unused setting.
#   * Campaign-wide constants (domain, Coriolis, EOS, forcing magnitudes, scalar
#     closure numerics) stay in code, recorded by the git commit in every output
#     file.
#
# Every run saves its resolved config as `config_toml` in its output files and
# as config.toml in its run directory, which is runs/<config file name>/.
# File names end in _spinup<Nspinup>_<Ntimesteps>steps.toml, checked against
# [run] (check_config_name). An existing run directory is overwritten.
#
# Values are SI (m, s, psu, m²/s). Pure Julia plus TOML and JLD2: it is
# included first, so a bad config fails before any model is built.

using TOML
using JLD2

if !@isdefined(_RUN_CONFIG_LOADED)
    _RUN_CONFIG_LOADED = true

    #####
    ##### Typed access with errors that name the file, table and key
    #####

    as_config_type(v, ::Type{Float64}, w) =
        (v isa Real && !(v isa Bool)) ? Float64(v) : error("$w: expected a number, got $(repr(v))")
    as_config_type(v, ::Type{Int}, w) =
        (v isa Integer && !(v isa Bool)) ? Int(v) : error("$w: expected an integer, got $(repr(v))")
    as_config_type(v, ::Type{Bool}, w) = v isa Bool ? v : error("$w: expected true or false, got $(repr(v))")
    as_config_type(v, ::Type{String}, w) =
        v isa AbstractString ? String(v) : error("$w: expected a string, got $(repr(v))")
    as_config_type(v, ::Type{Vector{String}}, w) =
        (v isa AbstractVector && all(x -> x isa AbstractString, v)) ? String[String(x) for x in v] :
        error("$w: expected a list of strings, got $(repr(v))")

    function config_value(d, key, T, w; check = nothing, why = "")
        haskey(d, key) || error("$w: missing required key `$key`")
        v = as_config_type(d[key], T, "$w.$key")
        check === nothing || check(v) || error("$w.$key = $(repr(v)): $why")
        return v
    end

    config_choice(d, key, options, w) =
        config_value(d, key, String, w; check = in(options), why = "must be one of $(join(repr.(options), ", "))")

    function config_table(d, key, w)
        haskey(d, key) || error("$w: missing required table [$key]")
        d[key] isa AbstractDict || error("$w: `$key` must be a table")
        return d[key]
    end

    function only_config_keys(d, allowed, w)
        extra = sort!(collect(setdiff(keys(d), allowed)))
        isempty(extra) ||
            error("$w: unknown key(s) $(join(extra, ", ")) (allowed here: $(join(sort(collect(allowed)), ", ")))")
        return nothing
    end

    #####
    ##### Sections
    #####

    # [model]: the numerics of a run. Nx even: the transport section sits at the
    # x-face Nx/2 + 1.
    function parse_model_config(d, w)
        keys_ = ("Nx", "Ny", "dt", "advection", "vertical_mixing", "fill_kappa_halos", "gmredi")
        only_config_keys(d, keys_, w)
        return (; Nx = config_value(d, "Nx", Int, w; check = n -> n ≥ 2 && iseven(n), why = "must be an even integer ≥ 2"),
                  Ny = config_value(d, "Ny", Int, w; check = ≥(2), why = "must be ≥ 2"),
                  dt = config_value(d, "dt", Float64, w; check = >(0), why = "must be > 0 (seconds)"),
                  advection = config_choice(d, "advection", ("centered", "weno"), w),
                  vertical_mixing = config_choice(d, "vertical_mixing", ("catke", "scalar_surface_taper"), w),
                  fill_kappa_halos = config_value(d, "fill_kappa_halos", Bool, w),
                  gmredi = config_value(d, "gmredi", Bool, w))
    end

    # [run]: step counts, and where the spinup comes from (spunup_state = ""
    # spins up in-process; a path loads a saved state).
    function parse_run_config(d, w)
        only_config_keys(d, ("Nspinup", "Ntimesteps", "spunup_state"), w)
        steps(k) = config_value(d, k, Int, w; check = ≥(1), why = "must be ≥ 1 (each loop's Euler step comes first)")
        return (; Nspinup = steps("Nspinup"), Ntimesteps = steps("Ntimesteps"),
                  spunup_state = config_value(d, "spunup_state", String, w))
    end

    # [spice]: the density-compensated salinity anomaly (spice_anomaly,
    # channel_common.jl). dS = 0 turns it off; H = inf makes it depth-independent.
    function parse_spice_config(d, w)
        only_config_keys(d, ("S0", "dS", "y", "L", "z", "H"), w)
        return (; S0 = config_value(d, "S0", Float64, w; check = >(0), why = "must be > 0 (psu)"),
                  dS = config_value(d, "dS", Float64, w),
                  y  = config_value(d, "y", Float64, w),
                  L  = config_value(d, "L", Float64, w; check = >(0), why = "must be > 0 (m)"),
                  z  = config_value(d, "z", Float64, w),
                  H  = config_value(d, "H", Float64, w; check = >(0), why = "must be > 0 (m), or inf"))
    end

    # [kappa]: the GM/Redi κ [m²/s] of each ISSD coefficient, constant in space
    # and time: the same value in the spinup and in the window.
    function parse_kappa_config(d, w)
        only_config_keys(d, ("skew", "symmetric"), w)
        κ(k) = config_value(d, k, Float64, w; check = >(0), why = "must be > 0 (m²/s)")
        return (; skew = κ("skew"), symmetric = κ("symmetric"))
    end

    # [checks]: the AD verification run after the model is built (ad_checks.jl).
    #   determinism_probe    — J is a pure function of its inputs (bitwise);
    #   fd_check             — compare the AD gradient with central finite
    #                          differences along random directions;
    #   fd_seed              — seeds every FD direction (reproducible);
    #   fd_random_directions — directions over ALL perturbable inputs at once;
    #   fd_components        — inputs to check one at a time, one direction
    #                          each (names checked against the model's at
    #                          startup; the log lists the available ones).
    function parse_checks_config(d, w)
        only_config_keys(d, ("determinism_probe", "fd_check", "fd_seed", "fd_random_directions", "fd_components"), w)
        checks = (; determinism_probe = config_value(d, "determinism_probe", Bool, w),
                    fd_check = config_value(d, "fd_check", Bool, w),
                    fd_seed = config_value(d, "fd_seed", Int, w),
                    fd_random_directions = config_value(d, "fd_random_directions", Int, w; check = ≥(0), why = "must be ≥ 0"),
                    fd_components = config_value(d, "fd_components", Vector{String}, w;
                                                 check = allunique, why = "names must not repeat"))
        checks.fd_check && checks.fd_random_directions == 0 && isempty(checks.fd_components) &&
            error("$w: fd_check = true but no directions (fd_random_directions = 0 and fd_components = [])")
        return checks
    end

    run_directory(config_path) = joinpath("runs", splitext(basename(config_path))[1]) * "/"

    # CONFIG FILE NAMES carry the step counts, `<name>_spinup<Nspinup>_<Ntimesteps>steps.toml`,
    # so runs that differ only in spinup or window length get distinct names and
    # run directories. The name restates two config values, so it is checked
    # against [run]: a mismatched or missing tag is an error, never a mislabelled
    # run.
    function check_config_name(path, run)
        m = match(r"_spinup(\d+)_(\d+)steps\.toml$", basename(path))
        tag = "_spinup$(run.Nspinup)_$(run.Ntimesteps)steps.toml"
        m === nothing && error("$path: config file names must end in _spinup<Nspinup>_<Ntimesteps>steps.toml; " *
                               "for this run's [run] that is …$tag")
        (parse(Int, m[1]), parse(Int, m[2])) == (run.Nspinup, run.Ntimesteps) ||
            error("$path: the file name says spinup $(m[1]), $(m[2]) steps, but the run's [run] has " *
                  "Nspinup = $(run.Nspinup), Ntimesteps = $(run.Ntimesteps); rename it to end in …$tag")
        return nothing
    end

    #####
    ##### Whole configs
    #####

    # A parsed config from its dictionary. [kappa] exists exactly when the run
    # has GM/Redi.
    function parse_config(d, w)
        only_config_keys(d, ("model", "run", "spice", "kappa", "checks"), w)
        model = parse_model_config(config_table(d, "model", w), "$w [model]")
        kappa = if model.gmredi
            parse_kappa_config(config_table(d, "kappa", w), "$w [kappa]")
        else
            haskey(d, "kappa") && error("$w: [kappa] given, but [model] gmredi = false")
            nothing
        end
        return (; model,
                  run    = parse_run_config(config_table(d, "run", w), "$w [run]"),
                  spice  = parse_spice_config(config_table(d, "spice", w), "$w [spice]"),
                  kappa,
                  checks = parse_checks_config(config_table(d, "checks", w), "$w [checks]"))
    end

    config_text(d) = sprint(io -> TOML.print(io, d; sorted = true))

    """
        load_run_config(path) -> NamedTuple

    The run config at `path`, validated: its sections, plus `path`, the run
    directory `dir` (runs/<file name>/) and `toml`, the resolved text to save
    with the run.
    """
    function load_run_config(path)
        isfile(path) || error("no such config file: $path")
        d = TOML.parsefile(path)
        cfg = parse_config(d, path)
        check_config_name(path, cfg.run)
        return merge(cfg, (; path, dir = run_directory(path), toml = config_text(d)))
    end

    #####
    ##### Command line
    #####

    """
        config_path_from_args(args, script) -> String

    The one command-line argument the driver takes: its config file.
    """
    function config_path_from_args(args, script)
        usage = "usage: julia -O0 --project $script <config.toml>"
        length(args) == 1 || error("$usage\ngive exactly one argument, the run's config file; got $(length(args)): $(join(args, " "))")
        endswith(args[1], ".toml") || error("$usage\nthe argument must be a .toml config file, got $(args[1])")
        return args[1]
    end

    # Write the resolved config into the run directory, for humans; the output
    # files carry it as `config_toml`.
    function write_run_config(cfg)
        mkpath(cfg.dir)
        write(joinpath(cfg.dir, "config.toml"), cfg.toml)
        return nothing
    end

end # include guard
