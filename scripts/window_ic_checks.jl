# window_ic_checks.jl
#
# The consistency checks between a run and the source of a LOADED window state
# ([run] spunup_state: an earlier run's data_init.jld2 or data_final.jld2), in
# one place. They run at startup, before anything is compiled, so a stale or
# mismatched file fails in seconds rather than after the multi-minute compile.
#
# WHAT IS COMPARED, and how:
#   * Config sections: the run's [model], [run], [kappa] and [spice] against the
#     source's saved config (config_toml). Compared as parsed configs
#     (run_config.jl), not field by field against metadata.
#   * Code-level constants: code_constants() (channel_common.jl, the ONE list,
#     also what every output file saves), entry by entry, plus XLA_FLAGS and
#     the Julia version.
#   * The steps the loaded state has run, against this run's Nspinup.
#
# SEVERITIES (one report, three levels):
#   * abort: a different ocean or a state that cannot be this run's. The EOS
#     (α, β_S, g), the spice anomaly, the grid, the steps run, or a source
#     without a saved config.
#   * warn: numerics that differ (the other [model] settings, κ, the other code
#     constants, XLA_FLAGS, the Julia version). Kept as warnings for deliberate
#     stress tests.
#   * info: deliberate differences (none at present; kept for new checks).
# All findings are logged in one block. The aborts then stop the run, before
# the compile.
#
# Files without a saved config are refused, not interpreted: re-run their
# source.
#
# Needs channel_common.jl (code_constants, LAUNCH_PROVENANCE) and run_config.jl
# included first.

using TOML
using JLD2

if !@isdefined(_WINDOW_IC_CHECKS_LOADED)
    _WINDOW_IC_CHECKS_LOADED = true

    struct CheckReport
        abort :: Vector{String}
        warn  :: Vector{String}
        info  :: Vector{String}
    end
    CheckReport() = CheckReport(String[], String[], String[])

    # Log every finding in one block, then stop on any abort.
    function finish_checks!(r::CheckReport, what)
        isempty(r.info) || @info "$what: deliberate differences:\n    " * join(r.info, "\n    ")
        isempty(r.warn) || @warn "$what: differences (numerics; continuing, as for a deliberate stress test):\n    " *
                                 join(r.warn, "\n    ")
        isempty(r.abort) || error("$what: ABORTING before anything is built:\n    " * join(r.abort, "\n    "))
        isempty(r.warn) && isempty(r.info) && @info "$what: consistent (config sections, code constants, files)."
        return r
    end

    # The report as text, saved with the run (aborts never reach a save).
    check_report_text(r::CheckReport) = join(vcat(["warn: " * m for m in r.warn], ["info: " * m for m in r.info]), "\n")

    #####
    ##### Reading a source run's files
    #####

    # A save file's metadata, falling back to the data_init.jld2 beside it for
    # keys a data_final.jld2 may lack. Only the keys the checks read.
    function source_metadata(path)
        wanted = Set(vcat(String.(collect(keys(code_constants()))), ["config_toml", "xla_flags", "julia_version", "Nspinup", "Ntimesteps"]))
        meta = Dict{String,Any}()
        for p in unique(filter(isfile, [path, joinpath(dirname(path), "data_init.jld2")]))
            jldopen(p, "r") do f
                for k in keys(f)
                    (k in wanted && !haskey(meta, k)) && (meta[k] = read(f, k))
                end
            end
        end
        return meta
    end

    # The config sections a run is defined by, as typed NamedTuples: from this
    # run's RUN_CONFIG (run_sections), or from a source's saved config
    # (saved_sections).
    run_sections(cfg) = (; cfg.model, run = (; cfg.run.Nspinup, cfg.run.Ntimesteps), cfg.kappa, cfg.spice)

    # Configs saved before the `kind` key was dropped carry kind = "truth";
    # ignore it so those runs can still serve as a window IC source.
    function saved_sections(d, w)
        get(d, "kind", nothing) == "truth" && (d = delete!(copy(d), "kind"))
        t = parse_config(d, w)
        return (; t.model, run = (; t.run.Nspinup, t.run.Ntimesteps), t.kappa, t.spice)
    end

    # "key: this run = …, source = …" for each field that differs.
    section_diffs(mine, theirs, who) =
        ["$k: this run = $(repr(mine[k])), $who = $(repr(theirs[k]))" for k in keys(mine) if !isequal(mine[k], theirs[k])]

    # The source's saved sections, or `nothing` (with an abort) for a file that
    # predates saved configs.
    function source_sections!(r, meta, path, who)
        haskey(meta, "config_toml") && return saved_sections(TOML.parse(meta["config_toml"]), "config saved in $path")
        push!(r.abort, "$who $path has no saved run config (no config_toml): re-run it from a config")
        return nothing
    end

    #####
    ##### Code-level constants
    #####

    # code_constants() against the source's saved values: the EOS aborts, the
    # rest warns, a value the source does not record warns as unverified.
    # `forcing_parameters` is compared field by field. Plus XLA_FLAGS, which
    # changes XLA's fusion and hence round-off, and the Julia version, which
    # changes the package versions the Manifest resolves to and Enzyme's code
    # generation, so bitwise agreement with the source is not expected.
    function check_code_constants!(r, meta, who)
        for (k, live) in pairs(code_constants())
            key = String(k)
            if !haskey(meta, key)
                push!(r.warn, "$k: cannot verify, the $who records none")
            elseif k === :forcing_parameters
                saved = meta[key]
                for f in propertynames(live)
                    if !hasproperty(saved, f)
                        push!(r.warn, "forcing parameter $f: cannot verify, the $who records none")
                    elseif getproperty(saved, f) != getproperty(live, f)
                        push!(r.warn, "forcing parameter $f: this run = $(getproperty(live, f)), $who = $(getproperty(saved, f))")
                    end
                end
            elseif k in (:alpha, :beta_S, :g)
                isapprox(meta[key], live; rtol = 1e-12) ||
                    push!(r.abort, "equation of state $k: this run = $live, $who = $(meta[key]) (a different ocean)")
            elseif meta[key] != live
                push!(r.warn, "$k: this run = $live, $who = $(meta[key])")
            end
        end
        live_xla = LAUNCH_PROVENANCE.xla_flags
        haskey(meta, "xla_flags") && meta["xla_flags"] != live_xla &&
            push!(r.warn, "XLA_FLAGS: this run = \"$live_xla\", $who = \"$(meta["xla_flags"])\" (changes round-off)")
        live_julia = LAUNCH_PROVENANCE.julia_version
        if !haskey(meta, "julia_version")
            push!(r.warn, "Julia version: cannot verify, the $who records none")
        elseif meta["julia_version"] != live_julia
            push!(r.warn, "Julia version: this run = $live_julia, $who = $(meta["julia_version"]) (changes round-off and packages)")
        end
        return r
    end

    #####
    ##### A loaded window state
    #####

    """
        check_window_ic!(r, path, cfg)

    Check that the state file at `path` (a data_init.jld2 or data_final.jld2)
    fits the run configured by `cfg`:
      * abort: no saved config, a different grid (Nx, Ny) or spice anomaly, a
        different EOS, or a state that has not run exactly cfg.run.Nspinup steps
        (its Nspinup, plus its Ntimesteps for a data_final.jld2);
      * warn: other [model] numerics, κ, the other code constants.
    Whether the state holds the same tracers as this run is checked when it is
    loaded (load_spunup_state!, channel_common.jl).
    """
    function check_window_ic!(r, path, cfg)
        isfile(path) || (push!(r.abort, "window IC: no such file: $path"); return r)
        meta = source_metadata(path)
        src  = source_sections!(r, meta, path, "window IC source")
        src === nothing && return r
        final = jldopen(f -> "T_final" in keys(f), path, "r")

        steps = src.run.Nspinup + (final ? src.run.Ntimesteps : 0)
        steps == cfg.run.Nspinup || push!(r.abort, "window IC $path has run $steps steps " *
            (final ? "(its Nspinup $(src.run.Nspinup) + its $(src.run.Ntimesteps)-step window) " : "") *
            "but this run's [run] Nspinup is $(cfg.run.Nspinup)")

        mine = run_sections(cfg)
        for d in section_diffs(mine.spice, src.spice, "window IC source")
            push!(r.abort, "[spice] $d (a different ocean)")
        end
        for d in section_diffs(mine.model, src.model, "window IC source")
            # A different grid cannot load; other numerics are a stress test.
            push!(startswith(d, "Nx") || startswith(d, "Ny") ? r.abort : r.warn, "[model] $d")
        end
        isequal(mine.kappa, src.kappa) ||
            push!(r.warn, "[kappa]: this run = $(mine.kappa), window IC source = $(src.kappa)")
        check_code_constants!(r, meta, "window IC source $path")
        return r
    end

    """
        validate_window_ic(path, cfg)

    check_window_ic! on its own, finished (logged, aborting on errors): for a
    run that loads its spun-up state from a file. Returns the report text.
    """
    function validate_window_ic(path, cfg)
        r = check_window_ic!(CheckReport(), path, cfg)
        finish_checks!(r, "Window IC $path")
        return check_report_text(r)
    end

end # include guard
