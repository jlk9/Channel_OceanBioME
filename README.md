# Channel_OceanBioME

A platform for testing the `OceanBioME.jl` package in an Oceananigans workflow.
- This uses a re-entrant channel configuration similar to the one presented in DJ4Earth (Moses et al., 2026) as a base template, then adds biogeochemistry.
- It is simply for debugging and evaluating correctness via estimates like finite difference checks, not scientific results (at least for now!).

## To se this package:

The main driver for the re-entrant channel is in `abernathey_channel.jl`, so run from the repo's root with:

```
julia -O0 --project scripts/abernathey_channel.jl <configuration of interest>
```

Two sample configurations are included in the `configs` folder.

To run on your machine, you might need to delete the included `Manifest.toml` file and install / precompile all the packages for your machine. I've included the Manifest to be through in citing package versions - I'll do this in a better way soon!