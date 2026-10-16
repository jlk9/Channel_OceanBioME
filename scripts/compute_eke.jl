# compute_eke.jl
#
# Offline resolved eddy kinetic energy (EKE) diagnostic for the reentrant-channel
# runs produced by abernathey_channel.jl.
#
# Purpose: answer "is the coarse model spuriously generating resolved eddies that
# the GM/Redi closure would double-count?" The eddy/mean split used here is the
# ZONAL (x) mean — the natural decomposition for a statistically x-homogeneous
# reentrant channel, and it needs only a single equilibrated snapshot (the u/v
# already dumped to the run's JLD2). It counts both transient and topographically
# forced standing eddies, which is what we want (both do the eddy flux GM
# represents). A time-mean split would isolate transient eddies only and needs
# accumulation over the run; it is deliberately not done here.
#
#   ū(y,z) = ⟨u⟩ₓ ,   u′ = u − ū
#   EKE(y,z) = ½( ⟨u′²⟩ₓ + ⟨v′²⟩ₓ )      via the variance identity ⟨u²⟩ₓ − ū²
#   MKE(y,z) = ½( ū² + v̄² )
#
# The headline number is EKE relative to MKE (both averaged over WET cells, so the
# ratio is unaffected by dry-cell dilution): a small ratio ⇒ effectively
# non-eddy-resolving (GM/Redi regime, the resolution choice is fine); an O(1) or
# larger ratio ⇒ the dynamics resolve eddies the closure also parameterizes, and
# you should coarsen.
#
# No Oceananigans dependency — pure array ops on the saved interior fields.
#
# Usage:
#   julia --project scripts/compute_eke.jl <path/to/data.jld2>
#
# Notes / approximations (adequate for a qualitative check, not a publication
# diagnostic):
#   * Dry (immersed) cells carry identically-zero velocity; a cell is treated wet
#     if either co-located velocity component is non-zero. Cells straddling the
#     topography are slightly over-included.
#   * Vertical averaging for the EKE(y) profile is unweighted in z (the grid is
#     stretched). EKE and MKE are similarly surface-intensified, so the EKE/MKE
#     RATIO is fairly insensitive to this; the absolute profile is a rough proxy.

using JLD2
using Statistics
using Printf

# ---- input ----------------------------------------------------------------
path = length(ARGS) ≥ 1 ? ARGS[1] :
    error("give the JLD2 output path, e.g.\n  julia --project scripts/compute_eke.jl runs/<config name>/data_final.jld2")

isfile(path) || error("no such file: $path")

data = jldopen(path, "r") do f
    Dict(k => read(f, k) for k in keys(f))
end

(haskey(data, "u") && haskey(data, "v")) ||
    error("file has no 'u'/'v' fields (run through the final jldsave first). keys = $(sort(collect(keys(data))))")

u = Float64.(data["u"])
v = Float64.(data["v"])

Nx = haskey(data, "Nx") ? Int(data["Nx"]) : size(u, 1)
Ny = haskey(data, "Ny") ? Int(data["Ny"]) : size(u, 2)
Nz = haskey(data, "Nz") ? Int(data["Nz"]) : size(u, 3)

# ---- co-locate velocities to cell centers (Center, Center, Center) --------
# u is (Face, Center, Center) on periodic x → average adjacent x-faces (wrap).
# v is (Center, Face, Center) on bounded  y → average adjacent y-faces.
# center_y_bounded tolerates either interior convention for v's y-length.

function center_x_periodic(u)
    Nx = size(u, 1)
    uc = similar(u)
    @inbounds for i in 1:Nx
        ip = i == Nx ? 1 : i + 1
        uc[i, :, :] .= 0.5 .* (u[i, :, :] .+ u[ip, :, :])
    end
    return uc
end

function center_y_bounded(v, Ny)
    ny = size(v, 2)
    if ny == Ny + 1                       # (Center, Face, Center): Ny+1 faces
        return 0.5 .* (v[:, 1:Ny, :] .+ v[:, 2:Ny+1, :])
    elseif ny == Ny                       # already collapsed to centers
        return v
    else
        error("unexpected v y-length $ny; expected $Ny or $(Ny + 1)")
    end
end

uc = center_x_periodic(u)[:, 1:Ny, :]
vc = center_y_bounded(v, Ny)

size(uc) == size(vc) ||
    error("co-located u $(size(uc)) and v $(size(vc)) shapes disagree")

# ---- wet mask (dry/immersed cells are zero-velocity) ----------------------
wet   = (uc .!= 0) .| (vc .!= 0)          # (Nx, Ny, Nz)
wetyz = dropdims(any(wet, dims = 1), dims = 1)   # (Ny, Nz): column wet if any x is wet
count(wetyz) > 0 || error("no wet cells detected")

# ---- zonal-mean eddy/mean decomposition -----------------------------------
ū  = mean(uc, dims = 1)                    # (1, Ny, Nz) mean flow
v̄  = mean(vc, dims = 1)
u2 = mean(uc .^ 2, dims = 1)              # ⟨u²⟩ₓ
v2 = mean(vc .^ 2, dims = 1)

eke = dropdims(0.5 .* ((u2 .- ū .^ 2) .+ (v2 .- v̄ .^ 2)), dims = 1)   # (Ny, Nz) ≥ 0
mke = dropdims(0.5 .* (ū .^ 2 .+ v̄ .^ 2),                dims = 1)

# ---- scalar summaries (over wet cells) ------------------------------------
eke_wet  = eke[wetyz]
mke_wet  = mke[wetyz]
mean_eke = mean(eke_wet)
mean_mke = mean(mke_wet)
ratio    = mean_eke / max(mean_mke, eps())

maxeke, imax = findmax(eke .* wetyz)
jmax, kmax   = Tuple(imax)

# EKE(y): mean over wet z at each y (unweighted in z — see header note)
eke_y = [begin
             row = eke[j, @view wetyz[j, :]]
             isempty(row) ? 0.0 : mean(row)
         end for j in 1:Ny]

# ---- report ---------------------------------------------------------------
@info "Resolved EKE diagnostic (zonal-mean split)  —  $path"
@printf("  grid                    : Nx=%d  Ny=%d  Nz=%d  (%d wet columns of %d)\n",
        Nx, Ny, Nz, count(wetyz), Ny * Nz)
@printf("  mean EKE  (wet cells)   : %.4e m²/s²   = %.2f cm²/s²\n", mean_eke, 1e4 * mean_eke)
@printf("  mean MKE  (wet cells)   : %.4e m²/s²   = %.2f cm²/s²\n", mean_mke, 1e4 * mean_mke)
@printf("  EKE / MKE               : %.3f\n", ratio)
@printf("  rms eddy speed √(2·EKE) : %.4f m/s\n", sqrt(2 * mean_eke))
@printf("  rms mean  speed √(2·MKE): %.4f m/s\n", sqrt(2 * mean_mke))
@printf("  peak EKE                : %.4e m²/s²  at (j=%d, k=%d)\n", maxeke, jmax, kmax)
@info(ratio < 0.1  ? "EKE ≪ MKE: effectively non-eddy-resolving — GM/Redi regime, resolution choice is safe." :
      ratio < 0.5  ? "EKE < MKE but not negligible: mild eddy activity; watch for partial double-counting." :
                     "EKE comparable to or exceeding MKE: the run resolves eddies GM/Redi also parameterizes — consider coarsening.")

# ---- save fields for plotting ---------------------------------------------
outpath = joinpath(dirname(path), "eke_diagnostics.jld2")
jldsave(outpath;
        Nx, Ny, Nz,
        eke,            # (Ny, Nz) zonal-mean eddy kinetic energy
        mke,            # (Ny, Nz) mean-flow kinetic energy
        eke_y,          # (Ny,)    z-averaged EKE profile
        wetyz,          # (Ny, Nz) wet-cell mask
        mean_eke, mean_mke, ratio)
@info "wrote $outpath"
