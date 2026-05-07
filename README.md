# surgical-sim

Physics-based simulation of biological tissue under mechanical load.
Long-term goal: digital twin for surgical procedures — muscle, bone, and cutting mechanics.

## Roadmap
- [x] Neo-Hookean material model for muscle tissue
- [ ] Stress field output (von Mises)
- [ ] Anisotropy along muscle fiber direction
- [ ] Bone model (linear elastic, orthotropic)
- [ ] Cutting mechanics (Cohesive Zone Model)
- [ ] Patient-specific geometry from CT data

## Stack
- FEniCSx / DOLFINx 0.10
- UFL (Unified Form Language)
- PETSc / MUMPS

## Getting started
```bash
docker run -it \
  -v /path/to/surgical-sim:/workspace \
  -w /workspace \
  dolfinx/dolfinx:stable

python3 simulations/tissue_compression.py
```
