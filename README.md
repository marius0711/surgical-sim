# surgical-sim

Physics-based simulation of biological tissue under mechanical load.
Long-term goal: digital twin for surgical procedures — muscle, bone, and cutting mechanics.

## Roadmap
- [x] Neo-Hookean material model for muscle tissue
- [x] Von Mises stress field output (ParaView XDMF)
- [x] Holzapfel-Ogden anisotropic extension (fiber reinforcement)
- [x] Force-displacement curves: Neo-Hookean vs Holzapfel-Ogden
- [x] Orthotropic bone model (linear elastic, cortical bone parameters)
- [ ] Two-material mesh: muscle + bone in contact
- [ ] Cutting mechanics (Cohesive Zone Model)
- [ ] CT-based patient-specific geometry

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
python3 simulations/tissue_anisotropic.py
python3 simulations/force_displacement.py
python3 simulations/bone_compression.py
```
