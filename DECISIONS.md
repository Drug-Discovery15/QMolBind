# DECISIONS

Each entry: decision / alternatives / reason.

## D1. Isolated venv, Python 3.12 (not conda, not 3.11)
- Decision: project-local `.venv` (user instruction: never touch the global env); Python 3.12.7 is the only interpreter present.
- Alternatives: conda-forge Python 3.11 env as in the prompt (conda not installed).
- Reason: user constraint + availability. `environment.yml` is kept for conda users, but the tested path is pip in `.venv`.

## D2. No `make` on this machine
- Decision: `Makefile` as specified, delegating to `scripts/tasks.py`; on Windows run `python scripts/tasks.py <target>`.
- Reason: stock Windows has no make.

## D3. OpenFF toolkit is not on PyPI
- Decision: `openff-toolkit`, `openff-nagl`, `openff-interchange` have no pip distributions (verified with `pip index versions`), and AmberTools (antechamber/AM1-BCC) is not pip-installable. The prompt's default `openff-2.1.0` via `SystemGenerator` is therefore unavailable in the venv. Ligand parametrization approach is decided in Stage 3 and recorded below.
- Alternatives: project-local micromamba env (would violate the venv instruction); skip.

## D4. Installed versions (first successful install)
torch 2.14.1+cpu, pennylane 0.45.1, pennylane-lightning 0.45.0, openmm 8.6.1, rdkit 2026.03.6, qiskit 2.5.2, qiskit-aer 0.17.2,
qiskit-ibm-runtime 0.50.0, torch_geometric 2.8.0.post1, transformers 5.19.0, pdbfixer 1.12.0, openmmforcefields 0.15.1, numpy 2.5.3.
Lockfile: `requirements-lock.txt` (pip freeze of `.venv`).

## D5. Torch build
- Decision: CPU torch. GPU (RTX 5070 Ti) present but 8-qubit sims and small MLPs run faster on CPU; GPU left optional.

## D6. Target selection (programmatic screen; see docs/DATA.md for the full table)
- 1STP (streptavidin-biotin), the prompt's suggestion, is REJECTED: RCSB resolution 2.6 A > 2.5 A.
- Screen of 8 candidates (resolution <= 2.5, exactly one non-additive ligand type, 10-45 heavy atoms, 3-8 RDKit rotatable bonds, no LINK/<1.7 A contact):
  accepted 1CIL (carbonic anhydrase II / ETS, 3 rotors), 3ERT (ER-alpha / OHT, 8), 1UYD (HSP90 / PU8, 8), 1EVE, 2HYY.
- Decision: use 1CIL (smoke target, small ligand), 3ERT, 1UYD. Alternatives: 1EVE (AChE gorge, large), 2HYY (Abl/imatinib, 37 heavy atoms, DFG-out) - dropped for oracle cost.
- Caveat: 1CIL has only K=3 torsions (the prompt's lower bound), so CMA-ES in 3-D will be near-trivial there; the 8-rotor targets are the informative ones.

## D7. Data storage: manifest with sha256 instead of DVC
- Decision: HDF5 per target in data_cache/processed/ + manifest.json (sha256). Alternative: DVC (extra dependency, needs remote). Prompt allows either.
- BindingDB: censored values (<, >) dropped; dedupe by canonical SMILES preferring Kd > Ki > IC50, median pAffinity within the preferred type; `measurement` and `n_measurements` kept.

## D8. Protein prep
- pdbfixer (heavy atoms, no invented loops; pH 7.4 hydrogens via Modeller). Chains kept = those with a heavy atom within 6 A of the ligand.
- Catalytic/structural metals ZN, MG, CA, MN within the pocket are KEPT (prompt says remove heteroatoms except the ligand): 1CIL's sulfonamide binds the active-site Zn, and without it the target is meaningless. They are modelled as non-bonded ions (amber14 ion templates).
- Pocket truncation to residues within 12 A of the ligand is applied by default for ALL targets (not only if slow), with ACE/NME caps built from the real coordinates of the removed neighbours and gaps <= 3 residues filled. Reason: keeps oracle cost low; protein is frozen anyway. 1CIL: 142 res / 2009 atoms; 3ERT: 132 / 2051; 1UYD: 147 / 2150.

## D9. Torsion application: own Rodrigues rotation, not RDKit SetDihedral
- Alternative: rdMolTransforms.SetDihedralDeg (mutates a Conformer, ~10x slower in Python, ambiguous about which side moves for ring-adjacent bonds). Mine rotates the side not containing the root fragment about the current bond axis; tests verify bond lengths preserved, exact dihedral change, +theta/-theta restore (< 1e-6 A) and agreement with RDKit's GetDihedralDeg.
- Rotatable bonds: RDKit *strict* SMARTS (no amides, no terminal CF3/tBu); K cap keeps the torsions that move the most atoms. Root fragment = largest heavy-atom component after cutting rotatable bonds.

## D10. RMSD symmetry
- Automorphisms are computed on the topology-only heavy-atom graph (bond orders/charges/aromaticity ignored) so carboxylate, sulfonyl and nitro oxygens are interchangeable. RDKit's CalcRMS uses typed matching and would count a carboxylate O swap as an error; tests compare against CalcRMS on a molecule where the two coincide. RMSD is in-place (no superposition), evaluation only.
