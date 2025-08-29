# Power-Law Galactic Rotation Curves from Tsallis Non-Extensive Statistics

[![DOI](https://zenodo.org/badge/DOI/16997832.svg)](https://zenodo.org/records/16997832))
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Abstract

We present a comprehensive theoretical framework demonstrating that power-law rotation curves V = Ar^α emerge as the unique self-consistent scale-free solutions for incompletely relaxed self-gravitating systems described by Tsallis non-extensive statistics. Starting from first principles, we derive α = −1/(n−1) where n is the polytropic index, yielding α = (q−1)/(q+3/2) for isotropic systems with Tsallis parameter q. 

## Author

**Johann Anton Michael Tupay**  
London, United Kingdom  
Contact: [jamtupay@icloud.com] 

## Paper

The full paper is available in the [`paper/`](paper/) directory:
- **PDF**: [A Complete Theoretical Framework for Power-Law Galactic Rotation Curves: From Tsallis Non-Extensive Statistics to Observable Predictions.pdf](paper/A Complete Theoretical Framework for Power-Law Galactic Rotation Curves: From Tsallis Non-Extensive Statistics to Observable Predictions.pdf)
- **LaTeX source**: [main.tex](paper/main.tex)

## Key Results

1. **Uniqueness proof**: Power laws are the only self-consistent scale-free solutions for the coupled polytrope-Poisson system
2. **Parameter relations**: α = (q−1)/(q+3/2) for isotropic systems, generalized for anisotropy
3. **Empirical validation**: Tests on 175 SPARC galaxies show:
   - Parameter-mass correlations: R² = 0.610 (A-mass), R² = 0.482 (α-mass)
   - Tully-Fisher relation preserved: R² = 0.879
   - Distinct lensing predictions for dwarf galaxies

## Repository Contents

- `paper/`: Contains the LaTeX source and compiled PDF
- `figs/`: All figures used in the paper
- `data/`: (Optional) Analysis scripts and processed data

## Compilation Instructions

To compile the paper from source:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
## Requirements

LaTeX distribution (TeXLive, MiKTeX, or MacTeX)
BibTeX
Standard LaTeX packages: amsmath, amssymb, amsthm, graphicx, hyperref, geometry, booktabs, float, biblatex

## Citation
If you use this work, please cite:
@article{Tupay2025,
  author  = {Tupay, Johann Anton Michael},
  title   = {A Complete Theoretical Framework for Power-Law Galactic 
            Rotation Curves: From Tsallis Non-Extensive Statistics 
            to Observable Predictions},
  year    = {2025},
  month   = {8},
  doi     = {[10.5281/zenodo.16997832]}
}

## Related Work
This paper builds on:

Companion paper: "An Entropy-Inspired Phenomenological Relation Competes with NFW on 175 SPARC Rotation Curves" (Tupay 2025, submitted)
SPARC database: Lelli et al. (2016) - 175 galaxies with accurate rotation curves

## License
This work is licensed under a Creative Commons Attribution 4.0 International License.

## Acknowledgments
We thank the SPARC team for making their rotation curve data publicly available.

## Keywords
galactic rotation curves Tsallis statistics non-extensive thermodynamics dark matter alternative polytropic models self-gravitating systems
