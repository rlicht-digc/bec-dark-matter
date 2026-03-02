# Paper 3 Cosmology + Phase Potential Note (Starter)

## Scope and claim boundary
This module is a bridge layer, not a full cosmology fit. It formalizes a minimum EFT structure that can host both:
1. a cosmological vacuum component (`rho_Lambda`, `w ~= -1`), and
2. a dense-phase structure relevant to throat/impedance constraints.

## Why contact BEC interaction energy cannot be `Lambda`
For a contact-interaction BEC term, the effective pressure is non-negative (`w >= 0` in the fluid description). A component with `w >= 0` redshifts with expansion and cannot drive accelerated expansion (`w < -1/3` required). Therefore, contact interaction energy alone cannot be identified with dark-energy `Lambda`.

## Minimal potential structure required
Two explicit toy families are used:

1. Family 1 (symmetry breaking + vacuum offset)
   `V(phi) = lambda/4 * (phi^2 - v^2)^2 + V0`
   This makes transparent that a nonzero vacuum offset is an explicit input scale.

2. Family 2 (two-minima phenomenology)
   `V(phi) = a phi^2 + b phi^4 + c phi^6 + V0`
   With `c > 0` and suitable `(a,b)`, two-phase minima and a barrier are straightforwardly realized.

Neither family is claimed unique; both are bookkeeping devices to show what must exist at minimum.

## Bridge constraints from throat-budget anchors
The loaded Stage-A/Stage-B anchors (`Delta*`, `alpha*`, `kappa/A1`, PRIMARY `A1` bands) constrain a gradient/impedance channel associated with compact-object throat physics. They do not, by themselves, determine the absolute cosmological offset `V0`.

Interpretation for Paper 3:
- one scale is tied to cosmological vacuum bookkeeping (meV-level `V0^(1/4)` from `rho_Lambda`),
- another scale can control dense-phase/barrier structure (often much higher, potentially EW/TeV-ish in toy EFT terms).

This is presented as a two-scale EFT requirement, not a solved cosmological-constant derivation.

## Cosmology observable targets used here
- `rho_Lambda = 6e-10 J/m^3` (input)
- `w_target ~= -1` (enforced only as vacuum dominance condition)

No `w(z)` fit, perturbation analysis, or Boltzmann-level consistency test is done in this starter module.

## Galaxy-side lever for `rho_c` (handoff to high-density program)
A practical lever is the onset density where high-density systems show a robust residual-shift/phase-indicator transition (inner-region behavior versus inferred local density). That transition density can be operationalized as a data-side estimator for `rho_c` and fed back into the phase-potential prior set.

## Deferred items
- microphysical UV derivation of `(a,b,c,lambda,v)`
- full cosmology fit (`w(z)`, growth, CMB/LSS constraints)
- unified inference combining compact-object and galaxy likelihoods
