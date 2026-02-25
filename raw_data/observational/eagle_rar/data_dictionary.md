# Data Dictionary

Key tabular files and column-level notes used in this project.
Schema source: local `ReadMe` (CDS byte-by-byte descriptions).

## `tablea1e.dat` (EAGLE Ref-L0100N1504 sample)

| column | unit | notes |
|---|---|---|
| `ID` | --- | EAGLE galaxy identifier. |
| `logMs` | log10(Msun) | Stellar mass (within 30 kpc spherical aperture). |
| `logMh` | log10(Msun) | Halo mass. |
| `vflat` | km/s | Circular speed where the rotation curve flattens. |
| `Reff` | kpc | Half-mass radius. |
| `Rs` | --- | Median azimuthal speed / vertical dispersion for stars. |
| `Fs` | --- | Fraction of non counter-rotating stars. |

## `tablea1t.dat` (IllustrisTNG TNG100-1 sample)

| column | unit | notes |
|---|---|---|
| `ID` | --- | IllustrisTNG galaxy identifier. |
| `logMs` | log10(Msun) | Stellar mass (within 30 kpc spherical aperture). |
| `logMh` | log10(Msun) | Halo mass. |
| `vflat` | km/s | Circular speed where the rotation curve flattens. |
| `Reff` | kpc | Half-mass radius. |
| `Rs` | --- | Median azimuthal speed / vertical dispersion for stars. |
| `Fs` | --- | Fraction of non counter-rotating stars. |

## `tablea2.dat` (SPARC comparison sample from Marasco+2020)

| column | unit | notes |
|---|---|---|
| `Galaxy` | --- | Galaxy name. |
| `logMs50` | log10(Msun) | Stellar mass posterior median. |
| `logMs16` | log10(Msun) | Stellar mass posterior 16th percentile. |
| `logMs84` | log10(Msun) | Stellar mass posterior 84th percentile. |
| `logMh50` | log10(Msun) | Halo mass posterior median. |
| `logMh16` | log10(Msun) | Halo mass posterior 16th percentile. |
| `logMh84` | log10(Msun) | Halo mass posterior 84th percentile. |
| `vflat` | km/s | Flat rotation velocity. |
| `e_vflat` | km/s | Uncertainty on `vflat`. |
| `Reff` | kpc | Effective radius from 3.6 micron photometry. |
| `e_Reff` | kpc | Uncertainty on `Reff`. |
