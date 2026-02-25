# Data Dictionary

Key tabular files and column-level notes used in this project.
Schema source: `SPARC_ReadMe.txt` (CDS/VizieR byte-by-byte descriptions for table1/table2).

## `SPARC_table1_vizier.dat`

| column | unit | notes |
|---|---|---|
| `Name` | --- | Galaxy name. |
| `Type` | code | Numerical Hubble type (0-11). |
| `Dist` | Mpc | Assumed distance. |
| `e_Dist` | Mpc | Mean uncertainty on distance. |
| `f_Dist` | code | Distance-method flag (1-5). |
| `i` | deg | Assumed inclination angle. |
| `e_i` | deg | Mean uncertainty on inclination. |
| `L3.6` | GLsun | Total luminosity at 3.6 micron. |
| `e_L3.6` | GLsun | Mean uncertainty on `L3.6`. |
| `Reff` | kpc | Effective radius enclosing half of total 3.6 micron light. |
| `SBeff` | Lsun/pc2 | Mean surface brightness within `Reff`. |
| `Rdisk` | kpc | Exponential disk scale length at 3.6 micron. |
| `SBdisk` | Lsun/pc2 | Extrapolated central disk surface brightness. |
| `MHI` | GMsun | Total HI mass. |
| `RHI` | kpc | HI radius at face-on surface density of 1 Msun/pc2. |
| `Vflat` | km/s | Asymptotic flat rotation speed. |
| `e_Vflat` | km/s | Mean uncertainty on `Vflat`. |
| `Qual` | code | Rotation-curve quality flag (1 high, 2 medium, 3 low). |
| `Ref` | --- | Reference key to HI/Hα data source. |

## `SPARC_table2_rotmods.dat`

| column | unit | notes |
|---|---|---|
| `Name` | --- | Galaxy identifier. |
| `Dist` | Mpc | Assumed distance. |
| `Rad` | kpc | Galactocentric radius. |
| `Vobs` | km/s | Observed circular velocity. |
| `e_Vobs` | km/s | Uncertainty on observed velocity (random/non-circular contribution). |
| `Vgas` | km/s | Gas contribution to circular speed (includes helium factor 1.33). |
| `Vdisk` | km/s | Stellar disk velocity contribution for M/L = 1 at 3.6 micron. |
| `Vbulge` | km/s | Bulge velocity contribution for M/L = 1 at 3.6 micron. |
| `SBdisk` | Lsun/pc2 | Disk surface brightness profile. |
| `SBbulge` | Lsun/pc2 | Bulge surface brightness profile. |

## `sparc_pgc_crossmatch.csv`

| column | unit | notes |
|---|---|---|
| `sparc_name` | --- | SPARC galaxy name. |
| `pgc` | --- | Principal Galaxy Catalogue identifier. |
| `method` | --- | Crossmatch method tag. |
| `T` | code | Numerical Hubble type proxy copied from SPARC metadata. |
| `D` | Mpc | Distance. |
| `eD` | Mpc | Distance uncertainty. |
| `fD` | code | Distance-method flag. |
| `Inc` | deg | Inclination angle. |
| `eInc` | deg | Inclination uncertainty. |
| `L36` | GLsun | 3.6 micron luminosity. |
| `Reff` | kpc | Effective radius. |
| `Rdisk` | kpc | Exponential disk scale length. |
| `MHI` | GMsun | HI mass. |
| `RHI` | kpc | HI radius. |
| `Vflat` | km/s | Flat rotation velocity. |
| `eVflat` | km/s | Uncertainty on `Vflat`. |
| `Q` | code | Rotation-curve quality flag. |
