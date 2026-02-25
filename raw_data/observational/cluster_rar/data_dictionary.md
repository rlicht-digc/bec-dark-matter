# Data Dictionary

Key tabular files and column-level notes used in this project.
Schema source: in-file CDS table headers for `tian2020_fig2.dat` and `tian2020_table1.dat`.

## `tian2020_fig2.dat`

| column | unit | notes |
|---|---|---|
| `AName` | --- | Compact cluster identifier used across Tian+2020 tables. |
| `Rad` | kpc | Radius at which accelerations are reported. |
| `log_gbar` | log10(m/s2) | Log baryonic acceleration. |
| `log_gtot` | log10(m/s2) | Log total (observed) acceleration. |
| `e_log_gbar` | dex | Uncertainty on `log_gbar`. |
| `e_log_gtot` | dex | Uncertainty on `log_gtot`. |

## `tian2020_table1.dat`

| column | unit | notes |
|---|---|---|
| `Name` | --- | Full cluster name. |
| `z` | --- | Cluster redshift. |
| `RAh` | h | Right ascension hour component. |
| `RAm` | min | Right ascension minute component. |
| `RAs` | s | Right ascension second component. |
| `DEd` | deg | Declination degree component. |
| `DEm` | arcmin | Declination arcminute component. |
| `DEs` | arcsec | Declination arcsecond component. |
| `Band` | --- | HST band used for photometry/modeling. |
| `n` | --- | Sersic index for BCG light-profile modeling. |
| `Re` | TODO: requires upstream docs not present locally. | Effective radius from the table header; exact physical unit requires upstream table notes not present locally. |
| `e_Re` | TODO: requires upstream docs not present locally. | Uncertainty on `Re`; exact physical unit requires upstream table notes not present locally. |
| `Rad` | kpc | Reference radius used for acceleration evaluation in Tian+2020. |
| `Mstar` | TODO: requires upstream docs not present locally. | Stellar-mass estimate as tabulated by Tian+2020; unit scale requires upstream table notes not present locally. |
| `Mgas` | TODO: requires upstream docs not present locally. | Gas-mass estimate as tabulated by Tian+2020; unit scale requires upstream table notes not present locally. |
| `e_Mgas` | TODO: requires upstream docs not present locally. | Uncertainty on gas mass. |
| `Mtot` | TODO: requires upstream docs not present locally. | Total mass estimate as tabulated by Tian+2020; unit scale requires upstream table notes not present locally. |
| `e_Mtot` | TODO: requires upstream docs not present locally. | Uncertainty on total mass. |
| `AName` | --- | Short cluster identifier matching `tian2020_fig2.dat`. |
