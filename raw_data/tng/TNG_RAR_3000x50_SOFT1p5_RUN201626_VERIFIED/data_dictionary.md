# Data Dictionary

Key tabular files and column-level notes used in this project.

## `galaxy_scatter_dm.csv`

| column | unit | notes |
|---|---|---|
| `SubhaloID` | --- | Integer identifier. |
| `n_dm_pts` | count | Count of objects/points/voxels. |
| `sigma_dm_std` | dex | Scatter/residual metric in log-space units unless documented otherwise. |
| `sigma_dm_robust` | dex | Scatter/residual metric in log-space units unless documented otherwise. |
| `Mstar_Msun` | Msun | Mass quantity. |
| `SFR` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `logMstar` | log10(Msun) | Logarithmic mass quantity. |
| `logSFRp` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |

## `galaxy_scatter_dm_with_env.csv`

| column | unit | notes |
|---|---|---|
| `SubhaloID` | --- | Integer identifier. |
| `n_dm_pts` | count | Count of objects/points/voxels. |
| `sigma_dm_std` | dex | Scatter/residual metric in log-space units unless documented otherwise. |
| `sigma_dm_robust` | dex | Scatter/residual metric in log-space units unless documented otherwise. |
| `Mstar_Msun` | Msun | Mass quantity. |
| `SFR` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `logMstar` | log10(Msun) | Logarithmic mass quantity. |
| `logSFRp` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `GroupID` | --- | Integer identifier. |
| `M200c_Msun` | Msun | Mass quantity. |
| `logM200c` | log10(Msun) | Logarithmic mass quantity. |
| `env` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |

## `meta/master_catalog.csv`

| column | unit | notes |
|---|---|---|
| `SubhaloID` | --- | Integer identifier. |
| `SubhaloGrNr` | --- | Integer identifier. |
| `Mstar_Msun` | Msun | Mass quantity. |
| `SFR` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `Rhalf_star_kpc` | kpc | Radius-like quantity inferred from column name; verify against upstream docs. |
| `profile_csv` | --- | Identifier or categorical label. |
| `n_radii` | kpc | Radius-like quantity inferred from column name; verify against upstream docs. |
| `any_lowres` | flag | Quality/control flag from source catalog. |
| `n_dm_max` | count | Count of objects/points/voxels. |
| `n_star_max` | count | Count of objects/points/voxels. |
| `n_gas_max` | count | Count of objects/points/voxels. |
| `dm_cutout_count` | count | Count of objects/points/voxels. |
