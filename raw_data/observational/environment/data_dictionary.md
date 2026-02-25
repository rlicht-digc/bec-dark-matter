# Data Dictionary

Key tabular files and column-level notes used in this project.
Schema source: CSV headers in this folder and derivation notes in `TERMS_OF_USE.md`.

## `sparc_environment_catalog.csv`

| column | unit | notes |
|---|---|---|
| `sparc_name` | --- | SPARC galaxy identifier used for joins to SPARC tables. |
| `source` | --- | Upstream source label for the environment assignment. |
| `group_name` | --- | Matched group or cluster name from group catalogs. |
| `logMh` | log10(Msun) | Host halo mass proxy in log10 solar masses. |
| `richness` | count | Group richness metric from upstream group catalog. |
| `central` | flag | Central/satellite indicator from group catalog. |
| `env_class` | category | Environment class label (field/group/cluster style taxonomy). |
| `env_dense` | flag | Binary dense-environment flag used by analysis scripts. |
| `separation_arcsec` | arcsec | Angular separation between matched entries. |

## `wallaby_environment_catalog.csv`

| column | unit | notes |
|---|---|---|
| `name` | --- | WALLABY source identifier. |
| `ra` | deg | Right ascension (J2000). |
| `dec` | deg | Declination (J2000). |
| `Vsys` | km/s | Systemic velocity used in the WALLABY environment join. |
| `env_class` | category | Environment class label. |
| `group_name` | --- | Matched group name. |
| `env_dense` | flag | Binary dense-environment flag used by analysis scripts. |
| `logMh` | log10(Msun) | Host halo mass proxy in log10 solar masses. |
| `local_density` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `team_release` | --- | Team release/version tag from source catalog. |
