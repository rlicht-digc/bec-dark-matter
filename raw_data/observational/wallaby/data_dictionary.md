# Data Dictionary

Key tabular files and column-level notes used in this project.

## `wallaby_dr1_kinematic_catalogue.csv`

| column | unit | notes |
|---|---|---|
| `name` | --- | Identifier or categorical label. |
| `ra` | deg | Sky coordinate (J2000) from source catalog. |
| `dec` | deg | Sky coordinate (J2000) from source catalog. |
| `freq` | MHz | Spectral frequency reported by source catalog. |
| `team_release` | --- | Identifier or categorical label. |
| `team_release_kin` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `Vsys_model` | km/s | Velocity-like quantity. |
| `e_Vsys_model` | km/s | Uncertainty on velocity-like quantity. |
| `X_model` | pixel | Cube/image coordinate index from source processing. |
| `e_X_model` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `Y_model` | pixel | Cube/image coordinate index from source processing. |
| `e_Y_model` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `RA_model` | deg | Sky coordinate (J2000) from source catalog. |
| `e_RA_model` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `DEC_model` | deg | Sky coordinate (J2000) from source catalog. |
| `e_DEC_model` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `Inc_model` | deg | Position/inclination angle. |
| `e_Inc_model` | deg | Uncertainty on angular quantity. |
| `PA_model` | deg | Position/inclination angle. |
| `e_PA_model` | deg | Position/inclination angle. |
| `PA_model_g` | deg | Position/inclination angle. |
| `e_PA_model_g` | deg | Position/inclination angle. |
| `QFlag_model` | flag | Quality/control flag from source catalog. |
| `Rad` | kpc | Radius-like quantity inferred from column name; verify against upstream docs. |
| `Vrot_model` | km/s | Velocity-like quantity. |
| `e_Vrot_model` | km/s | Uncertainty on velocity-like quantity. |
| `e_Vrot_model_inc` | km/s | Uncertainty on velocity-like quantity. |
| `Rad_SD` | kpc | Radius-like quantity inferred from column name; verify against upstream docs. |
| `SD_model` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `e_SD_model` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `SD_FO_model` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `e_SD_FO_model_inc` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |

## `wallaby_dr1_source_catalogue.csv`

| column | unit | notes |
|---|---|---|
| `name` | --- | Identifier or categorical label. |
| `ra` | deg | Sky coordinate (J2000) from source catalog. |
| `dec` | deg | Sky coordinate (J2000) from source catalog. |
| `freq` | MHz | Spectral frequency reported by source catalog. |
| `f_sum` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `err_f_sum` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `rms` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `w20` | km/s | Velocity-like quantity. |
| `w50` | km/s | Velocity-like quantity. |
| `kin_pa` | deg | Position/inclination angle. |
| `rel` | flag | Quality/control flag from source catalog. |
| `qflag` | flag | Quality/control flag from source catalog. |
| `kflag` | flag | Quality/control flag from source catalog. |
| `n_pix` | count | Count of objects/points/voxels. |
| `f_min` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `f_max` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `ell_maj` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `ell_min` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `ell_pa` | deg | Position/inclination angle. |
| `ell3s_maj` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `ell3s_min` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `ell3s_pa` | deg | Position/inclination angle. |
| `x` | pixel | Cube/image coordinate index from source processing. |
| `err_x` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `y` | pixel | Cube/image coordinate index from source processing. |
| `err_y` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `z` | pixel | Cube/image coordinate index from source processing. |
| `err_z` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `x_min` | pixel | Cube/image coordinate index from source processing. |
| `x_max` | pixel | Cube/image coordinate index from source processing. |
| `y_min` | pixel | Cube/image coordinate index from source processing. |
| `y_max` | pixel | Cube/image coordinate index from source processing. |
| `z_min` | pixel | Cube/image coordinate index from source processing. |
| `z_max` | pixel | Cube/image coordinate index from source processing. |
| `comments` | --- | Identifier or categorical label. |
| `team_release` | --- | Identifier or categorical label. |
| `dist_h` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `log_m_hi` | log10(Msun) | Logarithmic mass quantity. |

## `wallaby_dr2_high_res_catalogue.csv`

| column | unit | notes |
|---|---|---|
| `name` | --- | Identifier or categorical label. |
| `ra` | deg | Sky coordinate (J2000) from source catalog. |
| `dec` | deg | Sky coordinate (J2000) from source catalog. |
| `freq` | MHz | Spectral frequency reported by source catalog. |
| `f_sum` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `err_f_sum` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `f_sum_corr` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `err_f_sum_corr` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `f_sum_corr_30` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `err_f_sum_corr_30` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `rms` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `w20` | km/s | Velocity-like quantity. |
| `w50` | km/s | Velocity-like quantity. |
| `kin_pa` | deg | Position/inclination angle. |
| `rel` | flag | Quality/control flag from source catalog. |
| `qflag` | flag | Quality/control flag from source catalog. |
| `kflag` | flag | Quality/control flag from source catalog. |
| `n_pix` | count | Count of objects/points/voxels. |
| `f_min` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `f_max` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `ell_maj` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `ell_min` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `ell_pa` | deg | Position/inclination angle. |
| `ell3s_maj` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `ell3s_min` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `ell3s_pa` | deg | Position/inclination angle. |
| `x` | pixel | Cube/image coordinate index from source processing. |
| `err_x` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `y` | pixel | Cube/image coordinate index from source processing. |
| `err_y` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `z` | pixel | Cube/image coordinate index from source processing. |
| `err_z` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `x_min` | pixel | Cube/image coordinate index from source processing. |
| `x_max` | pixel | Cube/image coordinate index from source processing. |
| `y_min` | pixel | Cube/image coordinate index from source processing. |
| `y_max` | pixel | Cube/image coordinate index from source processing. |
| `z_min` | pixel | Cube/image coordinate index from source processing. |
| `z_max` | pixel | Cube/image coordinate index from source processing. |
| `dist_h` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `log_m_hi` | log10(Msun) | Logarithmic mass quantity. |
