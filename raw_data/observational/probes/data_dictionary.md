# Data Dictionary

Key tabular files and column-level notes used in this project.

## `main_table.csv`

| column | unit | notes |
|---|---|---|
| `name` | --- | Identifier or categorical label. |
| `RA` | deg | Sky coordinate (J2000) from source catalog. |
| `DEC` | deg | Sky coordinate (J2000) from source catalog. |
| `morphology` | --- | Identifier or categorical label. |
| `redshift_helio` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `redshift_helio_e` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `redshift_cmb` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `redshift_cmb_e` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `distance` | Mpc | Distance estimate. |
| `distance_e` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `distance_method` | --- | Identifier or categorical label. |
| `RC_survey` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `ext_f` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `ext_n` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `ext_g` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `ext_r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `ext_z` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `has_f-band` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `has_n-band` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `has_g-band` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `has_r-band` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `has_z-band` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `has_w1-band` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `has_w2-band` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `AutoProf_flags` | flag | Quality/control flag from source catalog. |

## `model_fits.csv`

| column | unit | notes |
|---|---|---|
| `name` | --- | Identifier or categorical label. |
| `rc_model:Tan:r_t` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `rc_model:Tan:v_c` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `rc_model:Tan:v0` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `rc_model:Tan:x0` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `rc_model:Tanh:r_t` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `rc_model:Tanh:v_c` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `rc_model:Tanh:v0` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `rc_model:Tanh:x0` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `rc_model:C97:r_t` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `rc_model:C97:v_c` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `rc_model:C97:beta` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `rc_model:C97:gamma` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `rc_model:C97:v0` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `rc_model:C97:x0` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:Re|f` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:Ie|f` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:M|f` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:n|f` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:Re|n` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:Ie|n` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:M|n` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:n|n` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:Re|g` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:Ie|g` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:M|g` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:n|g` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:Re|r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:Ie|r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:M|r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:n|r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:Re|z` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:Ie|z` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:M|z` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:n|z` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:Re|w1` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:Ie|w1` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:M|w1` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:n|w1` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `sersic:Re|w2` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |

## `structural_parameters.csv`

| column | unit | notes |
|---|---|---|
| `name` | --- | Identifier or categorical label. |
| `RA` | deg | Sky coordinate (J2000) from source catalog. |
| `DEC` | deg | Sky coordinate (J2000) from source catalog. |
| `redshift_helio` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `redshift_helio_e` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR|Ri22:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR:E|Ri22:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR|Ri22.5:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR:E|Ri22.5:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR|Ri23:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR:E|Ri23:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR|Ri23.5:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR:E|Ri23.5:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR|Ri24:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR:E|Ri24:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR|Ri24.5:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR:E|Ri24.5:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR|Ri25:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR:E|Ri25:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR|Ri25.5:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR:E|Ri25.5:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR|Ri26:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR:E|Ri26:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR|Rp20:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR:E|Rp20:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR|Rp30:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR:E|Rp30:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR|Rp40:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR:E|Rp40:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR|Rp50:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR:E|Rp50:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR|Rp60:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR:E|Rp60:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR|Rp70:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR:E|Rp70:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR|Rp80:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR:E|Rp80:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR|Re1.5:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR:E|Re1.5:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
| `appR|Re2:r` | TODO: requires upstream docs not present locally. | TODO: requires upstream docs not present locally. |
