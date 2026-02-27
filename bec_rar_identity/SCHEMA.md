# Common Schema

Each asset in this hub follows a standard metadata schema in manifests:
- `source_path`: original absolute path
- `hub_link`: canonical path in `bec_rar_identity/`
- `exists`: source existence boolean
- `valid`: basic viability check result (syntax for `.py`, parse for `.json`, non-zero size otherwise)
- `error`: validation error message if invalid
- `size_bytes`: file size in bytes if present

Test inventory columns:
- `category`: `core` or `extended`
- `script_name`
- `syntax_ok`
- `mentions_48133`: whether script text references the 48k dataset marker
