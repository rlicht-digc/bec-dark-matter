#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import py_compile
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path('/Users/russelllicht/bec-dark-matter').resolve()
HUB = ROOT / 'bec_rar_identity'
STAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

IGNORE_PATTERNS = ('outputs_bh', 'outputs_bh3', '/.git/', 'singularity', 'blackhole', 'black_hole', 'bh_')


@dataclass
class Action:
    kind: str
    source: Optional[str]
    target: Optional[str]
    note: str


actions: List[Action] = []


def log_action(kind: str, source: Optional[Path], target: Optional[Path], note: str) -> None:
    actions.append(
        Action(
            kind=kind,
            source=str(source) if source else None,
            target=str(target) if target else None,
            note=note,
        )
    )


def ensure_dirs() -> None:
    dirs = [
        HUB,
        HUB / 'datasets',
        HUB / 'datasets' / 'sparc_unified',
        HUB / 'datasets' / 'tng_3000x50_verified',
        HUB / 'datasets' / 'tng_48133x50_big_base',
        HUB / 'tests',
        HUB / 'tests' / 'core',
        HUB / 'tests' / 'core' / 'scripts',
        HUB / 'tests' / 'core' / 'outputs',
        HUB / 'tests' / 'extended',
        HUB / 'tests' / 'extended' / 'scripts',
        HUB / 'tests' / 'extended' / 'notes',
        HUB / 'runs',
        HUB / 'runs' / 'referee_battery',
        HUB / 'runs' / 'universality_audit',
        HUB / 'runs' / 'legacy_outputs',
        HUB / 'artifacts',
        HUB / 'artifacts' / 'summaries',
        HUB / 'artifacts' / 'figures',
        HUB / 'artifacts' / 'tables',
        HUB / 'artifacts' / 'markdown',
        HUB / 'manifests',
        HUB / 'archive',
        HUB / 'archive' / 'root_orphans',
        HUB / 'archive' / 'root_orphans' / STAMP,
        HUB / 'archive' / 'nonviable',
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def safe_unlink(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def symlink_force(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        safe_unlink(dst)
    os.symlink(str(src), str(dst))
    log_action('symlink', src, dst, 'linked into canonical hub')


def ignored(path: Path) -> bool:
    s = str(path).lower().replace('\\', '/')
    return any(p in s for p in IGNORE_PATTERNS)


def find_test_scripts() -> List[Path]:
    pipeline = ROOT / 'analysis' / 'pipeline'
    scripts = sorted(pipeline.glob('test_*.py'))
    out = []
    for p in scripts:
        if ignored(p):
            continue
        out.append(p)
    return out


def find_summary_jsons() -> List[Path]:
    out: List[Path] = []
    for p in ROOT.rglob('summary_*.json'):
        if ignored(p):
            continue
        if str(HUB) in str(p):
            continue
        out.append(p)
    return sorted(out)


def find_artifacts() -> Dict[str, List[Path]]:
    figures: List[Path] = []
    tables: List[Path] = []
    markdown: List[Path] = []

    for p in ROOT.rglob('*'):
        if not p.is_file():
            continue
        if str(HUB) in str(p):
            continue
        if ignored(p):
            continue
        low = p.suffix.lower()
        pstr = str(p)
        is_relevant = (
            'analysis/results' in pstr
            or 'rerun_outputs' in pstr
            or 'outputs/universality_audit' in pstr
            or p.name.startswith('fig_')
            or p.name.startswith('summary_')
            or p.name in {'dataset_lineage.md', 'summary_rar_tng.md'}
        )
        if not is_relevant:
            continue
        if low == '.png':
            figures.append(p)
        elif low in {'.csv', '.parquet'}:
            tables.append(p)
        elif low in {'.md'}:
            markdown.append(p)

    return {
        'figures': sorted(figures),
        'tables': sorted(tables),
        'markdown': sorted(markdown),
    }


def compile_status(py_path: Path) -> Tuple[bool, Optional[str]]:
    try:
        py_compile.compile(str(py_path), doraise=True)
        return True, None
    except Exception as exc:  # noqa
        return False, str(exc)


def json_status(json_path: Path) -> Tuple[bool, Optional[str]]:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json.load(f)
        return True, None
    except Exception as exc:  # noqa
        return False, str(exc)


def ensure_dataset_links() -> List[Dict[str, object]]:
    mappings = [
        (ROOT / 'analysis' / 'results' / 'rar_points_unified.csv', HUB / 'datasets' / 'sparc_unified' / 'rar_points_unified.csv'),
        (ROOT / 'analysis' / 'results' / 'galaxy_results_unified.csv', HUB / 'datasets' / 'sparc_unified' / 'galaxy_results_unified.csv'),
        (ROOT / 'datasets' / 'TNG_RAR_3000x50_SOFT1p5_RUN201626_VERIFIED' / 'rar_points_CLEAN.parquet', HUB / 'datasets' / 'tng_3000x50_verified' / 'rar_points_CLEAN.parquet'),
        (ROOT / 'datasets' / 'TNG_RAR_3000x50_SOFT1p5_RUN201626_VERIFIED' / 'rar_points.csv', HUB / 'datasets' / 'tng_3000x50_verified' / 'rar_points.csv'),
        (ROOT / 'datasets' / 'TNG_RAR_3000x50_SOFT1p5_RUN201626_VERIFIED' / 'meta' / 'master_catalog.csv', HUB / 'datasets' / 'tng_3000x50_verified' / 'master_catalog.csv'),
        (ROOT / 'datasets' / 'TNG_RAR_48133x50_SOFT1p5_RUN20260223_061026_BIG_BASE' / 'rar_points.parquet', HUB / 'datasets' / 'tng_48133x50_big_base' / 'rar_points.parquet'),
        (ROOT / 'datasets' / 'TNG_RAR_48133x50_SOFT1p5_RUN20260223_061026_BIG_BASE' / 'meta' / 'master_catalog.csv', HUB / 'datasets' / 'tng_48133x50_big_base' / 'master_catalog.csv'),
    ]

    rows: List[Dict[str, object]] = []
    for src, dst in mappings:
        exists = src.exists()
        if exists:
            symlink_force(src, dst)
        else:
            log_action('missing_dataset', src, dst, 'dataset source missing')
        rows.append(
            {
                'source_path': str(src),
                'hub_link': str(dst),
                'exists': exists,
                'size_bytes': src.stat().st_size if exists else None,
            }
        )
    return rows


def link_runs() -> None:
    candidates = [
        (ROOT / 'rerun_outputs', HUB / 'runs' / 'referee_battery' / 'rerun_outputs'),
        (ROOT / 'outputs' / '20260223_192210', HUB / 'runs' / 'legacy_outputs' / '20260223_192210'),
        (ROOT / 'outputs' / 'universality_audit', HUB / 'runs' / 'universality_audit' / 'universality_audit'),
        (ROOT / 'analysis' / 'results' / 'tng_sparc_composition_sweep', HUB / 'runs' / 'universality_audit' / 'tng_sparc_composition_sweep'),
    ]
    for src, dst in candidates:
        if src.exists():
            symlink_force(src, dst)
        else:
            log_action('missing_run', src, dst, 'run directory missing')


def link_scripts_and_build_inventory() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    core_sources = [
        ROOT / 'run_all_referee_tests.py',
        ROOT / 'common_rar.py',
        ROOT / 'analysis' / 'pipeline' / 'run_referee_required_tests.py',
        ROOT / 'analysis' / 'pipeline' / 'run_universality_audit.py',
        ROOT / 'analysis' / 'pipeline' / 'tng_dataset_lineage.py',
        ROOT / 'analysis' / 'pipeline' / 'test_tng_sparc_composition_sweep.py',
        ROOT / 'analysis' / 'pipeline' / 'test_tng_sparc_fairness_gap.py',
    ]
    for src in core_sources:
        if not src.exists():
            rows.append(
                {
                    'category': 'core',
                    'script_name': src.name,
                    'source_path': str(src),
                    'hub_link': None,
                    'exists': False,
                    'syntax_ok': None,
                    'syntax_error': 'missing file',
                    'mentions_48133': None,
                }
            )
            continue
        dst = HUB / 'tests' / 'core' / 'scripts' / src.name
        symlink_force(src, dst)
        ok, err = compile_status(src)
        txt = src.read_text(encoding='utf-8', errors='ignore')
        rows.append(
            {
                'category': 'core',
                'script_name': src.name,
                'source_path': str(src),
                'hub_link': str(dst),
                'exists': True,
                'syntax_ok': ok,
                'syntax_error': err,
                'mentions_48133': ('48133' in txt or 'BIG_BASE' in txt),
            }
        )

    for src in find_test_scripts():
        dst = HUB / 'tests' / 'extended' / 'scripts' / src.name
        symlink_force(src, dst)
        ok, err = compile_status(src)
        txt = src.read_text(encoding='utf-8', errors='ignore')
        rows.append(
            {
                'category': 'extended',
                'script_name': src.name,
                'source_path': str(src),
                'hub_link': str(dst),
                'exists': True,
                'syntax_ok': ok,
                'syntax_error': err,
                'mentions_48133': ('48133' in txt or 'BIG_BASE' in txt),
            }
        )

    return rows


def link_artifacts_and_build_inventory() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    # Summaries (JSON)
    for src in find_summary_jsons():
        rel = src.relative_to(ROOT)
        dst = HUB / 'artifacts' / 'summaries' / rel
        symlink_force(src, dst)
        ok, err = json_status(src)
        rows.append(
            {
                'type': 'summary_json',
                'source_path': str(src),
                'hub_link': str(dst),
                'exists': True,
                'valid': ok,
                'error': err,
                'size_bytes': src.stat().st_size,
            }
        )

    artifacts = find_artifacts()
    for kind, paths in artifacts.items():
        for src in paths:
            rel = src.relative_to(ROOT)
            dst = HUB / 'artifacts' / kind / rel
            symlink_force(src, dst)
            rows.append(
                {
                    'type': kind,
                    'source_path': str(src),
                    'hub_link': str(dst),
                    'exists': True,
                    'valid': src.stat().st_size > 0,
                    'error': None,
                    'size_bytes': src.stat().st_size,
                }
            )

    return rows


def move_root_orphans() -> List[Dict[str, str]]:
    """Move ambiguous root-level output artifacts into canonical archive."""
    root_orphans = [
        ROOT / 'fig_alpha_star.png',
        ROOT / 'fig_mass_matched_phase.png',
        ROOT / 'fig_phase_null.png',
        ROOT / 'fig_xi_organizing.png',
        ROOT / 'summary_alpha_star_convergence.json',
        ROOT / 'summary_alpha_star_convergence_table.csv',
        ROOT / 'summary_dataset_lineage.json',
        ROOT / 'summary_mass_matched_phase.json',
        ROOT / 'summary_phase_peak_null.json',
        ROOT / 'summary_xi_organizing.json',
        ROOT / 'dataset_lineage.md',
    ]

    moved: List[Dict[str, str]] = []
    dst_root = HUB / 'archive' / 'root_orphans' / STAMP
    for src in root_orphans:
        if not src.exists():
            continue
        dst = dst_root / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved.append({'source': str(src), 'target': str(dst)})
        log_action('move', src, dst, 'moved ambiguous root artifact into archive/root_orphans')
    return moved


def clean_generated_clutter() -> List[str]:
    removed: List[str] = []

    # Remove pycache under analysis/pipeline only (safe generated clutter)
    for p in ROOT.rglob('__pycache__'):
        if str(HUB) in str(p):
            continue
        if '/outputs_bh' in str(p).lower() or '/outputs_bh3' in str(p).lower():
            continue
        shutil.rmtree(p, ignore_errors=True)
        removed.append(str(p))
        log_action('remove', p, None, 'removed generated __pycache__ directory')

    # Remove macOS DS_Store files in repo root tree except hidden system areas
    for p in ROOT.rglob('.DS_Store'):
        if str(HUB) in str(p):
            continue
        try:
            p.unlink()
            removed.append(str(p))
            log_action('remove', p, None, 'removed generated .DS_Store file')
        except Exception:
            pass

    return removed


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_readme(test_rows: List[Dict[str, object]], artifact_rows: List[Dict[str, object]], dataset_rows: List[Dict[str, object]]) -> None:
    n_core = sum(1 for r in test_rows if r['category'] == 'core' and r['exists'])
    n_ext = sum(1 for r in test_rows if r['category'] == 'extended' and r['exists'])
    n_tests_total = n_core + n_ext
    n_summ = sum(1 for r in artifact_rows if r['type'] == 'summary_json')
    n_fig = sum(1 for r in artifact_rows if r['type'] == 'figures')
    n_tab = sum(1 for r in artifact_rows if r['type'] == 'tables')
    n_md = sum(1 for r in artifact_rows if r['type'] == 'markdown')

    txt = f"""# BEC = RAR Identity Hub

This directory is the canonical, BH/singularity-excluded organization for BEC↔RAR identity work.

## Schema
- `datasets/`: canonical dataset links (SPARC unified, TNG 3000x50 verified, TNG 48133x50 big-base)
- `tests/core/scripts/`: core referee battery orchestration scripts
- `tests/extended/scripts/`: all discovered `test_*.py` BEC↔RAR analysis scripts
- `runs/`: run-level output roots (referee battery, universality audit, legacy outputs)
- `artifacts/summaries/`: linked `summary_*.json` files (mirrored by original path)
- `artifacts/figures/`, `artifacts/tables/`, `artifacts/markdown/`: linked plots/tables/docs relevant to tests
- `manifests/`: inventories, viability checks, relocation and cleanup logs
- `archive/root_orphans/`: previously ambiguous root-level artifacts moved out of project root
- `archive/nonviable/`: reserved for invalid/unusable assets if encountered

## Current inventory snapshot
- test scripts organized: `{n_tests_total}` (core `{n_core}`, extended `{n_ext}`)
- summary JSON artifacts indexed: `{n_summ}`
- figure artifacts indexed: `{n_fig}`
- table artifacts indexed: `{n_tab}`
- markdown artifacts indexed: `{n_md}`
- datasets linked: `{sum(1 for r in dataset_rows if r['exists'])}` / `{len(dataset_rows)}`

## 48k dataset coverage
- `datasets/tng_48133x50_big_base/rar_points.parquet` points to the 48,133×50 big-base per-point dataset.

## Manifests to use
- `manifests/test_inventory.csv`
- `manifests/artifact_inventory.csv`
- `manifests/dataset_inventory.csv`
- `manifests/viability_report.json`
- `manifests/relocation_map.json`
- `manifests/cleanup_actions.json`
"""
    (HUB / 'README.md').write_text(txt, encoding='utf-8')

    schema = """# Common Schema

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
"""
    (HUB / 'SCHEMA.md').write_text(schema, encoding='utf-8')


def main() -> None:
    ensure_dirs()

    dataset_rows = ensure_dataset_links()
    link_runs()
    test_rows = link_scripts_and_build_inventory()
    artifact_rows = link_artifacts_and_build_inventory()

    moved_orphans = move_root_orphans()
    removed_clutter = clean_generated_clutter()

    # inventories
    write_csv(
        HUB / 'manifests' / 'test_inventory.csv',
        test_rows,
        ['category', 'script_name', 'source_path', 'hub_link', 'exists', 'syntax_ok', 'syntax_error', 'mentions_48133'],
    )
    write_csv(
        HUB / 'manifests' / 'artifact_inventory.csv',
        artifact_rows,
        ['type', 'source_path', 'hub_link', 'exists', 'valid', 'error', 'size_bytes'],
    )
    write_csv(
        HUB / 'manifests' / 'dataset_inventory.csv',
        dataset_rows,
        ['source_path', 'hub_link', 'exists', 'size_bytes'],
    )

    viable_tests = sum(1 for r in test_rows if r['exists'] and r['syntax_ok'])
    invalid_tests = [r for r in test_rows if r['exists'] and not r['syntax_ok']]
    valid_json = sum(1 for r in artifact_rows if r['type'] == 'summary_json' and r['valid'])
    invalid_json = [r for r in artifact_rows if r['type'] == 'summary_json' and not r['valid']]

    viability = {
        'status': 'OK',
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'scope': 'BEC=RAR identity audit (BH/singularity excluded)',
        'counts': {
            'test_scripts_total': len(test_rows),
            'test_scripts_syntax_ok': viable_tests,
            'test_scripts_syntax_fail': len(invalid_tests),
            'summary_json_total': sum(1 for r in artifact_rows if r['type'] == 'summary_json'),
            'summary_json_valid': valid_json,
            'summary_json_invalid': len(invalid_json),
        },
        'invalid_tests': invalid_tests,
        'invalid_summary_json': invalid_json,
        'notes': [
            'Out-of-place root artifacts were moved into archive/root_orphans and indexed.',
            'Generated clutter (__pycache__, .DS_Store) was removed.',
            'No BH/singularity outputs were linked into this hub.',
        ],
    }
    (HUB / 'manifests' / 'viability_report.json').write_text(json.dumps(viability, indent=2), encoding='utf-8')

    relocation_map = {
        'status': 'OK',
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'hub_root': str(HUB),
        'moved_root_orphans': moved_orphans,
        'run_links': [
            {
                'source_path': str(ROOT / 'rerun_outputs'),
                'hub_link': str(HUB / 'runs' / 'referee_battery' / 'rerun_outputs'),
            },
            {
                'source_path': str(ROOT / 'outputs' / '20260223_192210'),
                'hub_link': str(HUB / 'runs' / 'legacy_outputs' / '20260223_192210'),
            },
            {
                'source_path': str(ROOT / 'outputs' / 'universality_audit'),
                'hub_link': str(HUB / 'runs' / 'universality_audit' / 'universality_audit'),
            },
            {
                'source_path': str(ROOT / 'analysis' / 'results' / 'tng_sparc_composition_sweep'),
                'hub_link': str(HUB / 'runs' / 'universality_audit' / 'tng_sparc_composition_sweep'),
            },
        ],
    }
    (HUB / 'manifests' / 'relocation_map.json').write_text(json.dumps(relocation_map, indent=2), encoding='utf-8')

    cleanup = {
        'status': 'OK',
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'removed_generated_clutter': removed_clutter,
        'actions_logged': [a.__dict__ for a in actions],
    }
    (HUB / 'manifests' / 'cleanup_actions.json').write_text(json.dumps(cleanup, indent=2), encoding='utf-8')

    write_readme(test_rows, artifact_rows, dataset_rows)

    print(f'hub={HUB}')
    print(f'test_scripts={len(test_rows)} syntax_ok={viable_tests} syntax_fail={len(invalid_tests)}')
    print(f'summary_json={sum(1 for r in artifact_rows if r["type"]=="summary_json")} valid={valid_json} invalid={len(invalid_json)}')
    print(f'moved_root_orphans={len(moved_orphans)}')
    print(f'removed_generated_clutter={len(removed_clutter)}')


if __name__ == '__main__':
    main()
