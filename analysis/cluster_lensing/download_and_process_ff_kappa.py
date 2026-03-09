#!/usr/bin/env python3
"""
Download and process Frontier Fields convergence (kappa) maps for the 6 HFF clusters.

Uses Zitrin LTM-Gauss models from the MAST archive (latest available version per cluster).
Computes azimuthally averaged kappa profiles, surface mass density, and enclosed mass.

Requirements: numpy, scipy, astropy
"""

import os
import sys
import urllib.request
import urllib.error
import numpy as np
from scipy.integrate import cumulative_trapezoid

try:
    from astropy.io import fits
    from astropy.cosmology import FlatLambdaCDM
    import astropy.units as u
    import astropy.constants as const
except ImportError:
    print("ERROR: astropy is required. Install with: pip install astropy")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Cosmology (Planck 2018)
cosmo = FlatLambdaCDM(H0=67.4, Om0=0.315)

# Fiducial source redshift for Sigma_cr calculation
Z_SOURCE = 2.0

# Output paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, '..', '..'))
DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, 'raw_data', 'observational',
                            'cluster_rar', 'frontier_fields')
OUTPUT_CSV = os.path.join(PROJECT_ROOT, 'raw_data', 'observational',
                          'cluster_rar', 'frontier_fields_kappa_profiles.csv')

# Cluster metadata: (directory_name, redshift, version_available)
# Versions determined by checking MAST archive directory listings (2026-03-08):
#   abell2744, macs0416 have v3; the rest have v1
CLUSTERS = {
    'abell2744':  {'z': 0.308, 'version': 'v3'},
    'abell370':   {'z': 0.375, 'version': 'v1'},
    'abells1063': {'z': 0.348, 'version': 'v1'},
    'macs0416':   {'z': 0.396, 'version': 'v3'},
    'macs0717':   {'z': 0.548, 'version': 'v1'},
    'macs1149':   {'z': 0.544, 'version': 'v1'},
}

MAST_BASE = "https://archive.stsci.edu/pub/hlsp/frontier"

# Radial binning
R_MIN_KPC = 10.0    # inner radius in kpc
N_RADIAL_BINS = 60  # number of logarithmic bins


def build_url(cluster, version):
    """Build the MAST download URL for a cluster's kappa FITS file."""
    fname = f"hlsp_frontier_model_{cluster}_zitrin-ltm-gauss_{version}_kappa.fits"
    url = f"{MAST_BASE}/{cluster}/models/zitrin-ltm-gauss/{version}/{fname}"
    return url, fname


def download_kappa_maps():
    """Download kappa FITS files from MAST. Returns dict of {cluster: filepath}."""
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    downloaded = {}
    failed = []

    for cluster, info in CLUSTERS.items():
        version = info['version']
        url, fname = build_url(cluster, version)
        local_path = os.path.join(DOWNLOAD_DIR, fname)

        # Skip if already downloaded
        if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
            print(f"  [CACHED] {cluster}: {fname}")
            downloaded[cluster] = local_path
            continue

        print(f"  Downloading {cluster} ({version})...")
        print(f"    URL: {url}")

        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (BEC-Dark-Matter research project)'
            })
            with urllib.request.urlopen(req, timeout=120) as response:
                data = response.read()
                with open(local_path, 'wb') as f:
                    f.write(data)
            size_mb = len(data) / 1e6
            print(f"    OK ({size_mb:.1f} MB)")
            downloaded[cluster] = local_path
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            print(f"    FAILED: {e}")
            failed.append((cluster, url, fname))

    if failed:
        print("\n" + "=" * 70)
        print("MANUAL DOWNLOAD REQUIRED for the following files:")
        print("=" * 70)
        for cluster, url, fname in failed:
            print(f"\n  Cluster: {cluster}")
            print(f"  URL:     {url}")
            print(f"  Save as: {os.path.join(DOWNLOAD_DIR, fname)}")
        print(f"\nAlternatively, visit: {MAST_BASE}/")
        print("Navigate to each cluster -> models -> zitrin-ltm-gauss -> version")
        print("and download the *_kappa.fits file.")
        print("=" * 70 + "\n")

    return downloaded


def get_pixel_scale(header):
    """
    Extract pixel scale in arcsec/pixel from FITS header.
    Tries CD matrix first, then CDELT keywords, then falls back to 0.065"/pix.
    """
    # Try CD matrix
    if 'CD1_1' in header and 'CD1_2' in header:
        cd11 = header['CD1_1']
        cd12 = header['CD1_2']
        # Pixel scale = sqrt(cd11^2 + cd12^2) in degrees
        scale_deg = np.sqrt(cd11**2 + cd12**2)
        return scale_deg * 3600.0  # arcsec/pixel

    # Try CDELT
    if 'CDELT1' in header:
        return abs(header['CDELT1']) * 3600.0  # arcsec/pixel

    # Fallback: typical HFF model pixel scale
    print("    WARNING: No WCS pixel scale found, using default 0.065 arcsec/pixel")
    return 0.065


def compute_sigma_cr(z_l, z_s):
    """
    Compute the critical surface mass density Sigma_cr in kg/m^2.

    Sigma_cr = (c^2 / 4*pi*G) * (D_s / (D_l * D_ls))

    Parameters
    ----------
    z_l : float
        Lens (cluster) redshift.
    z_s : float
        Source redshift.

    Returns
    -------
    sigma_cr : float
        Critical surface mass density in kg/m^2.
    """
    D_l = cosmo.angular_diameter_distance(z_l).to(u.m).value
    D_s = cosmo.angular_diameter_distance(z_s).to(u.m).value
    D_ls = cosmo.angular_diameter_distance_z1z2(z_l, z_s).to(u.m).value

    c_val = const.c.to(u.m / u.s).value       # m/s
    G_val = const.G.to(u.m**3 / (u.kg * u.s**2)).value  # m^3/(kg*s^2)

    sigma_cr = (c_val**2 / (4.0 * np.pi * G_val)) * (D_s / (D_l * D_ls))
    return sigma_cr


def azimuthal_average(kappa_2d, pixel_scale_arcsec, z_l, r_min_kpc, n_bins):
    """
    Azimuthally average a 2D kappa map in logarithmic radial bins.

    Parameters
    ----------
    kappa_2d : 2D ndarray
        Convergence map.
    pixel_scale_arcsec : float
        Pixel scale in arcsec/pixel.
    z_l : float
        Cluster (lens) redshift.
    r_min_kpc : float
        Minimum radius in kpc.
    n_bins : int
        Number of logarithmic radial bins.

    Returns
    -------
    r_kpc : 1D ndarray
        Bin-center radii in kpc.
    kappa_profile : 1D ndarray
        Azimuthally averaged kappa at each radius.
    """
    ny, nx = kappa_2d.shape

    # Find the center: use peak kappa pixel
    # Smooth slightly to avoid noise peaks
    from scipy.ndimage import gaussian_filter
    kappa_smooth = gaussian_filter(kappa_2d.astype(float), sigma=3)
    cy, cx = np.unravel_index(np.nanargmax(kappa_smooth), kappa_smooth.shape)

    # Convert pixel scale to kpc/pixel
    # angular_diameter_distance gives physical transverse distance
    D_A = cosmo.angular_diameter_distance(z_l).to(u.kpc).value  # kpc
    arcsec_per_kpc = 1.0 / (D_A * np.pi / (180.0 * 3600.0))  # arcsec per kpc
    kpc_per_pixel = pixel_scale_arcsec / arcsec_per_kpc

    # Build distance array from center in kpc
    y_idx, x_idx = np.mgrid[0:ny, 0:nx]
    r_pixels = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
    r_kpc_map = r_pixels * kpc_per_pixel

    # Determine max radius (distance to nearest edge from center)
    r_max_kpc = min(cx, nx - cx - 1, cy, ny - cy - 1) * kpc_per_pixel
    r_max_kpc = max(r_max_kpc, r_min_kpc * 2)  # ensure at least some range

    # Logarithmic radial bins
    r_edges = np.logspace(np.log10(r_min_kpc), np.log10(r_max_kpc), n_bins + 1)
    r_centers = np.sqrt(r_edges[:-1] * r_edges[1:])  # geometric mean

    kappa_profile = np.zeros(n_bins)
    valid_mask = np.isfinite(kappa_2d)

    for i in range(n_bins):
        mask = (r_kpc_map >= r_edges[i]) & (r_kpc_map < r_edges[i + 1]) & valid_mask
        if np.any(mask):
            kappa_profile[i] = np.nanmean(kappa_2d[mask])
        else:
            kappa_profile[i] = np.nan

    return r_centers, kappa_profile


def process_cluster(cluster, fits_path, z_l, z_s=Z_SOURCE):
    """
    Process a single cluster's kappa map.

    Returns
    -------
    dict with keys: r_kpc, kappa, sigma_kg_m2, M_2D_Msun, M_3D_approx_Msun
    """
    print(f"\n  Processing {cluster} (z={z_l:.3f})...")

    # Read FITS
    with fits.open(fits_path) as hdul:
        # Kappa map is typically in the primary HDU or first extension
        kappa_2d = None
        header = None
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 2:
                kappa_2d = hdu.data.astype(np.float64)
                header = hdu.header
                break

        if kappa_2d is None:
            print(f"    ERROR: No 2D data found in {fits_path}")
            return None

    print(f"    Map shape: {kappa_2d.shape}")
    print(f"    kappa range: [{np.nanmin(kappa_2d):.4f}, {np.nanmax(kappa_2d):.4f}]")

    # Pixel scale
    pixel_scale = get_pixel_scale(header)
    print(f"    Pixel scale: {pixel_scale:.4f} arcsec/pixel")

    # Azimuthal average
    r_kpc, kappa_prof = azimuthal_average(
        kappa_2d, pixel_scale, z_l, R_MIN_KPC, N_RADIAL_BINS
    )

    # Remove bins with NaN
    good = np.isfinite(kappa_prof) & (kappa_prof > 0)
    if not np.any(good):
        print(f"    ERROR: No valid radial bins for {cluster}")
        return None

    r_kpc = r_kpc[good]
    kappa_prof = kappa_prof[good]

    # Critical surface density
    sigma_cr = compute_sigma_cr(z_l, z_s)
    print(f"    Sigma_cr = {sigma_cr:.4e} kg/m^2")

    # Surface mass density
    sigma_kg_m2 = kappa_prof * sigma_cr

    # Convert to Msun/kpc^2 for mass integration
    Msun_kg = const.M_sun.to(u.kg).value
    kpc_to_m = u.kpc.to(u.m)
    sigma_Msun_kpc2 = sigma_kg_m2 * (kpc_to_m**2) / Msun_kg  # Msun/kpc^2

    # Enclosed projected mass M_2D(<R)
    # M_2D(<R) = integral_0^R Sigma(R') * 2*pi*R' dR'
    # Use cumulative trapezoidal integration
    integrand = sigma_Msun_kpc2 * 2.0 * np.pi * r_kpc  # Msun/kpc^2 * kpc = Msun/kpc
    M_2D = np.zeros_like(r_kpc)
    if len(r_kpc) > 1:
        M_2D_cum = cumulative_trapezoid(integrand, r_kpc, initial=0.0)
        # Add contribution from inner region (0 to r_min) assuming constant density
        M_inner = sigma_Msun_kpc2[0] * np.pi * r_kpc[0]**2
        M_2D = M_2D_cum + M_inner
    else:
        M_2D[0] = sigma_Msun_kpc2[0] * np.pi * r_kpc[0]**2

    # Approximate 3D mass: M_3D ~ M_2D * (4/pi) for NFW-like profiles
    M_3D = M_2D * (4.0 / np.pi)

    print(f"    Radial range: {r_kpc[0]:.1f} - {r_kpc[-1]:.1f} kpc")
    print(f"    M_2D at outermost bin: {M_2D[-1]:.3e} Msun")
    print(f"    M_3D at outermost bin: {M_3D[-1]:.3e} Msun")

    return {
        'r_kpc': r_kpc,
        'kappa': kappa_prof,
        'sigma_kg_m2': sigma_kg_m2,
        'M_2D_Msun': M_2D,
        'M_3D_approx_Msun': M_3D,
    }


def save_profiles(all_results):
    """Save all cluster profiles to a single CSV file."""
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    rows = []
    for cluster, info in CLUSTERS.items():
        if cluster not in all_results or all_results[cluster] is None:
            continue
        res = all_results[cluster]
        z = info['z']
        for i in range(len(res['r_kpc'])):
            rows.append({
                'cluster': cluster,
                'z': z,
                'r_kpc': res['r_kpc'][i],
                'kappa': res['kappa'][i],
                'sigma_kg_m2': res['sigma_kg_m2'][i],
                'M_2D_Msun': res['M_2D_Msun'][i],
                'M_3D_approx_Msun': res['M_3D_approx_Msun'][i],
            })

    if not rows:
        print("\nERROR: No data to save.")
        return

    # Write CSV
    header = 'cluster,z,r_kpc,kappa,sigma_kg_m2,M_2D_Msun,M_3D_approx_Msun'
    with open(OUTPUT_CSV, 'w') as f:
        f.write(header + '\n')
        for row in rows:
            line = (f"{row['cluster']},{row['z']:.3f},{row['r_kpc']:.4f},"
                    f"{row['kappa']:.6e},{row['sigma_kg_m2']:.6e},"
                    f"{row['M_2D_Msun']:.6e},{row['M_3D_approx_Msun']:.6e}")
            f.write(line + '\n')

    print(f"\nSaved {len(rows)} rows to {OUTPUT_CSV}")
    print(f"  Clusters included: {sorted(set(r['cluster'] for r in rows))}")


def main():
    print("=" * 70)
    print("Frontier Fields Kappa Map Download & Processing")
    print("  Model: Zitrin LTM-Gauss (latest version per cluster)")
    print(f"  Source redshift (fiducial): z_s = {Z_SOURCE}")
    print(f"  Cosmology: H0={cosmo.H0.value}, Om0={cosmo.Om0}")
    print("=" * 70)

    # Step 1: Download
    print("\nStep 1: Downloading kappa maps from MAST...")
    downloaded = download_kappa_maps()

    if not downloaded:
        print("\nNo files downloaded. Cannot proceed with processing.")
        print("Please download the files manually and re-run this script.")
        return

    print(f"\n  Successfully obtained {len(downloaded)}/{len(CLUSTERS)} kappa maps.")

    # Step 2: Process
    print("\nStep 2: Processing kappa maps...")
    all_results = {}
    for cluster, fits_path in downloaded.items():
        z_l = CLUSTERS[cluster]['z']
        result = process_cluster(cluster, fits_path, z_l)
        all_results[cluster] = result

    # Step 3: Save
    print("\nStep 3: Saving profiles...")
    save_profiles(all_results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for cluster in CLUSTERS:
        status = "OK" if cluster in all_results and all_results[cluster] is not None else "MISSING"
        if status == "OK":
            res = all_results[cluster]
            print(f"  {cluster:12s}: {len(res['r_kpc']):3d} radial bins, "
                  f"M_3D(max) = {res['M_3D_approx_Msun'][-1]:.2e} Msun")
        else:
            print(f"  {cluster:12s}: {status}")
    print("=" * 70)


if __name__ == '__main__':
    main()
