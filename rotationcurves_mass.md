# Datasets bridging black hole masses, rotation curves, and BEC dark matter

**The overlap between direct black hole mass measurements and published galaxy rotation curves is small but scientifically potent: roughly 7–10 galaxies currently satisfy both criteria from SPARC-adjacent samples, with NGC 4258 standing as the single most valuable target.** This report maps every major dataset relevant to the proposed BEC dark matter framework, identifies specific cross-match galaxies, and provides access details for each resource. The landscape is shifting rapidly — JWST stellar-dynamical measurements added three new spiral-galaxy BH masses in 2024–2026 (M81, M94, NGC 4826), and BIG-SPARC (~4,000 rotation curves) will dramatically expand overlap when released. For the broader BEC framework, PHANGS CO rotation curves at 150 pc resolution offer the best existing data for soliton core testing, while Mistele et al. 2024 provides the most refined weak lensing RAR extending kinematic results by 2.5 dex.

---

## Part 1: Direct black hole mass catalogs and their rotation curve overlap

### 1A. Reverberation mapping databases contain ~100+ AGN but minimal rotation curve overlap

The **AGN Black Hole Mass Database** (Bentz & Katz 2015, PASP 127, 67) at **http://www.astro.gsu.edu/AGNmass/** is the definitive RM compilation — continuously updated, now listing **~100+ unique AGN** with downloadable CSV/TSV files. Users can adjust the virial scale factor ⟨f⟩ (default 4.3 ± 1.1 from Grier+2013). Columns include log M_BH with asymmetric uncertainties, coordinates, redshift, and individual lag/width measurements per emission line.

Recent RM campaigns have expanded the sample substantially. The SDSS-RM project (Shen et al. 2024, ApJS 272, 26) produced **339 lag detections** across 849 quasars over 11 years, with final data at https://ariel.astro.illinois.edu/sdssrm/. The SEAMBH project (Du et al. 2015–2025 series, now Paper XV) added ~20+ super-Eddington AGN. However, both SDSS-RM quasars (z = 0.1–4.5) and most SEAMBH targets are far too distant for resolved HI rotation curves.

The RM–rotation curve overlap is sparse. **NGC 4051** (log M_BH = 5.89) is the only confirmed RM galaxy in the SPARC 175-galaxy sample. NGC 4151 (log M_BH = 7.37) has VLA HI data from Mundell et al. 1999 but complex bar-driven kinematics and is not in SPARC. NGC 3227 (log M_BH = 6.68) has some HI data but is tidally interacting with NGC 3226, complicating the rotation curve. **NGC 4395** (log M_BH = 5.45), an extremely nearby (d ≈ 4.4 Mpc) bulgeless dwarf Seyfert, is a promising candidate worth checking against SPARC and BIG-SPARC. No published study has combined RM black hole masses with galaxy-scale HI rotation curves — this appears to be genuinely novel.

### 1B. Water maser galaxies offer the most precise BH masses but few have extended rotation curves

The **Megamaser Cosmology Project** (MCP; led by Braatz, Reid et al. at NRAO) has produced ~15 galaxies with robust Keplerian maser-disk BH masses at **<10% uncertainty**. Key compilations are Kuo et al. 2011 (ApJ 727, 20; 7 BH masses), Gao et al. 2017 (ApJ 834, 52; 3 new masses plus compiled list of 15), and Pesce et al. 2020 (ApJL 891, L1; 6 galaxies for H₀). The full publication list is at https://safe.nrao.edu/wiki/bin/view/Main/MegamaserCosmologyProject. Data are published as tables in ApJ papers; there is no unified downloadable database.

Among maser galaxies with published rotation curves:

- **NGC 4258 is the gold standard** — maser BH mass (3.82 ± 0.01) × 10⁷ M☉ with <1% precision (Herrnstein+2005, Reid+2019), geometric maser distance 7.576 ± 0.082 Mpc, and rotation curve data from van Albada & van der Hulst (1980, WSRT HI), Burbidge et al. 1963 (optical), and Sofue compilations (1997, 1999). The RC extends to the flat part at ~200–210 km/s out to ~15–20 kpc. Critically, **NGC 4258 is NOT in SPARC or THINGS** despite being at 7.6 Mpc within THINGS distance range. A modern THINGS-quality HI rotation curve would need to be assembled from VLA archival data or awaited from BIG-SPARC.
- **NGC 1068** has a maser BH mass (~1.5 × 10⁷ M☉, complex torus geometry) and HI observations (Brinks+1997 VLA). A brand-new **Cepheid distance** of 10.72 ± 0.52 Mpc was published in February 2026 (Bentz et al., arXiv:2602.22407).
- **NGC 3079** has a maser BH mass (~2.9 × 10⁶ M☉, Gao+2017) and HI/CO rotation curves (Irwin & Sofue 1992), but no Cepheid/TRGB distance.
- **NGC 4388** has a maser mass (8.5 × 10⁶ M☉, Kuo+2011, but only 5 maser spots — "use with caution") and some HI data (Lu+2003), though ram-pressure stripping in Virgo complicates kinematics.

The remaining MCP galaxies — UGC 3789, NGC 2960, NGC 6264, NGC 6323, NGC 5765b, CGCG 074-064 — are either **too distant** (>50 Mpc) or **HI-poor** for useful extended rotation curves.

### 1C. Dynamical BH mass compilations identify ~7 confirmed SPARC overlaps

Three major compilations cover stellar and gas dynamical BH masses:

| Compilation | Galaxies | Year | Best access |
|---|---|---|---|
| **Kormendy & Ho 2013** (ARA&A 51, 511) | 85 dynamical | 2013 | Machine-readable tables from ARA&A website; supplement at arXiv:1308.6483 |
| **Saglia et al. 2016** (ApJ 818, 47) | 97 (31 core-E, 17 power-E, 30 classical bulges, 19 pseudobulges) | 2016 | PDF at MPE website; journal supplementary tables |
| **van den Bosch 2016** (ApJ 831, 134) | **294 total** (230 used in fits; includes RM) | 2016 | **GitHub FITS file** at https://github.com/remco-space/Black-Hole-Mass-compilation |

The van den Bosch 2016 GitHub repository is the **most accessible machine-readable compilation**, containing `BHcompilation.fits` with galaxy names, BH masses, velocity dispersions, luminosities, distances, and quality flags. Kormendy & Ho 2013 Table 3 is the definitive reference for disk-galaxy BH classifications (classical vs. pseudobulge), with ~30–35 disk galaxies. No continuously updated Kormendy online database exists beyond the 2013 tables.

Cross-referencing the researcher's 15 candidate SPARC galaxies against these compilations yields **7 confirmed galaxies with both direct BH mass and rotation curve data**: NGC 224 (M31, stellar dynamics, ~1.4 × 10⁸ M☉), NGC 1068 (maser, ~1.5 × 10⁷), NGC 2748 (gas dynamics, ~4.4 × 10⁷ from Atkinson+2005, large uncertainty), NGC 3031/M81 (stellar dynamics, new JWST: 4.78 × 10⁷), NGC 4258 (maser, 3.82 × 10⁷), NGC 4736/M94 (stellar dynamics, new JWST: 1.60 × 10⁷), and NGC 4826 (stellar dynamics, Kormendy+2024: 8.4 × 10⁶). Two S0 galaxies — NGC 2787 (gas dynamics, ~4.1 × 10⁷) and NGC 7457 (stellar dynamics, ~3.5 × 10⁶) — have marginal overlap. **NGC 2841, NGC 3198, NGC 5055, NGC 7331, and NGC 7814 have no published direct BH mass measurements** despite having excellent rotation curves and, in several cases, Cepheid distances.

### 1D. Post-2020 compilations and JWST are transforming the field

The total count of galaxies with direct dynamical BH masses is now approximately **150–200**, depending on quality cuts. Graham & Sahu (2023, multiple MNRAS papers) use ~150 direct measurements for updated scaling relations; Davis & Graham (2019, ApJ 873, 85) specifically compiled **40 spiral galaxies** with direct BH masses — the best starting point for systematic SPARC cross-matching. The ETHER sample (Nagar et al., in preparation) claims 233 galaxies but is not yet public. The WISE2MBH project (Ramos Padilla+2024, MNRAS 531, 4503) provides scaling-based BH mass estimates for millions of galaxies using a control sample of 152 direct measurements.

**JWST is the most important new contributor.** Three spiral-galaxy BH masses were measured or updated with JWST/NIRSpec in 2024–2026: NGC 3031/M81 (Nguyen et al. 2026, arXiv:2601.17439), NGC 4736/M94 (2025, arXiv:2505.09941), and NGC 4258 as a benchmark (2025, arXiv:2509.20519). The ALMA/WISDOM project has added ~30 CO-dynamical BH masses but predominantly in early-type galaxies. GRAVITY+ at VLT resolved the BLR of quasar J0920 at z = 2.3 (Nature 2024), the most distant direct BH mass, but this has no rotation curve overlap. The EHT has imaged only M87 and Sgr A*; next-generation EHT targets (~12 prime candidates from Zhang+2024) are **all ellipticals/lenticulars** with zero SPARC overlap.

---

## Part 2: Rotation curve surveys — what exists and what resolution they achieve

### 2A. Major public HI and CO rotation curve databases

| Survey | Galaxies (RCs) | Resolution | Status | Public data URL |
|---|---|---|---|---|
| **SPARC** | 175 (175) | Varies | Complete | https://astroweb.case.edu/SPARC/ |
| **THINGS** | 34 (19 RCs from de Blok+2008) | 6–12″ (~200 pc) | Complete | https://www.mpia.de/THINGS/ |
| **LITTLE THINGS** | 41 dwarfs (26 RCs) | 6″ (~200 pc) | Complete | https://science.nrao.edu/science/surveys/littlethings |
| **WHISP** | ~500 (62–70 published RCs) | 30–60″ | Complete (cubes) | http://wow.astron.nl |
| **PROBES** | **3,163** (compilation) | Varies | Published 2022 | https://connorjstone.com (Stone+2022, ApJS 262, 33) |
| **PHANGS-ALMA** | 90 (67 CO RCs) | **1.5″ (150 pc)** | Complete | https://phangs.org/data and https://almascience.eso.org/alma-data/lp/PHANGS |
| **MHONGOOSE** | 30 (ongoing) | 7–90″ | Ongoing (survey paper 2024) | Partial; de Blok+2024, A&A 688, A109 |
| **MIGHTEE-HI** | ~200 expected | 12″ | Ongoing | Early data available |
| **BIG-SPARC** | ~4,000 target | Varies | In development | Not yet released (Lelli+, arXiv:2411.13329) |

**WHISP** observed ~500 galaxies with WSRT; HI data cubes are public at Westerbork-on-the-Web, but rotation curves are published only for subsets — 62 dwarfs (Swaters+2009) and 70 galaxies (van Eymeren+2011). The full survey lacks a unified rotation curve catalog. **THINGS** remains the gold standard for high-resolution HI kinematics of nearby galaxies: 19 rotation curves from de Blok et al. 2008 with 6″ beams (~100–500 pc). Data cubes and moment maps are public; rotation curve tables are available upon request from de Blok. **LITTLE THINGS** extends to 41 dwarf irregulars with 26 published rotation curves (Oh+2015; independently derived by Iorio+2017 using 3D-BAROLO). **PROBES** is a homogenized compilation of 3,163 late-type spiral rotation curves (from Hα and HI sources) plus photometry — the largest single catalog, publicly available.

For new radio telescopes: **MHONGOOSE** (de Blok+2024) is observing 30 nearby galaxies with MeerKAT at extraordinary HI column density sensitivity (~5 × 10¹⁷ cm⁻²), but the full survey data release is pending. **MIGHTEE-HI** has demonstrated rotation curves for resolved galaxies out to z ~ 0.09. Sofue maintains a compilation of rotation curves at https://www.ioa.s.u-tokyo.ac.jp/~sofue/h-rot.htm with tabular data for many nearby spirals.

### 2B. Inner rotation curves for soliton core testing

**PHANGS CO rotation curves at 150 pc resolution are the most valuable new dataset for testing soliton cores.** Lang et al. 2020 (ApJ 897, 122) derived CO rotation curves for 67 galaxies at 1″ (~100 pc projected) angular resolution, with 150 pc radial bins — well-matched to predicted soliton core radii of ~100 pc–1 kpc for ultralight axion masses ~10⁻²² eV. CO traces the molecular gas-dominated inner regions where HI is deficient, making PHANGS complementary to THINGS/SPARC for inner mass profiles. PHANGS-MUSE achieves even better **~50 pc** resolution for 19 galaxies.

The major IFU surveys achieve coarser inner resolution than needed for sub-kpc soliton core detection:

- **MaNGA** (DR17, ~10,000 galaxies): typical PSF ~2.5″ corresponds to ~1.5 kpc at median z ~ 0.03 — insufficient for individual soliton fits, though useful for statistical studies.
- **CALIFA** (DR3, 667 galaxies): ~2.7″ fiber diameter, ~1 kpc resolution at ~65 Mpc median distance — explicitly unable to detect kinematically decoupled nuclear components.
- **SAMI** (DR3, 3,068 galaxies): ~2″ seeing corresponds to ~2 kpc at z ~ 0.05 — too coarse.

**De Blok et al. 2008 THINGS rotation curves remain the HI inner-profile gold standard**, reaching ~100–200 pc from center for the nearest galaxies (e.g., NGC 2403 at 3.2 Mpc). For a combined approach, the optimal strategy pairs PHANGS CO data in inner regions with THINGS/SPARC HI data at larger radii.

### 2C. No survey simultaneously targets both rotation curves and BH masses

The **MASSIVE survey** (Ma+2014; https://blackhole.berkeley.edu/) studies 116 of the most massive early-type galaxies with IFS kinematics and ~15+ BH masses, but these are pressure-supported systems without disk rotation curves. **EDGE-CALIFA** (Bolatto+2017; 126 galaxies, 17 with CO rotation curves) has no systematic BH mass overlap. No existing survey was designed to jointly measure rotation curves and BH masses. The most productive approach is cross-matching existing BH mass compilations (especially Davis & Graham 2019's 40 spirals) against SPARC, PROBES, or PHANGS.

---

## Part 3: Datasets specifically for the BEC dark matter framework

### 3A. No unified soliton core radius catalog exists, but several partial compilations are available

Published soliton/FDM profile fits to rotation curves come from scattered sources with no single comprehensive catalog:

- **Bernal et al. 2018** (MNRAS 475, 1447): Fitted soliton+NFW profiles to **24 galaxies** (18 LSB + 6 SPARC). **Publishes core radii** (r_c = 0.33–8.96 kpc), boson mass values, and NFW parameters in journal tables. Best-fit boson mass: m_ψ = 0.554 × 10⁻²³ eV.
- **Fernandez-Hernandez et al. 2019** (MNRAS 488, 5127): Non-parametric reconstruction for **88 spiral galaxies** comparing soliton, Burkert, NFW, and pseudo-isothermal profiles. Publishes fitted characteristic surface density and mass within 300 pc for each galaxy.
- **Khelashvili et al. 2023** (A&A 677, A46): Full Bayesian soliton+NFW fits to **LITTLE THINGS dwarf irregulars**. Publishes core radii and central densities. Found **>5σ tension** between core-mass scaling and soliton predictions — a key result for BEC DM testing.
- **Pozo, Broadhurst et al. 2021/2023/2024**: Joint stellar phase-space analysis of **~20+ Local Group dSphs and UFDs**. Derive m_B = (8.1 ± 1.6) × 10⁻²³ eV. Publish soliton core radii and densities, and the scaling ρ_c ∝ R_c⁻⁴.
- **Bar, Blum & Sun 2022** (PRD 105, 083015): Uses all 175 SPARC galaxies to derive upper bounds on soliton mass, but does **not** publish individual galaxy core radius fits.

For classical cored profiles (proxies for soliton cores): **Donato et al. 2009** found the constant central surface density ρ₀·r₀ ≈ 141 M☉/pc² across ~1,000 spirals using Burkert/pseudo-isothermal fits. Burkert 2015 (ApJ 808) fits 8 MW dSphs with core radii 280 pc–1.3 kpc. The Burkert profile closely approximates the soliton profile (noted by Schive+2014), so these classical core-radius measurements serve as BEC-relevant proxies.

### 3B. Cluster RAR data is anchored by Tian+2020 with several newer additions

**Tian et al. 2020** (ApJ 896, 70) remains the primary cluster RAR dataset: 20 massive CLASH clusters with combined weak lensing + strong lensing + X-ray profiles. Found g_tot ∝ g_bar^0.51 with acceleration scale g‡ = (2.02 ± 0.11) × 10⁻⁹ m/s², **17× the galactic value**. Data are on **VizieR** at J/ApJ/896/70 with radial g_tot and g_bar profiles for each cluster.

Post-2020 cluster RAR datasets include:

- **Eckert et al. 2022** (A&A 662, A123): Definitive mass profiles of **12 X-COP clusters** from deep XMM-Newton + SZ, with full decomposition into gas, BCG, satellites, and DM. Finds cluster RAR "strongly departs" from spiral RAR. Available through the X-COP project — the highest-quality individual cluster mass profiles.
- **Tian et al. 2024**: Extends cluster RAR using dynamical (MaNGA IFS) measurements of **50 BCGs**, confirming the elevated acceleration scale with an independent kinematic method.
- **Pradyumna et al. 2021**: Independent RAR from 12 Chandra + 12 X-COP clusters; 0.11–0.14 dex scatter.
- **eROSITA eRASS1** (Bulbul+2024, A&A 685, A106): Catalog of **12,247 clusters** with X-ray luminosity, temperature, and total mass within R₅₀₀. Public at https://erosita.mpe.mpg.de/dr1/. Individual mass profiles use isothermal assumptions, limiting their quality for detailed RAR work, but the enormous sample enables statistical studies. A recent paper (arXiv:2411.09735) presents joint X-ray + dynamical profiles for 22 eRASS1 clusters including RAR analysis.

For hydrostatic mass reconstruction: the **ACCEPT database** (Cavagnolo+2009; http://www.pa.msu.edu/astro/MC2/accept/) provides temperature, density, and entropy profiles for ~240 Chandra clusters. **HIFLUGCS** offers the 64 brightest X-ray clusters as a complete flux-limited sample.

### 3C. KiDS weak lensing remains the only published RAR; DES and HSC have untapped potential

**Mistele et al. 2024** (JCAP 2024, 020) provides the **most refined weak lensing RAR** to date, extending the kinematic RAR by 2.5 decades using KiDS-1000 data with improved deprojection and consistent stellar mass estimates. This resolves the 6σ early/late-type discrepancy from Brouwer+2021 — when strict isolation criteria and consistent SPS models are applied, both types lie on the **same RAR**. A companion paper (Mistele+2024, ApJL 969, L3) derives circular velocity curves from weak lensing out to ~1 Mpc, finding velocities remain flat to hundreds of kpc with no decline.

The original **Brouwer et al. 2021** (A&A 650, A113) KiDS-1000 data products (ESD profiles) are at https://kids.strw.leidenuniv.nl/sciencedata.php.

**No dedicated RAR analysis has been published using DES or HSC data.** DES Y6 covers ~5,000 deg² with 140 million source galaxies; galaxy-galaxy lensing measurements are public for Y1 and Y3 at https://des.ncsa.illinois.edu/releases, but all papers focus on cosmological parameters rather than RAR. HSC covers ~1,400 deg² with public data releases at https://hsc.mtk.nao.ac.jp/ssp/. Both surveys could yield RAR measurements using the Mistele deprojection method, representing a significant untapped opportunity. **Euclid** (launched July 2023) has no RAR results yet but will likely produce the definitive weak lensing RAR. A new kinematic RAR from **MIGHTEE-HI** (Vărășteanu+2025) extends to low accelerations at z ≤ 0.08.

---

## Part 4: Data access summary for all major datasets

| Dataset | N objects | Format | Public? | Access URL / Location | VizieR? |
|---|---|---|---|---|---|
| Bentz & Katz AGN BH Mass DB | ~100+ RM AGN | CSV/TSV | Yes | http://www.astro.gsu.edu/AGNmass/ | No |
| SDSS-RM final | 849 quasars, 339 lags | FITS/tables | Yes | https://ariel.astro.illinois.edu/sdssrm/ | No |
| MCP maser BH masses | ~15 galaxies | Tables in papers | Yes (via ADS) | NRAO wiki for pub list | No |
| van den Bosch 2016 compilation | 294 galaxies | **FITS** | Yes | **https://github.com/remco-space/Black-Hole-Mass-compilation** | No |
| Kormendy & Ho 2013 tables | 85 dynamical | Machine-readable zip | Yes | ARA&A website; arXiv:1308.6483 | No |
| Saglia+2016 | 97 dynamical | PDF/journal tables | Yes | MPE website; ApJ supplementary | No |
| Davis & Graham 2019 (spirals) | 40 spirals | Journal tables | Yes | ApJ supplementary material | No |
| SPARC | 175 late-type | ASCII (.mrt) | Yes | https://astroweb.case.edu/SPARC/ | No |
| THINGS | 34 (19 RCs) | FITS cubes | Yes | https://www.mpia.de/THINGS/ | Partial |
| LITTLE THINGS | 41 (26 RCs) | FITS cubes | Yes | https://science.nrao.edu/science/surveys/littlethings | No |
| WHISP | ~500 (cubes) | FITS cubes | Yes | http://wow.astron.nl | No |
| PROBES | 3,163 RCs | CSV | Yes | connorjstone.com; ApJS supplementary | No |
| PHANGS-ALMA | 90 (67 CO RCs) | FITS/CSV | Yes | https://phangs.org/data | No |
| PHANGS-MUSE | 19 galaxies | FITS cubes | Yes | ESO archive + CADC | No |
| MHONGOOSE | 30 | FITS cubes | Partial | Contact PI (de Blok) | No |
| MaNGA (DR17) | ~10,000 | FITS cubes | Yes | https://www.sdss4.org/dr17/manga/ | No |
| CALIFA (DR3) | 667 | FITS cubes | Yes | http://califa.caha.es/ | Yes |
| Tian+2020 cluster RAR | 20 CLASH clusters | Machine-readable | Yes | **VizieR J/ApJ/896/70** | **Yes** |
| Brouwer+2021 KiDS lensing RAR | ~10⁵ lenses | Text files | Yes | https://kids.strw.leidenuniv.nl/sciencedata.php | No |
| eROSITA eRASS1 clusters | 12,247 | FITS catalog | Yes | https://erosita.mpe.mpg.de/dr1/ | Yes |
| ACCEPT cluster profiles | ~240 | Online tables | Yes | http://www.pa.msu.edu/astro/MC2/accept/ | No |
| Bernal+2018 soliton fits | 24 galaxies | Journal tables | Yes (via MNRAS) | ADS | No |
| Khelashvili+2023 soliton fits | ~10–20 dIrr | Journal tables | Yes (via A&A) | ADS | No |

---

## Part 5: The 20 highest-priority cross-match galaxies

The galaxies below are ranked by how completely they satisfy all three criteria: (1) direct BH mass measurement, (2) published rotation curve extending to the flat part, and (3) accurate distance (Cepheid or TRGB). Bold entries in the BH mass column indicate the most precise measurements.

### Tier 1 — All three criteria met

| Rank | Galaxy | BH mass (M☉) | BH method | RC source | Distance (Mpc) | Dist. method | In SPARC? |
|---|---|---|---|---|---|---|---|
| 1 | **NGC 4258** | **(3.82 ± 0.01) × 10⁷** | Maser (Herrnstein+05) | van Albada+1980; Sofue compilations | 7.576 ± 0.08 | Maser geometric + Cepheid + TRGB | No |
| 2 | **NGC 3031 (M81)** | **(4.78 ± 0.1) × 10⁷** | Stellar dyn. (JWST 2026) | THINGS; Sofue compilations | 3.63 ± 0.14 | Cepheid + TRGB (×4) | No |
| 3 | **NGC 4736 (M94)** | **(1.60 ± 0.16) × 10⁷** | Stellar dyn. (JWST 2025) | THINGS (de Blok+08) | ~4.66 | TRGB | No |
| 4 | **NGC 4826 (M64)** | **8.4 (+1.7/−0.6) × 10⁶** | Stellar dyn. (Kormendy+24) | HI (Braun+1992); Sofue | ~7.27 | SBF (verify TRGB) | No |
| 5 | **NGC 224 (M31)** | ~1.4 × 10⁸ | Stellar dyn. (Bender+05) | HI (Carignan+06; Chemin+09) | 0.765 ± 0.03 | Cepheid + TRGB + EB | No |
| 6 | **NGC 1068 (M77)** | ~1.5 × 10⁷ | Maser (complex torus) | HI (Brinks+97); Sofue | 10.72 ± 0.52 | **Cepheid (Bentz+2026, new!)** | No |
| 7 | **Circinus** | ~1.7 × 10⁶ | Maser (warped disk) | HI (Jones+99; Koribalski+04) | ~4.2 | TRGB | No |

### Tier 2 — Two of three criteria, or reduced quality on one

| Rank | Galaxy | BH mass | BH method | RC source | Distance | Notes |
|---|---|---|---|---|---|---|
| 8 | NGC 2748 | (4.4 ± 3.5) × 10⁷ | Gas dynamics (Atkinson+05) | **SPARC** | Cepheid (SN Ia host) | BH mass has large uncertainty |
| 9 | NGC 4051 | ~1.7 × 10⁶ | RM (Bentz & Katz DB) | **SPARC** | ~10 Mpc (redshift) | **Only confirmed RM+SPARC overlap**; lacks Cepheid/TRGB |
| 10 | NGC 4151 | ~1.8 × 10⁷ | RM + gas dyn. + stellar | Limited HI (Mundell+99) | 15.8 ± 0.4 (Cepheid) | Excellent BH mass + distance; needs better RC |
| 11 | NGC 3079 | (2.9 ± 0.3) × 10⁶ | Maser (Gao+17) | HI/CO (Irwin & Sofue 92) | ~20 Mpc (redshift) | No accurate distance |
| 12 | NGC 2787 | ~4.1 × 10⁷ | Gas dynamics (Sarzi+01) | Limited (S0) | ~7.5 (SBF) | S0 galaxy — minimal HI |
| 13 | NGC 4388 | 8.5 × 10⁶ | Maser (tentative) | Limited (Virgo, stripped) | ~17 (redshift) | Caution: only 5 maser spots |

### Tier 3 — Excellent rotation curves needing BH mass measurements

| Galaxy | RC quality | Distance | BH status | Prospect |
|---|---|---|---|---|
| NGC 7331 | THINGS, extended, flat | 14.7 Mpc (Cepheid) | No direct measurement | JWST could measure |
| NGC 5055 (M63) | SPARC showcase | ~8.9 Mpc | No direct measurement | JWST/ALMA candidate |
| NGC 2841 | Classic, extended | 14.1 Mpc (Cepheid) | No direct measurement | Important MOND test galaxy |
| NGC 3198 | Iconic flat RC | 13.8 Mpc (Cepheid) | No direct measurement | Most famous RC in literature |
| NGC 598 (M33) | THINGS, extended | 0.84 Mpc (Cepheid + TRGB) | Upper limit ≤1,500 M☉ | Null detection — interesting constraint |

---

## Conclusions and strategic recommendations

The current overlap between direct BH masses and high-quality rotation curves is a sample of **~7–10 galaxies**, small but sufficient for an initial BH mass–healing length study. NGC 4258 is the irreplaceable cornerstone — the only galaxy with a sub-percent maser BH mass, geometric distance, and extended rotation curve, though assembling a modern-standard HI rotation curve from VLA archives remains an open task. Three 2024–2026 JWST stellar-dynamical BH masses (M81, M94, NGC 4826) have just expanded the usable sample by ~40%.

The most impactful near-term action is cross-matching the **Davis & Graham 2019 catalog of 40 spirals with direct BH masses** against the **van den Bosch 2016 GitHub compilation** and **PROBES 3,163 rotation curves**, which likely yields additional overlapping systems beyond the 15 candidates already checked. When **BIG-SPARC** releases ~4,000 rotation curves, the overlap should grow substantially — NGC 4258 almost certainly enters the sample, and several more maser and dynamical-BH galaxies may gain usable rotation curves.

For the BEC framework, the three highest-value datasets not yet in use are (1) **PHANGS CO rotation curves at 150 pc** for inner soliton-core testing in 67 galaxies, (2) **Eckert+2022 X-COP cluster mass profiles** for cluster-scale healing-length tests complementing Tian+2020, and (3) **Mistele et al. 2024 refined weak lensing RAR** as a significant upgrade over Brouwer+2021. The absence of any DES or HSC RAR analysis represents a conspicuous gap — applying the Mistele deprojection method to DES Y6 (~5,000 deg²) would yield the most powerful weak lensing RAR test of BEC predictions at ultra-low accelerations.