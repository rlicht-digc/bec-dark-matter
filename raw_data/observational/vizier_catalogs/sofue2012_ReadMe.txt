J/ApJS/199/26          The 2MASS Redshift Survey (2MRS)          (Huchra+, 2012)
================================================================================
The 2MASS Redshift Survey - Description and data release.
    Huchra J.P., Macri L.M., Masters K.L., Jarrett T.H., Berlind P.,
    Calkins M., Crook A.C., Cutri R., Erdogdu P., Falco E., George T.,
    Hutcheson C.M., Lahav O., Mader J., Mink J.D., Martimbeau N., Schneider S.,
    Skrutskie M., Tokarz S., Westover M.
   <Astrophys. J. Suppl. Ser., 199, 26 (2012)>
   =2012ApJS..199...26H
================================================================================
ADC_Keywords: Galaxies, IR ; Extinction ; Photometry, infrared ; Redshifts ;
              Surveys ; Galaxy catalogs
Keywords: catalogs - galaxies: distances and redshifts - surveys

Abstract:
    We present the results of the 2MASS Redshift Survey (2MRS), a ten-year
    project to map the full three-dimensional distribution of galaxies in
    the nearby universe. The Two Micron All Sky Survey (2MASS) was
    completed in 2003 and its final data products, including an extended
    source catalog (XSC), are available online. The 2MASS XSC contains
    nearly a million galaxies with Ks<=13.5mag and is essentially complete
    and mostly unaffected by interstellar extinction and stellar confusion
    down to a galactic latitude of |b|=5{deg} for bright galaxies.
    Near-infrared wavelengths are sensitive to the old stellar populations
    that dominate galaxy masses, making 2MASS an excellent starting point
    to study the distribution of matter in the nearby universe. We
    selected a sample of 44599 2MASS galaxies with Ks<=11.75mag and
    |b|>=5{deg} (>=8{deg} toward the Galactic bulge) as the input catalog
    for our survey. We obtained spectroscopic observations for 11000
    galaxies and used previously obtained velocities for the remainder of
    the sample to generate a redshift catalog that is 97.6% complete to
    well-defined limits and covers 91% of the sky. This provides an
    unprecedented census of galaxy (baryonic mass) concentrations within
    300Mpc. Earlier versions of our survey have been used in a number of
    publications that have studied the bulk motion of the Local Group,
    mapped the density and peculiar velocity fields out to 50h^-1^Mpc,
    detected galaxy groups, and estimated the values of several
    cosmological parameters. Additionally, we present morphological types
    for a nearly complete sub-sample of 20860 galaxies with Ks<=11.25mag
    and |b|>=10{deg}.

Description:
    We obtained spectra for 11000 galaxies that met the selection
    criteria, plus an additional 2898 galaxies beyond the catalog limits.
    Observations were carried out between 1997 September and 2011 January
    using a variety of facilities (Fred L. Whipple, 1.5m; Cerro Tololo
    1.5m and 4m; McDonald, 2.1m; Hobby-Eberly, 9.2m). The majority of the
    spectra obtained for this survey were acquired at the Fred L. Whipple
    Observatory (FLWO) 1.5m telescope, which mostly targeted galaxies in
    the northern hemisphere. In the southern hemisphere, we relied heavily
    on observations by the 6dFGS project (Jones et al. 2009, Cat. VII/259)
    but also carried out our own observations using the Cerro Tololo
    Interamerican Observatory (CTIO) 1.5m telescope.

File Summary:
--------------------------------------------------------------------------------
 FileName     Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe           80        .   This file
table3.dat      233    44599   2MRS catalog
table4.dat       73      590   2MRS Catalog - bibliographic references
table6.dat       68     4291   2MRS redshifts for 2MASS XSC galaxies beyond the
                               main catalog limits
table7.dat       68       14   Redshifts for galaxies not in the 2MASS XSC
                               which were observed serendipitously
table8.dat       53      324   2MASS XSC or LGA objects removed from input
                               catalog
table9.dat       71       74   2MASS XSC objects with reprocessed photometry
table10.dat      16       87   2MASS XSC objects with suspect photometry,
                               flagged for reprocessing at a later date
table11.dat      16      155   2MASS XSC objects with compromised photometry,
                               flagged for reprocessing and removed from catalog
table12.dat     166      334   Alternative redshifts chosen over NED default
                               redshifts
table13.dat      48     4857   Redshifts from 6dFGS, SDSS or NED for galaxies
                               also observed by 2MRS
--------------------------------------------------------------------------------

See also:
  II/306  : The SDSS Photometric Catalog, Release 8 (Adelman-McCarthy+, 2011)
  VII/259 : 6dF galaxy survey final redshift release (Jones+, 2009)
  VII/233 : The 2MASS Extended sources (IPAC/UMass, 2003-2006)
  J/AJ/126/63   : Host galaxies of 2MASS-QSOs with z<=3 (Hutchings+, 2003)
  J/ApJ/560/566 : K-band galaxy luminosity function from 2MASS (Kochanek+, 2001)
  http://tdc-www.cfa.harvard.edu/2mrs/ : 2MRS at the SAO telescope data center
  http://www.ipac.caltech.edu/2mass/   : 2MASS home page at IPAC
  http://www.aao.gov.au/local/www/6df/ : 6dF Galaxy Survey home page
  http://www-wfau.roe.ac.uk/6dFGS/     : 6dF Galaxy Redshift Survey -DR3
  http://www.sdss3.org/                : SDSS-III home page

Byte-by-byte Description of file: table3.dat
--------------------------------------------------------------------------------
   Bytes Format Units    Label   Explanations
--------------------------------------------------------------------------------
   1- 16  A16   ---      ID      2MASS ID from XSC (Cat. VII/233) or 2MASS
                                 Large Galaxy Atlas (LGA) databases (Jarrett
                                 et al. 2003AJ....125..525J)
  18- 26  F9.5  deg      RAdeg   Right Ascension in decimal degrees (J2000)
  28- 36  F9.5  deg      DEdeg   Declination in decimal degrees (J2000)
  38- 46  F9.5  deg      GLON    Galactic longitude
  48- 56  F9.5  deg      GLAT    Galactic latitude
  58- 63  F6.3  mag      Kcmag   Extinction-corrected 2MASS Ks magnitude (1)
  65- 70  F6.3  mag      Hcmag   Extinction-corrected 2MASS H magnitude (1)
  72- 77  F6.3  mag      Jcmag   ?=99.999 Extinction-corrected 2MASS J mag. (1)
  79- 84  F6.3  mag      Ktmag   Extinction-corrected total extrapolated Ks mag
  86- 91  F6.3  mag      Htmag   Extinction-corrected total extrapolated H mag
  93- 98  F6.3  mag      Jtmag   Extinction-corrected total extrapolated J mag
 100-104  F5.3  mag    e_Kcmag   Uncertainty in Kcmag
 106-110  F5.3  mag    e_Hcmag   Uncertainty in Hcmag
 112-116  F5.3  mag    e_Jcmag   ?=9.999 Uncertainty in Jcmag
 118-122  F5.3  mag    e_Ktmag   Uncertainty in Ktmag
 124-128  F5.3  mag    e_Htmag   Uncertainty in Htmag
 130-134  F5.3  mag    e_Jtmag   Uncertainty in Jtmag
 136-140  F5.3  mag      E(B-V)  Foreground galactic extinction (2)
 142-146  F5.3 [arcsec]  Riso    Log of K=20 mag/sq arcsec elliptical isophote
                                 semimajor axis
 148-152  F5.3 [arcsec]  Rext    Log of total elliptical aperture semimajor axis
 154-158  F5.3  ---      b/a     Axis ratio of the J+H+Ks co-added image at the
                                 3{sigma} isophote
 160-163  A4    ---      flags   [Z0-6n] 2MASS XSC pipeline photometric
                                 confusion flags (3)
 165-169  A5    ---      type    Morphological type, expressed using the ZCAT
                                 convention (G1)
 171-172  A2    ---    r_type    Source of morphological type (4)
 174-178  I5    km/s     cz      ? Redshift, corrected to solar system
                                 barycenter reference frame
 180-182  I3    km/s   e_cz      ? Uncertainty in cz
     184  A1    ---    n_cz      Code for source of cz (G2)
 186-204  A19   ---    r_cz      Reference for cz; see table4.dat file
 206-233  A28   ---      CAT     Galaxy identification in input redshift catalog
--------------------------------------------------------------------------------
Note (1): measured at the K=20mag/sq.arcsec isophote.
Note (2): From Schlegel, Finkbeiner & Davis (1998ApJ...500..525S).
Note (3): concatenation of 2MASS XSC pipeline photometry flags cc_flg
     (k_flg_k20fe, h_flg_k20fe, j_flg_k20fe). A "Z" in the first character
     indicates galaxy photometry comes from the 2MASS Large Galaxy Atlas.
     The figures are, from the 2MASX (Cat. VII/233):
     0 = no other sources within aperture used
     1 = aperture contained pixels masked off in coadd
     2 = aperture contained pixels masked off due to neighboring sources
     3 = aperture contained pixels that had a point source flux subtracted off
     4 = aperture contained pixels within bright star mask
     5 = aperture contained pixels masked off due to persistence
     6 = aperture ran into coadd boundary 
     n = not found in the filter
Note (4): Morphological type source as follows:
     JH = newly typed galaxy by John Huchra
     ZC = previously listed in J. Huchra's personal compilation of redshifts
          (ZCAT) (from multiple sources including RC3 and NED)

Byte-by-byte Description of file: table4.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 19  A19   ---     Bibcode   ADS bibcode (1)
  21- 73  A53   ---     Aut       Author(s) and (year)
--------------------------------------------------------------------------------
Note (1): The last 12 entries are 2MRS internal codes.
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table[67].dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label   Explanations
--------------------------------------------------------------------------------
   1- 16  A16   ---     ID      2MASS ID from XSC or LGA databases (pseudo
                                2MASS id based on galaxy coordinates for table7)
  18- 26  F9.5  deg     RAdeg   Right Ascension in decimal degrees (J2000)
  28- 36  F9.5  deg     DEdeg   Declination in decimal degrees (J2000)
  38- 42  I5    km/s    cz      Redshift
  44- 46  I3    km/s  e_cz      Uncertainty in cz
      48  A1    ---   n_cz      Source code for cz (G2)
  50- 68  A19   ---   r_cz      Reference for cz; see table4.dat file
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table8.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 16  A16   ---     ID        2MASS ID from XSC or LGA databases
  18- 53  A36   ---     Reason    Reason for removal from 2MRS catalog
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table9.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label   Explanations
--------------------------------------------------------------------------------
   1- 16  A16   ---     ID      Original 2MASS ID from XSC or LGA databases
  18- 23  F6.3  mag     Komag   Original Ks isophotal magnitude
  25- 29  F5.3 [arcsec] Roiso   Log of original Ks=20mag/sq arcsec
                                isophotal radius
  31- 35  F5.3  ---     b/ao    Original axis ratio of the J+H+Ks co-added
                                image at the 3{sigma} isophote
  37- 52  A16   ---     2MRS    Reprocessed 2MASS ID (HHMMSSss+DDMMSSs)
  54- 59  F6.3  mag     Kmag    Reprocessed Ks isophotal magnitude (1)
  61- 65  F5.3 [arcsec] Riso    Log of reprocessed Ks=20mag/sq arcsec
                                isophotal radius
  67- 71  F5.3  ---     b/a     Reprocessed axis ratio of the J+H+Ks co-added
                                image at the 3{sigma} isophote
--------------------------------------------------------------------------------
Note (1): All properties of the reprocessed galaxies are listed in Table 3 under
          the corresponding 2MASS ID. The contents of this Table are only
          intended to provide an overview of the changes due to reprocessing.
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table10.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 16  A16   ---     ID        2MASS ID of galaxy (1)
--------------------------------------------------------------------------------
Note (1): All properties of the flagged galaxies are listed in Table 3. This
          Table is only intended to provide an index of the galaxies with
          suspect photometry.
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table11.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label     Explanations
--------------------------------------------------------------------------------
   1- 16  A16   ---     ID        2MASS ID of galaxy (1)
--------------------------------------------------------------------------------
Note (1): These galaxies are not part of our catalog and therefore we do not
          list their photometric properties. This Table is only intended to
          provide other users of the 2MASS XSC an index of galaxies that we
          consider to have compromised photometry.
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table12.dat
--------------------------------------------------------------------------------
   Bytes Format Units    Label     Explanations
--------------------------------------------------------------------------------
   1- 16  A16   ---      ID        2MASS ID from XSC or LGA databases
  18- 22  I5    km/s     czNED     ? Default NED redshift
  24- 26  I3    km/s   e_czNED     ? Reported uncertainty in czNED
  28- 31  A4    ---    q_czNED     Qualifier for czNED (1)
  33- 51  A19   ---    r_czNED     Bibliographic code for czNED
  53- 57  I5    km/s     cz        Alternative redshift adopted by 2MRS
  59- 61  I3    km/s   e_cz        ? Reported uncertainty in cz
      63  A1    ---    f_cz        Source code for cz (G2)
  65- 67  F3.1  arcmin   sep       Angular separation (2)
  69- 87  A19   ---    r_cz        Reference for cz; see table4.dat file
  89-166  A78   ---    n_cz        Comment on choice of cz
--------------------------------------------------------------------------------
Note (1): Qualifier as follows:
  PRED = "predicted" redshift listed in NED
  PHOT = photometric redshift listed in NED
Note (2): Between 2MASS source and alternative redshift entry in original
          publication.
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table13.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label   Explanations
--------------------------------------------------------------------------------
   1- 16  A16   ---     ID      2MASS ID from XSC or LGA databases
  18- 22  I5    km/s    cz      Alternative redshift
  24- 26  I3    km/s  e_cz      ? Reported uncertainty cz
      28  A1    ---   n_cz      Source code for cz (G2)
  30- 48  A19   ---   r_cz      Reference for cz; see table4.dat file
--------------------------------------------------------------------------------

Global notes:
Note (G1): the morphological information is encoded following the 5-digit ZCAT
   convention (I2, A1, I1, A1) (T-type, bar, luminosity class, pecularities)
   -------------------------------------------------------------
   (1) T-type
   -------------------------------------------------------------
   -9 = QSO/AGN
   -7 = Unclassified Elliptical
   -6 = Compact Elliptical
   -5 = E, and dwarf E
   -4 = E/S0
   -3 = L-, S0-
   -2 = L, S0
   -1 = L+, S0+
    0 = S0/a, S0-a
    1 = Sa
    2 = Sab
    3 = Sb
    4 = Sbc
    5 = Sc
    6 = Scd
    7 = Sd
    8 = Sdm
    9 = Sm
   10 = Im, Irr I, Magellanic Irregular, Dwarf Irregular
   11 = Compact Irregular, Extragalactic HII Region
   12 = Extragalactic HI cloud (no galaxy visible)
   15 = Peculiar, Unclassifiable
   16 = Irr II
   19 = Unclassified galaxy (visually confirmed to be a galaxy, but not typed)
   20 = S..., Sc-Irr, Unclassified Spiral
   98 = Galaxy that has never been visually examined.
   -------------------------------------------------------------
   (2) Bar type
   -------------------------------------------------------------
    A = unbarred (A)
    X = mixed type (AB)
    B = barred (B)
   -------------------------------------------------------------
   (3) Luminosity classes (for spirals and irregulars)
   -------------------------------------------------------------
    1 = I
    2 = I-II
    3 = II
    4 = II-II
    5 = III
    6 = III-IV
    7 = IV
    8 = IV-V
    9 = V
   -------------------------------------------------------------
   (4) Peculiarities
   -------------------------------------------------------------
    D = Double or Multiple
    P = Peculiar
    R = Outer Ring
    r = Inner Ring
    s = S-shaped
    t = Mixed (Inner ring/S-shaped)
    T = Pseudo outer ring
    / = Spindle
   -------------------------------------------------------------

Note (G2): Provenance of redshift as follows:
   C = CTIO (Cerro-Tololo)
   D = McDonald Obs. (Texas)
   F = FLWO (Fred Lawrence Whipple Observatory, Arizona)
   M = alternative NED redshift
   N = default NED redshift
   O = alternative redshift from ZCAT (Previous observations by Huchra and
       collaborators)
   S = SDSS-DR8 (see Cat. II/306)
   6 = 6dFGS-DR3 (see Cat. VII/259)
--------------------------------------------------------------------------------

History:
    From electronic version of the journal

================================================================================
(End)                 Greg Schwarz [AAS], Emmanuelle Perret [CDS]    11-Jun-2012
