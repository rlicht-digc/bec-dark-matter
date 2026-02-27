J/A+A/370/765       HI synthesis observations in UMa cluster (Verheijen+, 2001)
================================================================================
The Ursa Major cluster of galaxies. IV. HI synthesis observations.
    Verheijen M.A.W., Sancisi R.
   <Astron. Astrophys. 370, 765 (2001)>
   =2001A&A...370..765V
================================================================================
ADC_Keywords: Clusters, galaxy ; H I data ; Galaxies, rotation ; Photometry
Keywords: galaxies: fundamental parameters - galaxies: kinematics and dynamics -
          galaxies: spiral - galaxies: structure

Abstract:
    In this data paper we present the results of an extensive 21 cm-line
    synthesis imaging survey of 43 spiral galaxies in the nearby Ursa
    Major cluster using the Westerbork Synthesis Radio Telescope. Detailed
    kinematic information in the form of position-velocity diagrams and
    rotation curves is presented in an atlas together with HI channel
    maps, 21 cm continuum maps, global HI profiles, radial HI surface
    density profiles, integrated HI column density maps, and HI velocity
    fields. The relation between the corrected global HI linewidth and the
    rotational velocities Vmax and Vflat as derived from the rotation
    curves is investigated. Inclination angles obtained from the optical
    axis ratios are compared to those derived from the inclined HI disks
    and the HI velocity fields. The galaxies were not selected on the
    basis of their HI content but solely on the basis of their cluster
    membership and inclination which should be suitable for a kinematic
    analysis. The observed galaxies provide a well-defined, volume limited
    and equidistant sample, useful to investigate in detail the
    statistical properties of the Tully-Fisher relation and the dark
    matter halos around them.

File Summary:
--------------------------------------------------------------------------------
 FileName  Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe        80        .   This file
table1.dat    86       52   All galaxies in the UMa cluster brighter than
                             Mb,i(B)=-16.8 and more inclined than 45 degrees
table2.dat    92       52   Photometry of all galaxies in the UMa cluster
                             brighter than M_b,i_(B)=-16.8 and more inclined
                             than 45 degrees
table3.dat    68       57   A comparison of the widths and integrated fluxes
                             from the present WSRT survey and from the
                             literature
table4.dat    48      437   Rotation curves derived from velocity fields
                             and XV-diagrams
table5.dat   104       43   Results from the HI synthesis observations
                               literature
--------------------------------------------------------------------------------

See also:
    J/AJ/112/2471 : The Ursa Major cluster. I. (Tully+ 1996)

Byte-by-byte Description of file: table1.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label   Explanations
--------------------------------------------------------------------------------
   1-  2  A2    ---     Sample  Samples (G1)
   4- 10  A7    ---     Name    Galaxy name (U for UGC, N for NGC)
      11  A1    ---   n_Name    [fi] Note on the galaxy (2)
  13- 14  I2    h       RAh     Right ascension (B1950)
  16- 17  I2    min     RAm     Right ascension (B1950)
  19- 22  F4.1  s       RAs     Right ascension (B1950)
      24  A1    ---     DE-     Declination sign (B1950)
  25- 26  I2    deg     DEd     Declination (B1950)
  28- 29  I2    arcmin  DEm     Declination (B1950)
  31- 32  I2    arcsec  DEs     Declination (B1950)
  34- 39  F6.2  deg     GLON    Galactic longitude
  41- 45  F5.2  deg     GLAT    Galactic latitude
  47- 50  A4    ---     MType   Morphological type
  52- 55  F4.2  arcmin  D25(B)  Observed major axis diameter of the
                                 25th mag/arcsec^2^ blue isophote
  57- 59  I3    deg     PA      Position angle of the receding side of the
                                 galaxy (3)
  61- 64  F4.2  ---     1-b/a   Observed ellipticity of the optical galaxy image
  66- 67  I2    deg     iopt    Inclination as derived from the observed axis
                                 ratio (b/a).
  69- 70  I2    deg     iadopt  Adopted inclination angle as derived from
                                 several methods
      72  I1    deg   e_iadopt  rms uncertainty on iadopt
  74- 76  A3    ---     SB      Low (LSB) or high surface brightness (HSB)
                                 galaxy, according to Paper II
  78- 81  F4.2  mag     [BH]    Galactic extinction in the B-band according to
                                 Burstein & Heiles (1984ApJS...54...33B)
  83- 86  F4.2  mag     [SFD]   Galactic extinction in the B-band according to
                                 Schlegel et al. (1998ApJ...500..525S)
--------------------------------------------------------------------------------
Note (2): There are 3 additional galaxies in the tables which do not meet the
           luminosity (f) and inclination (i) criteria but happened to be in the
           same WSRT fields as galaxies from the complete sample.
Note (3): For galaxies which are not observed or not detected in HI, this is the
           smallest position angle of the major axis measured eastward from the
           north.
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table2.dat
--------------------------------------------------------------------------------
   Bytes Format Units   Label    Explanations
--------------------------------------------------------------------------------
   1-  2  A2    ---     Sample   Samples (G1)
   4- 10  A7    ---     Name     Galaxy name (U for UGC, N for NGC)
      11  A1    ---   n_Name     [fi] Note on the galaxy (2)
  13- 17  F5.2  mag     Bmag     Total B magnitude (3)
  19- 23  F5.2  mag     Rmag     Total R magnitude (3)
  25- 29  F5.2  mag     Imag     Total I magnitude (3)
  31- 35  F5.2  mag     K'mag    ? Total K'mag magnitude (3)
  37- 39  I3    km/s    WHI      ? Corrected HI line width at the 20% level,
                                    used to calculate the internal extinction
  41- 44  F4.2  mag     AB       Calculated internal extinction corrections in
                                  the B passband  (4)
  46- 49  F4.2  mag     AR       Calculated internal extinction corrections in
                                  the R passband  (4)
  51- 54  F4.2  mag     AI       Calculated internal extinction corrections in
                                  the I passband  (4)
  56- 59  F4.2  mag     AK'      Calculated internal extinction corrections in
                                  the K' passband  (4)
  61- 66  F6.2  mag     BMAG     Ttal absolute B magnitude (5)
  68- 73  F6.2  mag     RMAG     Total absolute R magnitude (5)
  75- 80  F6.2  mag     IMAG     Total absolute I magnitude (5)
  82- 87  F6.2  mag     K'MAG    ? Total absolute K' magnitude (5)
  89- 92  F4.2  arcmin  D25      Diameter of the 25th mag/arcsec^2^ blue
                                  isophote (6)
--------------------------------------------------------------------------------
Note (2): There are 3 additional galaxies in the tables which do not meet the
           luminosity (f) and inclination (i) criteria but happened to be in the
           same WSRT fields as galaxies from the complete sample.
Note (3): Total magnitudes form paper I (Tully et al., 1996AJ....112.2471T)
Note (4): Calculated internal extinction corrections toward face-on
           A^i->0^_lambda_ according to Tully et al. (1998AJ....115.2264T):
           A^i->0^_lambda_ = {gamma}_{lambda}_*log(a/b)
           where a/b is the observed axis ratio of the galaxy as an indication
           of inclination while {gamma}_{lambda}_ depends on the luminosity and
           is calculated according to:
                {gamma}_B_ = 1.57 + 2.75(logW^i^_R,I_ - 2.5)
                {gamma}_R_ = 1.15 + 1.88(logW^i^_R,I_ - 2.5)
                {gamma}_I_ = 0.92 + 1.63(logW^i^_R,I_ - 2.5)
                {gamma}_K'_= 0.22 + 0.40(logW^i^_R,I_ - 2.5)
           where W^i^_R,I_ is the distance independent HI line width corrected
           for instrumental resolution, corrected for turbulent motion according
           to Tully & Fouqui (1985ApJS...58...67T) (TFq hereafter) with
           W_t,20_=22km/s as motivated in Sect. 4 and corrected for inclination
           using i_adopt_ from table1.
Note (5): Total absolute magnitudes corrected for Galactic and internal
           extinction and a distance modulus of 31.35 corresponding to a
           distance to the Ursa Major cluster of 18.6 Mpc
Note (6): Diameter corrected for both galactic and internal extinction and
           projection
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table3.dat
--------------------------------------------------------------------------------
   Bytes Format Units     Label  Explanations
--------------------------------------------------------------------------------
   1-  5  A5    ---       Name   Galaxy name (U for UGC, N for NGC)
   6- 10  A5    ---     n_Name   Note on Name (1)
  12- 16  F5.1  km/s      W20    HI line width at the 20% level, this work
  18- 20  F3.1  km/s    e_W20    rms uncertainty on W20
  22- 25  F4.1  km/s      Res    Velocity resolution, this work
  27- 31  F5.1  Jy.km/s   Int    Integrated flux, this work
  33- 35  F3.1  Jy.km/s e_Int    rms uncertainty on Int
  37- 39  I3    km/s      W20l   ? HI line width at the 20% level, from Ref
      40  A1    ---     n_W20l   [m?] Note on W20l (1)
  42- 43  I2    km/s    e_W20l   ? rms uncertainty on W20l
  45- 48  F4.1  km/s      Resl   ?  Velocity resolution, from Ref
  50- 54  F5.1  Jy.km/s   Intl   ? Integrated flux, from Ref
      55  A1    ---     n_Intl   [?!] Note on Intl (1)
  57- 60  F4.1  Jy.km/s e_Intl   ? rms uncertainty on Intl
  62- 63  I2    ---     r_Intl   ? References (2)
  65- 68  A4    ---       Obs    Synthesis observation with the WRST or the VLA
--------------------------------------------------------------------------------
Note (1): Notes:
     (c): the authors suggest possible confusion with a dwarf companion.
       c: flagged by the authors as confused with near companion.
       l: large correction factor (>1.20) applied for primary beam
             flux attenuation.
       i: flagged by the authors as possibly interacting.
    noSD: no useful single dish profile available due to obvious confusion.
       m: line width directly measured from the published HI profile.
       !: the integrated flux as quoted by the author is a factor 2
           larger than is quoted by any other source. Therefore, half the
           integrated flux was adopted from this source.
Note (2): References:
       1: Fisher & Tully (1981ApJS...47..139F)
       2: Appleton & Davies (1982MNRAS.201.1073A)
       3: Richter & Huchtmeier (1991A&AS...87..425R)
       4: Oosterloo & Shostak (1993A&AS...99..379O)
       5: Huchtmeier & Richter (1986A&AS...63..323H)
       6: Schneider et al. (1992ApJS...81....5S)
       7: Jore et al. (1996AJ....112..438J)
       8: Schwarz (1985A&A...142..273S)
       9: Thuan & Martin (1981ApJ...247..823T)
      10: Magri (1994AJ....108..896M)
      11: Van der Burg (1987, Ph.D. Thesis, University of Groningen)
      12: Gottesman et al. (1984ApJ...286..471G)
      13: Van Moorsel (1983A&AS...54....1V)
      14: Grewing & Mebold (1975A&A....42..119G)
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table4.dat
--------------------------------------------------------------------------------
  Bytes Format Units   Label    Explanations
--------------------------------------------------------------------------------
  1-  2  A2    ---     Sample   [F X] Sample (1)
  4-  8  A5    ---     Name     Galaxy name (N for NGC, U for UGC)
 10- 12  I3    arcsec  Rad      Radius
 14- 16  I3    km/s    VrotApp  ? Approaching rotation velocity
 18- 20  I3    km/s    VrotAppM ? Maximum value of approaching rotation velocity
 22- 24  I3    km/s    VrotAppm ? Minimum value of approaching rotation velocity
 26- 28  I3    km/s    VrotRec  ? Receding rotation velocity
 30- 32  I3    km/s    VrotRecM ? Maximum value of receding rotation velocity
 34- 36  I3    km/s    VrotRecm ? Minimum value of receding rotation velocity
 38- 40  I3    km/s    Vrot     ? Average rotational velocity
 42- 44  I3    deg     Incl     Inclination
 46- 48  I3    deg     PA       [13/381] Position angle
--------------------------------------------------------------------------------
Note (1): Samples:
      FX: Rotation curves derived from velocity fields and XV-diagrams
       X: Rotation curves derived from XV-diagrams only
--------------------------------------------------------------------------------

Byte-by-byte Description of file: table5.dat
--------------------------------------------------------------------------------
   Bytes Format Units     Label   Explanations
--------------------------------------------------------------------------------
   1-  2  A2    ---       Sample  [FPH] Sample  (G1)
   4-  8  A5    ---       Name    Galaxy name (N for NGC, U for UGC)
  10- 14  F5.1  km/s      W20     Uncorrected widths with formal errors of the
                                   global profiles at 20% of the peak flux
  16- 18  F3.1  km/s    e_W20     rms uncertainty on W20
  20- 24  F5.1  km/s      W50     Uncorrected widths with formal errors of the
                                   global profiles at 50% of the peak flux
  26- 28  F3.1  km/s    e_W50     rms uncertainty on W50
  30- 33  F4.1  km/s      Res     Instrumental velocity resolution at which the
                                   global profiles were observed
  35- 40  F6.1  km/s      HRV     Heliocentric systemic velocity as derived from
                                   the global profiles
  42- 44  F3.1  km/s    e_HRV     rms uncertainty on HRV
  46- 50  F5.1  Jy.km/s   Int     Integrated HI flux
  52- 54  F3.1  Jy.km/s e_Int     rms uncertainty on Int
      56  A1    ---     l_F21cm   Limit flag on F21cm
  57- 61  F5.1  mJy       F21cm   21cm (1400MHz) continuum flux density (2)
  63- 65  F3.1  mJy     e_F21cm   ? rms uncertainty on F21cm
  67- 70  F4.2  arcmin    RHI     ? Radius of the HI disk (3)
  72- 75  F4.2  arcmin    Rlmp    ? Radius of the last measured point of the
                                     rotation curve (4)
  77- 79  I3    km/s      Vlmp    ? Rotational velocity of the last measured
                                     point
  81- 82  I2    km/s    e_Vlmp    ? rms uncertainty on Vrot
  84- 90  A7    ---       Shape   Information on the overall shape of the
                                   rotation curve (5)
  92- 94  I3    km/s      Vmax    ? Maximum observed rotational velocity (6)
  96- 97  I2    km/s    e_Vmax    ? rms uncertainty on Vmax
  99-101  I3    km/s      Vflat   ? Average rotational velocity of the flat part
                                     of the rotation curve (7)
 103-104  I2    km/s    e_Vflat   ? rms uncertainty on Vflat
--------------------------------------------------------------------------------
Note (2): In case no continuum flux was detected, a 3{sigma} upper limit 
      for extended emission is given.
Note (3): Radius at the azimuthally averaged surface density of 1M_{sun}/pc^2^, 
     measured from the radial surface density profiles.
Note (4): The differences between Rlmp and RHI depend on the sensitivity 
    of the measurement and the distribution of the HI gas along the
    kinematic major axis.
Note (5): Overal shape of the rotation curve:
       R: rising rotation curve
       F: the rotation curve shows a flat part
       D: the rotation curve shows a declining part
       L: lopsided
Note (6): For galaxies with a rising rotation curve (R) Vmax=Vlmp
Note (7): For galaxies with a flat rotation curve (F) Vflat=Vmax may deviate
       from Vlmp because V_flat was averaged over the flat part of the
       rotation curve while Vlmp was measured at a single point.
--------------------------------------------------------------------------------

Global Notes:
Note (G1): the sample values are:
    FH: Galaxies with fully analyzed HI data
    PH: Galaxies with partially analyzed HI data
    CH: Galaxies with confused HI data
    NH: Not observed or too little HI content

History:
    From electronic version of the journal

References:
    Tully et al.,      Paper I   1996AJ....112.2471T, Cat. <J/AJ/112/2471>
    Tully & Verheijen, Paper II  1997ApJ...484..145T
================================================================================
(End)                                        Patricia Bauer [CDS]    05-Jun-2001
