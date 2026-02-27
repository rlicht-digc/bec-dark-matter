J/A+A/623/A123      Einasto parameters for SPARC galaxies        (Ghari+, 2019)
================================================================================
Dark matter-baryon scaling relations from Einasto halo fits to SPARC galaxy
rotation curves.
    Ghari A., Famaey B., Laporte C., Haghi H.
    <Astron. Astrophys. 623, A123 (2019)>
    =2019A&A...623A.123G        (SIMBAD/NED BibCode)
================================================================================
ADC_Keywords: Galaxies, rotation
Keywords: galaxies: kinematics and dynamics  - galaxies: spiral - dark matter

Abstract:
    Dark matter-baryon scaling relations in galaxies are important in
    order to constrain galaxy formation models. Here, we provide a modern
    quantitative assessment of those relations, by modelling the rotation
    curves of galaxies from the Spitzer Photometry and Accurate Rotation
    Curves (SPARC) database with the Einasto dark halo model. We focus in
    particular on the comparison between the original SPARC parameters,
    with constant mass-to-light ratios for bulges and disks, and the
    parameters for which galaxies follow the tightest radial acceleration
    relation. We show that fits are improved in the second case, and that
    the pure halo scaling relations also become tighter. We report that
    the density at the radius where the slope is -2 is strongly
    anticorrelated to this radius, and to the Einasto index. The latter is
    close to unity for a large number of galaxies, indicative of large
    cores. In terms of dark matter-baryon scalings, we focus on relations
    between the core properties and the extent of the baryonic component,
    which are relevant to the cusp-core transformation process. We report
    a positive correlation between the core size of halos with small
    Einasto index and the stellar disk scale-length, as well as between
    the averaged dark matter density within 2kpc and the baryon-induced
    rotational velocity at that radius. This finding is related to the
    consequence of the radial acceleration relation on the diversity of
    rotation curve shapes, quantified by the rotational velocity at 2kpc.
    While a tight radial acceleration relation slightly decreases the
    observed diversity compared to the original SPARC parameters, the
    diversity of baryon-induced accelerations at 2kpc is sufficient to
    induce a large diversity, incompatible with current hydrodynamical
    simulations of galaxy formation, while maintaining a tight radial
    acceleration relation.

Description:
    Maximum posterior dark matter halo parameters of individual rotation
    curve fits to 160 SPARC galaxies with Q<3 and parameters from Li et
    al. (2018A&A...615A...3L).

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
tablea1.dat       92      160   Rotation curve fit results
--------------------------------------------------------------------------------

See also:
   J/AJ/152/157 : Mass models for 175 disk galaxies with SPARC (Lelli+, 2016)
   http://astroweb.cwru.edu/SPARC/ : SPARC database

Byte-by-byte Description of file: tablea1.dat
--------------------------------------------------------------------------------
   Bytes Format Units          Label    Explanations
--------------------------------------------------------------------------------
   1- 13  A13   ---            Name     Name of the galaxies
  15- 21  F7.3 10-3Msun/pc+3   rho-2    Einasto parameter rho_-2_,
                                         density at radius r-2
  23- 28  F6.3 10-3Msun/pc+3 e_rho-2    Error on rho-2 (lower value,
                                         68 percent confidence interval)
  30- 35  F6.3 10-3Msun/pc+3 E_rho-2    Error on rho-2 (upper value,
                                         68 percent confidence interval)
  37- 43  F7.3  kpc            r-2      Einasto parameter r_-2_, radius at which
                                         the density profile has a slope of -2
  45- 50  F6.3  kpc          e_r-2      Error on r-2 (lower value,
                                         68 percent confidence interval)
  52- 57  F6.3  kpc          E_r-2      Error on r-2 (upper value,
                                         68 percent confidence interval)
  59- 64  F6.3  ---            n        Einasto index n, sets the general
                                         shape of the density profile
  66- 70  F5.3  ---          e_n        Error on n (lower value,
                                         68 percent confidence interval)
  72- 76  F5.3  ---          E_n        Error on n (upper value,
                                         68 percent confidence interval)
  78- 84  F7.3  km/s           Vmax     Maximum circular velocity of the halo
  86- 92  F7.3  km/s           Vrot     Rotational velocity at 2kpc from
                                          the center
--------------------------------------------------------------------------------

Acknowledgements:
     Amir Ghari, amir.ghari(at)iasbs.ac.ir

================================================================================
(End)                                        Patricia Vannier [CDS]  31-Jan-2019
