import os

import pandas as pd
import numpy as np

from load_galaxy import load_galaxy


def read_galaxies(
    table_path="./", profile_path="profiles/", has=[], limit=np.inf, skip_flagged=np.inf
):
    """
    Function to read in galaxy data for all galaxies with specified restrictions.

    data_path: file path to PROBES data [string]
    has: require all loaded galaxies to have the listed photometry bands [list of strings]
    limit: only load in this many galaxies, useful for testing [int]
    skip_flagged: skip any galaxy with this many photometry flags [int]

    returns: dictionary with galaxy name as keys, and galaxy data as values. The data is as returned by load_galaxy.py
    """

    # Load the data tables
    MainTable = pd.read_csv(os.path.join(table_path, "main_table.csv"), skiprows=1)
    Model = pd.read_csv(os.path.join(table_path, f"model_fits.csv"))
    Struct = pd.read_csv(os.path.join(table_path, f"structural_parameters.csv"), skiprows=1)
    Galaxies = {}

    # Loop over galaxies
    for g in range(len(MainTable["name"])):
        # Check if galaxy has all required photometry bands
        hasall = True
        for h in has:
            if not MainTable[f"has_{h}-band"][g]:
                hasall = False
        if not hasall:
            continue

        # Check if galaxy has too many photometry flags
        flags = 0
        if MainTable[f"AutoProf_flags"][g] != "-":
            flags += 1
        flags += MainTable[f"AutoProf_flags"][g].count(";")
        if flags >= skip_flagged:
            continue

        # Load the galaxy
        Galaxies[MainTable["name"][g]] = load_galaxy(
            table_path=table_path,
            profile_path=profile_path,
            name=MainTable["name"][g],
            MainTable=MainTable,
            Model=Model,
            Struct=Struct,
        )

        # Stop if we've loaded enough galaxies
        if len(Galaxies) >= limit:
            break

    return Galaxies


if __name__ == "__main__":
    Galaxies = read_galaxies(limit=10)
    print(Galaxies.keys())
    gal = tuple(Galaxies.keys())[0]
    print(Galaxies[gal].keys())
    print(Galaxies[gal]["main table"].keys())
    print(Galaxies[gal]["photometry"].keys())
    print(Galaxies[gal]["rotation curve"].keys())
    print(Galaxies[gal]["model fits"].keys())

    import matplotlib.pyplot as plt

    plt.scatter(
        Galaxies[gal]["rotation curve"]["R"],
        Galaxies[gal]["rotation curve"]["V"],
        s=10,
        c="k",
        linewidth=0,
    )
    plt.grid()
    plt.xlabel("R [arcsec]")
    plt.ylabel("V [km/s]")
    plt.show()

    plt.figure()
    for band in Galaxies[gal]["photometry"].keys():
        plt.plot(
            Galaxies[gal]["photometry"][band]["R"],
            Galaxies[gal]["photometry"][band]["SB"],
            label=band,
        )
    plt.ylim([None, 35])
    plt.xlabel("Radius (arcsec)")
    plt.ylabel("Surface Brightness (mag/arcsec^2)")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()
