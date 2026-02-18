import os
from collections import defaultdict

import pandas as pd


def load_galaxy(
    table_path="./",
    profile_path="profiles/",
    name="UGC10909",
    MainTable=None,
    RC=None,
    SB=None,
    Model=None,
    Struct=None,
):
    """
    Function to load a single galaxy and all its associated data.

    data_path: file path to PROBES data [string]
    name: name of galaxy to load [string]

    returns: dictionary with the tables and profiles for the galaxy:
        Galaxy['main table'] = table of galaxy properties
        Galaxy['photometry'] = dictionary of photometry profiles. Keys are the photometry bands.
        Galaxy['rotation curve'] = rotation curve profile
        Galaxy['model fits'] = dictionary of model fits
        Galaxy['structural parameters'] = dictionary of structural parameters
    """

    Galaxy = defaultdict(dict)

    # Load main table data
    if MainTable is None:
        MainTable = pd.read_csv(os.path.join(table_path, "main_table.csv"), skiprows=1)
    g = list(MainTable["name"]).index(name)
    Galaxy["main table"] = dict((h, MainTable[h][g]) for h in MainTable.keys())

    # Load photometry and rotation curve
    if RC is None:
        RC = pd.read_csv(os.path.join(profile_path, f"{name}_rc.prof"), skiprows=1)
    Galaxy["rotation curve"] = RC
    SB = {}
    for band in ["f", "n", "g", "r", "z", "w1", "w2"]:
        try:
            if band not in SB:
                SB[band] = pd.read_csv(
                    os.path.join(profile_path, f"{name}_{band}.prof"), skiprows=1
                )
            Galaxy["photometry"][band] = SB[band]
        except FileNotFoundError:
            pass

    # Load model fits
    try:
        if Model is None:
            Model = pd.read_csv(os.path.join(table_path, f"model_fits.csv"))
        m = list(Model["name"]).index(name)
        Galaxy["model fits"] = dict((h, Model[h][m]) for h in Model.keys())
    except ValueError:
        pass

    # Load Structural Parameters
    try:
        if Struct is None:
            Struct = pd.read_csv(os.path.join(table_path, f"structural_parameters.csv"), skiprows=1)
        s = list(Struct["name"]).index(name)
        Galaxy["structural parameters"] = dict((h, Struct[h][s]) for h in Struct.keys())
    except ValueError:
        pass

    return Galaxy


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    Galaxy = load_galaxy(name="UGC10909")
    print(Galaxy.keys())
    print(Galaxy["main_table"].keys())
    print(Galaxy["photometry"].keys())
    print(Galaxy["rotation curve"].keys())
    print(Galaxy["model fits"].keys())
    print(Galaxy["structural parameters"].keys())

    plt.figure()
    plt.scatter(
        Galaxy["rotation curve"]["R"], Galaxy["rotation curve"]["V"], s=10, c="k", linewidth=0
    )
    plt.grid()
    plt.xlabel("Radius (arcsec)")
    plt.ylabel("Velocity (km/s)")
    plt.show()

    plt.figure()
    for band in Galaxy["photometry"].keys():
        plt.plot(Galaxy["photometry"][band]["R"], Galaxy["photometry"][band]["SB"], label=band)
    plt.ylim([None, 35])
    plt.xlabel("Radius (arcsec)")
    plt.ylabel("Surface Brightness (mag/arcsec^2)")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()
