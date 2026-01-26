###
# This code hosts the catalog manipulation to assemble the master dataset 
#for the COSMIC VLASS exoplanet host toy project
###

import pandas as pd
import numpy as np
import pyvo as vo


def main():
    print('Testing TAP')
    tap_service = vo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
    query = """
        SELECT TOP 5
            sy_name, hostname, ra, dec, gaia_dr2_id
        FROM
            stellarhosts
        WHERE 
            sy_pnum = 2
        """
    results = tap_service.search(query)
    print(results)
    return

if __name__ == "__main__":
    main()