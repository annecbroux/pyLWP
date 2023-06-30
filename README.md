# pyLWP

This repo (previously private repo on LTE github) contains the codes of the Liquid Water Path (LWP) and Integrated Water Vapor (IWV, sometimes also refered to as "Precipitable Water Vapor" or PWV) retrieval presented in Billault-Roux and Berne, 2021 (https://doi.org/10.5194/amt-14-2749-2021).

This uses as input measurements from the cloud radar/radiometer "WProf" (Küchler et al. 2017, https://doi.org/10.1175/JTECH-D-17-0019.1), namely the 89-GHz brightness temperature (TB).

Optional inputs are information on the geographical location, surface atmospheric conditions, and reanalysis products (ERA5, https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5).

A neural network is used for this retrieval, implemented through the keras python library (Chollet 2015, https://keras.io).
The requirements for running the retrieval are listed in requirements.txt (Note: more packages might be needed for fully re-training the model: this was not tested).

The user is most likely interested by the code contained in the juypter notebook **`implementation_lwp_retrieval.ipynb`** which illustrates how to implement the retrieval using real data.

The repo additionally contains the following directories and files:
- The "parameters" directory contains the neural network parameters + values to use for normalization (tree of subdirectories which are browsed through depending on the chosen input features).
- The "training_scripts" directory contains a few scripts that were used during training of the model.
- "download_ERA5_data.py" is a script for the download of reanalysis data, which should be adapted by the 
- ICEGENESIS_Jan_ERA5_for_LWP.nc: example of ERA5 data in NetCDF format
- tools.py: a few useful functions


Questions should be directed at anne-claire.billault-roux@epfl.ch

