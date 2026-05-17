# marshmallow-msi
A Python project for processing mass spectrometry imaging data (imzml) acquired from Bruker or Waters instruments.

Please cite: doi: https://doi.org/10.64898/2025.12.25.694368
A computational workflow for microscopy-guided ion identification in clinical mass spectrometry imaging datasets
Ana-Maria Năstase, Ping K. Yip,  Christopher E.G. Uff, Irma O’Meara,  Hervé Barjat


The central class imat (which stands for image matrix)
stores information about a mass spectrometry imaging dataset including its image size
and mass channel intensities. This is the backbone of the rest of the modules of this
Python project.

The rest of the modules inlcude:
1. process_imat.py - Marshmallow class - which converts imzml files to imat object.
2. synth_msi.py -  SynthMSI class - generates synthetic MSI dataset when given 2 binary masks.

Note: This repository is not actively maintained. For the latest version, please contact the author.

Shield: [![CC BY-NC-ND 4.0][cc-by-nc-nd-shield]][cc-by-nc-nd]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License][cc-by-nc-nd].

[![CC BY-NC-ND 4.0][cc-by-nc-nd-image]][cc-by-nc-nd]

[cc-by-nc-nd]: http://creativecommons.org/licenses/by-nc-nd/4.0/
[cc-by-nc-nd-image]: https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png
[cc-by-nc-nd-shield]: https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg



