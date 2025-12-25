# marshmallow-msi
A Python project for processing mass spectrometry imaging data (imzml) acquired from Bruker or Waters instruments.

The central class imat (which stands for image matrix)
stores information about a mass spectrometry imaging dataset including its image size
and mass channel intensities. This is the backbone of the rest of the modules of this
Python project.

The rest of the modules inlcude:
1. process_imat.py - Marshmallow class - which converts imzml files to imat object.
2. synth_msi.py -  SynthMSI class - generates synthetic MSI dataset when given 2 binary masks.




This project is set to be packaged soon.


Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

