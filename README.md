# README #

Rough version of Expresso data analysis GUI. The GUI is intended to take time series measurements output by the Expresso *Drosophila* feeding measurement system and identify feeding bouts 

### Dependencies ###

The Expresso GUI takes advantage of many of the standard Python modules, e.g. scipy, numpy, Tkinter, etc. However, it also includes a view non-standard dependencies. These can call be installed via pip or conda (if applicable).

* h5py (http://www.h5py.org/) deals with the hdf5 data file format
* pywt (https://pywavelets.readthedocs.io/en/latest/) package for using wavelets in Python
* changepy (https://github.com/ruipgil/changepy) python implementation of the PELT change point detection algorithm
* statsmodels (http://statsmodels.sourceforge.net/)

### To do ###

Since this is a rough version of the code, there still remain quite a few things to do:

* Replace user-specified free parameters with values derived from data itself
* Test analysis results versus user-annotated data
* Integrate video tracking
* Rearrange GUI components to be as user-friendly as possible
* Further analysis for bouts (beyond just duration)
* Write methods for analyzing many flies and/or many trials