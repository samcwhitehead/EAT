# EXPRESSO ANALYSIS TOOLBOX (EAT) #

**E**xpresso **A**nalysis **T**oolbox (EAT) is a Python toolbox for analyzing data from Expresso behavioral assays, in which the food consumption of individual *Drosophila* is monitored on the nanoliter scale (described in [Yapici et al., 2016](https://doi.org/10.1016/j.cell.2016.02.061). EAT expands on the original Expresso assay by combining high-resolution food intake measurements with motion tracking, allowing a multi-faceted analysis of *Drosophila* feeding and foraging behavior. For more information, see our manuscript: [to be determined](http://yapicilab.com/research-projects.html) 

### Expresso Hardware and Firmware ###

EAT is designed to be used for analyzing data from Expresso behavioral assays. The resources listed below can be used to build the apparatus for Expresso experiments and collect data during Expresson experiments (prbably need to update these):
	* [Open source plans for Expresso sensor banks](http://public.iorodeo.com/docs/expresso/hardware_design_files.html)
	* [Source code for Expresso data acquisition software](http://public.iorodeo.com/docs/expresso/device_software.html)

### EAT Installation ###

EAT can be run on Windows, Linux, or MacOS. We recommend an **Anaconda environment** based installation, as it provides the easiest route to get things running, but the installation process can be adapted to other Python distributions as needed. The steps to install EAT using an Anaconda environment are described below:

0. **Install Anaconda** If you do not already have a Python installation on your machine, we recommend downloading and installing Anaconda to do so. Installation files for Anaconda can be found here: [Anaconda installation file download](https://www.anaconda.com/products/individual), and operating-system-specific installation instructions can be found here [Anaconda installation guide](https://docs.anaconda.com/anaconda/install/) (find your OS on the left hand column under *Anaconda Individual Edition* or appropriate Anaconda version. Note that EAT works with both Python 2 and Python 3, but Python 3 is recommended.

1. **Create an Anaconda environment for running EAT** This is used to prevent conflicts within the base installation, and allow users a clean installation of EAT. To do so:
	* Download EAT code from the Bitbucket repository: [EAT code repository](https://bitbucket.org/samcwhitehead/visual_expresso_gui/src/master/) [need to update once we put on github]
	* Extract EAT files wherever you'd like to store Python scripts. We'll refer to the folder containg the EAT repository as *../Visual Expresso/* [need to update code repository name]
	* Navigate to **./Visual Expresso/** in the Anaconda terminal (recommended), cmd, or Unix terminal
	* Create the EAT Anaconda environment using the eat environment .yaml file by running `conda env create --file eat.yaml` in the terminal. 

2. **Activate the Anaconda environment for EAT** Once the `eat` environment from Step 1 has been successfully created, activate it by running `conda activate eat` in the terminal

3. **Configure TkDND** TkDND is a tool to allow drag and drop functionality for the EAT GUI interfaces. To configure this appropriately:
	* Confirm that your EAT repository contains a folder called *TkinterDnD2* (*./Visual Expresso/src/TkinterDnD2*). This should be included by default. If this folder is not present in your EAT repository , you can download a new version of TkDND via sourceforge: [TkDND download](https://sourceforge.net/projects/tkdnd/)
	* Locate the proper tkdnd binaries for your operating system. These can be found in the compressed folder *tkdnd_binaries* (*./Visual Expresso/src/tkdnd_binaries*). Extract the contents of the *tkdnd_binaries* folder and locate the subfolder corresponding to your operating system (e.g. for a 64-bit Windows installation, locate the *Win64* folder). In the appropriate folder for your operating system, you should find a compressed file of the form *tkdnd2.8-(OS-INFO).tar.gz*. If this file is not present, it can also be downloaded from sourceforge: [tkdnd binaries download](https://sourceforge.net/projects/tkdnd/files/) -- navigate to the appropriate operating system binaries folder on this sourceforge page, select TkDnD2.8, and download the relevant .tar.gz file for your OS.
	* Extract the contents of the tkdnd binary file, *tkdnd2.8-(OS-INFO).tar.gz*. The extraction can be performed with WinRar, 7Zip, default OS extraction tool, etc.
	* Move the extracted tkdnd binaries folder, which should be called *tkdnd2.8*, into the *tcl* folder of the Anaconda `eat` environment. For Windows, this *tcl* folder may look like *./Anaconda3/envs/eat/tcl/*. Note: if installing without using an Anaconda environment, the *tkdnd2.8* folder should be moved to the *tcl* folder of whatever Python installation you're using.

4. **Open IDE of choice to run/edit EAT code** While the EAT code can be run from the terminal as any python script, using an integrated development environment (IDE) to open/edit/run EAT code can make the process more intuitive. The IDE that you are most comfortable with is best, but, if you do not already have a preferred IDE, we provide a few suggestions below:
	* Spyder, the default Anaconda IDE, is installed in the `eat` environment by default. To open Spyder, run the command `spyder` in the Anaconda terminal (with the `eat` environment activated, as in Step 2). To avoid issues with plots produced by the EAT code, we suggest changing the default graphics backend in Spyder by going to *Tools --> Preferences --> IPython Console --> Graphics (tab)* and switching the default, "Inline", to "Tkinter".
	* PyCharm is added as an optional installation alongside Anaconda. To get PyCharm to work along with EAT, open the EAT code as a Project, then set the `eat` environment as the Project Interpreter. That process is described here: [Using PyCharm and Anaconda environment](https://docs.anaconda.com/anaconda/user-guide/tasks/pycharm/).
	
### Usage ###
(Needs to be fleshed out)

The main tools from EAT are three GUI interfaces which deal with pre-processing, processing, and post-processing data from Visual Expresso behavioral assays. These three are briefly described below:

* **`crop_and_save_gui.py`** is used to crop and save videos taken during Visual Expresso experiments. The cropped videos assist in data sorting and computation time for future analyses, and are thus a vital part of the analysis pipeline.
* **`visual_expresso_gui_main.py`** is the main GUI interfact use to analyze Expresso data. It is used to perform meal bout detection, video tracking, plotting data, and saving analysis results.
* **`post_processing_gui.py`** is used with analysis summary files generated by `visual_expresso_gui_main.py` to perform groupwise comparisons both via generating plots and performing hypothesis testing.

### To do ###
*