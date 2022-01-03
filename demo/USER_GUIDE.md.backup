# EXPRESSO ANALYSIS TOOLBOX (EAT) USER GUIDE #

**E**xpresso **A**nalysis **T**oolbox (EAT) is a Python toolbox for analyzing data from Expresso behavioral assays, in which the food consumption of individual *Drosophila* is monitored on the nanoliter scale (described in [Yapici et al., 2016](https://doi.org/10.1016/j.cell.2016.02.061). In addition to high-resolution food intake measurements, EAT facilitates the analysis of simultaneously captured motion tracking data, allowing a multi-faceted analysis of Drosophila feeding and foraging behavior. The source code for EAT can be found on GitHub ([here](https://github.com/scw97/EAT)), and an example of this combined feeding/tracking analysis can be seen in our forthcoming manuscript [TBD](http://yapicilab.com/research-projects.html).

For installation and assay setup information, see the repository [README](https://github.com/scw97/EAT/edit/master/README.md) file.

## Overview ##

In a typical run of the Visual Expresso assay, the user will collect both feeding and video data via the Expresso banks and an external camera, respectively. Feeding data is saved in .hdf5 format, while video data is typically saved in .avi format. The goals of EAT are to identity meal bouts within the feeding data, track the fly's position using the video data, and synthesize these two data types for combined analyses.

The following is a rough outline of the steps used to achieve these goals, which we will expand upon in subsequent sections:

1. Preprocess raw video data to obtained cropped video files containing single flies, with the tip of the food capillary located manually.
2. Perform combined meal bout detection and tracking analysis on each fly.
3. Examine groupwise statistics for movement/eating variables.

Note that EAT can be used to analyze feeding and video data separately, but is best used when both are available. 

## Video preprocessing ##

![](gifs/pre_process_vid.gif)

The video data collected in our Visual Expresso experiments consists of overhead footage of two Expresso banks, each containing five channels. Thus, while feeding data is collected and stored at the per-fly level, video data typically contains multiple flies in frame. To match up feeding and tracking data, the video preprocessing GUI (`crop_and_save_gui.py`) allows the user to manually identify ROIs to generate per-fly video files, as well as determine the capillary tip location in each video and estimate the convertion from pixels to centimeters. 

Each Expresso bank has an assigned name --- e.g. "XP04" --- and the channels within each bank are numbered 1--5. Each individual fly in an experimental run can thus be identified by its combination of bank name and channel number, e.g. XP04/channel_1. For a video from an experimental run --- let's call it `example_data.avi` --- we'll use the preprocessing GUI to produce a series of video files of the form:
*  `example_data_XP04_channel_1.avi`
*  `example_data_XP04_channel_2.avi`
*  ...
*  `example_data_XP05_channel_1.avi`
*  ...

This process also generates a `example_data_VID_INFO.hdf5` file containing information on the cropping ROIs, pixel to cm conversion, etc. 

To perform video pre-processing, first open the preprocessing GUI by running **`crop_and_save_gui.py`**. Then do the following:

1. **Load videos for preprocessing into the GUI.** This can be accomplished in one of two ways:  
    * Click the \<Add Directory> button to locate and select directories 	containing video files for preprocessing. Then highlight desired directories in the "Directory list" box and click \<Get Video Files> to populate the "Detected files" box with all video files located within the selected directories.
    * Drag and drop videos files directly into the "Detected files" box. (It is also possible to drag and drop directories in the "Directory list" box and click \<Get Video Files> to populate the "Detected files").
 
2. **Select videos to preprocess.** Videos are selected by highlighting them within the "Detected files" box and clicking \<Add Video(s) to Batch> (located underneath the "Batch analyze list" box). This should populate the "Batch analyze list" box with the selected videos

3. **Enter Expresso banks names and channel numbers.** As mentioned above, each Expresso bank has a string identifier (e.g. "XP04") and a set number of channels --- this information should be stored in the same .hdf5 output file that contains the feeding data. For the video(s) you'd like to preprocess, enter the bank names and channel numbers into the "Bank 1 Name," "Bank 1 Channels," "Bank 2 Name," and "Bank 2 Channels" text fields, which channel numbers separated by columns. Ensure that all videos in the "Batch analyze list" box have the same bank names and channel numbers because, in subsequent steps, the preprocessing pipeline will ask you to identify the ROI for (Bank 1)/(Channel 1), (Bank 1)/(Channel 2), ..., (Bank 2)/(Channel 1), (Bank 2)/(Channel 2), ...

4. **Begin manual preprocessing input.** Click \<Process Videos> to begin. A single frame from the first video in the "Batch analyze list" box will then open, and you will be prompted to perform several tasks:
    * Define the ROI for (Bank 1)/(Channel 1) by drawing a rectangle with the mouse and pressing Enter. Note that the region highlighted in the video **must** correspond to (Bank 1)/(Channel 1) in the Expresso .hdf5 feeding data output --- we recommend adding bank name and channel numbers to the filming area to avoid ambiguity. 
    * Identify the capillary tip location for the (Bank 1)/(Channel 1) video. After defining the first ROI, a new window should open with a single frame showing only the ROI. Using this, double click on the location of the capillary tip and press Enter. NB: Estimates of capillary location can be edited later using the \<Refine Tip Estimate> button in the same preprocessing GUI.
    * Draw lines corresponding to the vial length and width in the (Bank 1)/(Channel 1) ROI. You will be prompted to draw these two lines separately; for each, draw the corresponding line on the cropped image and press enter to continue. These two distances within the image --- measured in pixels --- are then matched with stored values for the real-world lengths in centimeters to obtain the pixel/cm conversion factor. To update the real-world vial length measurements, change these values in the `v_expresso_gui_params.py` file (`vial_length_cm` and `vial_width_cm` in the `trackingParams` dictionary object).
    * Repeat the first two steps (drawing ROI and identifying capillary tip) for the remaining bank and channel combinations: (Bank 1)/(Channel 2), ..., (Bank 2)/(Channel 1), (Bank 2)/(Channel 2), ... Note that the vial dimensions only need to be measured once
    

After each bank/channel combination has had an ROI and capillary tip location defined, new cropped video files will be generated corresponding to each fly. These will be used in subsequent steps to extract fly body tracking data.


## Feeding and tracking analyses ##

Having preprocessed our video data such that we know have an individual video file for each fly, we can now use the main EAT GUI to extract information about fly meal bouts, movement, etc. To begin, run **`visual_expresso_gui_main.py`**, which loads an instance of the main EAT GUI window. The following steps will walk through the use of this GUI.

### Importing data ###
To analyze data with EAT, we first need to import it into the GUI. This can be accomplished in one of several ways: 
* Click the \<Add Directory> button and use it to select a folder containing Visual Expresso data --- this will populate the "Directory list" box. From there, highlight a directory (or set of directories) and click either \<Get HDF5 Files> or \<Get Video Files> to populate the "Channel data files" or "Video files" boxes. In the case of "Video files," this box will be populated with cropped .avi files, as generated from the preprocessing steps described in the previous section. In the case of "Channel data files," this box will be populated by .hdf5 files containing liquid level data output by the Expresso system. To access individual fly data from these .hdf5 files, use the \<Unpack XP> button to populate the 'XP list' box with the list of Expresso banks in the selected .hdf5 files and the \<Unpack XP> button to populate the "Channel list" box with individual fly channels (see GIF below). 
* Follow the same steps as above, but drag and drop folders into the "Directory list" box.
* Drag and drop .hdf5 files directly into the "Channel data files" box. Use the same steps as above to subdivide into bank and channel information. 
* Drag and drop cropped .avi files directly into the "Video files" box.

![](gifs/access_channel_data.gif)

To assist in the populating of the different data type list boxes, the options under the "Combined Analysis Tools" dropdown menu can be used to synchronize the contents of the "Channel list" and "Video files" boxes via either set union or intersection, as well as to match selected (highlighted) entries.

### Analyzing single channels ###

After populating the "Channel list" box, the buttons to its left can be used to analyze data from single flies. In particular, we can generate time series plots of liquid level with detected meal bouts indicated (\<Plot Channel>) or save meal bout information --- e.g. meal start, stop, duration, and volume --- in .csv format (\<Save CSV>) for a selected "Channel list" entry:

![](gifs/plot_and_save_channel.gif)

### Analyzing single videos ###

After populating the "Video files" box, the buttons to its left can be used to perform video tracking (\<Analyze Video>) and plot various kinematic parameters (\<Plot Results>). NB: tracking results are saved automatically in .hdf5 format with the filename suffix "_TRACKING" or "_TRACKING\_PROCESSED." 

![](gifs/analyze_vid.gif)

### Batch analysis ###

To analyze data from multiple flies at once, we use the batch analysis window on the righthand side of the main EAT GUI. The tabs on the top left of this window correspond to batch analysis of 1) just liquid level data ("Batch Channel Analysis), 2) just video data ("Batch Video Analysis"), and 3) combined liquid level and video data ("Batch Combined Analysis"). We'll discuss only the "Batch Combined Analysis" here, as it is typically the default, and since the analyses of single data types mirror the options available for combined data. 

To begin batch analysis, we need to populate the batch analysis list box. To do this, highlight of entries in either the "Channel list" or "Video files" box and click the \<Add Data to Batch> button. Underneath the batch analysis list box are three columns containing buttons, as well as a column of input fields on the lefthand side. The input fields are used solely as time limits and bin sizes for plots of feeding --- they will not affect the output of any saved analyses. The leftmost column of buttons deals with populating the batch analysis list box (adding entries as described above, as well as removing some or all entries). The more functionally useful buttons are found in the center and rightmost columns.

The central column of batch analysis buttons contains tools for analyzing an 