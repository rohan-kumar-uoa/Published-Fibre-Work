This repository contains Data files and data analysis files for work on optical fibre sensing.

Data File Info:


```/Fibre Data Files/23.06 files```:

Contains datafiles from experiments done on a 3 FBG array, where each FBG was put into a strain cycle. The FBG array was made up of interrogator -> 01 -> 06 ->07. The folder names with (manual) in the title are from manually resetting the force to zero at the end of every cycle. This was overriding the Labview VI which gave non-zero force readings between each cycle.
Run 4, 6, 7 are the main data files, and have been analysed in Test_Notebook_3.ipynb

```/Fibre Data Files/17.06 strain array 07```:

Testing data for the above experiment. FBG 07 was strained for 2 cycles, and then 4 cycles. 

```/Fibre Data Files/Stationary Spectra```:

Contains stationary spectra from each of the individual 5 FBGs. Stationary spectra was collected on 17.04 and 24.04. The 'FBGs comb attempt' contains just FBG 01, 02, and their combined response. Should be used to create a spectral_summation() class for fitting a sum of FBG responses to an array.
This is shown in 'Test_Notebook_1'

```/Fibre Data Files/Time Dependent Spectra/Initial FBG array strain testing```:

These are the run files from strain testing the FBG array using epoxy to hold the Fibre (where there was poor force transfer to the actual fibre). This is analysed in Test_Notebook_2.ipynb. The FBG array was made up of interrogator -> 01 -> 02 -> 04 -> 06 -> 07.

```/Fibre Data Files/10.06```:

These files are not analysed in the notebooks, but are from manual attempts at generating strain on individual FBGs in an FBG array. The array was just FBG 01,06,07.


Use 
```
pip install requirements.txt
```
to install all necessary packages.
