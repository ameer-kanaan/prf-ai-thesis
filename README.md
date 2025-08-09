# Search for a Biomarker of Neural Feedback in the Human Brain: Optimization with Evolutionary Artificial Intelligence Methods and Evaluation of Neural Outcomes

Complete code implementations from my Master thesis in AI investigating population receptive field (pRF) models. A study at the intersection of cognitive neuroscience and AI.
### Introduction
I implemented a DoG–based pRF model to fit fMRI data from our study, which experimentally manipulated feedback processing in humans using Memantine. This is summarized in the following diagram: 
<img width="1667" height="1558" alt="0" src="https://github.com/user-attachments/assets/35f19a46-2221-4363-9f68-8d635c6e7f95" />

Additionally, I made and benchmarked a non-gradient based AI fitting routine that focuses on efficient runtime-accuracy tradeoff. The newly introduced method greatly reduced runtime by \~75% with minimal impact on accuracy (\~ -3%) under default settings as compared to the incumbant grid search method. This Evolutionary-Strategy-inspired method relied on archival data from real experiments to initialize well placed priors that greatly enhance the parameter fitting process. Additionally, it is designed to work with the defaults to enable "on the go" fits, yet it also keeps the power of customization with the researcher in order to adjust the runtime-accuracy tradeoff as needed, with the potential to beat the grid search approach (accuracy-wise) if the researcher is willing to be patient (ie. sacrifice runtime).

The name I gave to this approach is: **Hybridized BiPOPulational Covariance Matrix Adaptation Evolutionary Strategy (H-BIPOP-CMA-ES)**, which is made of two phases; a BIPOP-CMA-ES phase which implements a state-of-the-art variant of Evolutionary Strategies for global search, then forwards the results to a SciPy Nelder-Mead optimizer for fine fitting.

### Repository Contents
The main directory includes has the following files and folders:
* H_CMA_ES.py (File): This python script includes my implementation of H-BIPOP-CMA-ES, an opitimization method for pRF fitting that serves as a more scalable alternative to the standard grid search approach used in the field.
* H-BIPOP-CMA-ES vs prfpy (Folder): scripts, spreadsheets and slurm jobs for benchmarking H-BIPOP-CMA-ES against the default approach (grid fit based coarse-to-fine optimization from the _prfpy_ library).
* Notebooks (Folder): Includes my main experimental "playground" to develop the visualizations, make sense of the data, and such, loosly structured, as it was my development scratchsheet.
* pRF fitting scripts (Folder): Includes the code used to generate the pRF modelling results mentioned in the Memantine study in the thesis. aggregate_data_merge.py is the script used to concatenate all subjects in one dataset. Slurm job submission are provided too.

Any .sh files in th repo are slurm job submissions forwarded to Snellius (The Dutch National Supercomputer).

### Memantine Study Pipeline
First, the entire dataset is fitted to an averaged version across conditions, using the all_brain_fits_free.py script. The resulting parameters (X, Y, and HRF1) from these free fits are then used to generate another model with these parameters fixed, via the all_brain_fits_fixed.py script. This latter model is used to report the results.

### Acknowledgement
This thesis, titled "_Search for a Biomarker of Neural Feedback in the Human Brain: Optimization with Evolutionary Artificial Intelligence Methods and Evaluation of Neural Outcomes_", authored by Emir Kenanoglu and supervised by Dr. Maartje de Jong and Dr. Tomas Knappen, was submitted in fulfillment of the requirements for the Master of Artificial Intelligence program (Cognitive Sciences Track) at Vrije Universiteit Amsterdam. The research was conducted as part of a thesis internship at Spinoza Centre for Neuroimaging, a subsidiary of VU Amsterdam, UMC Amsterdam, and Netherlands Institute for Neuroscience (NIN) - Computational Cognitive Neuroscience & Neuroimaging group.

## Repository References and Credits
My work was built on the following libraries:
* _prfpy_ for pRF modelling: https://github.com/VU-Cog-Sci/prfpy/
* _pycma_ for BIPOP-CMA-ES optimization: https://github.com/CMA-ES/pycma
* Special thanks to Marcus Daghlian's utility functions for pRF fitting support: https://github.com/mdaghlian/dag_prf_utils/tree/main/dag_prf_utils

