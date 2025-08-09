import os
import sys
import time
import csv
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import io
import yaml
from prfpy import stimulus, model, fit

print("Start of script: ", datetime.now().strftime("%H:%M:%S"))

vertices_num = [1,2,11,101,1001,10001]
task_id = int(sys.argv[1])
num_vertices= vertices_num[task_id-1]

# Aliases & constants
opj = os.path.join
basedir = '/home/ekenanoglu/DoG'
eps = 1e-1
inf = np.inf
njobs = 192
nbatches = njobs

# Directory paths
dir_contents_path = opj(basedir, "crossvalidation", "raw_data")
dir_contents_subs = os.listdir(dir_contents_path)

# Subjects & sessions
subjects = ["002"] 
sessions = ["avg"]

# Load analysis config
with open(opj(basedir, 'prf_analysis.yml')) as f:
    analysis_info = yaml.safe_load(f)

# Load design matrix
dm = io.loadmat(opj(basedir, 'design_task-2R.mat'))['stim'][:, :, 5:]

# Create stimulus object
prf_stim = stimulus.PRFStimulus2D(
    screen_size_cm=analysis_info['screen_size_cm'],
    screen_distance_cm=analysis_info['screen_distance_cm'],
    design_matrix=dm, 
    TR=analysis_info['TR']
)

# Precompute parameters
ss = prf_stim.screen_size_degrees
max_ecc_size = ss / 2.0
grid_nr = 30
sizes = max_ecc_size * np.linspace(0.25, 1, grid_nr)**2
eccs = max_ecc_size * np.linspace(0.1, 1, grid_nr)**2
polars = np.linspace(0, 2*np.pi, grid_nr)

# Thresholds and grids from config
rsq_threshold = analysis_info['rsq_threshold']
xtol = analysis_info['xtol']
ftol = analysis_info['ftol']
hrf_pars = analysis_info['hrf']['pars']
surround_amplitude_grid = analysis_info['norm']['surround_amplitude_grid']
surround_size_grid = analysis_info['norm']['surround_size_grid']
neural_baseline_grid = analysis_info['norm']['neural_baseline_grid']
surround_baseline_grid = analysis_info['norm']['surround_baseline_grid']

# Process data for each session and subject
for session_id in sessions:
    for subject in subjects:
        print(f"Loading subject {subject} data! - session {session_id}")

        labels_rois = opj(basedir, "data", f'sub-{subject}')
        data_full = np.load(opj(labels_rois, f'ses-{session_id}', 
                                f"sub-{subject}_ses-{session_id}_task-2R_hemi-LR_desc-avg_bold.npy"))

        # Load ROIs for left and right hemispheres
        rois =  ["V1", "V2", "V3", "FEF"]
        lh_labels = {roi: nib.freesurfer.read_label(opj(labels_rois, "rois", "corrections", f'lh.{roi}.label')) for roi in rois}
        rh_labels = {roi: nib.freesurfer.read_label(opj(labels_rois, "rois", "corrections", f'rh.{roi}.label')) for roi in rois}
        all_lh = nib.freesurfer.read_geometry(opj(labels_rois, "rois", "surf", 'lh.inflated'))
        offset = len(all_lh[0])

        # Combine and offset RH labels
        combined_vertices = {
            roi: np.sort(np.concatenate([lh_labels[roi], rh_labels[roi] + offset]))
            for roi in rois
        }

        # Aggregate ROIs
        roi_vertices = np.sort(np.concatenate(list(combined_vertices.values())))[:num_vertices]

# Iterative search parameters
iterative_search_params_file = np.load(
    opj(basedir, "output", f"sub-{subject}", "fine grid params", 
        f"dog_sub{subject}_sesavg_itrsrchparams_free_final.npy")
)
hrf_pars = iterative_search_params_file[:, 5:7]

# Load V1 ROI for left & right hemispheres
cw = labels_rois  # Alias for clarity
V1_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.V1.label'))
all_lh = nib.freesurfer.read_geometry(opj(cw, "rois", "surf", 'lh.inflated'))
V1_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'rh.V1.label'))
V1_rh += len(all_lh[0])
V1_vertices = np.sort(np.concatenate([V1_lh, V1_rh]))

# Load and extract data
data_full = np.load(opj(cw, 'ses-avg', f"sub-{subject}_ses-avg_task-2R_hemi-LR_desc-avg_bold.npy"))
print(f"Loading subject {subject} data! - session avg")
ind_of_interst = roi_vertices
data_roi = data_full[:, ind_of_interst]
y_real = data_roi

# Define Gaussian bounds
gauss_bounds = [
    (-1.5 * max_ecc_size, 1.5 * max_ecc_size),  # x
    (-1.5 * max_ecc_size, 1.5 * max_ecc_size),  # y
    (0.2, 1.5 * ss),                            # prf size
    tuple(analysis_info['prf_ampl_gauss']),     # prf amplitude
    tuple(analysis_info['bold_bsl']),           # bold baseline
    tuple(analysis_info['hrf']['deriv_bound']), # hrf derivative
    tuple(analysis_info['hrf']['disp_bound'])   # hrf dispersion
]


def save_fit_results(filename, results):
    """
    Save or append fit results to a CSV file.
    """
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[
            'num_vertices',
            'coarsefit_time',
            'iterfit_time',
            'total_time',
            'coarsefit_r2',
            'iterfit_r2',
            'library',
            'iterfit_optimizer'
        ])
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

# --------------------- GRID FIT ROUTINE ------------------------

gauss_bounds = [(-1.5*max_ecc_size, 1.5*max_ecc_size),  # x
                (-1.5*max_ecc_size, 1.5*max_ecc_size),  # y
                (0.2, 1.5*ss),                          # prf size
                tuple(analysis_info['prf_ampl_gauss']), # prf amplitude
                tuple(analysis_info['bold_bsl']),       # bold baseline
                tuple(analysis_info['hrf']['deriv_bound']), # hrf derivative
                tuple(analysis_info['hrf']['disp_bound'])]  # hrf dispersion

print(f"Now starting Gaussian gridfit at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}", flush=True)
start = time.time()

fixed_gauss_model = model.Iso2DGaussianModel(stimulus=prf_stim, hrf=analysis_info['hrf']['pars'])
fixed_gauss_fitter = fit.Iso2DGaussianFitter(data=data_roi.T, model=fixed_gauss_model, n_jobs=njobs)

fixed_gauss_fitter.grid_fit(
    ecc_grid=eccs,
    polar_grid=polars,
    size_grid=sizes,
    hrf_1_grid=np.linspace(0, 5, 5),
    hrf_2_grid=np.array([0]),
    n_batches=nbatches,
    fixed_grid_baseline=None,
    grid_bounds=[tuple(analysis_info['prf_ampl_gauss'])],
    verbose=True
)

grid_elapsed = time.time() - start
fixed_gauss_fitter.grid_fit_elapsed = grid_elapsed

fixed_gauss_grid = np.nan_to_num(fixed_gauss_fitter.gridsearch_params)
mean_rsq_grid = np.mean(fixed_gauss_grid[fixed_gauss_grid[:, -1] > rsq_threshold, -1])

print(f"Completed Gaussian gridfit at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}. "
      f"Voxels/vertices above {rsq_threshold}: {np.sum(fixed_gauss_grid[:, -1] > rsq_threshold)}/{fixed_gauss_fitter.data.shape[0]}", flush=True)
print(f"Gridfit took {timedelta(seconds=grid_elapsed)} | Mean rsq>{rsq_threshold}: {round(mean_rsq_grid, 4)}", flush=True)

# ---------------------- ITERATIVE FIT ---------------------------

print(f"Now starting Gaussian iterfit at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}", flush=True)
start = time.time()

fixed_gauss_fitter.iterative_fit(
    rsq_threshold=rsq_threshold,
    bounds=gauss_bounds,
    constraints=[]
)

iter_elapsed = time.time() - start
fixed_gauss_fitter.iter_fit_elapsed = iter_elapsed

fixed_gauss_iter = np.nan_to_num(fixed_gauss_fitter.iterative_search_params)
mean_rsq_iter = np.nanmean(fixed_gauss_iter[fixed_gauss_fitter.rsq_mask, -1])

print(f"Completed Gaussian iterfit at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}. "
      f"Mean rsq>{rsq_threshold}: {round(mean_rsq_iter, 4)}", flush=True)
print(f"Iterfit took {timedelta(seconds=iter_elapsed)}", flush=True)


# ##################### DoG MODEL #############################

print(f"Now starting DoG gridfit at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}", flush=True)

DoG_bounds = [(-1.5*max_ecc_size, 1.5*max_ecc_size),  # x
              (-1.5*max_ecc_size, 1.5*max_ecc_size),  # y
              (0.2, 1.5*ss),                          # prf size
              tuple(analysis_info['prf_ampl_dog']),   # prf amplitude
              tuple(analysis_info['bold_bsl']),       # bold baseline
              tuple(analysis_info['dog']['surround_amplitude_bound']),
              (eps, 3*ss),                             # surround size
              tuple(analysis_info['hrf']['deriv_bound']), # hrf derivative
              tuple(analysis_info['hrf']['disp_bound'])]  # hrf dispersion

dog_model = model.DoG_Iso2DGaussianModel(stimulus=prf_stim, hrf=analysis_info['hrf']['pars'])
dog_fitter = fit.DoG_Iso2DGaussianFitter(data=data_roi.T, model=dog_model, n_jobs=njobs,
                                         previous_gaussian_fitter=fixed_gauss_fitter)

start = time.time()

dog_fitter.grid_fit(
    surround_amplitude_grid=surround_amplitude_grid,
    surround_size_grid=surround_size_grid,
    n_batches=nbatches,
    rsq_threshold=rsq_threshold,
    fixed_grid_baseline=None,
    grid_bounds=DoG_bounds,
    verbose=True
)

dog_grid_elapsed = time.time() - start
dog_fitter.grid_fit_elapsed = dog_grid_elapsed

dog_grid = np.nan_to_num(dog_fitter.gridsearch_params)
mean_rsq_grid_dog = np.mean(dog_grid[dog_grid[:, -1] > rsq_threshold, -1])

print(f"Completed DoG gridfit at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}. "
      f"Voxels/vertices above {rsq_threshold}: {np.sum(dog_grid[:, -1] > rsq_threshold)}/{dog_fitter.data.shape[0]}", flush=True)
print(f"Gridfit took {timedelta(seconds=dog_grid_elapsed)} | Mean rsq>{rsq_threshold}: {round(mean_rsq_grid_dog, 4)}", flush=True)

# --------------------- DoG ITERATIVE FIT -------------------------

print(f"Now starting DoG iterfit at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}", flush=True)
start = time.time()

dog_fitter.iterative_fit(
    rsq_threshold=rsq_threshold,
    bounds=DoG_bounds,
    constraints=[],
    method="Nelder-Mead"
)

dog_iter_elapsed = time.time() - start
dog_fitter.iter_fit_elapsed = dog_iter_elapsed

dog_iter = np.nan_to_num(dog_fitter.iterative_search_params)
mean_rsq_iter_dog = np.nanmean(dog_iter[dog_fitter.rsq_mask, -1])

print(f"Completed DoG iterfit at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}. "
      f"Mean rsq>{rsq_threshold}: {round(mean_rsq_iter_dog, 4)}", flush=True)
print(f"Iterfit took {timedelta(seconds=dog_iter_elapsed)}", flush=True)

# ---------------------- SAVE DoG RESULTS -------------------------
dog_results = {
    'num_vertices': len(roi_vertices),
    'coarsefit_time': round(dog_fitter.grid_fit_elapsed + fixed_gauss_fitter.grid_fit_elapsed, 2),
    'iterfit_time': round(dog_fitter.iter_fit_elapsed + fixed_gauss_fitter.iter_fit_elapsed, 2),
    'total_time': round(dog_fitter.grid_fit_elapsed + dog_fitter.iter_fit_elapsed + fixed_gauss_fitter.grid_fit_elapsed + fixed_gauss_fitter.iter_fit_elapsed, 2),
    'coarsefit_r2': round(mean_rsq_grid_dog, 4),
    'iterfit_r2': round(mean_rsq_iter_dog, 4),
    'library': 'prfpy (High Order ROIs)',
    'iterfit_optimizer': 'Nelder-Mead'
}
save_fit_results('prfpy_fit_results.csv', dog_results)
