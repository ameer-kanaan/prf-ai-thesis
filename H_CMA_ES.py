import os
import sys
import time
import csv
from datetime import datetime
import numpy as np
import nibabel as nib
from scipy import io
import yaml
from prfpy import stimulus, model
from joblib import Parallel, delayed
from scipy.optimize import minimize
import cma


#### H-BIPOP-CMA-ES utility dependencies ####
def sanitize_bounds(bounds, epsilon=1e-8):
    '''
    Sanitize the bounds by ensuring that the lower bound is less than the upper bound.
    '''
    if bounds is None:
        return None
    sanitized = []
    for lb, ub in bounds:
        lb_f = float(lb)
        ub_f = float(ub)
        if lb_f >= ub_f:
            ub_f = lb_f + epsilon
        sanitized.append((lb_f, ub_f))
    return sanitized

class IsoWrapper:
    def __init__(self, prf_stim, hrf_pars, model_type="Gaussian"):
        if model_type == "Gaussian":
            self.model = model.Iso2DGaussianModel(stimulus=prf_stim, hrf=hrf_pars)
        elif model_type == "DoG":
            self.model = model.DoG_Iso2DGaussianModel(stimulus=prf_stim, hrf=hrf_pars)
        elif model_type == "DN":
            self.model = model.Norm_Iso2DGaussianModel(stimulus=prf_stim, hrf=hrf_pars)

    def predict_batch(self, batch_params):
        batch_params = np.array(batch_params)
        preds = self.model.return_prediction(*batch_params.T)
        return np.array(preds)

def compute_r2(y, preds):
    ss_res = np.sum((preds - y) ** 2, axis=1)
    ss_tot = np.sum((y - np.mean(y, axis=1, keepdims=True)) ** 2, axis=1)
    return 1 - ss_res / ss_tot


def create_fitness_function_batch(y_voxel_array, wrapper):
    def fitness_batch(params_batch):
        preds = wrapper.predict_batch(params_batch)
        y_expanded = np.repeat(y_voxel_array[np.newaxis, :], preds.shape[0], axis=0)
        r2 = compute_r2(y_expanded, preds)
        return -r2
    return fitness_batch


def optimize_voxel_cma(voxel_idx, initial_param, y_real, prf_stim, hrf_pars, max_iter, model_type,
                       sigma=1, popsize=16, restarts=1, bipop=True, sigma_vec=None, bounds=None):

    y_voxel_array = y_real[:, voxel_idx]
    wrapper = IsoWrapper(prf_stim, hrf_pars, model_type)
    fitness_batch = create_fitness_function_batch(y_voxel_array, wrapper)

    opts = {'maxiter': max_iter, 'verbose': -9}
    if popsize is not None:
        opts['popsize'] = popsize
    if sigma_vec is not None:
        opts['CMA_stds'] = sigma_vec
    if bipop:
        opts['CMA_active'] = True
    if bounds is not None:
        lower_bounds, upper_bounds = zip(*bounds)
        opts['bounds'] = [list(lower_bounds), list(upper_bounds)]

    best_params = None
    best_fitness = np.inf
    for _ in range(restarts + 1):
        es = cma.CMAEvolutionStrategy(initial_param, sigma, opts)
        while not es.stop():
            solutions = es.ask()
            fitnesses = fitness_batch(solutions)
            es.tell(solutions, fitnesses)
        if es.result.fbest < best_fitness:
            best_fitness = es.result.fbest
            best_params = es.result.xbest

    best_r2 = -best_fitness
    return voxel_idx, best_params, best_r2

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


def hybrid_optimize_voxel_cma(voxel_idx, initial_param, y_real, prf_stim, hrf_pars, model_type,
                              max_iter_cma, sigma, popsize, restarts, bipop, sigma_vec, bounds):
    voxel_idx, cma_best_params, cma_best_r2 = optimize_voxel_cma(
        voxel_idx, initial_param, y_real, prf_stim, hrf_pars, max_iter_cma, model_type,
        sigma=sigma, popsize=popsize, restarts=restarts, bipop=bipop, sigma_vec=sigma_vec, bounds=bounds)
    return voxel_idx, cma_best_params, cma_best_r2


def hybrid_optimize_voxel_finefit(voxel_idx, cma_best_params, y_real, prf_stim, hrf_pars, model_type,
                                  max_iter_scipy, bounds):
    model_instance = {
        "Gaussian": model.Iso2DGaussianModel(prf_stim, hrf_pars),
        "DoG": model.DoG_Iso2DGaussianModel(prf_stim, hrf_pars),
        "DN": model.Norm_Iso2DGaussianModel(prf_stim, hrf_pars)
    }[model_type]

    y = y_real[:, voxel_idx]

    def objective(params):
        pred = model_instance.return_prediction(*params).flatten()
        ss_res = np.sum((pred - y) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return ss_res / ss_tot

    res = minimize(objective, cma_best_params, method='Nelder-Mead', bounds=bounds, options={'maxiter': max_iter_scipy})
    final_params = res.x
    final_r2 = 1 - res.fun

    return voxel_idx, final_params, final_r2


#### This is the Main H-BIPOP-CMA-ES Function #####

def hybrid_optimize_all_voxels(y_real, prf_stim, hrf_pars, model_type, initial_params=None, max_iter_cma=20, max_iter_scipy=200, n_jobs=-1,
                               sigma=1, popsize=16, restarts=2, bipop=True, sigma_vec=None, bounds=None):
    '''
    y_real: The real observed data for the voxel.
    prf_stim: The stimulus used for the PRF model (a stimulus object from prfpy).
    hrf_pars: The hemodynamic response function parameters (per SPM method).
    model_type: The type of model to use (e.g., "Gaussian", "DoG", "DN"). "None" automatically defaults to a Gaussian.
    initial_params: Initial parameters for the optimization. If None, defaults to the prior means.
    max_iter_cma: Maximum iterations for the coarse fitting phase (ie. BIPOP-CMA-ES.. without the H-).
    max_iter_scipy: Maximum iterations for the fine fitting phase (ie. the SciPy optimizer Nelder-Mead.. this is the H- in H-BIPOP-CMA-ES).
    n_jobs: Number of parallel jobs to run.
    sigma: Initial global standard deviation for the CMA-ES optimizer.
    popsize: Population size for the BIPOP-CMA-ES optimizer.
    restarts: Number of restarts for the BIPOP-CMA-ES optimizer.
    bipop: Whether to use a bi-population strategy (Boolean).
    sigma_vec: Vector of standard deviations for each parameter. If None, defaults to the prior standard deviations.
    bounds: Bounds for the parameters.

    Returns: The optimized modeled voxels with their corresponding parameters and R² value.
    '''


    # Clean the bounds for mathematical stability
    bounds = sanitize_bounds(bounds)

    if initial_params is None:
        # this the mutation mean prior, it was experimentally deduced from the Spinoza center for neuroimaging archive
        if model_type in ("Gaussian", None):
            initial_params = [[0, 0, 1, 0, 0, 1, 0]]
        elif model_type == "DoG":
            initial_params = [[0, 0, 1, 0, 0, 0.001, 5.5, 1, 0]]
        elif model_type == "DN":
            initial_params = [[0, 0, 1, 0, 0, 0.001, 5.5, 1.5, 20, 1, 0]]

    if sigma_vec is None:
        # this the mutation variance (diagonal) prior, it was experimentally deduced from the Spinoza center for neuroimaging archive
        if model_type in ("Gaussian", None):
            sigma_vec = np.array([1.7, 1.7, 1.25, 0.065, 0.1, 3.0, 0.05])
        elif model_type == "DoG":
            sigma_vec = np.array([1.7, 1.7, 1.25, 0.065, 0.1, 0.01, 6.0, 3.0, 0.05])
        elif model_type == "DN":
            sigma_vec = np.array([1.7, 1.7, 1.25, 0.065, 0.1, 0.01, 6.0, 2.0, 30.0, 3.0, 0.05])

    n_voxels = y_real.shape[1]

    # Start CMA fit phase
    cma_fit_start = time.time()
    cma_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(hybrid_optimize_voxel_cma)(
            i, initial_params[0], y_real, prf_stim, hrf_pars, model_type, max_iter_cma,
            sigma, popsize, restarts, bipop, sigma_vec, bounds
        )
        for i in range(n_voxels)
    )
    cma_fit_elapsed = time.time() - cma_fit_start

    cma_output = np.array([np.concatenate([best_params.reshape(-1), [best_r2]]) for _, best_params, best_r2 in cma_results])
    avg_r2_cma = np.round(cma_output[:, -1].mean(), 4)

    print(f"CMA fit took: {cma_fit_elapsed:.2f} seconds ({cma_fit_elapsed/60:.2f} minutes), Avg R² is {avg_r2_cma}")

    # Start finefit phase
    finefit_start = time.time()

    finefit_results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(hybrid_optimize_voxel_finefit)(
            voxel_idx, cma_best_params, y_real, prf_stim, hrf_pars, model_type, max_iter_scipy, bounds
        )
        for voxel_idx, cma_best_params, _ in cma_results
    )
    finefit_elapsed = time.time() - finefit_start

    finefit_output = np.array([np.concatenate([best_params.reshape(-1), [best_r2]]) for _, best_params, best_r2 in finefit_results])
    avg_r2_finefit = np.round(finefit_output[:, -1].mean(), 4)

    print(f"Finefit took: {finefit_elapsed:.2f} seconds ({finefit_elapsed/60:.2f} minutes), Avg R² is {avg_r2_finefit}")

    # ---------------------- SAVE HYBRID RESULTS --------------------
    hybrid_results = {
        'num_vertices': n_voxels,
        'coarsefit_time': round(cma_fit_elapsed, 2),
        'iterfit_time': round(finefit_elapsed, 2),
        'total_time': round(cma_fit_elapsed + finefit_elapsed, 2),
        'coarsefit_r2': avg_r2_cma,
        'iterfit_r2': avg_r2_finefit,
        'library': 'H-CMA-ES(V ROIs - Unlimited NM & CMA iter & 2 restarts)',
        'iterfit_optimizer': 'Nelder-Mead'
    }
    save_fit_results('HCMAES_fit_results.csv', hybrid_results)

    return finefit_output


## --- Execution ---
## EXAMPLE USAGE

#### Data loading and objects initialization - Just standard loading of pRF —normalized and post-processed— data ####
# print("Start of script: ", datetime.now().strftime("%H:%M:%S"))

# vertices_num = [1,2,11,101,1001,10001]
# task_id = int(sys.argv[1])
# num_vertices= vertices_num[task_id-1]

# # Aliases & constants
# opj = os.path.join
# basedir = '/home/ekenanoglu/DoG'
# eps = 1e-1
# inf = np.inf
# njobs = 192
# nbatches = njobs

# # Directory paths
# dir_contents_path = opj(basedir, "crossvalidation", "raw_data")
# dir_contents_subs = os.listdir(dir_contents_path)

# # Subjects & sessions
# subjects = ["002"] 
# sessions = ["avg"]

# # Load analysis config
# with open(opj(basedir, 'prf_analysis.yml')) as f:
#     analysis_info = yaml.safe_load(f)

# # Load design matrix
# dm = io.loadmat(opj(basedir, 'design_task-2R.mat'))['stim'][:, :, 5:]

# # Create stimulus object
# prf_stim = stimulus.PRFStimulus2D(
#     screen_size_cm=analysis_info['screen_size_cm'],
#     screen_distance_cm=analysis_info['screen_distance_cm'],
#     design_matrix=dm, 
#     TR=analysis_info['TR']
# )

# # Precompute parameters
# ss = prf_stim.screen_size_degrees
# max_ecc_size = ss / 2.0
# grid_nr = 30
# sizes = max_ecc_size * np.linspace(0.25, 1, grid_nr)**2
# eccs = max_ecc_size * np.linspace(0.1, 1, grid_nr)**2
# polars = np.linspace(0, 2*np.pi, grid_nr)

# # Thresholds and grids from config
# rsq_threshold = analysis_info['rsq_threshold']
# xtol = analysis_info['xtol']
# ftol = analysis_info['ftol']
# hrf_pars = analysis_info['hrf']['pars']
# surround_amplitude_grid = analysis_info['norm']['surround_amplitude_grid']
# surround_size_grid = analysis_info['norm']['surround_size_grid']
# neural_baseline_grid = analysis_info['norm']['neural_baseline_grid']
# surround_baseline_grid = analysis_info['norm']['surround_baseline_grid']

# # Process data for each session and subject
# for session_id in sessions:
#     for subject in subjects:
#         print(f"Loading subject {subject} data! - session {session_id}")

#         labels_rois = opj(basedir, "data", f'sub-{subject}')
#         data_full = np.load(opj(labels_rois, f'ses-{session_id}', 
#                                 f"sub-{subject}_ses-{session_id}_task-2R_hemi-LR_desc-avg_bold.npy"))

#         # Load ROIs for left and right hemispheres
#         rois = ["V1", "V2", "V3", "FEF"]
#         lh_labels = {roi: nib.freesurfer.read_label(opj(labels_rois, "rois", "corrections", f'lh.{roi}.label')) for roi in rois}
#         rh_labels = {roi: nib.freesurfer.read_label(opj(labels_rois, "rois", "corrections", f'rh.{roi}.label')) for roi in rois}
#         all_lh = nib.freesurfer.read_geometry(opj(labels_rois, "rois", "surf", 'lh.inflated'))
#         offset = len(all_lh[0])  # Offset for RH vertices

#         # Combine and offset RH labels
#         combined_vertices = {
#             roi: np.sort(np.concatenate([lh_labels[roi], rh_labels[roi] + offset]))
#             for roi in rois
#         }

#         # Aggregate ROIs
#         roi_vertices = np.sort(np.concatenate(list(combined_vertices.values())))[:num_vertices]

# # Iterative search parameters
# iterative_search_params_file = np.load(
#     opj(basedir, "output", f"sub-{subject}", "fine grid params", 
#         f"dog_sub{subject}_sesavg_itrsrchparams_free_final.npy")
# )
# hrf_pars = iterative_search_params_file[:, 5:7]

# # Load V1 ROI for left & right hemispheres
# cw = labels_rois  # Alias for clarity

# # Load and extract data
# data_full = np.load(opj(cw, 'ses-avg', f"sub-{subject}_ses-avg_task-2R_hemi-LR_desc-avg_bold.npy"))
# print(f"Loading subject {subject} data! - session avg")
# ind_of_interst = roi_vertices
# data_roi = data_full[:, ind_of_interst]
# y_real = data_roi

# # Define Gaussian bounds
# gauss_bounds = [
#     (-1.5 * max_ecc_size, 1.5 * max_ecc_size),  # x
#     (-1.5 * max_ecc_size, 1.5 * max_ecc_size),  # y
#     (0.2, 1.5 * ss),                            # prf size
#     tuple(analysis_info['prf_ampl_gauss']),     # prf amplitude
#     tuple(analysis_info['bold_bsl']),           # bold baseline
#     tuple(analysis_info['hrf']['deriv_bound']), # hrf derivative
#     tuple(analysis_info['hrf']['disp_bound'])   # hrf dispersion
# ]


# # Assumes max_ecc_size, ss, and analysis_info are defined earlier in your script
# gauss_bounds = [(-1.5 * max_ecc_size, 1.5 * max_ecc_size),  # x
#                 (-1.5 * max_ecc_size, 1.5 * max_ecc_size),  # y
#                 (0.2, 1.5 * ss),  # prf size
#                 tuple(analysis_info['prf_ampl_gauss']),  # prf amplitude
#                 tuple(analysis_info['bold_bsl']),  # bold baseline
#                 tuple(analysis_info['hrf']['deriv_bound']),  # hrf derivative
#                 tuple(analysis_info['hrf']['disp_bound'])]


# DoG_bounds = [(-1.5 * max_ecc_size, 1.5 * max_ecc_size),  # x
#               (-1.5 * max_ecc_size, 1.5 * max_ecc_size),  # y
#               (0.2, 1.5 * ss),  # prf size
#               tuple(analysis_info['prf_ampl_dog']),  # prf amplitude
#               tuple(analysis_info['bold_bsl']),  # bold baseline
#               tuple(analysis_info['dog']['surround_amplitude_bound']),
#               (eps, 3 * ss),  # surround size
#               tuple(analysis_info['hrf']['deriv_bound']),  # hrf derivative
#               tuple(analysis_info['hrf']['disp_bound'])]  # hrf dispersion

# DN_bounds = [(-1.5 * max_ecc_size, 1.5 * max_ecc_size),  # x
#               (-1.5 * max_ecc_size, 1.5 * max_ecc_size),  # y
#               (0.2, 1.5 * ss),  # prf size
#               tuple(analysis_info['prf_ampl_dog']),  # prf amplitude
#               tuple(analysis_info['bold_bsl']),  # bold baseline
#               tuple(analysis_info['dog']['surround_amplitude_bound']),
#               (1e-8, 3 * ss),  # surround size (use small epsilon instead of eps variable)
#               tuple(analysis_info['norm']['neural_baseline_bound']),
#               tuple(analysis_info['norm']['surround_baseline_bound']),
#               tuple(analysis_info['hrf']['deriv_bound']),  # hrf derivative
#               tuple(analysis_info['hrf']['disp_bound'])]  # hrf dispersion


#### Now call the H-BIPOP-CMA-ES Function
# hybrid_res = hybrid_optimize_all_voxels(
#     y_real, prf_stim, hrf_pars, model_type="DoG", initial_params=None,
#     max_iter_cma=None, max_iter_scipy=None, n_jobs=-1, sigma=1, popsize=16, restarts=2, bipop=True,
#     sigma_vec=None, bounds=DoG_bounds)
