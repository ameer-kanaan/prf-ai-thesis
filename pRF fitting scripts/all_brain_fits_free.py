import yaml
import os
from prfpy import stimulus, model, fit
import numpy as np
from scipy import io
from datetime import datetime, timedelta
import time
import sys
import nibabel as nib

opj = os.path.join
basedir = '/home/ekenanoglu/DoG'
eps = 1e-1
inf = np.inf

dir_contents = os.listdir("/home/ekenanoglu/DoG/data")

subjects =[i.split("-")[1] for i in dir_contents]
task_id = int(sys.argv[1])  # This gets the SLURM_ARRAY_TASK_ID

# Ensure task ID matches subject index
subject = subjects[task_id - 1] 

njobs = 192
nbatches = njobs

with open(opj(basedir, 'prf_analysis.yml')) as f:
    analysis_info = yaml.safe_load(f)

dm = io.loadmat(opj(basedir, 'design_task-2R.mat'))['stim']
dm = dm[:,:,5:]

cw = opj(basedir, "data", f'sub-{subject}')

sessions = ["2", "3", "avg"]

prf_stim = stimulus.PRFStimulus2D(
    screen_size_cm=analysis_info['screen_size_cm'],
    screen_distance_cm=analysis_info['screen_distance_cm'],
    design_matrix=dm, 
    TR=analysis_info['TR'])

ss = prf_stim.screen_size_degrees
max_ecc_size = ss/2.0

grid_nr = 30
sizes, eccs, polars = max_ecc_size * np.linspace(0.25, 1, grid_nr)**2, \
    max_ecc_size * np.linspace(0.1, 1, grid_nr)**2, \
    np.linspace(0, 2*np.pi, grid_nr)

#rsq thresh
rsq_threshold = analysis_info['rsq_threshold']
hrf_pars = analysis_info['hrf']['pars']

surround_amplitude_grid = analysis_info['norm']['surround_amplitude_grid']
surround_size_grid = analysis_info['norm']['surround_size_grid']
neural_baseline_grid = analysis_info['norm']['neural_baseline_grid']
surround_baseline_grid = analysis_info['norm']['surround_baseline_grid']

xtol = analysis_info['xtol']
ftol = analysis_info['ftol']

for session_id in sessions:
########### DATA LOADING ###########

    data_full = np.load(opj(cw, f'ses-{session_id}', f"sub-{subject}_ses-{session_id}_task-2R_hemi-LR_desc-avg_bold.npy"))
    print(f"Loading subject {subject} data! - session {session_id}")

    #Load ROI vertices from left hemisphere, together with the number of vertices in the left hemisphere so indexing for right hemisphere works appropriately; the .label files start from 0 and work per hemisphere, whereas Inkscape merges the two hemispheres together.

    V1_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.V1.label'))
    V2_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections",'lh.V2.label'))
    V3_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.V3.label'))

    #If you only want V1, V2 and v3, comment the lines under
    FEF_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.FEF.label'))
    IA_IPS_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.IA-IPS.label'))
    LO_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.LO-cluster.label'))
    P_IPS_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.P-IPS.label'))
    SA_IPS_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.SA-IPS.label'))
    V3AB_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.V3AB.label'))
    VO_CLUSTER_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.VO-cluster.label'))
    HMT_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.hMT+.label'))
    HV4_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.hV4.label'))

    all_lh = nib.freesurfer.read_geometry(opj(cw,"rois", "surf", 'lh.inflated'))

    #Load ROI vertices from right hemisphere. 
    V1_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections",'rh.V1.label'))
    V2_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections",'rh.V2.label'))
    V3_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections",'rh.V3.label'))

    #If you only want V1, V2 and v3, comment the lines under
    FEF_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'rh.FEF.label'))
    IA_IPS_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'rh.IA-IPS.label'))
    LO_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'rh.LO-cluster.label'))
    P_IPS_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'rh.P-IPS.label'))
    SA_IPS_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'rh.SA-IPS.label'))
    V3AB_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'rh.V3AB.label'))
    VO_CLUSTER_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'rh.VO-cluster.label'))
    HMT_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'rh.hMT+.label'))
    HV4_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'rh.hV4.label'))

    all_rh = nib.freesurfer.read_geometry(opj(cw, "rois", "surf",'rh.inflated'))

    V1_rh = V1_rh + len(all_lh[0])
    V2_rh = V2_rh + len(all_lh[0])
    V3_rh = V3_rh + len(all_lh[0])

    #If you only want V1, V2 and v3, comment the lines under
    FEF_rh += len(all_lh[0])
    IA_IPS_rh += len(all_lh[0])
    LO_rh += len(all_lh[0])
    P_IPS_rh += len(all_lh[0])
    SA_IPS_rh += len(all_lh[0])
    V3AB_rh += len(all_lh[0])
    VO_CLUSTER_rh += len(all_lh[0])
    HMT_rh += len(all_lh[0])
    HV4_rh += len(all_lh[0])

    V1_vertices = np.sort(np.concatenate([V1_lh, V1_rh]))
    V2_vertices = np.sort(np.concatenate([V2_lh, V2_rh]))
    V3_vertices = np.sort(np.concatenate([V3_lh, V3_rh]))

    #If you only want V1, V2 and v3, comment the lines under
    FEF_verticies = np.sort(np.concatenate([FEF_lh, FEF_rh]))
    IA_IPS_verticies = np.sort(np.concatenate([IA_IPS_lh, IA_IPS_rh]))
    LO_verticies = np.sort(np.concatenate([LO_lh, LO_rh]))
    P_IPS_verticies = np.sort(np.concatenate([P_IPS_lh, P_IPS_rh]))
    SA_IPS_verticies = np.sort(np.concatenate([SA_IPS_lh, SA_IPS_rh]))
    V3AB_verticies = np.sort(np.concatenate([V3AB_lh, V3AB_rh]))
    VO_CLUSTER_verticies = np.sort(np.concatenate([VO_CLUSTER_lh, VO_CLUSTER_rh]))
    HMT_verticies = np.sort(np.concatenate([HMT_lh, HMT_rh]))
    HV4_verticies = np.sort(np.concatenate([HV4_lh, HV4_rh]))

    # If you only want V1, V2 and V3, comment this.
    roi_vertices = np.sort(np.concatenate([V1_vertices, V2_vertices, V3_vertices, FEF_verticies, IA_IPS_verticies,
    LO_verticies, P_IPS_verticies, SA_IPS_verticies, V3AB_verticies, VO_CLUSTER_verticies, HMT_verticies, HV4_verticies]))

    indices_in_roi = np.searchsorted(roi_vertices, np.concatenate([V1_vertices, V2_vertices, V3_vertices]))
    mask = np.isin(roi_vertices[indices_in_roi], np.concatenate([V1_vertices, V2_vertices, V3_vertices]))
    roi_vertices = indices_in_roi[mask]
    data_roi = data_full[:, roi_vertices]

    if len(np.where(np.isnan(data_roi).all(axis=0))[0]) > 0: #Check if there are fully NaN data and deal with them
        roi_vertices = np.delete(roi_vertices.reshape(-1,1), np.where(np.isnan(data_roi).all(axis=0))[0], axis=0).ravel() #This excludes the nans from the roi_vertices
        data_roi = np.delete(data_roi, np.where(np.isnan(data_roi).all(axis=0))[0], axis=1) #This excludes the nans from the data_roi


    # If you only want V1, V2, and V3 vertices, uncomment this
    # roi_vertices = np.sort(np.concatenate([V1_vertices, V2_vertices, V3_vertices]))

    np.save(opj(basedir, 'output', f"sub-{subject}", f'roivertices_{subject}'), roi_vertices)

    data_roi = data_full[:, roi_vertices]


    # First, perform grid fit
    print(f"Now starting Gaussian gridfit at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}", flush=True)
    start = time.time()
##################### Gaussian Model ########################
    gauss_bounds = [(-1.5*max_ecc_size, 1.5*max_ecc_size),  # x
                (-1.5*max_ecc_size, 1.5*max_ecc_size),  # y
                (0.2, 1.5*ss),  # prf size
                tuple(analysis_info['prf_ampl_gauss']),  # prf amplitude
                tuple(analysis_info['bold_bsl']), # bold baseline SHOULD THIS BE 0 OR 1000?
                tuple(analysis_info['hrf']['deriv_bound']), # hrf derivative
                tuple(analysis_info['hrf']['disp_bound'])]  # hrf dispersion

    
    gauss_model = model.Iso2DGaussianModel(stimulus=prf_stim, hrf=hrf_pars)

    gauss_fitter = fit.Iso2DGaussianFitter(data=data_roi.T, model=gauss_model, n_jobs=njobs)

    print(f"Now starting Gaussian gridfit at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}", flush=True)
    start = time.time()

    gauss_fitter.grid_fit(ecc_grid=eccs,
                    polar_grid=polars,
                    size_grid=sizes, 
                    hrf_1_grid=np.linspace(0,10,11),
                    hrf_2_grid= np.array([0]),
                    n_batches=nbatches,
                    fixed_grid_baseline=0,
                    grid_bounds=[tuple(analysis_info['prf_ampl_gauss'])],
                    verbose = True)

    elapsed = (time.time() - start)

    gauss_grid = np.nan_to_num(gauss_fitter.gridsearch_params)
    mean_rsq = np.mean(gauss_grid[gauss_grid[:, -1]>rsq_threshold, -1])
    start_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    nr = np.sum(gauss_grid[:, -1]>rsq_threshold)
    total = gauss_fitter.data.shape[0]
    print(f"Completed Gaussian gridfit at {start_time}. Voxels/vertices above {rsq_threshold}: {nr}/{total}",flush=True)
    print(f"Gridfit took {timedelta(seconds=elapsed)} | Mean rsq>{rsq_threshold}: {round(mean_rsq,2)}",flush=True)

    print(f"Now starting Gaussian iterfit at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}",flush=True)
    start = time.time()

    gauss_fitter.iterative_fit(
        rsq_threshold=rsq_threshold, 
        bounds=gauss_bounds,
        constraints=[],
        method = "Nelder-Mead")

    # print summary
    elapsed = (time.time() - start)              
    gauss_iter = np.nan_to_num(gauss_fitter.iterative_search_params)

    # verbose stuff
    mean_rsq = np.nanmean(gauss_iter[gauss_fitter.rsq_mask, -1])
    start_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')

    print(f"Completed Gaussian iterfit at {start_time}. Mean rsq>{rsq_threshold}: {round(mean_rsq,2)}",flush=True)
    print(f"Iterfit took {timedelta(seconds=elapsed)}",flush=True)

    g_params = [gauss_fitter.iterative_search_params[:,i] for i in range(7)]
    gauss_modeled = gauss_model.return_prediction(*g_params)
    # save parameters

    os.makedirs(opj(basedir, "output", f'sub-{subject}'), exist_ok=True)
    os.makedirs(opj(basedir, "output", f'sub-{subject}', "coarse grid params"), exist_ok=True)
    os.makedirs(opj(basedir, "output", f'sub-{subject}', "fine grid params"), exist_ok=True)

    np.save(opj(basedir, "output", f'sub-{subject}', "coarse grid params", f"gauss_sub{subject}_ses{session_id}_gridparams_free_final_final.npy"), gauss_fitter.gridsearch_params)
    np.save(opj(basedir, "output", f'sub-{subject}', "fine grid params", f"gauss_sub{subject}_ses{session_id}_itrsrchparams_free_final_final.npy"), gauss_fitter.iterative_search_params)
    np.save(opj(basedir, "output", f'sub-{subject}', f'gauss_ses{session_id}_free_final_final.npy'), gauss_modeled)

#################### DoG Model #############################

    DoG_bounds = [(-1.5*max_ecc_size, 1.5*max_ecc_size),  # x
                    (-1.5*max_ecc_size, 1.5*max_ecc_size), # y
                    (0.2, 1.5*ss), # prf size
                    tuple(analysis_info['prf_ampl_dog']),  # prf amplitude
                    tuple(analysis_info['bold_bsl']),# bold baseline SHOULD THIS BE 0 OR 1000?
                    tuple(analysis_info['dog']['surround_amplitude_bound']),
                    (eps, 3*ss),  # surround size
                    tuple(analysis_info['hrf']['deriv_bound']), # hrf derivative
                    tuple(analysis_info['hrf']['disp_bound'])]  # hrf dispersion


    dog_model = model.DoG_Iso2DGaussianModel(stimulus=prf_stim, hrf=hrf_pars)

    # Instantiate the DoG_Iso2DGaussianFitter
    dog_fitter = fit.DoG_Iso2DGaussianFitter(data=data_roi.T, model=dog_model, n_jobs=njobs, 
    previous_gaussian_fitter = gauss_fitter)

    dog_fitter.grid_fit(
        surround_amplitude_grid=surround_amplitude_grid,
        surround_size_grid=surround_size_grid,
        n_batches=nbatches,
        rsq_threshold=rsq_threshold,
        fixed_grid_baseline=0,
        grid_bounds = [tuple(analysis_info['prf_ampl_norm']), tuple(analysis_info['dog']['surround_amplitude_bound'])], #DoG_bounds,
        hrf_1_grid=np.linspace(0,10,11),
        hrf_2_grid = np.array([0]),
        verbose=True
    )

    elapsed = (time.time() - start)

    dog_grid = np.nan_to_num(dog_fitter.gridsearch_params)
    mean_rsq = np.mean(dog_grid[dog_grid[:, -1]>rsq_threshold, -1])
            
    # verbose stuff
    start_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    nr = np.sum(dog_grid[:, -1]>rsq_threshold)
    total = dog_fitter.data.shape[0]
    print(f"Completed DoG gridfit at {start_time}. Voxels/vertices above {rsq_threshold}: {nr}/{total}",flush=True)
    print(f"Gridfit took {timedelta(seconds=elapsed)} | Mean rsq>{rsq_threshold}: {round(mean_rsq,2)}",flush=True)

    # now, iterative fit
    print(f"Now starting DoG iterfit at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}",flush=True)
    start = time.time()

    dog_fitter.iterative_fit(
        rsq_threshold=rsq_threshold, 
        bounds=DoG_bounds,
        constraints=[],
        method = "Nelder-Mead")

    # print summary
    elapsed = (time.time() - start)              
    dog_iter = np.nan_to_num(dog_fitter.iterative_search_params)

    # verbose stuff
    mean_rsq = np.nanmean(dog_iter[dog_fitter.rsq_mask, -1])
    start_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')

    print(f"Completed DoG iterfit at {start_time}. Mean rsq>{rsq_threshold}: {round(mean_rsq,2)}",flush=True)
    print(f"Iterfit took {timedelta(seconds=elapsed)}", flush=True)

    dog_params = [dog_fitter.iterative_search_params[:,i] for i in range(9)]
    dog_modeled = dog_model.return_prediction(*dog_params)
#     # save parameters

    os.makedirs(opj(basedir, "output", f'sub-{subject}'), exist_ok=True)
    os.makedirs(opj(basedir, "output", f'sub-{subject}', "coarse grid params"), exist_ok=True)
    os.makedirs(opj(basedir, "output", f'sub-{subject}', "fine grid params"), exist_ok=True)

    np.save(opj(basedir, "output", f'sub-{subject}', "coarse grid params", f"dog_sub{subject}_ses{session_id}_gridparams_free_final_final.npy"), dog_fitter.gridsearch_params)
    np.save(opj(basedir, "output", f'sub-{subject}', "fine grid params", f"dog_sub{subject}_ses{session_id}_itrsrchparams_free_final_final.npy"), dog_fitter.iterative_search_params)
    np.save(opj(basedir, "output", f'sub-{subject}', f'dog_ses{session_id}_free_final_final.npy'), dog_modeled)

####### DN Model #############
    norm_bounds = [(-1.5*max_ecc_size, 1.5*max_ecc_size),  # x
                    (-1.5*max_ecc_size, 1.5*max_ecc_size),  # y
                    (0.2, 1.5*ss),  # prf size
                    tuple(analysis_info['prf_ampl_norm']),  # prf amplitude
                    tuple(analysis_info['bold_bsl']),  # bold baseline SHOULD THIS BE 0 OR 1000?
                    tuple(analysis_info['norm']['surround_amplitude_bound']),  # surround amplitude
                    (eps, 3*ss),  # surround size
                    tuple(analysis_info['norm']['neural_baseline_bound']),  # neural baseline
                    tuple([float(item) for item in analysis_info['norm']['surround_baseline_bound']]), # surround baseline
                    tuple(analysis_info['hrf']['deriv_bound']), # hrf derivative
                    tuple(analysis_info['hrf']['disp_bound'])]  # hrf dispersion

    norm_model = model.Norm_Iso2DGaussianModel(stimulus=prf_stim, hrf=hrf_pars)
    norm_fitter = fit.Norm_Iso2DGaussianFitter(data=data_roi.T, model=norm_model, previous_gaussian_fitter=gauss_fitter, 
    n_jobs=njobs)

    # first, gridfit
    print(f"Now starting DN gridfit at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}",flush=True)
    start = time.time()

    norm_fitter.grid_fit(surround_amplitude_grid=surround_amplitude_grid,
                    surround_size_grid=surround_size_grid,
                    neural_baseline_grid=neural_baseline_grid,
                    surround_baseline_grid=surround_baseline_grid, gaussian_params=gauss_fitter.iterative_search_params,
                    n_batches=nbatches,
                    rsq_threshold=rsq_threshold,
                    hrf_1_grid=np.linspace(0,10,11),
                    hrf_2_grid = np.array([0]),
                    fixed_grid_baseline= 0, #0,
                    grid_bounds= [tuple(analysis_info['prf_ampl_norm']), tuple(analysis_info['norm']['neural_baseline_bound'])], #norm_bounds, 
                    verbose=True)

    elapsed = (time.time() - start)

    norm_grid = np.nan_to_num(norm_fitter.gridsearch_params)
    mean_rsq = np.mean(norm_grid[norm_grid[:, -1]>rsq_threshold, -1])
            
    # verbose stuff
    start_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    nr = np.sum(norm_grid[:, -1]>rsq_threshold)
    total = norm_fitter.data.shape[0]
    print(f"Completed DN gridfit at {start_time}. Voxels/vertices above {rsq_threshold}: {nr}/{total}",flush=True)
    print(f"Gridfit took {timedelta(seconds=elapsed)} | Mean rsq>{rsq_threshold}: {round(mean_rsq,2)}",flush=True)

    # now, iterative fit
    print(f"Now starting DN iterfit at {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}",flush=True)
    start = time.time()

    norm_fitter.iterative_fit(
        rsq_threshold=rsq_threshold, 
        bounds=norm_bounds,
        constraints=[],
        method = "Nelder-Mead"
        )

    # print summary
    elapsed = (time.time() - start)              
    norm_iter = np.nan_to_num(norm_fitter.iterative_search_params)

    # verbose stuff
    mean_rsq = np.nanmean(norm_iter[norm_fitter.rsq_mask, -1])
    start_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')

    print(f"Completed DN iterfit at {start_time}. Mean rsq>{rsq_threshold}: {round(mean_rsq,2)}",flush=True)
    print(f"Iterfit took {timedelta(seconds=elapsed)}", flush=True)

    dn_params = [norm_fitter.iterative_search_params[:,i] for i in range(11)]
    
    dn_modeled = norm_model.return_prediction(*dn_params)
    # save parameters
    np.save(opj(basedir, "output", f'sub-{subject}', "coarse grid params", f"dn_sub{subject}_ses{session_id}_gridparams_free_final_final.npy"), norm_fitter.gridsearch_params)
    np.save(opj(basedir, "output", f'sub-{subject}', "fine grid params", f"dn_sub{subject}_ses{session_id}_itrsrchparams_free_final_final.npy"), norm_fitter.iterative_search_params)
    np.save(opj(basedir, "output", f'sub-{subject}', f'dn_ses{session_id}_free_final_final.npy'), dn_modeled)