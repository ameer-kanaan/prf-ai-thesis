import os
import numpy as np
import nibabel as nib
import pandas as pd

opj = os.path.join
basedir = '/home/ekenanoglu/DoG'

dir_contents = os.listdir("/home/ekenanoglu/DoG/data")

subjects = [i.split("-")[1] for i in dir_contents]

session_key = {
    "001": {"2": "placebo", "3": "memantine"},
    "002": {"2": "placebo", "3": "memantine"},
    "003": {"2": "memantine", "3": "placebo"},
    "004": {"2": "memantine", "3": "placebo"},
    "005": {"2": "placebo", "3": "memantine"},
    "007": {"2": "memantine", "3": "placebo"},
    "008": {"2": "placebo", "3": "memantine"},
    "010": {"2": "placebo", "3": "memantine"},
    "012": {"2": "memantine", "3": "placebo"},
    "013": {"2": "placebo", "3": "memantine"},
    "015": {"2": "placebo", "3": "memantine"},
    "016": {"2": "memantine", "3": "placebo"}
}

df_fixed_ses2 = pd.DataFrame()
df_fixed_ses3 = pd.DataFrame()
df_free_ses2 = pd.DataFrame()
df_free_ses3 = pd.DataFrame()

df_fixed_params_ses2 = pd.DataFrame()
df_fixed_params_ses3 = pd.DataFrame()
df_free_params_ses2 = pd.DataFrame()
df_free_params_ses3 = pd.DataFrame()

df_fixed_pred_ses2 = pd.DataFrame()
df_fixed_pred_ses3 = pd.DataFrame()
df_free_pred_ses2 = pd.DataFrame()
df_free_pred_ses3 = pd.DataFrame()

df_fixed_params_pred_ses2 = pd.DataFrame()
df_fixed_params_pred_ses3 = pd.DataFrame()
df_free_params_pred_ses2 = pd.DataFrame()
df_free_params_pred_ses3 = pd.DataFrame()
df_free_params_sesavg = pd.DataFrame()

for subject in subjects:
#Load V1 vertices from left hemisphere, together with the number of vertices in the left hemisphere so indexing for right hemisphere works appropriately; the .label files start from 0 and work per hemisphere, whereas Inkscape merges the two hemispheres together.
    cw = f'/home/ekenanoglu/DoG/data/sub-{subject}/'

    V1_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.V1.label'))
    V2_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections",'lh.V2.label'))
    V3_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.V3.label'))
    # FEF_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.FEF.label'))
    # IA_IPS_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.IA-IPS.label'))
    # LO_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.LO-cluster.label'))
    # P_IPS_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.P-IPS.label'))
    # V3AB_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.V3AB.label'))
    # VO_CLUSTER_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.VO-cluster.label'))
    # HMT_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.hMT+.label'))
    # HV4_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.hV4.label'))
    # SA_IPS_lh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'lh.SA-IPS.label'))

    all_lh = nib.freesurfer.read_geometry(opj(cw,"rois", "surf", 'lh.inflated'))

    # #Load V1 vertices from right hemisphere. 
    V1_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections",'rh.V1.label'))
    V2_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections",'rh.V2.label'))
    V3_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections",'rh.V3.label'))
    # FEF_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'rh.FEF.label'))
    # IA_IPS_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'rh.IA-IPS.label'))
    # LO_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'rh.LO-cluster.label'))
    # P_IPS_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'rh.P-IPS.label'))
    # V3AB_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'rh.V3AB.label'))
    # VO_CLUSTER_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'rh.VO-cluster.label'))
    # HMT_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'rh.hMT+.label'))
    # HV4_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'rh.hV4.label'))
    # SA_IPS_rh = nib.freesurfer.read_label(opj(cw, "rois", "corrections", 'rh.SA-IPS.label'))

    all_rh = nib.freesurfer.read_geometry(opj(cw, "rois", "surf",'rh.inflated'))

    V1_rh = V1_rh + len(all_lh[0])
    V2_rh = V2_rh + len(all_lh[0])
    V3_rh = V3_rh + len(all_lh[0])
    # FEF_rh += len(all_lh[0])
    # IA_IPS_rh += len(all_lh[0])
    # LO_rh += len(all_lh[0])
    # P_IPS_rh += len(all_lh[0])
    # V3AB_rh += len(all_lh[0])
    # VO_CLUSTER_rh += len(all_lh[0])
    # HMT_rh += len(all_lh[0])
    # HV4_rh += len(all_lh[0])
    # SA_IPS_rh += len(all_lh[0])

    V1_vertices = np.sort(np.concatenate([V1_lh, V1_rh]))
    V2_vertices = np.sort(np.concatenate([V2_lh, V2_rh]))
    V3_vertices = np.sort(np.concatenate([V3_lh, V3_rh]))
    # FEF_verticies = np.sort(np.concatenate([FEF_lh, FEF_rh]))
    # IA_IPS_verticies = np.sort(np.concatenate([IA_IPS_lh, IA_IPS_rh]))
    # LO_verticies = np.sort(np.concatenate([LO_lh, LO_rh]))
    # P_IPS_verticies = np.sort(np.concatenate([P_IPS_lh, P_IPS_rh]))
    # V3AB_verticies = np.sort(np.concatenate([V3AB_lh, V3AB_rh]))
    # VO_CLUSTER_verticies = np.sort(np.concatenate([VO_CLUSTER_lh, VO_CLUSTER_rh]))
    # HMT_verticies = np.sort(np.concatenate([HMT_lh, HMT_rh]))
    # HV4_verticies = np.sort(np.concatenate([HV4_lh, HV4_rh]))
    # SA_IPS_verticies = np.sort(np.concatenate([SA_IPS_lh, SA_IPS_rh]))

    regions = {
    "V1": V1_vertices,
    "V2": V2_vertices,
    "V3": V3_vertices,
    "FEF": FEF_verticies,
    "IA_IPS": IA_IPS_verticies,
    "LO": LO_verticies,
    "P_IPS": P_IPS_verticies,
    "V3AB": V3AB_verticies,
    "VO_CLUSTER": VO_CLUSTER_verticies,
    "HMT": HMT_verticies,
    "HV4": HV4_verticies,
    "SA_IPS": SA_IPS_verticies
    }

    # Concatenate vertices and build region labels
    roi_vertices = []
    roi_regions = []

    for region_name, vertices in regions.items():
        roi_vertices.append(vertices)
        roi_regions.extend([region_name] * len(vertices))  # one label per vertex

    # Convert to numpy arrays
    roi_vertices = np.sort(np.concatenate(roi_vertices))
    roi_regions = np.array(roi_regions)[np.argsort(np.concatenate(list(regions.values())))]  # Sort regions to match sorted vertices

    # Remove NaN-containing vertices and mask both arrays accordingly
    data_full_avg = np.load(f"/home/ekenanoglu/DoG/data/sub-{subject}/ses-avg/sub-{subject}_ses-avg_task-2R_hemi-LR_desc-avg_bold.npy") 
    data_full_ses2 = np.load(f"/home/ekenanoglu/DoG/data/sub-{subject}/ses-2/sub-{subject}_ses-2_task-2R_hemi-LR_desc-avg_bold.npy")
    data_full_ses3 = np.load(f"/home/ekenanoglu/DoG/data/sub-{subject}/ses-3/sub-{subject}_ses-3_task-2R_hemi-LR_desc-avg_bold.npy")

    # roi_vertices = np.sort(np.concatenate([V1_vertices, V2_vertices, V3_vertices]))
    # # , FEF_verticies, IA_IPS_verticies,
    # # LO_verticies, P_IPS_verticies, SA_IPS_verticies, V3AB_verticies, VO_CLUSTER_verticies, HMT_verticies, HV4_verticies]))


    # indices_in_roi = np.searchsorted(roi_vertices, np.concatenate([V1_vertices, V2_vertices, V3_vertices]))
    # mask = np.isin(roi_vertices[indices_in_roi], np.concatenate([V1_vertices, V2_vertices, V3_vertices]))
    # roi_vertices = indices_in_roi[mask]

    # data_roi_ses2 = data_full_ses2[:, roi_vertices]
    # data_roi_ses3 = data_full_ses3[:, roi_vertices]

    # valid_mask_2 = mask
    # valid_mask_3 = mask
    # # This whole upcoming chunk is to capture NaNs as they can break the code
    # if len(np.where(np.isnan(data_roi_ses2).all(axis=0))[0]) > 0: #Check if there are fully NaN rows in data_roi_ses2
    #     print("Careful! you have NaNs in your data! I am trying to take care of this now (by deleting the NaN vertices)")
    #     valid_mask_3 = ~np.isnan(data_roi_ses2).all(axis=0)
    #     roi_vertices = np.delete(roi_vertices.reshape(-1,1), np.where(np.isnan(data_roi_ses2).all(axis=0))[0], axis=0).ravel() #This excludes the nans from the roi_vertices
    #     data_roi_ses3 = np.delete(data_roi_ses3, np.where(np.isnan(data_roi_ses2).all(axis=0))[0], axis=1) #This excludes the nans from the data_roi_ses3, aligning it with data_roi_ses2
    #     data_roi_ses2 = np.delete(data_roi_ses2, np.where(np.isnan(data_roi_ses2).all(axis=0))[0], axis=1) #This excludes the nans from the data_roi_ses2
    #     valid_mask_2 = np.ones(data_roi_ses2.shape[1], dtype=bool)
    # session_condition = session_key[subject]["2"]
    # # np.save(f"/home/ekenanoglu/DoG/aggregated_data/ROI_vertices/roi_vertices_sub-{subject}_{session_condition}.npy", roi_vertices)

    # if len(np.where(np.isnan(data_roi_ses3).all(axis=0))[0]) > 0: #Check if there are fully NaN rows in data_roi_ses3
    #     print("Careful! you have NaNs in your data! I am trying to take care of this now (by deleting the NaN vertices)")
    #     valid_mask_2 = ~np.isnan(data_roi_ses3).all(axis=0)
    #     roi_vertices = np.delete(roi_vertices.reshape(-1,1), np.where(np.isnan(data_roi_ses3).all(axis=0))[0], axis=0).ravel() #This excludes the nans from the roi_vertices
    #     data_roi_ses2 = np.delete(data_roi_ses2, np.where(np.isnan(data_roi_ses3).all(axis=0))[0], axis=1) #This excludes the nans from the data_roi_ses2, aligning it with data_roi_ses3
    #     data_roi_ses3 = np.delete(data_roi_ses3, np.where(np.isnan(data_roi_ses3).all(axis=0))[0], axis=1) #This excludes the nans from the data_roi_ses3
    #     valid_mask_3 = np.ones(data_roi_ses3.shape[1], dtype=bool)
    # session_condition = session_key[subject]["3"]
    # np.save(f"/home/ekenanoglu/DoG/aggregated_data/ROI_vertices/roi_vertices_sub-{subject}_{session_condition}.npy", roi_vertices)

    # roi_regions = roi_regions[roi_vertices]



    roi_vertices = np.sort(np.concatenate([V1_vertices, V2_vertices, V3_vertices]))

    # Filter out any vertices that are all‐NaN in either session to maintain perfect alignment
    mask2 = ~np.all(np.isnan(data_full_ses2[:, roi_vertices]), axis=0)
    mask3 = ~np.all(np.isnan(data_full_ses3[:, roi_vertices]), axis=0)

    
    valid_mask = mask2 & mask3 
    roi_vertices = roi_vertices[valid_mask]

    data_roi_ses2 = data_full_ses2[:, roi_vertices]
    data_roi_ses3 = data_full_ses3[:, roi_vertices]
    data_roi_sesavg = data_full_avg[:, roi_vertices]

    #### This loads the fixed fits of the DoG
    dog_fixed_ses2 = np.load(f"/home/ekenanoglu/DoG/output/sub-{subject}/dog_ses2_fixed_final_final.npy")
    dog_fixed_ses3 = np.load(f"/home/ekenanoglu/DoG/output/sub-{subject}/dog_ses3_fixed_final_final.npy")

    dog_fixed_params_ses2 = np.load(f"/home/ekenanoglu/DoG/output/sub-{subject}/fine grid params/dog_sub{subject}_ses2_itrsrchparams_fixed_final_final.npy")[valid_mask, :]
    dog_fixed_params_ses3 = np.load(f"/home/ekenanoglu/DoG/output/sub-{subject}/fine grid params/dog_sub{subject}_ses3_itrsrchparams_fixed_final_final.npy")[valid_mask, :]
    
    ### This loads the free fits of the DoG
    dog_free_ses2 = np.load(f"/home/ekenanoglu/DoG/output/sub-{subject}/dog_ses2_free_final_final.npy")[valid_mask, :]
    dog_free_ses3 = np.load(f"/home/ekenanoglu/DoG/output/sub-{subject}/dog_ses3_free_final_final.npy")[valid_mask, :]

    dog_free_params_ses2 = np.load(f"/home/ekenanoglu/DoG/output/sub-{subject}/fine grid params/dog_sub{subject}_ses2_itrsrchparams_free_final_final.npy")[valid_mask, :]
    dog_free_params_ses3 = np.load(f"/home/ekenanoglu/DoG/output/sub-{subject}/fine grid params/dog_sub{subject}_ses3_itrsrchparams_free_final_final.npy")[valid_mask, :]
    dog_free_params_sesavg = np.load(f"/home/ekenanoglu/DoG/output/sub-{subject}/fine grid params/dog_sub{subject}_sesavg_itrsrchparams_free_final_final.npy")[valid_mask, :]

    #############################
    def make_metadata(subject, n_rows, roi_vertices=roi_vertices):
        return pd.DataFrame({
            'subject': [subject] * n_rows,
            'original_row': list(range(n_rows)),
            'vertex_id': roi_vertices,
            'vertex_region': roi_regions
        })

    # Append each dataset directly
    df_fixed_ses2 = pd.concat([df_fixed_ses2, pd.concat([make_metadata(subject, dog_fixed_ses2.shape[0]), pd.DataFrame(dog_fixed_ses2)], axis=1)], ignore_index=True)
    df_fixed_ses3 = pd.concat([df_fixed_ses3, pd.concat([make_metadata(subject, dog_fixed_ses3.shape[0]), pd.DataFrame(dog_fixed_ses3)], axis=1)], ignore_index=True)


    df_free_ses2 = pd.concat([df_free_ses2, pd.concat([make_metadata(subject, dog_free_ses2.shape[0]), pd.DataFrame(dog_free_ses2)], axis=1)], ignore_index=True)
    df_free_ses3 = pd.concat([df_free_ses3, pd.concat([make_metadata(subject, dog_free_ses3.shape[0]), pd.DataFrame(dog_free_ses3)], axis=1)], ignore_index=True)

    df_fixed_params_ses2 = pd.concat([df_fixed_params_ses2, pd.concat([make_metadata(subject, dog_fixed_params_ses2.shape[0]), pd.DataFrame(dog_fixed_params_ses2)], axis=1)], ignore_index=True)
    df_fixed_params_ses3 = pd.concat([df_fixed_params_ses3, pd.concat([make_metadata(subject, dog_fixed_params_ses3.shape[0]), pd.DataFrame(dog_fixed_params_ses3)], axis=1)], ignore_index=True)
    df_free_params_sesavg = pd.concat([df_free_params_sesavg, pd.concat([make_metadata(subject, dog_free_params_sesavg.shape[0]), pd.DataFrame(dog_free_params_sesavg)], axis=1)], ignore_index=True)

    df_free_params_ses2 = pd.concat([df_free_params_ses2, pd.concat([make_metadata(subject, dn_free_params_ses2.shape[0]), pd.DataFrame(dn_free_params_ses2)], axis=1)], ignore_index=True)
    df_free_params_ses3 = pd.concat([df_free_params_ses3, pd.concat([make_metadata(subject, dn_free_params_ses3.shape[0]), pd.DataFrame(dn_free_params_ses3)], axis=1)], ignore_index=True)
    df_free_params_ses2 = pd.concat([df_free_params_ses2, pd.concat([make_metadata(subject, dog_free_params_ses2.shape[0]), pd.DataFrame(dog_free_params_ses2)], axis=1)], ignore_index=True)
    df_free_params_ses3 = pd.concat([df_free_params_ses3, pd.concat([make_metadata(subject, dog_free_params_ses3.shape[0]), pd.DataFrame(dog_free_params_ses3)], axis=1)], ignore_index=True)

    df_fixed_pred_ses2 = pd.concat([df_fixed_pred_ses2, pd.concat([make_metadata(subject, dog_fixed_ses2.shape[0]), pd.DataFrame(dog_fixed_ses2)], axis=1)], ignore_index=True)
    df_fixed_pred_ses3 = pd.concat([df_fixed_pred_ses3, pd.concat([make_metadata(subject, dog_fixed_ses3.shape[0]), pd.DataFrame(dog_fixed_ses3)], axis=1)], ignore_index=True)
    
    df_free_pred_ses2 = pd.concat([df_free_pred_ses2, pd.concat([make_metadata(subject, dog_free_ses2.shape[0]), pd.DataFrame(dog_free_ses2)], axis=1)], ignore_index=True)
    df_free_pred_ses3 = pd.concat([df_free_pred_ses3, pd.concat([make_metadata(subject, dog_free_ses3.shape[0]), pd.DataFrame(dog_free_ses3)], axis=1)], ignore_index=True)

# Example function to filter by drug type using session_key
def filter_by_drug(data, session_num, drug_name, session_key):
    # Check if data is a NumPy array
    if isinstance(data, np.ndarray):
        subject_ids = [f"{int(subj):03d}" for subj in data[:, 0]]
        mask = [
            session_key.get(subj, {}).get(str(session_num)) == drug_name
            for subj in subject_ids
        ]
        return data[np.array(mask)]
    
    # If it's a DataFrame
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        df['subject_str'] = df.iloc[:, 0].apply(lambda x: f"{int(x):03d}")
        df_filtered = df[df['subject_str'].apply(
            lambda subj: session_key.get(subj, {}).get(str(session_num)) == drug_name
        )]
        return df_filtered.drop(columns=['subject_str'])
    
    else:
        raise TypeError("Input must be a pandas DataFrame or a NumPy array")
    

#### Isolate the aggregated Memantine Data and Placebo Data individually ####

# Filter memantine and placebo rows from both sessions

# Modeled Parameters
df_fixed_memantine_ses2 = filter_by_drug(df_fixed_params_ses2, 2, "memantine", session_key)
df_fixed_placebo_ses2 = filter_by_drug(df_fixed_params_ses2, 2, "placebo", session_key)
df_free_memantine_ses2 = filter_by_drug(df_free_params_ses2, 2, "memantine", session_key)
df_free_placebo_ses2 = filter_by_drug(df_free_params_ses2, 2, "placebo", session_key)

df_fixed_memantine_ses3 = filter_by_drug(df_fixed_params_ses3, 3, "memantine", session_key)
df_fixed_placebo_ses3 = filter_by_drug(df_fixed_params_ses3, 3, "placebo", session_key)
df_free_memantine_ses3 = filter_by_drug(df_free_params_ses3, 3, "memantine", session_key)
df_free_placebo_ses3 = filter_by_drug(df_free_params_ses3, 3, "placebo", session_key)

# Timecourse Predictions
df_fixed_memantine_ses2_preds = filter_by_drug(df_fixed_pred_ses2, 2, "memantine", session_key)
df_fixed_placebo_ses2_preds = filter_by_drug(df_fixed_pred_ses2, 2, "placebo", session_key)
df_free_memantine_ses2_preds = filter_by_drug(df_free_pred_ses2, 2, "memantine", session_key)
df_free_placebo_ses2_preds = filter_by_drug(df_free_pred_ses2, 2, "placebo", session_key)

df_fixed_memantine_ses3_preds = filter_by_drug(df_fixed_pred_ses3, 3, "memantine", session_key)
df_fixed_placebo_ses3_preds = filter_by_drug(df_fixed_pred_ses3, 3, "placebo", session_key)
df_free_memantine_ses3_preds = filter_by_drug(df_free_pred_ses3, 3, "memantine", session_key)
df_free_placebo_ses3_preds = filter_by_drug(df_free_pred_ses3, 3, "placebo", session_key)

# Now combine them in one dataframe
df_fixed_memantine = pd.concat([df_fixed_memantine_ses2, df_fixed_memantine_ses3])
df_fixed_placebo = pd.concat([df_fixed_placebo_ses2, df_fixed_placebo_ses3])
df_free_memantine = pd.concat([df_free_memantine_ses2, df_free_memantine_ses3])
df_free_placebo = pd.concat([df_free_placebo_ses2, df_free_placebo_ses3])

df_fixed_memantine_preds = pd.concat([df_fixed_memantine_ses2_preds, df_fixed_memantine_ses3_preds])
df_fixed_placebo_preds = pd.concat([df_fixed_placebo_ses2_preds, df_fixed_placebo_ses3_preds])
df_free_memantine_preds = pd.concat([df_free_memantine_ses2_preds, df_free_memantine_ses3_preds])
df_free_placebo_preds = pd.concat([df_free_placebo_ses2_preds, df_free_placebo_ses3_preds])

#### Save everything ####
output_dir = "/home/ekenanoglu/DoG/aggregated_data"
os.makedirs(output_dir, exist_ok=True)

# Set column names and save
def save_df(df, filename):
    df.to_parquet(os.path.join(output_dir, filename), index=False)

# Save according to session key
save_df(df_fixed_params_ses2, "aggregate_dog_fixed_params_ses2_final_final.parquet.gzip")
save_df(df_fixed_params_ses3, "aggregate_dog_fixed_params_ses3_final_final.parquet.gzip")
save_df(df_free_params_ses2, "aggregate_dog_free_params_ses2_final.parquet.gzip")
save_df(df_free_params_ses3, "aggregate_dog_free_params_ses3_final.parquet.gzip")

# Save according to condition key
save_df(df_fixed_memantine, "DoG_aggregate_fixed_memantine_iterparams_final_final.parquet.gzip")
save_df(df_fixed_placebo, "DoG_aggregate_fixed_placebo_iterparams_final_final.parquet.gzip")
save_df(df_free_params_sesavg, "DoG_aggregate_free_avg_iterparams_final_final.parquet.gzip")

# Save the time course predictions of the DoG model for both fixed and free data
save_df(df_fixed_memantine_preds, "DoG_aggregate_fixed_memantine_preds_final_final.parquet.gzip")
save_df(df_fixed_placebo_preds, "DoG_aggregate_fixed_placebo_preds_final_final.parquet.gzip")
save_df(df_free_memantine_preds, "DoG_aggregate_free_memantine_preds_final_final.parquet.gzip")
save_df(df_free_placebo_preds, "DoG_aggregate_free_placebo_preds_final_final.parquet.gzip")