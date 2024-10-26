#establishing environment
import numpy as np
import math
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def unnesting(df, explode, axis):
    '''
    code that expands lists in a column in a dataframe.
    '''
    if axis==1:
        idx = df.index.repeat(df[explode[0]].str.len())
        df1 = pd.concat([
            pd.DataFrame({x: np.concatenate(df[x].values)}) for x in explode], axis=1)
        df1.index = idx

        return df1.join(df.drop(explode, 1), how='right')
    else :
        df1 = pd.concat([
                         pd.DataFrame(df[x].tolist(), index=df.index).add_prefix(x) for x in explode], axis=1)
        return df1.join(df.drop(explode, 1), how='right')

def load_rid_forjson(ptname, data_directory):
    ptids = pd.read_csv(data_directory + '/pt_data/all_ptids.csv')
    rid = ptids['r_id'].loc[ptids['hup_id'] == ptname].astype('string')
    rid = np.array(rid)
    dkt_directory = data_directory + 'CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-research3T_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.csv'.format(rid[0],rid[0])
    brain_df = pd.read_csv(dkt_directory)
    brain_df['name'] = brain_df['name'].astype(str) + '-CAR'
    return rid[0], brain_df
        
def label_fix(pt, data_directory, threshold = 0.25):
    '''
    label_fix reassigns labels overlapping brain regions to "empty labels" in our DKTantspynet output from IEEG_recon
    input:  pt - name of patient. example: 'HUP100' 
            data_directory - directory containing CNT_iEEG_BIGS folder. (must end in '/')
            threshold - arbitrary threshold that r=2mm surround of electrode must overlap with a brain region. default: threshold = 25%, Brain region has a 25% or more overlap.
    output: relabeled_df - a dataframe that contains 2 extra columns showing the second most overlapping brain region and the percent of overlap. 
    '''

    rid, brain_df = load_rid_forjson(pt, data_directory)
    json_labels = data_directory + 'CNT_iEEG_BIDS/{}/derivatives/ieeg_recon/module3/{}_ses-research3T_space-T00mri_atlas-DKTantspynet_radius-2_desc-vox_coordinates.json'.format(rid,rid)
    workinglabels = pd.read_json(json_labels, lines=True)
    empty = (workinglabels[workinglabels['label'] == 'EmptyLabel'])
    empty = unnesting(empty, ['labels_sorted', 'percent_assigned'], axis=0)
    empty = empty[np.isnan(empty['percent_assigned1']) == False]
    changed = empty[empty['percent_assigned1'] >= threshold]
    
    brain_df['name'] = brain_df['name'].str.replace('-CAR','')
    relabeled_df = brain_df.merge(changed[['labels_sorted1', 'percent_assigned1']], left_on=brain_df['name'], right_on=changed['name'], how = 'left', indicator=True)

    return relabeled_df