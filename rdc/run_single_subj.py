import pandas as pd
import numpy as np
import glob as glob
import matplotlib.pyplot as plt
import seaborn as sns

import re
import os

from nltools.file_reader import onsets_to_dm
from nltools.data import Design_Matrix, Brain_Data

import nibabel as nib



TR = 1.25
UPSAMPLE_FROM_SECS = 1000


subj_dirs = glob.glob('/share/foxlab-backedup/RDC/RDC*')

for subj_dir in subj_dirs:
    print(f'DIR: {subj_dir}')

    fmri_files = glob.glob(subj_dir+'functional/*/*.nii.gz')
    fmri_files

    fmri_files_dict = {int( re.findall('MDARTTask(\d+)\/', f)[0]):f for f in fmri_files} 
    # print( fmri_files[0] )
    # re.findall('MDARTTask(\d+)\/', fmri_files[0])[0]
    fmri_files_dict
        
    print(fmri_files_dict)
    fmri_data = nib.load(fmri_files_dict[0])
    scan_length_secs = fmri_data.shape[3]*TR
    print(f'Total scan length: {scan_length_secs} seconds')


    motion_files = glob.glob(subj_dir+'functional/*/*BOLD_N468MOCOparams.csv')
    # !ls ../../data/processed/RDC102/functional/MDARTTask03/
    motion_files

    motion_files_dict = {int( re.findall('MDARTTask(\d+)\/', f)[0]):f for f in motion_files} 
    # print( fmri_files[0] )
    # re.findall('MDARTTask(\d+)\/', fmri_files[0])[0]
    motion_files_dict
    behavioral_files = glob.glob(subj_dir+'behavioral/*.txt')
    behavioral_files

    behavioral_files_dict = {int( re.findall('Run(\d+)\.', f)[0]):f for f in behavioral_files} 
    behavioral_files_dict
    # # check file keys... 
    # df = pd.read_csv(behavioral_files[0], sep='\t', comment='#')
    # df.head()



    dm_columns_to_model = ['cue_const_1s', 'cue_const', 'cue_p_risk', 'cue_r_risk', 'pe_risk','outcome', #'cue_const_amb', 
                    'cue_p_amb', 'cue_a_amb', 'pe_amb', 'rating', 'trs_to_drop']#, 'pe_rating']
    dm_columns_dict = {col:i for i, col in enumerate(dm_columns_to_model)}
        
    def create_design_matrix(df, dm_columns_dict, scan_length_secs, n_runs, UPSAMPLE_FROM_SECS ):    # create design matrix
        num_regressors = len(dm_columns_dict.keys())
        dm_array = np.zeros((num_regressors, int(scan_length_secs*n_runs*UPSAMPLE_FROM_SECS)))
    #     print(df)
        for i, row in df.iterrows():
            # print(row)

            # model cues
            dm_array[ dm_columns_dict['cue_const_1s'], row['TrialStartTime']:row['TrialStartTime']+(1*UPSAMPLE_FROM_SECS)] = 1
            dm_array[ dm_columns_dict['cue_const'], row['TrialStartTime']+(1*UPSAMPLE_FROM_SECS):row['ReinforcerInTime'] ] = 1
            # model risk cues (p/r)
            if row['TrialTypeCode'] != 3:
                dm_array[ dm_columns_dict['cue_p_risk'], row['TrialStartTime']:row['ReinforcerInTime']] = (row['ThreatPct']/100)-.5
                dm_array[ dm_columns_dict['cue_r_risk'], row['TrialStartTime']:row['ReinforcerInTime']] = np.power((row['ThreatPct']/100)-.5,2)
                
                dm_array[ dm_columns_dict['pe_risk'], row['TrialStartTime']:row['TrialStartTime']+(1*UPSAMPLE_FROM_SECS)] = (row['ThreatPct']/100) - .5
                dm_array[ dm_columns_dict['pe_risk'], row['ReinforcerInTime']:row['ReinforcerOutTime']] = row['IsaShockOut'] - row['ThreatPct']/100 - .5

            else:
                dm_array[ dm_columns_dict['cue_p_amb'], row['TrialStartTime']:row['ReinforcerInTime']] = (row['ThreatPct'] + row['AmbiguousPct']/2)/100 - .5
                dm_array[ dm_columns_dict['cue_a_amb'], row['TrialStartTime']:row['ReinforcerInTime']] = row['AmbiguousPct']/100 - .5

                dm_array[ dm_columns_dict['pe_amb'], row['TrialStartTime']:row['TrialStartTime']+(1*UPSAMPLE_FROM_SECS)] = (row['ThreatPct'] + row['AmbiguousPct']/2)/100 - .5
                dm_array[ dm_columns_dict['pe_amb'], row['ReinforcerInTime']:row['RatingStartTime']] = row['IsaShockOut'] - (row['ThreatPct'] + row['AmbiguousPct']/2)/100 - .5
            
            dm_array[ dm_columns_dict['outcome'], row['ReinforcerInTime']:row['RatingStartTime']] = row['IsaShockOut']
        
            dm_array[ dm_columns_dict['rating'], row['RatingStartTime']:row['RatingStartTime']+row['RatingRT'] ] = 1 # row['RatingValue']
            
                    
            # prep times to drop... 
            dm_array[ dm_columns_dict['trs_to_drop'], row['ReinforcerInTime']:row['ReinforcerInTime']+int(TR*2*UPSAMPLE_FROM_SECS) ] = 1
            
        return dm_array


    ### check this and delete this line of code!
    # motion = pd.read_csv(motion_files_dict[1])
    # motion_cov = m.filter(regex='MOCOparam')
    # dm.shape
    dm_list = []
    for run in np.arange(1,5): # keys were not in order... 
        df = pd.read_csv(behavioral_files_dict[run], sep='\t', comment='#')
        # df.head()

        dm_array = create_design_matrix(df, dm_columns_dict, scan_length_secs, 1, UPSAMPLE_FROM_SECS )
        dm = Design_Matrix(dm_array.T, columns=dm_columns_dict, sampling_freq=1*UPSAMPLE_FROM_SECS)

        dm_columns_to_conv = ['cue_const_1s', 'cue_const', 'cue_p_risk', 'cue_r_risk', 'pe_risk', 'outcome','cue_p_amb', 'cue_a_amb', 'pe_amb', 'rating']
        dm_conv = dm.convolve(columns=dm_columns_to_conv)

        dm_conv = dm_conv.downsample(1/TR)
        dm_conv_filt = dm_conv.add_poly(order=3, include_lower=True)
        dm_conv_filt = dm_conv_filt.add_dct_basis(duration=128)
        

        ### doesn't work... 
        
        # motion = pd.read_csv(motion_files_dict[1])
        # motion_cov = m.filter(regex='MOCOparam')
        # dm_motion = Design_Matrix(motion_cov, sampling_freq=1/TR)

        # dm_conv_filt.append(dm_motion, axis=1)
        
        dm_list.append(dm_conv_filt)

    for i, dm in enumerate(dm_list):
        if i == 0:
            dm_conv_filt = dm
        else:
            dm_conv_filt = dm_conv_filt.append( dm, axis=0) #, unique_cols=cov_cols )
    dm_motion_list = []
    for run in np.arange(1,5): # keys were not in order... 
        motion = pd.read_csv(motion_files_dict[run])
        motion_cov = m.filter(regex='MOCOparam')
        dm_motion = Design_Matrix(motion_cov, sampling_freq=1/TR)

        dm_motion_list.append(dm_motion)
        
    for i, dm in enumerate(dm_motion_list):
        if i == 0:
            dm_motion = dm
        else:
            dm_motion = dm_motion.append( dm, axis=0) #, unique_cols=cov_cols )

    dm_conv_filt_motion = pd.concat([dm_conv_filt, dm_motion], axis=1)
    dm_conv_filt_motion.shape


    # dm_conv_filt_motion.heatmap()
    
    # dm_conv_down = dm_conv.downsample(1/TR)
    # # plt.plot( dm_conv_down['cue_const_1s_c0'] )
    # # plt.plot( dm_conv_down['cue_r_risk_c0'] )
    # dm_conv_down.heatmap()
    def plot_vif( vif, this_title ):
        def add_value_label(x_list,y_list, ax):
            my_text = np.round(y_list,2)
            for i in range(len(x_list)):
                ax.text(i,y_list[i]-1,my_text[i], ha="center", color='white')

        f,a = plt.subplots(figsize=(6,3))

        a.bar(x=vif.index, height=vif.values )
        # a.plot(dm_conv_filt.vif(exclude_polys=False), linewidth=0, marker='o')
        # a.set_xticklabels(rotation=0)
        a.set_xticks(np.arange(len(vif.index)))
        a.set_xticklabels(vif.index, rotation = 90)
        a.set_ylabel('Variance Inflation Factor')

        # a.set_xlim( -.5, len( this_cols )-.5 )
        # a.set_ylim( 0,9 )
        # print(int(np.round(poly_vif)))

        add_value_label(vif.index,vif, a)

        plt.title('VIF while modeling poly/cosine '+this_title)

        # print(durations)
        
    # this_cols = dm_conv_filt.columns #dm_conv_filt.filter(regex='.*_c0$').columns
    # tmp = dm_conv_filt[this_cols].corr()
    # this_vif = np.diag(np.linalg.inv(tmp), 0)
    # # this_vif = poly_vif[:len(this_cols)]
    # this_vif = pd.Series(this_vif, index=this_cols)

    this_cols = dm_conv_filt.filter(regex='.*_c0$').columns
    tmp = dm_conv_filt.corr()
    this_vif = np.diag(np.linalg.inv(tmp), 0)
    this_vif = this_vif[:len(this_cols)]
    this_vif = pd.Series(this_vif, index=this_cols)

    # print( this_vif)


    # plot_vif( this_vif, 'test' )
    # plt.ylim((0,5))


    fmri_list = []
    print(fmri_files_dict)
    for run in np.arange(1,5): # keys were not in order... 
        print(run)
        fmri_list.append(fmri_files_dict[run])
    full_fmri = nib.funcs.concat_images(fmri_list, check_affines=True, axis=3)

        # fmri_data = nib.load(fmri_files_dict[run])
        # if run == 1:
        #     full_fmri = fmri_data.get_fdata()
        #     print(full_fmri.shape)
        # else:
        #     full_fmri = np.concatenate((full_fmri, fmri_data.get_fdata()), axis=3)    
    print( full_fmri.shape )


    fmri = Brain_Data(full_fmri)

        
    dm_conv_filt.shape
    fmri.X = dm_conv_filt_motion
    stats = fmri.regress(); # ; is important! otherwise it will recursion error... 

    newpath = subj_dir + 'beta_maps/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    for i, col in enumerate( dm_conv_filt_motion.filter(regex='_c0') ):
        print(col, dm_conv_filt_motion.columns[i])
        outfile = subj_dir + 'beta_maps/' + col + '.nii.gz'
        print(outfile)
        stats['beta'][i].write(outfile)
