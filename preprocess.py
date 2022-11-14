import numpy as np
import h5py
import pickle
import pandas as pd
import copy
from scipy import signal

class ECoGPreprocess:
    def __init__(
        self, session, base_path=None, load_exp_meta_data=True,
        bad_ch_file='bad_channels.pkl', tab_of_experiment='table_of_experiments.csv',
        electrode_pos='electrode_positions.pkl'
    ):
        """
        base_path: str
            base path of data. LFP measurment can be in subdirectories
        """
        
        self.base_path = base_path
        self.session = session
        self.lfp_mtx = None
        self.lfp_mtx_clean = None
        self.time = None
        self.stim1 = None # time of stim1
        self.stim2 = None # time of stim2
        self.t_start = None
        self.t_end = None
        self.fs = None
        self.bad_channels = None
        self.good_channels = None
        self.exp_meta = None
        self.electrode_pos = {}
        self.good_electrode_idx = {} # indices of good electrodes
        self.good_electrode_pos = {} # keys are indices
        self.stim1_idx_good = None
        self.stim2_idx_good = None
        self.test_rec_win = None

        
        if load_exp_meta_data:
            self.load_experiment_metadata(bad_ch_file, tab_of_experiment, electrode_pos)
        
    def set_base_path(self, base_path):
        self.base_path = base_path
        
    def load_experiment_metadata(self, bad_ch_file, tab_of_experiment, electrode_pos):
        try:
            bad_channels = pickle.load(open(self.base_path + '/' + bad_ch_file, "rb"))
            self.bad_channels = [int(ch) for ch in sorted(bad_channels[self.session[:-3]])]
            self.good_channels = [i for i in np.arange(1, 97, 1) if i not in self.bad_channels]
        except:
            print('File does not exist: ' + self.base_path + '/' + bad_ch_file)
        
        try:
            table_of_exp = pd.read_csv(self.base_path + '/' + tab_of_experiment)
            idx = np.where(table_of_exp['File Name'] == self.session + '.zip')[0][0]
            self.exp_meta = table_of_exp.iloc[idx]
        except:
            print('File does not exist: ' + self.base_path + '/' + tab_of_experiment)
        
        try:
            with open(self.base_path + '/' + electrode_pos, 'rb') as infile:
                electrode_pos = pickle.load(infile)
                self.electrode_pos = {}
                for key in electrode_pos:
                    self.electrode_pos[int(key)] = electrode_pos[key]
                
        except:
            print('File does not exist: ' + self.base_path + '/' + electrode_pos)
        
        
    def load_measurement_block(self, block, file_extension='.mat'):
        """
        session: str
            experimental session. E.g. "MonkeyG_20150908_Session2_M1"
        block: str
            measurement block. E.g. "CondBlock1" or "RecBlock1"
        """
        if 'Cond' in block or 'Rec' in block:

            if 'Cond' in block:
                sub_dir = 'ConditioningBlocks'
            elif 'Rec' in block:
                sub_dir = 'RecordingBlocks'

            file = self.base_path + '/' + self.session + '/' + sub_dir + '/' + block + file_extension

            # load LFP data
            data = h5py.File(file, 'r')
            keys = list(data.keys())
            keys.sort()
            i = 0
            make_signals = True
            for key in keys:
                if key.startswith('lfp'):
                    if make_signals:
                        signals = np.zeros((96, data[key].size))
                        make_signals = False
                    signals[i] = data[key][0]
                    i += 1

            self.lfp_mtx = signals

        elif 'Test' in block:
            sub_dir = 'TestingBlocks'
            # lfp_matrix should be dictionary with 2 keys (2 lasers)
            # each entry consitst of tensor with ch x time x trial
            file = self.base_path + '/' + self.session + '/' + sub_dir + '/' + block + file_extension

            # load LFP data
            data = h5py.File(file, 'r')
            keys = list(data.keys())
            keys.sort()
            make_signals1 = True
            make_signals2 = True
            i1 = 0
            i2 = 0
            for i, key in enumerate(keys):
                if key.startswith('lfp'):
                    if 'traces1' in key:
                        if make_signals1:
                            signals1 = np.zeros((96, data[key].shape[0], data[key].shape[1]))
                            make_signals1 = False
                        signals1[i1] = data[key][:]
                        i1 += 1
                    elif 'traces2' in key:
                        if make_signals2:
                            signals2 = np.zeros((96, data[key].shape[0], data[key].shape[1]))
                            make_signals2 = False
                        signals2[i2] = data[key][:]
                        i2 += 1

            self.lfp_mtx = {
                1: signals1,
                2: signals2
            }

        # load metadata
        self.fs = data['samp_freq'][0][0]

        if 'time' in keys:
            self.time = data['time'][0]
        if 'stim1' in keys:
            self.stim1 = data['stim1'][0]
        if 'stim2' in keys:
            self.stim2 = data['stim2'][0]
        if 'tstart' in keys:
            self.t_start = data['tstart'][0][0]
        if 'tend' in keys:
            self.t_end = data['tend'][0][0]
        if 'win' in keys:
            self.test_rec_win = data['win'][:]

        return self.lfp_mtx
    
    def remove_bad_channels(self):
        if self.lfp_mtx is None:
            print('Error: load LFP data first')
            return None
        else:
            data = copy.deepcopy(self.lfp_mtx)
            for bad_ch in self.bad_channels[::-1]:
                if type(data) is dict:
                    data[1] = np.delete(data[1], int(bad_ch - 1), 0)
                    data[2] = np.delete(data[2], int(bad_ch - 1), 0)
                else:
                    data = np.delete(data, int(bad_ch - 1), 0)
            self.lfp_mtx_clean = data
            
            # select good electrode positions
            for i, ch in enumerate(self.good_channels):
                self.good_electrode_idx[i] = ch
                self.good_electrode_pos[i] = self.electrode_pos[ch]

            # index of stim electrode
            stim1 = self.exp_meta.stim_Coh_from
            if stim1 != 0:
                self.stim1_idx_good = list(self.good_electrode_idx.values()).index(stim1)
            stim2 = self.exp_meta.stim_Coh_to
            if stim2 != 0:
                self.stim2_idx_good = list(self.good_electrode_idx.values()).index(stim2)
            
            return data

    def bp_filter(self, fmin, fmax, order=3, data_type='clean'):
        if data_type == 'clean':
            data = self.lfp_mtx_clean
        elif data_type == 'all':
            data = self.lfp_mtx
        else:
            raise ValueError('Invalid data_type. Use either "clean" or "all".')

        sos = signal.butter(
            N=order, Wn=[fmin, fmax], btype='bandpass', fs=self.fs, output='sos'
        )
        filtered_data = signal.sosfilt(sos, data)

        if data_type == 'clean':
            self.lfp_mtx_clean = filtered_data
        elif data_type == 'all':
            self.lfp_mtx = filtered_data

        return filtered_data
        
        
    def select_data_around_stim(self, stim_idx, L, which_stim=1, offset=0, data_type='clean', block_type='cond'):
        if block_type == 'cond':
            if which_stim == 1:
                if not isinstance(self.stim1, np.ndarray):
                    raise IndexError('No stimulation by laser 1')
                else:
                    idx = np.where(self.time - self.stim1[stim_idx] >= 0)[0][0]
                    stim_time = self.stim1[stim_idx]
            elif which_stim == 2:
                if not isinstance(self.stim2, np.ndarray):
                    raise IndexError('No stimulation by laser 2')
                else:
                    idx = np.where(self.time - self.stim2[stim_idx] >= 0)[0][0]
                    stim_time = self.stim2[stim_idx]
            else:
                raise ValueError('which_stim has to be either 1 or 2')

            if data_type == 'clean':
                data_short = self.lfp_mtx_clean[:,idx + offset:idx + offset + L]
            elif data_type == 'all':
                data_short = self.lfp_mtx[:,idx + offset:idx + offset + L]
            else:
                raise ValueError('Invalid data_type. Use either "clean" or "all".')

            t = self.time[idx + offset:idx + offset + L]
            return data_short, t, stim_time, idx

        elif block_type == 'test':
            samp_prior_stim = int(self.test_rec_win[0,0] * self.fs)

            # if less samples than requested are available adjust L
            if np.shape(self.lfp_mtx_clean[1])[1] < samp_prior_stim + offset + L:
                L -= (samp_prior_stim + offset + L) - np.shape(self.lfp_mtx_clean[1])[1]

            if which_stim == 1:
                if not isinstance(self.stim1, np.ndarray):
                    raise IndexError('No stimulation by laser 1')
            elif which_stim == 2:
                if not isinstance(self.stim2, np.ndarray):
                    raise IndexError('No stimulation by laser 2')
            else:
                raise ValueError('which_stim has to be either 1 or 2')

            if data_type == 'clean':
                if which_stim == 1:
                    data_short = self.lfp_mtx_clean[1][:,samp_prior_stim + offset:samp_prior_stim + offset + L,stim_idx]
                elif which_stim == 2:
                    data_short = self.lfp_mtx_clean[2][:,samp_prior_stim + offset:samp_prior_stim + offset + L,stim_idx]
            elif data_type == 'all':
                if which_stim == 1:
                    data_short = self.lfp_mtx[1][:,samp_prior_stim + offset:samp_prior_stim + offset + L,stim_idx]
                elif which_stim == 2:
                    data_short = self.lfp_mtx[2][:,samp_prior_stim + offset:samp_prior_stim + offset + L,stim_idx]
            else:
                raise ValueError('Invalid data_type. Use either "clean" or "all".')

            return data_short, samp_prior_stim, L

        else:
            raise ValueError('Invalid block_type. Use either "cond" or "test".')