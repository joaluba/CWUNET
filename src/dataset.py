import torch
import numpy as np
from torch.utils.data import Dataset
import helpers as hlp
import pandas as pd
import scipy.signal as signal
from os.path import dirname, basename, join


class DatasetReverbTransfer(Dataset):

    def __init__(self,config):
        # Load metadata from csv file
        df_ds = pd.read_csv(config["df_metadata"], index_col=None)
        # Choose only the split you want 
        df_ds = df_ds[df_ds["split"] == config["split"]].reset_index(drop=True)
        # Create a custom index with repeated values (0,0,1,1,2,2,...)
        # (1 datapoint is a pair of content and style audios)
        custom_index = np.repeat(np.arange(len(df_ds) // 2), 2)
        df_ds["pair_id"] = custom_index

        self.df_ds = df_ds
        self.sig_len=config["sig_len"] 
        self.fs=config["fs"] 
        self.split = config["split"] 
        self.device=config["device"]
        self.content_ir=config["content_rir"] 
        self.style_ir=config["style_rir"]
        self.p_noise=config["p_noise"] if config["split"]=="train" else 0
        self.has_clones=config["has_clones"] if config["split"]=="train" else False

        # --> see config/basic.yaml for more details about the parameters

    def __len__(self):
        return int(len(self.df_ds)/2)

    def __getitem__(self,index):
        # Pick pair of signals from metadata:
        df_pair=self.df_ds[self.df_ds["pair_id"]==index]
        df_pair=df_pair.reset_index()

        # Load signals (and resample if needed)
        s1 = hlp.torch_load_mono(df_pair["speech_file_path"][0],self.fs)
        s2 = hlp.torch_load_mono(df_pair["speech_file_path"][1],self.fs)

        # Crop signals to a desired length
        s1=hlp.get_nonsilent_frame(s1,self.sig_len)
        s2=hlp.get_nonsilent_frame(s2,self.sig_len)

        # Apply polarity or none
        s1*=np.random.choice([-1, 1])
        s2*=np.random.choice([-1, 1])

        # Load impulse responses
        # Note: If self.content_ir is not empty, it means that we want all content audios to have the same target ir,
        # and analogically for self.style_ir - we want only one target ir. Otherwise each style and each content audio
        # can have a different ir. This reflects if we want to learn one-to-one, many-to-one, one-to-many, or many-to-many. 

        if self.content_ir is None:
            # load either rir or its clone (same room, different position)
            load_clone = np.random.choice([True, False]) if self.has_clones else False

            if load_clone:
                r1 = hlp.torch_load_mono(df_pair["ir_clone_file_path"][0],self.fs)
            else:
                r1 = hlp.torch_load_mono(df_pair["ir_file_path"][0],self.fs)
            
        elif self.content_ir=="anechoic":
            r1 = torch.cat((torch.tensor([[1.0]]), torch.zeros((1,self.fs-1))),1)
        else: 
            r1 = hlp.torch_load_mono(self.content_ir,self.fs)
            
        if self.style_ir is None:
            # load either rir or its clone (same room, different position)
            load_clone = np.random.choice([True, False]) if self.has_clones else False
                
            if load_clone:
                r2 = hlp.torch_load_mono(df_pair["ir_clone_file_path"][1],self.fs)
            else:
                r2 = hlp.torch_load_mono(df_pair["ir_file_path"][1],self.fs)

        elif self.style_ir=="anechoic":
            r2 = torch.cat((torch.tensor([[1.0]]), torch.zeros((1,self.fs-1))),1)   
        else: 
            r2 = hlp.torch_load_mono(self.style_ir,self.fs)

        # truncate silence in all rirs:
        r1=hlp.truncate_ir_silence(r1, self.fs, threshold_db=20)
        r2=hlp.truncate_ir_silence(r2, self.fs, threshold_db=20)

        # Scale rirs so that the peak is at 1
        r1=hlp.torch_normalize_max_abs(r1) 
        r2=hlp.torch_normalize_max_abs(r2) 

        # Convolve signals with impulse responses
        s1r1 = torch.from_numpy(signal.fftconvolve(s1, r1,mode="full"))[:,:self.sig_len]
        s2r2 = torch.from_numpy(signal.fftconvolve(s2, r2,mode="full"))[:,:self.sig_len]
        s1r2 = torch.from_numpy(signal.fftconvolve(s1, r2,mode="full"))[:,:self.sig_len]

        # generate background noise samples
        n1=hlp.gen_rand_colored_noise(self.p_noise,self.sig_len)
        n2=hlp.gen_rand_colored_noise(self.p_noise,self.sig_len)

        # Add noise to content and style signal
        snr1=15 + (40 - 15) * torch.rand(1)
        snr2=15 + (40 - 15) * torch.rand(1)
        s1r1n1=hlp.torch_mix_and_set_snr(s1r1,n1,snr1)
        s2r2n2=hlp.torch_mix_and_set_snr(s2r2,n2,snr2)

        # normalize inputs
        s1r1n1=hlp.torch_normalize_max_abs(s1r1n1) # Reverberant content sound
        s2r2n2=hlp.torch_normalize_max_abs(s2r2n2) # Style sound
        s1r2=hlp.torch_normalize_max_abs(s1r2) # Target

        # s2r1=hlp.torch_standardize_max_abs(s2r1) # "Flipped" target
        s1=hlp.torch_normalize_max_abs(s1) # Anechoic content sound

        return s1r1n1, s2r2n2, s1r2, s1, s2
    
    def get_idx_with_rt60diff(self,diff_rt60_min,diff_rt60_max):
        # create column diff_rt60 to compute difference in rt60 between content and style audio
        self.df_ds["diff_rt60"] = self.df_ds["rt60_true"].diff()
        self.df_ds.loc[0::2, 'diff_rt60'] = self.df_ds['diff_rt60'].shift(periods=-1)
        # # check indices of datapoint where the rt60 for content is lower than rt60 for style
        selected=self.df_ds[(self.df_ds["diff_rt60"]>diff_rt60_min) & (self.df_ds["diff_rt60"]<diff_rt60_max)]
        selected=selected.iloc[::2]
        selected=selected["pair_id"].tolist()
        return selected

    def get_info(self,index,id="style"):
        df_pair=self.df_ds[self.df_ds["pair_id"]==index]
        if id=="style":
            styleorcontent_idx=1
        elif id=="content":
            styleorcontent_idx=0
        df=df_pair.iloc[styleorcontent_idx]
        return df