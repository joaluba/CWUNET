import torch
import numpy as np
import pandas as pd
import random
import os
from tqdm import tqdm
from os.path import join as pjoin
from torch.utils.data import Subset
import argparse
# modules with audio metrics
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio, SignalDistortionRatio, SpeechReverberationModulationEnergyRatio, PerceptualEvaluationSpeechQuality, ShortTimeObjectiveIntelligibility
from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
from pysepm import fwSNRseg
# my modules
import dataset
import loss_mel, loss_stft, loss_waveform, loss_embedd
import trainer
import helpers as hlp


class Evaluator(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.load_metrics()
        self.load_eval_dataset()
        self.failcount=0

    def load_metrics(self):
        device=self.config["device"]

        my_best_checkpoint_path="/home/ubuntu/guestxr2/home/ubuntu/joanna/CWUNET/results/big_1/checkpointbest.pt"

        self.measures= {
            "multi-stft" : loss_stft.MultiResolutionSTFTLoss().to(device), 
            "multi-mel"  : loss_mel.MultiMelSpectrogramLoss().to(device),
            "multi-wave" : loss_waveform.MultiWindowShapeLoss().to(device),
            "stft" : loss_stft.STFTLoss().to(device), 
            "logmel"  : loss_mel.LogMelSpectrogramLoss().to(device),
            "wave" : loss_waveform.WaveformShapeLoss(winlen=960).to(device),
            "mse" : torch.nn.MSELoss().to(device),
            "sisdr" : ScaleInvariantSignalDistortionRatio().to(device),
            "sdr" : SignalDistortionRatio().to(device),
            "srmr" : SpeechReverberationModulationEnergyRatio(16000).to(device),
            "pesq" : PerceptualEvaluationSpeechQuality(16000, 'wb').to(device),
            "stoi" : ShortTimeObjectiveIntelligibility(16000).to(device),
            "emb_cos": loss_embedd.EmbeddingLossCosine(my_best_checkpoint_path,device=device),
            "emb_euc": loss_embedd.EmbeddingLossEuclidean(my_best_checkpoint_path,device=device),
            "squim_obj" : SQUIM_OBJECTIVE.get_model().to(device),
            "squim_subj" : SQUIM_SUBJECTIVE.get_model().to(device)
        }

        # get speech reference for non-intrusive metrics:
        self.speechref = hlp.torch_load_mono("/home/ubuntu/guestxr2/home/ubuntu/joanna/CWUNET/audios/speech_VCTK_4_sentences.wav",48000)[:,:4*48000].unsqueeze(1)


    def load_eval_dataset(self):

        rt60diffmin = self.config["rt60diffmin"]
        rt60diffmax = self.config["rt60diffmax"]
        N_datapoints = self.config["N_datapoints"]
        batch_size_eval = self.config["batch_size_eval"]

        # load a test split from the dataset used for training
        self.config["split"] = self.config["eval_split"]
        self.config["p_noise"] = 0 # for evaluation, we do not want noise 

        # define testset to choose samples from in evaluation 
        self.testset_orig = dataset.DatasetReverbTransfer(self.config)

        # choose a subset from that testset based on RT60 difference
        indices_chosen = self.testset_orig.get_idx_with_rt60diff(rt60diffmin,rt60diffmax)

        # choose a specific number od d.p. (if N_datapoints is set, 0 means its not set)
        if N_datapoints>0:
            indices_chosen = range(0,N_datapoints)
        self.testset = Subset(self.testset_orig,indices_chosen)
        print(f"Preparing to evaluate {len(self.testset)} test datapoints")
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size_eval, shuffle=False, num_workers=6,pin_memory=True)


    def compute_metrics_oracle(self):
                
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)
        
        eval_dict_list=[]
        for j, data in tqdm(enumerate(self.testloader),total = len(self.testloader)):

            # get simple datapoint (can be used in batches)
            sContent, sStyle, sTarget, sAnecho, _ = data

            # get metrics
            # -> content : target 
            eval_dict_list.append(self.metrics4batch(j,"oracle","target:content",sTarget,sContent,sAnecho,nmref=self.speechref))
            # -> content : anechoic
            eval_dict_list.append(self.metrics4batch(j,"oracle","target:anecho",sTarget,sAnecho,sAnecho, nmref=self.speechref))
            # -> target : style
            eval_dict_list.append(self.metrics4batch(j,"oracle","target:style",sTarget,sStyle,sAnecho, nmref=self.speechref))
            # -> content : style
            eval_dict_list.append(self.metrics4batch(j,"oracle","content:style",sContent,sStyle,sAnecho, nmref=self.speechref))
            # -> targetclone : target (2 RIRs from the same room, different position)
            if self.config["has_clones"]:
                sTargetOrig, sTargetClone=self.testset_orig.get_target_clone(j,sAnecho) 
                eval_dict_list.append(self.metrics4batch(j,"oracle","target:targetclone",sTargetOrig,sTargetClone,sAnecho, nmref=self.speechref))
            

        return eval_dict_list
    
    def compute_metrics_checkpoint(self,checkpointpath,eval_tag):
                
        np.random.seed(0)
        random.seed(0)
        torch.manual_seed(0)

        # load training configuration
        device=self.config["device"]
        config_train=hlp.load_config(pjoin(os.path.dirname(checkpointpath),"train_config.yaml"))
        # load model architecture
        model=trainer.load_chosen_model(config_train,config_train["modeltype"]).to(device)
        # load weights from checkpoint
        train_results=torch.load(os.path.join(checkpointpath),map_location=device, weights_only=True)
        model.load_state_dict(train_results["model_state_dict"])

        eval_dict_list=[]
        for j, data in tqdm(enumerate(self.testloader),total = len(self.testloader)):

            # get datapoint 
            sContent, sStyle, sTarget, sAnecho, _ = data
            # get prediction
            _, _, _, sPrediction=trainer.infer(model, data, device)
            # get metrics
            # -> predicion : target
            eval_dict_list.append(self.metrics4batch(j,eval_tag,"prediction:target", sTarget, sPrediction, sAnecho, nmref=self.speechref))
            # -> predicion : content
            eval_dict_list.append(self.metrics4batch(j,eval_tag,"prediction:content",sContent, sPrediction, sAnecho,nmref=self.speechref))

        return eval_dict_list
    

    def metrics4batch(self, idx, label, comp_name, x1, x2, x_anecho, nmref,scale_in=True):

        device= self.config["device"]
        # prepare dimensions (B,C,N) -> (B,N)
        x1=x1.squeeze(1) 
        x2=x2.squeeze(1)
        x_anecho=x_anecho.squeeze(1)
        nmref=nmref.squeeze(1)
        # Resample - losses operate on 16kHz sampling rate
        x1=hlp.torch_resample_if_needed(x1,48000,16000).to(device)
        x2=hlp.torch_resample_if_needed(x2,48000,16000).to(device)
        x_anecho=hlp.torch_resample_if_needed(x_anecho,48000,16000).to(device)
        nmref=hlp.torch_resample_if_needed(nmref,48000,16000).to(device)

        if scale_in:
            x1=hlp.torch_normalize_max_abs(x1)
            x2=hlp.torch_normalize_max_abs(x2)
            x_anecho=hlp.torch_normalize_max_abs(x_anecho)

        # set all to the same energy
        rms_x1=torch.sqrt(torch.mean(x1**2)) # x1 is typically the target
        x2=set_rms(x2,rms_x1)
        x_anecho=set_rms(x_anecho,rms_x1)

        # ----- Compute perceptual losses -----
        L_pesq_x1=torch.tensor([float('nan')]).to(device)
        L_pesq_x2=torch.tensor([float('nan')]).to(device)
        L_pesq_ab=torch.tensor([float('nan')]).to(device)
        L_pesq_ba=torch.tensor([float('nan')]).to(device)
        try:
            L_pesq_x1=self.measures["pesq"](x1,x_anecho)
            L_pesq_x2=self.measures["pesq"](x2,x_anecho)
            L_pesq_ab=self.measures["pesq"](x1,x2)
            L_pesq_ba=self.measures["pesq"](x2,x1)
        except: 
            self.failcount+=1
            print("Could not compute pesq for this signal for " + str(self.failcount) + "times")

        L_stoi_x1=torch.tensor([float('nan')]).to(device)
        L_stoi_x2=torch.tensor([float('nan')]).to(device)
        L_stoi_ab=torch.tensor([float('nan')]).to(device)
        L_stoi_ba=torch.tensor([float('nan')]).to(device)
        try:
            L_stoi_x1=self.measures["stoi"](x1,x_anecho)
            L_stoi_x2=self.measures["stoi"](x2,x_anecho)
            L_stoi_ab=self.measures["stoi"](x1,x2)
            L_stoi_ba=self.measures["stoi"](x2,x1)
        except: 
            self.failcount+=1
            print("Could not compute stoi for this signal for " + str(self.failcount) + "times")

        L_sisdr_x1=torch.tensor([float('nan')]).to(device)
        L_sisdr_x2=torch.tensor([float('nan')]).to(device)
        L_sisdr_x1=self.measures["sisdr"](x1,x_anecho)
        L_sisdr_x2=self.measures["sisdr"](x2,x_anecho)

        # ----- Compute non-intrusive metrics -----
        try:
            L_pesqni_x1, L_stoini_x1, L_sisdrni_x1=self.measures["squim_obj"](x1)
            L_pesqni_x2, L_stoini_x2, L_sisdrni_x2=self.measures["squim_obj"](x2)
        except:
            print("Could not compute squim objective predictions")

        L_mos_x1=torch.tensor([float('nan')]).to(device)
        L_mos_x2=torch.tensor([float('nan')]).to(device)
        try:
            L_mos_x1=self.measures["squim_subj"](x1,nmref) 
            L_mos_x2=self.measures["squim_subj"](x2,nmref) 
        except:
            print("Could not compute squim subjective predictions")

        L_srmr_x1=self.measures["srmr"](x1)
        L_srmr_x2=self.measures["srmr"](x2)

        # ----- Compute metrics from spear -----
        L_fwsnr_ab=fwSNRseg(x1.squeeze(0).cpu().numpy(),x2.squeeze(0).cpu().numpy(),16000)
        L_fwsnr_ba=fwSNRseg(x2.squeeze(0).cpu().numpy(),x1.squeeze(0).cpu().numpy(),16000)

        multi_mag_ab, multi_coh_ab= self.measures["multi-stft"](x1,x2)
        multi_mag_ba, multi_coh_ba= self.measures["multi-stft"](x2,x1)
        L_multi_stft_ab=multi_mag_ab.item() + multi_coh_ab.item()
        L_multi_stft_ba=multi_mag_ba.item() + multi_coh_ba.item()

        mag_ab, coh_ab= self.measures["stft"](x1,x2)
        mag_ba, coh_ba= self.measures["stft"](x2,x1)
        L_stft_ab=mag_ab.item() + coh_ab.item()
        L_stft_ba=mag_ba.item() + coh_ba.item()

        # ----- Create a dictionary with loss values -----
        dict_row={ 
        'label': label,
        'idx':idx, 
        'compared': comp_name,
        # Type 1: similarity measured using symmetric metric
        # M = m(a,b) = m(b,a)
        '1L_multi-stft-mag': multi_mag_ab.item(), 
        '1L_stft-mag': mag_ab.item(),
        '1L_multi-wave': self.measures["multi-wave"](x1,x2).item(),
        '1L_wave': self.measures["wave"](x1,x2).item(),
        '1L_logmel': self.measures["logmel"](x1,x2).item(),
        '1L_multi-mel': self.measures["multi-mel"](x1,x2).item(),
        '1S_sisdr': self.measures["sisdr"](x1,x2).item(),
        '1L_emb_euc': self.measures["emb_euc"](x1,x2).item(),
        '1L_emb_cos': self.measures["emb_cos"](x1,x2).item(), 
        
        # Type 2: similarity measured using non-symmetric metric 
        # M = (m(a,b)+m(b,a))/2
        '2S_fwsnr': (L_fwsnr_ab + L_fwsnr_ba)/2,
        '2L_multi-stft': ((L_multi_stft_ab+L_multi_stft_ba)/2),
        '2L_stft': ((L_stft_ab+L_stft_ba)/2),
        '2S_pesq': ((L_pesq_ab+L_pesq_ba)/2).item(),
        '2S_stoi': ((L_stoi_ab+L_stoi_ba)/2).item(),

        # Type 3: similarity measured as distance in intrusive metric
        # M = abs(m(a,ref) - m(b,ref))
        '3D_pesq': torch.mean(torch.abs(L_pesq_x1-L_pesq_x2)).item(), 
        '3D_stoi': torch.mean(torch.abs(L_stoi_x1-L_stoi_x2)).item(),
        '3D_sisdr': torch.mean(torch.abs(L_sisdr_x1-L_sisdr_x2)).item(),
        '3D_mos_nidiff': torch.mean(torch.abs(L_mos_x1-L_mos_x2)).item(),
        '3D_pesq_nidiff': torch.mean(torch.abs(L_pesqni_x1-L_pesqni_x2)).item(),
        '3D_stoi_nidiff': torch.mean(torch.abs(L_stoini_x1-L_stoini_x2)).item(), 
        '3D_sisdr_nidiff': torch.mean(torch.abs(L_sisdrni_x1-L_sisdrni_x2)).item(),
        '3D_srmr_nidiff': torch.mean(torch.abs(L_srmr_x1-L_srmr_x2)).item(),  

        }
        
        return dict_row
    

def eval_experiment(config,checkpoints_list=None):

    eval_dict_list=[]
    eval_dir=config["eval_savedir"]
    

    # ---- initialize evaluator ----
    # (metrics, baselines, dataset, dataloader)
    myeval = Evaluator(config)

    # ---- evaluate oracle data  -----

    if config["compute_only"]=="oracle":
        eval_file_name="oracle_"+config["eval_file_name"]
        print(f"Computing metrics -> oracle ")
        eval_dict_list.extend(myeval.compute_metrics_oracle())
        # list -> df and save 
        pd.DataFrame(eval_dict_list).to_csv(eval_dir+eval_file_name, index=False)

    elif config["compute_only"]=="models":
        eval_file_name="models_"+config["eval_file_name"]
        # ---- evaluate my trained models ----
        print(f"Computing metrics -> models ")
        # a loop over all conditions of the considered experiment (eval_dir=unique for each experiment)
        # each condition contains a checkpoint for a different model version
        if checkpoints_list==None:
    
            for subdir in os.listdir(eval_dir):
                exp_subdir = os.path.join(eval_dir, subdir)
                if os.path.isdir(exp_subdir):
                    print(f"...using training results {exp_subdir}")
                    # checkpoint file path
                    tmp_checkpointpath=pjoin(exp_subdir,"checkpointbest.pt")
                    # name of this experiment condition
                    tmp_eval_tag=pjoin(exp_subdir).split('/')[-1]
                    # compute metrics for this model version
                    tmp_dict_eval=myeval.compute_metrics_checkpoint(tmp_checkpointpath,tmp_eval_tag)
                    # add the metrics to the main list containing all eval results
                    eval_dict_list.extend(tmp_dict_eval)
                    # list -> df and save 
                    pd.DataFrame(eval_dict_list).to_csv(eval_dir+eval_file_name, index=False)
        else: 
            for checkpoint in checkpoints_list:
                print(f"...using training results {checkpoint}")
                # name of this experiment condition
                tmp_eval_tag=checkpoint.split('/')[-2] + "_" + os.path.basename(checkpoint).split(".")[0]
                # compute metrics for this model version
                tmp_dict_eval=myeval.compute_metrics_checkpoint(checkpoint,tmp_eval_tag)
                # add the metrics to the main list containing all eval results
                eval_dict_list.extend(tmp_dict_eval)
                # list -> df and save 
                pd.DataFrame(eval_dict_list).to_csv(eval_dir+eval_file_name, index=False)


def set_rms(signal,rms_target):
    """ Normalize signal to a specific RMS level in dB FS """
    rms_current = torch.sqrt(torch.mean(signal**2))  # Compute current RMS

    if rms_current > 0:
        signal = signal * (rms_target / rms_current)  # Scale signal

    return signal


if __name__ == "__main__":


    # Parse command-line argument
    parser = argparse.ArgumentParser(description="Run evaluation with a specific compute_only setting.")
    parser.add_argument("--compute_only", type=str, default=None, help="Specify compute_only parameter.")
    args = parser.parse_args()

    # load default config
    config=hlp.load_config(pjoin("/home/ubuntu/guestxr2/home/ubuntu/joanna/CWUNET/config/basic.yaml"))

    config["compute_only"]=args.compute_only

    # set parameters for this experiment
    config["eval_savedir"] = "/home/ubuntu/guestxr2/home/ubuntu/joanna/CWUNET/results/big_1/"
    config["eval_file_name"] = "evaluation.csv"
    config["rt60diffmin"] = -3
    config["rt60diffmax"] = 3
    config["N_datapoints"] = 100 # if 0 - whole test set included 
    config["batch_size_eval"] = 1
    config["eval_split"] = "test"

    # make a list of checkpoints to compare (in addition to ground truth sounds)
    checkpoint_paths=[
                    "/home/ubuntu/guestxr2/home/ubuntu/joanna/CWUNET/results/big_1/checkpoint100.pt",
                    "/home/ubuntu/guestxr2/home/ubuntu/joanna/CWUNET/results/big_1/checkpointbest.pt"]
    

    eval_experiment(config,checkpoints_list=checkpoint_paths)