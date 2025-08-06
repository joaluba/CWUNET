import torch
import loss_mel
import loss_stft
import loss_waveform


class load_chosen_loss(torch.nn.Module):
    def __init__(self,config,losstype):
        super().__init__()
        self.config=config
        self.losstype=losstype
        self.load_criterions(config["device"])

    def load_criterions(self,device):
        if self.losstype=="stft":
            self.criterion_audio=loss_stft.MultiResolutionSTFTLoss().to(device)

        elif self.losstype=="logmel":
            self.criterion_audio=loss_mel.MultiMelSpectrogramLoss().to(device)
        
        elif self.losstype=="wave":
            self.criterion_audio=loss_waveform.MultiWindowShapeLoss().to(device)

        elif self.losstype=="logmel+wave":
            self.criterion_audio1=loss_mel.MultiMelSpectrogramLoss().to(device)
            self.criterion_audio2=loss_waveform.MultiWindowShapeLoss().to(device)

        elif self.losstype=="stft+wave":
            self.criterion_audio1=loss_stft.MultiResolutionSTFTLoss().to(device)
            self.criterion_audio2=loss_waveform.MultiWindowShapeLoss().to(device)


        elif self.losstype=="stft+wave+emb":
            self.criterion_audio1=loss_stft.MultiResolutionSTFTLoss().to(device)
            self.criterion_audio2=loss_waveform.MultiWindowShapeLoss().to(device)
            self.criterion_emb=torch.nn.MSELoss(dim=2,eps=1e-8).to(device)

        else:
            print("this loss is not implemented")

    def forward(self, epoch, data, model_combined, device):
        # get datapoint
        sContent_in = data[0].to(device) # s1r1 - content
        sStyle_in=data[1].to(device) # s2r2 - style
        sTarget=data[2].to(device) # s1r2 - target

        # forward pass - get prediction 
        embStyle=model_combined.conditioning_network(sStyle_in)
        sPrediction=model_combined(sContent_in,sStyle_in)

        if self.losstype=="stft":
            L_sc, L_mag = self.criterion_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
            L_stft = L_sc+ L_mag 
            L = [L_stft]
            L_names = ["L_stft"]
        
        elif self.losstype=="logmel":
            L_logmel = self.criterion_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
            L = [L_logmel]
            L_names =["L_logmel"]

        elif self.losstype=="logmel+wave":
            L_logmel = self.criterion_audio1(sTarget.squeeze(1), sPrediction.squeeze(1))
            L_wave = self.criterion_audio2(sTarget.squeeze(1), sPrediction.squeeze(1))
            L = [L_logmel,L_wave]
            L_names =["L_logmel","L_wave"]
        
        elif self.losstype=="stft+wave":
            L_sc, L_mag  = self.criterion_audio1(sTarget.squeeze(1), sPrediction.squeeze(1))
            L_stft = L_sc + L_mag 
            L_wave = self.criterion_audio2(sTarget.squeeze(1), sPrediction.squeeze(1))
            L = [L_stft,L_wave]
            L_names =["L_stft","L_wave"]
        
        elif self.losstype=="stft+wave+emb":
            L_sc, L_mag  = self.criterion_audio1(sTarget.squeeze(1), sPrediction.squeeze(1))
            L_stft = L_sc + L_mag 
            L_wave = self.criterion_audio2(sTarget.squeeze(1), sPrediction.squeeze(1))
            # get the embedding of the prediction
            embTarget = model_combined.conditioning_network(sTarget)
            L_emb = self.criterion_emb(embStyle,embTarget)
            L = [L_stft,L_wave, L_emb]
            L_names =["L_stft","L_wave","L_emb"]

        elif self.losstype=="wave":
            L_wave = self.criterion_audio(sTarget.squeeze(1), sPrediction.squeeze(1))
            L = [L_wave]
            L_names =["L_wave"]

        else:
            print("the forward for this loss is not implemented")

        return L, L_names
    

