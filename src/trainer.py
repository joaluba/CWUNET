import torch
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
# my modules
import dataset
import loss
import models
import helpers as hlp
from os.path import join as pjoin
from os import makedirs
import yaml

class Trainer(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.load_train_objects(config)

    def load_train_objects(self,config):
        
        modeltype=config["modeltype"]
        losstype=config["losstype"]
        device=config["device"]
        learn_rate=config["learn_rate"]
        batch_size=config["batch_size"]
        resume_from_checkpoint=config["resume_from_checkpoint"]
        resume_tboard_dir=config["savedir"]
        trainscheme=config["trainscheme"]
        savedir=config["savedir"]

        makedirs(savedir, exist_ok=True)

        # ---- MODELS: ----
        self.model=load_chosen_model(config,modeltype).to(device)

        # ---- OPTIMIZERS: ----
        if trainscheme=="joint":
            self.optimizer =  torch.optim.AdamW(self.model.parameters(), learn_rate)
        elif trainscheme=="separate":
            self.optimizer_AE =  torch.optim.AdamW(self.model.autoencoder.parameters(), learn_rate)
            self.optimizer_CN=  torch.optim.AdamW(self.model.conditioning_network.parameters(), learn_rate)

        # ---- LOSS CRITERION: ----

        # load training loss function
        self.criterion=loss.load_chosen_loss(config,losstype).to(device)

        # load other loss functions to be used as validation loss functions
        self.criterion_val_stft=loss.load_chosen_loss(config,"stft").to(device)
        self.criterion_val_logmel=loss.load_chosen_loss(config,"logmel").to(device)
        self.criterion_val_wave=loss.load_chosen_loss(config,"wave").to(device)

        # ---- DATASETS: ----
        config["split"]="val"
        self.valset=dataset.DatasetReverbTransfer(config)
        config["split"]="train"
        self.trainset=dataset.DatasetReverbTransfer(config)

        # ---- DATA LOADERS: ----
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle=True, num_workers=6,pin_memory=True)
        self.valloader = torch.utils.data.DataLoader(self.valset, batch_size=batch_size, shuffle=False, num_workers=6,pin_memory=True)

        # ---- PREPARE TRAINING LOOP (starting from scratch or resume training): ----
        if resume_from_checkpoint is None:
            print("Starting new training from scratch in : "+ savedir)
            self.writer=SummaryWriter(savedir) 
            self.loss_evol=[]
            self.start_epoch=0
            self.best_val_loss=float("inf")
        else:
            checkpoint_path=pjoin(savedir,resume_from_checkpoint)
            train_results=torch.load(checkpoint_path,map_location=device)
            self.model.load_state_dict(train_results["model_state_dict"])
            # ---- OPTIMIZERS: ----
            if trainscheme=="joint":
                self.optimizer.load_state_dict(train_results["optimizer_state_dict"])
            elif trainscheme=="separate":
                self.optimizer_AE.load_state_dict(train_results["optimizer_AE_state_dict"])
                self.optimizer_CN.load_state_dict(train_results["optimizer_CN_state_dict"])       

            self.writer=SummaryWriter(resume_tboard_dir)
            self.loss_evol = train_results['loss']
            self.start_epoch = train_results['epoch']+1
            self.best_val_loss=float("inf")
            print("Resume training from epoch "+ str(self.start_epoch) + " from training "+ savedir)

            
    def train(self):

        # trainng parameters
        store_outputs = self.config["store_outputs"]
        savedir = self.config["savedir"]
        num_epochs = self.config["num_epochs"]
        device = self.config["device"]
        loss_alphas=self.config["loss_alphas"]
        checkpoint_step =self.config["checkpoint_step"]
        trainscheme=self.config["trainscheme"]
        
        # save training config file in the savedir dedicated for this training
        if (bool(store_outputs)):
            with open(pjoin(savedir,'train_config.yaml'), 'w') as yaml_file:
                yaml.dump(self.config, yaml_file, default_flow_style=False)
            print(self.config)
        
        # ------------- TRAINING START: -------------
        start = time.time()
        for epoch in range(self.start_epoch,num_epochs):

            # ----- Update learning rate: -----
            if trainscheme=="joint":
                self.learn_rate_update(epoch,20,1,self.optimizer)
                print(f"Epoch {epoch}: Learning Rate: {self.optimizer.param_groups[0]['lr']}") 
            elif trainscheme=="separate":
                self.learn_rate_update(epoch,20,1,self.optimizer_CN)
                self.learn_rate_update(epoch,20,1,self.optimizer_AE)
                print(f"Epoch {epoch}: Learning Rate: {self.optimizer_CN.param_groups[0]['lr']}") 
                print(f"Epoch {epoch}: Learning Rate: {self.optimizer_AE.param_groups[0]['lr']}") 

            
            # ----- Training loop for this epoch: -----
            self.model.train()
            train_loss=0
            end = time.time()
            for j,data in tqdm(enumerate(self.trainloader),total = len(self.trainloader)):

                # # measure data loading time (how much time of the training loop is spent on waiting on the next batch)
                # # - should be zero if the data loading is not a bottleneck
                # print(f"Time to load data: {time.time() - end}")     

                # infer and compute loss
                loss_vals,loss_names=self.criterion(epoch, data, self.model, device)
                # log and sum all loss terms
                loss=0
                for i in range(0,len(loss_vals)):
                    loss_term=loss_alphas[i]*loss_vals[i]
                    loss+=loss_term
                    # self.writer.add_scalar(loss_names[i], loss_term.item(), epoch * len(self.trainloader) + j) # tensorboard
                    
                # empty gradient
                if trainscheme=="joint":
                    self.optimizer.zero_grad()
                elif trainscheme=="separate":
                    self.optimizer_AE.zero_grad()
                    self.optimizer_CN.zero_grad()
                
                # compute gradients 
                loss.backward()

                # update weights
                if trainscheme=="joint":
                    self.optimizer.step()
                elif trainscheme=="separate":
                    self.optimizer_AE.step()
                    self.optimizer_CN.step()

                # add current batch loss to total epoch loss
                train_loss += loss.item()

                # log variables and audios to tensorboard:
                self.writer.add_scalar('TrainLossPerBatch', loss.item(), epoch * len(self.trainloader) + j) # tensorboard

                if j==0:
                    self.logaudio_tboard(self.writer)# tensorboard

                # # measure time required to process one batch 
                # print(f"Time to process a batch: {time.time() - end}")
                # end = time.time()  

            # ----- Validation loop for this epoch: -----
            self.model.eval()
            with torch.no_grad():
                val_loss=0
                stft_loss=0
                logmel_loss=0
                wave_loss=0
                for j,data in tqdm(enumerate(self.valloader),total = len(self.valloader)):
                                
                    # infer and compute loss
                    loss_vals,loss_names=self.criterion(epoch, data, self.model, device)   
                    # log and sum all loss terms
                    loss=0
                    for i in range(0,len(loss_vals)):
                        loss_term=loss_alphas[i]*loss_vals[i]
                        loss+=loss_term
                        # self.writer.add_scalar("val_"+loss_names[i], loss_term.item(), epoch * len(self.valloader) + j) # tensorboard

                    # add current batch val loss to total epoch val loss
                    val_loss += loss.item()  
                    
                    # apart from regular validation loss, compute additional losses
                    # to easily compare between the conditions
                    stft_vals_stft,_ = self.criterion_val_stft(epoch, data, self.model, device)   
                    stft_loss += stft_vals_stft[0].item()

                    logmel_vals,_ = self.criterion_val_logmel(epoch, data, self.model, device)   
                    logmel_loss += logmel_vals[0].item()

                    wave_vals,_ = self.criterion_val_wave(epoch, data, self.model, device)   
                    wave_loss += wave_vals[0].item()
                    

            # Print stats at the end of the epoch
            num_samples_train=len(self.trainloader.sampler)
            num_samples_val=len(self.valloader.sampler)
            avg_train_loss = train_loss / num_samples_train
            avg_val_loss = val_loss / num_samples_val
            avg_stft_loss =stft_loss/num_samples_val
            avg_logmel_loss =logmel_loss/num_samples_val
            avg_wave_loss =wave_loss/num_samples_val
            print(f'Epoch: {epoch}, Train. Loss: {avg_train_loss:.5f}, Val. Loss: {avg_val_loss:.5f}')
            self.loss_evol.append((avg_train_loss,avg_val_loss)) 

            self.writer.add_scalar('TrainLoss', avg_train_loss, epoch) # tensorboard
            self.writer.add_scalar('ValLoss', avg_val_loss, epoch) # tensorboard
            self.writer.add_scalar('VAL_STFT',avg_stft_loss, epoch) # tensorboard
            self.writer.add_scalar('VAL_LOGMEL',avg_logmel_loss, epoch) # tensorboard
            self.writer.add_scalar('VAL_WAVE',avg_wave_loss, epoch) # tensorboard
            
            # Save checkpoint (overwrite)
            if (bool(store_outputs)) & (epoch % checkpoint_step ==0):
                self.save_checkpoint(epoch,self.loss_evol,str(epoch))
                
            # Save best checkpoint so far (its the one with the lowest val loss)
            if avg_val_loss < self.best_val_loss:
                # save the best checkpoint so far 
                if (bool(store_outputs)):
                    self.save_checkpoint(epoch,self.loss_evol,"best")
                self.best_val_loss = avg_val_loss
                counter = 0
                print(f'Loss decreased.')
            else:
                counter += 1
                print(f'Loss did not decrease for {counter} epochs.')
            
        end=time.time()
        print(f"Finished training after: {(end-start)} seconds")
        self.writer.close()


    def learn_rate_update(self,curr_epoch, step,factor,optimizer):
        # Update learning rate every "step" epochs
        if (curr_epoch + 1) % step == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= factor                
        
    def logaudio_tboard(self,writer):
        # Function to log specific audio samples in tensorboard
        fs = self.config["fs"]
        device=self.config["device"]

        with torch.no_grad():
            
            chosen_idx_derev=self.valset.get_idx_with_rt60diff(-3,-0.7)
            chosen_idx_rerev=self.valset.get_idx_with_rt60diff(0.7,3)

            chosen_idx=chosen_idx_derev[:3]+chosen_idx_rerev[:3]
            
            # inference for the chosen samples:
            for i in range(0,len(chosen_idx)):
                data=self.valset[chosen_idx[i]]
                data = [data[i].unsqueeze(0) for i in range(len(data))]
                sContent,_,sTarget, sPrediction=infer(self.model,data,device)

                # wave formatting:
                wave_content=sContent[0,:,:].squeeze(0)
                wave_target=sTarget[0,:,:].squeeze(0)
                wave_predict=sPrediction[0,:,:].squeeze(0)

                # save audio to tensorboard:
                writer.add_audio(f'Content_dp{i}', wave_content/wave_content.abs().max(), torch.tensor(fs))
                writer.add_audio(f'Target_dp{i}', wave_target/wave_target.abs().max(), torch.tensor(fs))
                writer.add_audio(f'Predict_dp{i}', wave_predict/wave_predict.abs().max(), torch.tensor(fs))

    def save_checkpoint(self,epoch,loss_evol,name):
        savedir = self.config["savedir"]
        trainscheme=self.config["trainscheme"]
        # update weights
        if trainscheme=="joint":
            torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss_evol,
            }, pjoin(savedir,'checkpoint' +name+'.pt'))
        elif trainscheme=="separate":
            torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_AE_state_dict': self.optimizer_AE.state_dict(),
            'optimizer_CN_state_dict': self.optimizer_CN.state_dict(),
            'loss': loss_evol,
            }, pjoin(savedir,'checkpoint' +name+'.pt'))


        

def load_train_results(datapath, exp_tag, train_tag,configtype="yaml"):

    if configtype=="yaml":
        config=hlp.load_config(pjoin(datapath,exp_tag,train_tag,"train_config.yaml"))
        config["device"]="cpu"
        train_results=torch.load(pjoin(datapath,exp_tag,train_tag,"checkpointbest.pt"),map_location=config["device"])
    elif configtype=="pt":
        args=torch.load(pjoin(datapath,exp_tag,train_tag,"trainargs.pt"))
        args.device="cpu"
        train_results=torch.load(pjoin(datapath,exp_tag,train_tag,"checkpointbest.pt"),map_location=args.device)
        config = {attr: getattr(args, attr) for attr in dir(args) if not attr.startswith('__') and not callable(getattr(args, attr))}
        config["modeltype"]="c_wunet"
        config_s=hlp.load_config("../config/old_params.yaml")
        # Identify new parameters in conf2 not present in conf1
        new_params = {key: config_s[key] for key in config_s if key not in config}
        # Update conf1 with these new parameters
        config.update(new_params)

    return config,train_results


def load_chosen_model(config,model_type):

    if model_type=="c_wunet":
        autoencoder=models.waveunet(config)
        condgenerator=models.ReverbEncoder(config)
        jointmodel=models.cond_reverb_transfer(autoencoder,condgenerator)
    else:
        print("This model type is not implemented")

    return jointmodel

def infer(model,data,device):
    # Function to infer target audio
    with torch.no_grad():
        sContent = data[0].to(device)
        sStyle=data[1].to(device)
        sTarget=data[2].to(device)
        sPrediction=model(hlp.unsqueezeif2D(sContent),hlp.unsqueezeif2D(sStyle))
        
    return sContent, sStyle, sTarget, sPrediction

if __name__ == "__main__":
    
    # ---- test training loop ----

    config=hlp.load_config("../CWUNET/config/basic.yaml")
    # set arguments for running a pilot training
    config["num_epochs"]=3
    config["checkpoint_step"]=1
    config["savedir"]="/home/ubuntu/joanna/CWUNET/results/pilot_1"
    config["modeltype"]="c_wunet"
    config["df_metadata"]="../CWUNET/dataset-metadata/ds1_metadata_example.csv"
    # config["resume_from_checkpoint"]="checkpoint0.pt"

    new_experiment=Trainer(config)
    new_experiment.train()














