import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import numpy as np
import helpers as hlp

# --------------------------------------------------------------------------------------------------
# ----------------------------------------- CONDITIONING: ------------------------------------------
# --------------------------------------------------------------------------------------------------

class FiLM(nn.Module):
    # -------- Definition of a FiLM layer: --------
    def __init__(self, zdim, n_channels):
        super().__init__()
        self.gamma = nn.Linear(zdim, n_channels)   
        self.beta = nn.Linear(zdim, n_channels)   

    def forward(self, x, z):
        
        gamma_shaped=self.gamma(z).permute(0,2,1).repeat(1,1,x.shape[-1])
        beta_shaped=self.beta(z).permute(0,2,1).repeat(1,1,x.shape[-1])
        assert gamma_shaped.shape==x.shape, f"wrong gamma shape"
        assert beta_shaped.shape==x.shape, f"wrong beta shape"
        x = gamma_shaped * x + beta_shaped

        return x
    
# --------------------------------------------------------------------------------------------------
# ------------------------------------ REVERB ENCODER NETWORK: -------------------------------------
# --------------------------------------------------------------------------------------------------


class ReverbEncoderBlock(nn.Module):
    # -------- One encoder block: --------
    def __init__(self, in_channels, out_channels, kernel_size=15, stride=2, padding=7):
        super().__init__()
        # direct connection
        self.direct_conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.direct_bn = nn.BatchNorm1d(out_channels)
        self.direct_prelu = nn.PReLU()
        # residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.residual_bn = nn.BatchNorm1d(out_channels)
    

    def forward(self, x):
        # residual connection
        residual = self.residual_conv(x)
        residual = self.residual_bn(residual)
        # direct connection
        x = self.direct_conv(x)
        x = self.direct_bn(x)
        x = self.direct_prelu(x)
        # add outputs of direct and residual connections
        x += residual
        return x

class ReverbEncoder(nn.Module):
    # -------- Encoder: --------
    def __init__(self, config):
        super().__init__() 
        # constants
        self.sig_len=config["sig_len"]
        self.z_len=config["rev_emb_len"]
        self.N_layers=config["n_blocks_revenc"]
        self.learnable_pooling=config["learnable_pooling"]
        # internal parameters of the network:
        kernel_size=15
        stride=2
        padding= (kernel_size - 1) // 2

        # convolutional layers are a series of encoder blocks with increasing channels
        conv_layers = []
        block_channels=[1, 32, 32, 64, 64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512]
        x_len=self.sig_len
        for i in range(self.N_layers):
            conv_layers.append(ReverbEncoderBlock(block_channels[i], block_channels[i+1], kernel_size=15, stride=stride,padding=padding))
            # compute heigth of the ouput (width=1,depth=block_channels) - used later for lear pooling
            if self.learnable_pooling:
                x_len=np.floor((x_len-kernel_size+2*padding)/stride)+1

        self.conv_layers = nn.Sequential(*conv_layers)

        # adaptive pooling layer to flatten and aggregate information
        if self.learnable_pooling:
            self.aggregate = nn.Conv1d(block_channels, self.z_len, kernel_size=int(x_len), stride=1, padding=0)
        else:
            self.aggregate = nn.AdaptiveAvgPool1d(1)
        
        # final mlp layers
        self.mlp = nn.Sequential(nn.Linear(block_channels[self.N_layers],self.z_len))
        
        
    def forward(self, x):
        # Convolutional residual blocks:
        x = self.conv_layers(x)
        # Aggregate info & flatten:
        x = self.aggregate(x)
        x = x.view([x.shape[0],1,-1]) 
        # Dense layers after aggregation:
        x = self.mlp(x)
        return x
    
# --------------------------------------------------------------------------------------------------
# -------------------------------------- REVERB TRANSFER NETWORK: ----------------------------------
# ------------------------------------------- (WAVE-U-NET) -----------------------------------------
# --------------------------------------------------------------------------------------------------

class waveunet_DownSamplingLayer(nn.Module):
    # ------- Downsampling (encoder) block ------
    def __init__(self, channel_in, channel_out, dilation=1, kernel_size=15, stride=1, padding=7, z_channels=512):
        super().__init__()
        self.conv=nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation)
        self.bn=nn.BatchNorm1d(channel_out)
        self.film=FiLM(z_channels,channel_out)
        self.acti=nn.PReLU()

    def forward(self, x, z):
        x=self.conv(x)
        x=self.bn(x)
        x=self.film(x,z)
        x=self.acti(x)
        return x 

class waveunet_UpSamplingLayer(nn.Module):
    # ------- Upsampling (decoder) block ------
    def __init__(self, channel_in, channel_out, kernel_size=5, stride=1, padding=2,z_channels=512):
        super().__init__()
        self.conv=nn.Conv1d(channel_in, channel_out, kernel_size=kernel_size,
                      stride=stride, padding=padding)
        self.bn=nn.BatchNorm1d(channel_out)
        self.film=FiLM(z_channels,channel_out)
        self.acti=nn.PReLU()

    def forward(self, x, z):
        x=self.conv(x)
        x=self.bn(x)
        x=self.film(x,z)
        x=self.acti(x)
        return x

class waveunet(nn.Module):
    # ------- Standard wave-u-net encoder-decoder ------
    def __init__(self, config):
        super().__init__()
        # constants:
        self.n_layers = config["n_blocks_enc"]
        self.z_channels = config["rev_emb_len"]
        self.channels_interval = 24
        # encoder:
        encoder_in_channels_list = [1] + [i * self.channels_interval for i in range(1, self.n_layers)]
        # [1, 24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264]
        encoder_out_channels_list = [i * self.channels_interval for i in range(1, self.n_layers + 1)]
        # [24, 48, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288]
        self.encoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.encoder.append(
                waveunet_DownSamplingLayer(
                    channel_in=encoder_in_channels_list[i],
                    channel_out=encoder_out_channels_list[i],
                    z_channels=self.z_channels)
                )            

        # layers in the middle:
        self.middle = nn.Sequential(
            nn.Conv1d(self.n_layers * self.channels_interval, self.n_layers * self.channels_interval, 15, stride=1,
                      padding=7),
            nn.BatchNorm1d(self.n_layers * self.channels_interval),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        # decoder:
        decoder_in_channels_list = [(2 * i + 1) * self.channels_interval for i in range(1, self.n_layers)] + [
            2 * self.n_layers * self.channels_interval]
        decoder_in_channels_list = decoder_in_channels_list[::-1]
        decoder_out_channels_list = encoder_out_channels_list[::-1]
        self.decoder = nn.ModuleList()
        for i in range(self.n_layers):
            self.decoder.append(
                waveunet_UpSamplingLayer(
                    channel_in=decoder_in_channels_list[i],
                    channel_out=decoder_out_channels_list[i],
                    z_channels=self.z_channels)
            )

        # final output layer:
        self.out = nn.Sequential(
            nn.Conv1d(1 + self.channels_interval, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )
    
    def forward(self, x, z_enc,z_dec):
        # x - input waveform
        # o - waveform flowing through the net
        # z_enc - conditioning vector for encoder
        # z_dec - conditioning vector for decoder
        skipfeats = []
        o = x

        # Down Sampling
        for i in range(self.n_layers):
            # encoder layer
            o = self.encoder[i](o,z_enc)
            # store skip features for later
            skipfeats.append(o)
            # decimate, [batch_size, T // 2, channels]
            o = o[:, :, ::2]

        # middle layer:
        o = self.middle(o)

        # Up Sampling
        for i in range(self.n_layers):
            # interpolate, [batch_size, T * 2, channels]:
            o = F.interpolate(o, scale_factor=2, mode="linear", align_corners=True)
            # concatenate with skip features
            o = torch.cat([o, skipfeats[self.n_layers - i - 1]], dim=1)
            # decoder layer 
            o = self.decoder[i](o,z_dec)
            
        # concatenate output with input
        o = torch.cat([o, x], dim=1)
        # final layer with tanh activation
        o = self.out(o)
        return o

# --------------------------------------------------------------------------------------------------
# -------------------------------- FULL APPROACH (CONDITIONED ENCODER-DECODER): --------------------
# --------------------------------------------------------------------------------------------------   

class cond_reverb_transfer(nn.Module):
    # ------- Full conditional architecture ------
    def __init__(self, autoencoder,conditioning_network):
        super(cond_reverb_transfer, self).__init__()
        self.autoencoder = autoencoder
        self.conditioning_network = conditioning_network

    def forward(self, content, style):
        style_emb = self.conditioning_network(style)
        content_emb = self.conditioning_network(content)
        output = self.autoencoder(content,content_emb,style_emb)
        return output


# --------------------------------------------------------------------------------------------------  



if __name__ == "__main__":
# ---- test definitions ----
    

    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # load default parameters
    config = hlp.load_config(os.path.join(script_dir, "../config/basic.yaml"))

    # specify parameters of the model
    config["fs"] = 48000
    config["sig_len"] = 98304
    config["rev_emb_len"] = 124
    config["n_blocks_revenc"] = 14
    config["n_blocks_enc"] = 12
    config["device"]="cpu"


    from fvcore.nn import FlopCountAnalysis

    # create random tensors with the size of the expected inputs
    s_content=torch.randn(1,1,config["sig_len"]).to(config["device"])
    s_style=torch.randn(1,1,config["sig_len"]).to(config["device"])
    reverb_emb=torch.randn(1,1,config["rev_emb_len"]).to(config["device"])

    # ----- check reverb encoder -----
    print(f"\n%%%%%%%%%%%%% SUMMARY FOR THE REVERB ENCODER (FINS) %%%%%%%%%%%%%%")
    model=ReverbEncoder(config).to(config["device"])
    model.eval()
    reverb_emb=model(s_style)
    summary(model,(1, 1, config["sig_len"]),device=config["device"])# torch summary expects 2 dim input for 1d conv
    print(f"reverb encoder network input shape: {s_style.shape}")
    print(f"reverb encoder network output shape: {reverb_emb.shape}")

    inputs = (
        torch.randn(1, 1, config["sig_len"]).to(config["device"]),
    )

    flops = FlopCountAnalysis(model, inputs)
    print(f"Total FLOPs in the reverb encoder: {flops.total():,}")
    print(f"Total GFLOPs in the reverb encoder: {flops.total() / 1e9:.3f}")

    # ------ check waveunet ------
    print(f"\n%%%%%%%%%%%%% SUMMARY FOR THE TRANSFER NETWORK (COND. WAVE-U-NET) %%%%%%%%%%%%%%")
    model=waveunet(config).to(config["device"])
    model.eval()
    s_target=model(s_content,reverb_emb,reverb_emb)
    summary(model,[(1, 1, config["sig_len"]),(1, 1, config["rev_emb_len"]),(1, 1, config["rev_emb_len"])],device=config["device"]) # torch summary expects 2 dim input for 1d conv
    print(f"waveunet input shape: {s_content.shape}")
    print(f"waveunet output shape: {s_target.shape}")


    inputs = (
        torch.randn(1, 1, config["sig_len"]).to(config["device"]),
        torch.randn(1, 1, config["rev_emb_len"]).to(config["device"]),
        torch.randn(1, 1, config["rev_emb_len"]).to(config["device"]),
    )

    flops = FlopCountAnalysis(model, inputs)
    print(f"Total FLOPs in the conditioned waveunet: {flops.total():,}")
    print(f"Total GFLOPs in the conditioned waveunet: {flops.total() / 1e9:.3f}")

    # ------ check full conditioned encoder-decoder model ------
    print(f"\n%%%%%%%%%%%%% SUMMARY FOR THE FULL APPROACH (REVERB ENCODER + TRANSFER NETWORK) %%%%%%%%%%%%%%")
    cond_generator = ReverbEncoder(config)
    autoencoder = waveunet(config)
    model=cond_reverb_transfer(autoencoder,cond_generator).to(config["device"])
    s_target=model(s_content,s_style)
    summary(model,[(1, 1, config["sig_len"]),(1, 1, config["sig_len"])],device=config["device"]) # torch summary expects 2 dim input for 1d conv
    print(f"conditioned encoder-decoder input shapes: {s_content.shape}")
    print(f"conditioned encoder-decoder output shape: {s_target.shape}")

    inputs = (
        torch.randn(1, 1, config["sig_len"]).to(config["device"]),
        torch.randn(1, 1, config["sig_len"]).to(config["device"]),
    )

    flops = FlopCountAnalysis(model, inputs)
    print(f"Total FLOPs: {flops.total():,}")
    print(f"Total GFLOPs: {flops.total() / 1e9:.3f}")
  