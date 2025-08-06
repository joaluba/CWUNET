# ---------------- GENERATE RIRS WITH MASP BASED ON  METADATA ---------------

from multiprocessing import Pool, Manager
import argparse
import numpy as np
import numpy as np
import pandas as pd
from os.path import join as pjoin
from masp import shoebox_room_sim as srs
from scipy.io import wavfile

import helpers as hlp

def render_mono_ir(df_rooms,i):
    # Multiprocessing function which, based on the room info,
    # computes room simulation, renders and stores a RIR, and 
    # calculates several acoustic parameters (either based on 
    # room or based on the synthetic RIR)
    # Inputs:
    # df_rooms - input dataframe with room information 
    # i - index for a specific room
    # -------------------------------------------------------
    # pick 1 row from the data frame
    a = df_rooms.iloc[i]
    # receiver position (mono mic): 
    rec = np.array([[a.mic_pos_x, a.mic_pos_y, a.mic_pos_z]])
    # first source position
    src1 = np.array([[a.src1_pos_x, a.src1_pos_y, a.src1_pos_z]])
    # second source position
    src2 = np.array([[a.src2_pos_x, a.src2_pos_y, a.src2_pos_z]])
    # room dimensions:
    room = np.array([a.room_x, a.room_y, a.room_z])
    # reverberation:
    rt60 = np.array([a.rt60_set])
    # Compute absorption coefficients for desired rt60 and room dimensions
    abs_walls,rt60_true = srs.find_abs_coeffs_from_rt(room, rt60)
    # Small correction for sound absorption coefficients:
    if sum(rt60_true-rt60>0.05*rt60_true)>0 :
        abs_walls,rt60_true = srs.find_abs_coeffs_from_rt(room, rt60_true + abs(rt60-rt60_true))
    # Generally, we simulate up to RT60:
    limits = np.minimum(rt60, maxlim)
    # Compute echogram:
    abs_echogram1 = srs.compute_echograms_mic(room, src1, rec, abs_walls, limits, mic_specs)
    abs_echogram2 = srs.compute_echograms_mic(room, src2, rec, abs_walls, limits, mic_specs)
    # Compute stats based on the room information: 
    rt60_stats,d_critical,d_mfpath= srs.room_stats(room, abs_walls, verbose=True)
    # Render RIR: 
    rir = srs.render_rirs_mic(abs_echogram1, band_centerfreqs, fs_rir)
    clone_rir = srs.render_rirs_mic(abs_echogram2, band_centerfreqs, fs_rir)
    # Create file names for rir and clone_rir:
    round_x=int(100*np.round(a.room_x,2))
    round_y=int(100*np.round(a.room_y,2))
    round_z=int(100*np.round(a.room_z,2))
    round_rt=int(100*np.round(rt60_true[0],2))
    filename_rir=f"monoRIR_x{round_x}y{round_y}z{round_z}_rtms{round_rt}.wav"
    filename_clone_rir="clone_" + filename_rir
    # filepaths 
    filepath_rir=pjoin(writepath, filename_rir)
    filepath_clone_rir=pjoin(writepath, filename_clone_rir)
    # save rirs
    wavfile.write(filepath_rir, fs_target, np.squeeze(rir).astype(np.float32))
    wavfile.write(filepath_clone_rir, fs_target, np.squeeze(clone_rir).astype(np.float32))
    # Compute stats based on the RIR: 
    bands=np.array([16, 31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000])
    rt30_meas,rt20_meas, edt_meas,c50_meas=hlp.compute_ir_stats(filepath_rir,bands) 
    # Fill df_rooms with computed stats:
    df_rooms.loc[i,"ir_file_name"] = filename_rir
    df_rooms.loc[i,"ir_clone_file_name"] = filename_clone_rir
    df_rooms.loc[i,"rt60_true"] = rt60_true[0]
    df_rooms.loc[i,"rt60_masp_stats"] = rt60_stats[0]
    df_rooms.loc[i,"cd_masp_stats"] = d_critical[0]
    df_rooms.loc[i,"mfp_masp_stats"] = d_mfpath
    df_rooms.loc[i,"rt30_meas"] =rt30_meas
    df_rooms.loc[i,"rt20_meas"] =rt20_meas
    df_rooms.loc[i,"edt_meas"] =edt_meas
    df_rooms.loc[i,"c50_meas"] =c50_meas
    return df_rooms.loc[i]


if __name__ == "__main__":

    fs_rir = 48000
    fs_target = 48000 
    maxlim = 1.8 # Maximum reverberation time
    band_centerfreqs=np.array([16000]) #change this for multiband
    # Mic characteristics
    mic_specs = np.array([[0, 0, 0, 1]]) # omni-directional

    # Argument parser
    parser = argparse.ArgumentParser(description="Run parallel RIR rendering.")
    parser.add_argument("writepath", type=str, help="Path to save the output files")
    parser.add_argument("df_rooms", type=str, help="Path to the CSV file containing room parameters")
    args = parser.parse_args()

    # Load the room parameters from CSV
    df_rooms = pd.read_csv(args.df_rooms)
    writepath = args.writepath

    # Read number of rirs
    N_rirs = len(df_rooms)  
    
    with Pool(processes=8) as pool:
        # Run render_mono_ir in parallel
        result = pool.starmap(render_mono_ir, [(df_rooms, idx) for idx in range(N_rirs)])

    df_rooms_with_stats = pd.concat(result, axis=1).T
    # store the information abou the dataset:
    # (rir file paths and all corresponding room and rir stats)
    df_rooms_with_stats.to_csv(pjoin(writepath,"rir_info.csv"))

    print(f"Processing completed. Results saved to {args.writepath}")

# Use this script as :
# python3 render_rirs.py "/home/ubuntu/Data/synth_rirs_new"  "/home/ubuntu/joanna/CWUNET/dataset-metadata/11-02-2025--17-16_room_info.csv"


