
from datetime import datetime
# load my modules
import trainer
import torch
import random
import numpy as np
import helpers as hlp


def exp_combinations_to_file(combinations, file_path):
    # Save combinations to text file
    with open(file_path, 'w') as file:
        for i, combination in enumerate(combinations, start=1):
            line = f"Scheduled combination: {i},"
            for condition_name, condition_value in combination.items():
                line += f" {condition_name}: {condition_value},"
            # Remove the trailing comma
            line = line.rstrip(',')
            file.write(line + '\n')
        file.write("\n")

    # Get lines of a text file (to track progress during experiment)
    with open(file_path, 'r') as file:
        lines = file.readlines()

    return lines

def note_down_finished_cond(file_path,line,wheresaved): 
    # Open the file again in write mode to overwrite its content with the appended word
    with open(file_path, 'a') as file:
            # Append the word to the end of each line and write it back to the file
            new_line = line.strip().replace("Scheduled", "Finished")
            file.write(new_line + ". \n"+ "-> Saved as:" + wheresaved + " \n" )


def get_msg_for_exp_log(message):
    # Get current date and time
    current_date = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    # Get user input
    user_input = input(message + ": ")
    # Format message with current date
    message_with_date = f"[{current_date}] {user_input}\n"
    return message_with_date


if __name__ == "__main__":

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    # # Create folder for storing results of this experiment
    # date_tag = "20-05-2024" #datetime.now().strftime("%d-%m-%Y")
    # runexp_savepath=os.path.join(config["savedir"],"runs-exp-"+date_tag)

    # load default parameters
    config = hlp.load_config("/home/ubuntu/Data/RESULTS-reverb-match-cond-u-net/runs-exp-20-05-2024/26-06-2024--22-52_c_wunet_stft+wave+emb_0.6_0.2_0.2/train_config.yaml")
    config["resume_checkpoint"]="checkpoint130.pt"
    config["resume_tboard"]=config["savedir"]

    # train with current parameters
    new_experiment=trainer.Trainer(config)
    new_experiment.train()







    








