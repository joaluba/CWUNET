import os
from itertools import product
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

    # Prompt user for input with a message
    user_message = get_msg_for_exp_log("Enter info for experiment log")

    # Write the message to a file
    with open("../CWUNET/experiment_log.txt", "a") as file:
        file.write(user_message)

    # load default parameters
    config = hlp.load_config("../CWUNET/config/basic.yaml")
    config["num_epochs"]=300

    # Permuting parameters
    perm_losses = [ "stft+wave", "logmel+wave"]
    perm_model_types=["c_wunet"]

    # Conditions combinations list
    from itertools import product
    cond_combinations = []
    for combo in product(perm_losses, perm_model_types):
        combination_dict = {
            'losstype': combo[0],
            'modeltype': combo[1]
            }
        cond_combinations.append(combination_dict)

    # Create folder for storing results of this experiment
    date_tag = datetime.now().strftime("%d-%m-%Y")
    runexp_savepath=os.path.join(config["savedir"],"runs-exp-"+date_tag)
    if not os.path.exists(runexp_savepath):
        os.makedirs(runexp_savepath)

    # Save parameter combinations list to file (to track progress of training)
    condfilepath = os.path.join(runexp_savepath,"expconds_" + date_tag +".txt")
    lines=exp_combinations_to_file(cond_combinations,condfilepath)

    cond_count=0 
    for i, combination in enumerate(cond_combinations, start=0):

        # prepare params for this combination 
        config["losstype"] =combination["losstype"]
        config["modeltype"] = combination["modeltype"]

        if (config["losstype"] == "stft") | (config["losstype"] == "logmel"): 
            config["loss_alphas"] = [1]
        elif (config["losstype"]=="stft+wave")| (config["losstype"]=="logmel+wave"):
            config["loss_alphas"] = [0.8,0.2]

        # create training tags based on date and params
        date_tag = datetime.now().strftime("%d-%m-%Y--%H-%M")
        loss_tag = "_"+ config["losstype"] 
        model_tag ="_"+ config["modeltype"]
        alpha_tag="_"+ '_'.join(map(str, config["loss_alphas"]))
        # create one long training tag
        tag = date_tag  + model_tag + loss_tag + alpha_tag

        # prepare diectory for this training combination
        config["savedir"]=os.path.join(runexp_savepath,tag) 

        # train with current parameters
        new_experiment=trainer.Trainer(config)
        new_experiment.train()

        # note down finished condition
        note_down_finished_cond(condfilepath,lines[cond_count], config["savedir"])
        cond_count+=1
        






    








