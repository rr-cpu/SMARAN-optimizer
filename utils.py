import matplotlib.pyplot as plt
import pandas as pd
import os
from glob import glob
import random
import numpy as np
import os
import torch


#function for seeding
def seed_everything(seed: int = 42):
    # Core seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Enforce deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

    print(f"Seeding complete: {seed}")


    

class graph_plot:
    def __init__(self):
        pass

    def plot_losses(self,train_losses, test_losses):
        epochs = range(1, len(train_losses) + 1)
    
        plt.figure(figsize=(8, 6))
        plt.plot(epochs, train_losses, label="Training Loss", marker='o', linestyle='-')
        plt.plot(epochs, test_losses, label="Test Loss", marker='s', linestyle='--')
    
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training vs Test Loss")
        plt.legend()
        plt.grid()
        plt.show()
    
    def plot_multiple_losses(self, loss_lists, optimizer_names,title,xlabel='Epochs',ylabel='loss'):
        epochs = range(1, len(loss_lists[0]) + 1)  # Assumes all optimizers have the same number of epochs
        
        plt.figure(figsize=(8, 6))
        
        for losses, name in zip(loss_lists, optimizer_names):
            plt.plot(epochs, losses, marker='o', linestyle='-', label=name)
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()

def save_losses_to_excel(train_loss, test_loss, accuracy_list, mAP_list, filename):
    """
    Save training and testing loss lists to an Excel file.

    Parameters:
    - train_loss (list of float): List of training loss values.
    - test_loss (list of float): List of testing loss values.
    - filename (str): Name of the Excel file to save (default: 'loss_log.xlsx')
    """
    if len(train_loss) != len(test_loss):
        raise ValueError("train_loss and test_loss must have the same length.")
    if len(accuracy_list) != len(test_loss):
        raise ValueError("accuracy_list and test_loss must have the same length.")
        

    df = pd.DataFrame({
        'Epoch': list(range(1, len(train_loss) + 1)),
        'Train Loss': train_loss,
        'Test Loss': test_loss,
        'Test Accuracy': accuracy_list,
        'mAP': mAP_list
    })

    df.to_excel(filename, index=False)
    print(f"Losses saved to '{filename}' successfully.")



def plot_losses_from_folder(folder_path):
    # Find all .xlsx files in the folder
    files = glob(os.path.join(folder_path, '*.xlsx'))

    # Plot training losses
    for file in files:
        df = pd.read_excel(file)
        label = os.path.splitext(os.path.basename(file))[0]
        plt.plot(df['Epoch'], df['Train Loss'],marker='.', label=f'{label} - Train')
    plt.title('Training Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot testing losses
    for file in files:
        df = pd.read_excel(file)
        label = os.path.splitext(os.path.basename(file))[0]
        plt.plot(df['Epoch'], df['Test Loss'], marker='.', label=f'{label} - Test')
    plt.title('Testing Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()