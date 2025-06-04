import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from training import MF_comp, training_loop
from data_generation import generate_data, generate_data_with_thinning

def single_param_multi_run(n_1, n_2, r, e, m, num_runs, num_epochs, lr, device, plot=True):
    # define the loss function: binary cross entropy
    criterion = nn.BCELoss()
    errors = []
    for _ in range(num_runs):
        print(f'Run {_+1}/{num_runs}')
        X, dataloader = generate_data(n_1, n_2, r, e, m, device)
        model = MF_comp(n_1, n_2, r)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        errors.append(training_loop(X, model, dataloader, optimizer, criterion, num_epochs, device))
    if plot:
        plot_error_trajectories(errors)
    return errors

def plot_error_trajectories(errors):
    for i in range(len(errors)):
        plt.plot(errors[i])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.yscale('log')
    plt.title('Error vs. Epoch')
    plt.show()

def sample_complexity_with_rank(n_1, n_2, r_range, m_range, num_runs, num_epochs, lr, device, plot=True):
    # define the loss function: binary cross entropy
    criterion = nn.BCELoss()
    errors = np.zeros((len(r_range), len(m_range), num_runs, num_epochs))
    e = 0
    for i, r in enumerate(r_range):
        for k in range(num_runs):
            X, dataloaders = generate_data_with_thinning(n_1, n_2, r, e, m_range, device)
            for j, m in enumerate(m_range):
                print(f'Rank {r}, Samples {m}, Run {k+1}/{num_runs}')
                model = MF_comp(n_1, n_2, r)
                model.to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                errors[i, j, k, :] = training_loop(X, model, dataloaders[m], optimizer, criterion, num_epochs, device)
    if plot:
        plot_r_m_heatmap(errors, r_range, m_range)
    return errors

import matplotlib.colors as colors

def plot_r_m_heatmap(errors, r_range, m_range):
    # Calculate mean of the last 10 time steps across all runs
    final_errors = np.mean(errors[:, :, :, -10:], axis=3)
    
    # Calculate average and standard deviation of errors
    avg_errors = np.mean(final_errors, axis=2)
    std_errors = np.std(final_errors, axis=2)
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Create heatmap using matplotlib with log normalization
    im = ax.imshow(avg_errors, 
                   cmap='YlGnBu', 
                   norm=colors.LogNorm(),  # Set colorbar to log scale
                   aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Average Error (Log Scale)')
    
    # Annotate each cell with average and standard error
    for i in range(avg_errors.shape[0]):
        for j in range(avg_errors.shape[1]):
            text = f'{avg_errors[i,j]:.4f}\n({std_errors[i,j]:.4f})'
            txt_clr = 'white' if avg_errors[i, j] > 0.075 else 'black'
            if avg_errors[i, j] < 1e-4:
                text = f'< 1e-4\n({std_errors[i,j]:.4f})'
            ax.text(j, i, text, 
                    ha='center', 
                    va='center', 
                    color=txt_clr)
    
    # Set x and y ticks
    ax.set_yticks(np.arange(len(r_range)))
    ax.set_yticklabels([str(r) for r in r_range], rotation=45)
    ax.set_xticks(np.arange(len(m_range)))
    ax.set_xticklabels([str(m) for m in m_range])
    
    # Label axes
    ax.set_ylabel('Rank $(r)$')
    ax.set_xlabel('Sample Size $(m)$')
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()

    # Save the figure
    fig.savefig('sample_complexity_heatmap.pdf', dpi=300, bbox_inches='tight')