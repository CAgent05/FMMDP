import argparse
import torch
from collections import deque
from utils import prepare_agent
import numpy as np
from src.net import GNNStack
import pandas as pd
import os
import warnings
import time
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='DRL Analysis based on TodyNet')
parser.add_argument('-a', '--arch', metavar='ARCH', default='dyGIN2d')
parser.add_argument('-d', '--dataset', metavar='DATASET', default='BipedalWalkerHCSA')
parser.add_argument('-n', '--nsteps', type=int, default=20)
parser.add_argument('-e', '--episodes', type=int, default=1000)
parser.add_argument('--alg', type=str, default="Todynet", help='the algorithm used for training')
parser.add_argument('--num_layers', type=int, default=3, help='the number of GNN layers')
parser.add_argument('--groups', type=int, default=4, help='the number of time series groups (num_graphs)')
parser.add_argument('--pool_ratio', type=float, default=0.2, help='the ratio of pooling for nodes')
parser.add_argument('--kern_size', type=str, default="9,5,3", help='list of time conv kernel size for each layer')
parser.add_argument('--in_dim', type=int, default=64, help='input dimensions of GNN stacks')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dimensions of GNN stacks')
parser.add_argument('--out_dim', type=int, default=256, help='output dimensions of GNN stacks')

# CLI Parse
args = parser.parse_args()

# make dir for exp result
result_save_dir = './result/' + args.alg 
if not os.path.exists(result_save_dir):
    os.makedirs(result_save_dir)
result_save_dir = result_save_dir + '/' + args.dataset + '_' + str(args.nsteps) + '.csv'

# prepare for agent and env
model_dir = './model/' + args.alg + '/' + args.dataset + '_' + str(args.nsteps) + '.pth'
if args.dataset[-3:] == "SAR":
    input_tag = "SAR"
    args.dataset = args.dataset[:-3]
elif args.dataset[-2:] == 'SA':
    input_tag = 'SA'
    args.dataset = args.dataset[:-2]
else:
    input_tag = "S"
    args.dataset = args.dataset[:-1]

env, model, num_nodes, alg_tag = prepare_agent(args.dataset, input_tag)
print(f"The dim of features: {num_nodes}\nThe alg used for training agent: {alg_tag}")

df = pd.DataFrame(columns=['Episode', 'Reward', 'Pre', 'True', 'Probabilities', 'Steps', 'T'])

args.kern_size = [ int(l) for l in args.kern_size.split(",") ]
    
seq_length = args.nsteps

# Model initialisation
todeynet = GNNStack(gnn_model_type=args.arch, 
                    num_layers=args.num_layers, 
                    groups=args.groups, 
                    pool_ratio=args.pool_ratio, 
                    kern_size=args.kern_size, 
                    in_dim=args.in_dim, 
                    hidden_dim=args.hidden_dim, 
                    out_dim=args.out_dim, 
                    seq_len=seq_length, 
                    num_nodes=num_nodes, 
                    num_classes=2)

todeynet.to('cuda:0')
todeynet.eval()

X_train = torch.load('./data/Train/' + args.dataset + 'SA_' + str(args.nsteps)  + '/X_train.pt')
Y_train = torch.load('./data/Train/' + args.dataset + 'SA_' + str(args.nsteps)  + '/y_train.pt')
print(X_train.shape)

X = X_train[0].float().unsqueeze(0)
label = Y_train[5].long()
print(X.shape, label)

# Load trained model parameters
print("Loading trained model parameters...")
todeynet.load_state_dict(torch.load(model_dir, map_location='cuda:0'))

# Get adjacency matrices after training
print("Getting adjacency matrices after training...")
_ = todeynet(X)
learned_adj_after = todeynet.get_learned_graph().clone()

# Print debug information
print(f"Sequence length: {seq_length}")
print(f"Number of time slices (groups): {args.groups}")
print(f"Steps per time slice: {seq_length // args.groups}")
print(f"After training adjacency matrix shape: {learned_adj_after.shape}")

def plot_trained_network(adj_after, save_path, threshold=0.1):
    """
    Plot all time slices after training (1x4 layout)
    
    Args:
        adj_after: Adjacency matrix after training
        save_path: Save path
        threshold: Threshold for displaying connections
    """
    if torch.is_tensor(adj_after):
        adj_after = adj_after.cpu().detach().numpy()
    
    num_slices = len(adj_after)
    
    # Create 1x4 subplot layout
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i in range(num_slices):
        # Process data after training
        adj_matrix_after = adj_after[i]
        num_nodes = adj_matrix_after.shape[0]
        
        # Create graph after training
        G_after = nx.Graph()
        for node in range(num_nodes):
            G_after.add_node(node)
        
        for x in range(num_nodes):
            for y in range(x+1, num_nodes):
                weight = adj_matrix_after[x, y]
                if abs(weight) > threshold:
                    G_after.add_edge(x, y, weight=weight)
        
        # Use same layout seed to ensure consistent node positions
        pos = nx.spring_layout(G_after, k=1.5, iterations=50, seed=42)
        
        # Draw graph after training
        for edge in G_after.edges(data=True):
            x, y, data = edge
            weight = data['weight']
            color = 'green' if weight > 0 else 'red'
            width = abs(weight) * 3
            alpha = min(abs(weight) * 2, 0.8)
            
            axes[i].plot([pos[x][0], pos[y][0]], [pos[x][1], pos[y][1]], 
                        color=color, linewidth=width, alpha=alpha)
        
        nx.draw_networkx_nodes(G_after, pos, ax=axes[i],
                              node_color='lightblue', 
                              node_size=300, 
                              alpha=0.9,
                              edgecolors='black',
                              linewidths=1)
        
        nx.draw_networkx_labels(G_after, pos, ax=axes[i],
                               font_size=6, 
                               font_weight='bold')
        
        # Calculate time range
        steps_per_slice = seq_length // args.groups
        start_step = i * steps_per_slice + 1
        end_step = (i + 1) * steps_per_slice
        
        # Set title
        axes[i].set_title(f'Time Slice {i+1} (Steps {start_step}-{end_step})\n'
                         f'{len(G_after.edges())} connections', 
                         fontweight='bold', fontsize=15)
        
        axes[i].axis('off')
    
    # Add overall legend
    green_line = plt.Line2D([0], [0], color='green', linewidth=3, label='Positive Correlation')
    red_line = plt.Line2D([0], [0], color='red', linewidth=3, label='Negative Correlation')
    
    fig.legend(handles=[green_line, red_line], 
              loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Trained network visualization saved to: {save_path}")

def plot_trained_heatmap(adj_after, save_path):
    """
    Plot heatmap after training (1x4 layout)
    """
    if torch.is_tensor(adj_after):
        adj_after = adj_after.cpu().detach().numpy()
    
    num_slices = len(adj_after)
    
    # Create 1x4 subplot layout
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Calculate global min and max values to maintain consistent color scale
    vmin = adj_after.min()
    vmax = adj_after.max()
    
    for i in range(num_slices):
        # Calculate time range
        steps_per_slice = seq_length // args.groups
        start_step = i * steps_per_slice + 1
        end_step = (i + 1) * steps_per_slice
        
        # Heatmap after training
        im = axes[i].imshow(adj_after[i], cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[i].set_title(f'Time Slice {i+1} (Steps {start_step}-{end_step})', 
                         fontweight='bold', fontsize=12)
        axes[i].set_xlabel('Target Node')
        axes[i].set_ylabel('Source Node')
    
    # Add colorbar
    fig.colorbar(im, ax=axes, shrink=0.6, label='Connection Strength')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Trained network heatmap saved to: {save_path}")

def analyze_trained_graph(adj_after):
    """
    Analyze statistical information of graph structure after training
    """
    if torch.is_tensor(adj_after):
        adj_after = adj_after.cpu().detach().numpy()
    
    analysis = []
    
    for i in range(len(adj_after)):
        # Statistical metrics
        mean_strength = np.mean(np.abs(adj_after[i]))
        std_strength = np.std(adj_after[i])
        max_strength = np.max(np.abs(adj_after[i]))
        
        # Calculate number of connections
        threshold = 0.1
        connections = np.sum(np.abs(adj_after[i]) > threshold) // 2  # Undirected graph, divide by 2
        
        # Number of positive and negative connections
        positive_connections = np.sum(adj_after[i] > threshold) // 2
        negative_connections = np.sum(adj_after[i] < -threshold) // 2
        
        analysis.append({
            'time_slice': i + 1,
            'mean_strength': mean_strength,
            'std_strength': std_strength,
            'max_strength': max_strength,
            'total_connections': connections,
            'positive_connections': positive_connections,
            'negative_connections': negative_connections
        })
        
        print(f"Time Slice {i+1}:")
        print(f"  Mean Connection Strength: {mean_strength:.4f}")
        print(f"  Std Connection Strength: {std_strength:.4f}")
        print(f"  Max Connection Strength: {max_strength:.4f}")
        print(f"  Total Connections: {connections}")
        print(f"  Positive Connections: {positive_connections}")
        print(f"  Negative Connections: {negative_connections}")
        print()
    
    return analysis

# Main visualization code
if len(learned_adj_after.shape) == 3:
    print("Found multiple time slices in trained model")
    
    # Plot trained network graph
    print("\nGenerating trained network visualization...")
    plot_trained_network(
        adj_after=learned_adj_after,
        save_path='./Todynet/log/trained_network_visualization.png',
        threshold=0.1
    )
    
    # Plot trained network heatmap
    print("\nGenerating trained network heatmap...")
    plot_trained_heatmap(
        adj_after=learned_adj_after,
        save_path='./Todynet/log/trained_network_heatmap.png'
    )
    
    # Analyze trained graph structure
    print("\nAnalyzing trained graph structure...")
    analysis = analyze_trained_graph(learned_adj_after)
    
else:
    print("Error: Expected 3D adjacency matrices for trained model")

print("\n" + "="*60)
print("Trained Network Graph Analysis Complete!")
print("="*60)