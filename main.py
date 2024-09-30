import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
from time import time
import os
import random
import configparser
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from models.ASTFGCN import make_model
from metrics import *
from DataLoader import load_data
from earlystopping import EarlyStopping
from utils import evaluate_model, predict_and_save
from utils import get_adjacency_matrix, get_adjacency_matrix2
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from tensorboardX import SummaryWriter


data_path = './data/PEMS04/PEMS04_h1_d0_w0.npz'
"""
PEMS03:
    train_x = (15711, 358, 1, 12)
    train_target = (15711, 358, 12)

PEMS04:
    train_x = (10181, 307, 3, 12)
    train_target = (10181, 307, 12)

PEMS07:
    train_x = (16920, 883, 1, 12)
    train_target = (16920, 883, 12)

PEMS08:
    train_x = (10699, 170, 3, 12)
    train_target = (10699, 170, 12)
"""

# Set random seeds
def seed_env(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_env(1)

# Set config
parser = argparse.ArgumentParser(description='Import parameter')

parser.add_argument('--config', default='config/PEMS04.conf', type=str, help='configuration file')

args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.config)

data_config = config['Data']
training_config = config['Training']

adj_file = data_config['adj_file']
graph_npz_file = data_config['graph_npz_file']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

dataset_name = data_config['dataset_name']
num_of_nodes = int(data_config['num_of_nodes'])
points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])
len_input = int(data_config['len_input'])

model_name = training_config['model_name']

batch_size = int(training_config['batch_size'])
learning_rate = float(training_config['learning_rate'])
weight_decay = float(training_config['weight_decay'])
dropout = float(training_config['dropout'])
epochs = int(training_config['epochs'])
T_max = int(training_config['T_max'])
eta_min = float(training_config['eta_min'])

K = int(training_config['K'])
hidden_size = int(training_config['hidden_size'])
num_layers = int(training_config['num_layers'])
num_blocks = int(training_config['num_blocks'])
num_heads = int(training_config['num_heads'])

num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])


if dataset_name == 'PEMS03' or 'PEMS04' or 'PEMS08' or 'PEMS07':
    adj_mx = get_adjacency_matrix2(adj_file, num_of_nodes, id_filename=id_filename)

print('adj_mx:', adj_mx)


folder_dir = '%s_h%dd%dw%d' % (model_name, num_of_hours, num_of_days, num_of_weeks)
params_path = os.path.join('garage', dataset_name, folder_dir)
print('params_path:', params_path)


# Load data
train_x_tensor, train_target_tensor, train_dataloader, _, val_target_tensor, val_dataloader, _, test_target_tensor, test_dataloader, mean, std = load_data(dataset_name, num_of_weeks, num_of_days, num_of_hours, device, batch_size)

adj_mx = torch.tensor(adj_mx).to(device)


# Create a model
model = make_model(in_c=len_input, hid_c=hidden_size, out_c=num_for_predict, skip_c=256, num_nodes=num_of_nodes, K=K, num_layers=num_layers, c1=32, c2=(32, 48, 64), c3=(8, 10, 12, 14, 16), c4=16, num_blocks=num_blocks,
                   num_heads=num_heads, dropout=dropout).to(device)

# EarlyStopping
early_stopping = EarlyStopping(save_path=params_path)

# Training model
def train():
    print(model)
    
    # Setup optimizer and scheduler
    optimizer = Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
    scheduler = CosineAnnealingLR(optimizer = optimizer, T_max = T_max, eta_min = eta_min)

    # Set loss function
    loss_fn = nn.SmoothL1Loss().to(device)

    total_step = 0
    min_val_epoch = 0
    start_time = time()
    writer = SummaryWriter(logdir=params_path, flush_secs=20)

    min_val_loss = np.Inf

    epoch_losses = []
    for epoch in range(1, epochs + 1):
        print('current epoch: ', epoch)
        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        val_loss = evaluate_model(model, val_dataloader, adj_mx, loss_fn, epoch, writer)
        print('val loss', val_loss)
        epoch_losses.append(val_loss)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            min_val_epoch = epoch
            torch.save(model.state_dict(), params_filename)
            print('min_val_epoch: ', min_val_epoch)
            print('min_val_loss: ', min_val_loss)
            print('save params to file: %s' % params_filename)

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        model.train()

        for batch_index, batch_data in enumerate(tqdm(train_dataloader, desc=f' {epoch} epoch, training', total=len(train_dataloader))):
            x, y = batch_data
            optimizer.zero_grad()
            y_pred = model(x, adj_mx)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            training_loss = loss.item()
            total_step += 1
            writer.add_scalar('training_loss', training_loss, total_step)

            if total_step % 1000 == 0:

                print('total_step: %s, training loss: %.2f, time: %.2fs' % (total_step, training_loss, time() - start_time))

    print('min_val_epoch:', min_val_epoch)
    # apply the best model on the test dataset
    predict_main(min_val_epoch, test_dataloader, test_target_tensor, adj_mx, mean, std, 'test')

    def plot_combined():
        num_samples = 2000
        node_indices = random.sample(range(num_of_nodes), 2)
        time_steps = np.arange(num_samples)

        fig = plt.figure(figsize=(16, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

        # fig1
        ax1 = fig.add_subplot(gs[0])
        for idx in node_indices:
            ax1.plot(time_steps, train_x_tensor[:num_samples, idx, 0, 0].cpu().numpy(), label=f'Node {idx}')
        ax1.set_xlabel('Time Steps')
        ax1.set_ylabel('Traffic Flow')
        ax1.legend()
        ax1.grid(True)

        # fig2
        ax2 = fig.add_subplot(gs[1])
        cax = ax2.matshow(adj_mx.cpu().numpy())
        fig.colorbar(cax, ax=ax2, fraction=0.046, pad=0.04)
        ax2.set_xlabel('Node')
        ax2.set_ylabel('Node')
        ax2.tick_params(axis='x', labelbottom=True, labeltop=False)
        ax2.tick_params(axis='y', labelleft=True, labelright=False)
        ax2.tick_params(axis='x', top=False)
        ax2.tick_params(axis='y', right=False)

        plt.text(0.5, -0.16, '(a) Traffic flow for 2 random nodes', ha='center', va='center', fontsize=16, transform=ax1.transAxes)
        plt.text(0.5, -0.28, '(b) Sensor connectivity matrix', ha='center', va='center', fontsize=16, transform=ax2.transAxes)

        plt.subplots_adjust(bottom=0.2)

        plot_filename = os.path.join(params_path, 'combined_traffic_and_connectivity.svg')
        plt.savefig(plot_filename, format='svg')
        plt.close()

    plot_combined()

    writer.close()


    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker=None, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plot_filename = os.path.join(params_path, 'epoch_losses.svg')
    plt.savefig(plot_filename, format='svg')
    plt.close()


def predict_main(total_step, data_loader, data_target_tensor, adj_mx, mean, std, type):

    params_filename = os.path.join(params_path, 'epoch_%s.params' % total_step)
    print('load weight from:', params_filename)

    model.load_state_dict(torch.load(params_filename))

    predict_and_save(model, data_loader, data_target_tensor, adj_mx, total_step, mean, std, params_path, type)


if __name__ == "__main__":
    
    train()

