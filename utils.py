import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from metrics import masked_mape



def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA

def get_adjacency_matrix2(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    type_: str, {connectivity, distance}

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type_ == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    return A


def inverse_normalization(x, mean, std):
    x = x * std + mean
    return x


def evaluate_model(model, val_dataloader, adj_mx, loss_fn, epoch, writer, limit=None):

    model.eval()

    with torch.no_grad():

        val_dataloader_length = len(val_dataloader) 
        tmp = [] 

        for batch_index, batch_data in enumerate(val_dataloader):
            x, y = batch_data
            y_pred = model(x, adj_mx)
            loss = loss_fn(y_pred, y)
            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_dataloader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)
        writer.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss


def predict_and_save(model, data_loader, data_target_tensor, adj_mx, total_step, mean, std, params_path, type):
    
    model.eval()

    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()
        loader_length = len(data_loader)
        input = []  # input
        prediction = []  # output

        for batch_index, batch_data in enumerate(data_loader):
            x, y = batch_data
            input.append(x[:, :, 0:1].cpu().numpy())
            y_pred = model(x, adj_mx)
            prediction.append(y_pred.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting dataset batch %s / %s' % (batch_index + 1, loader_length))

        input = np.concatenate(input, 0)
        input = inverse_normalization(input, mean, std)
        prediction = np.concatenate(prediction, 0)

        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (total_step, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        # compute error
        error_list = []
        prediction_length = prediction.shape[2]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == prediction.shape[0]
            print('current epoch: %s, predict %s-th point' % (total_step, i+1))
            mae = mean_absolute_error(data_target_tensor[:, :, i], prediction[:, :, i])
            mape = masked_mape(data_target_tensor[:, :, i], prediction[:, :, i], 0)
            rmse = mean_squared_error(data_target_tensor[:, :, i], prediction[:, :, i]) ** 0.5
            print('MAE: %.2f' % (mae))
            print('MAPE: %.2f' % (mape))
            print('RMSE: %.2f' % (rmse))
            error_list.extend([mae, mape, rmse])

        # print
        mae = mean_absolute_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1))
        mape = masked_mape(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1), 0)
        rmse = mean_squared_error(data_target_tensor.reshape(-1, 1), prediction.reshape(-1, 1)) ** 0.5
        print('all MAE: %.2f' % (mae))
        print('all MAPE: %.2f' % (mape))
        print('all RMSE: %.2f' % (rmse))
        error_list.extend([mae, mape, rmse])
        print(error_list)

        # Randomly select two nodes for comparison
        num_nodes = input.shape[1]
        nodes_to_plot = random.sample(range(num_nodes), 2)

        # Plot real vs predicted values for the randomly selected nodes in subplots
        num_steps = min(3000, input.shape[0])  # Ensure we don't exceed available time steps
        fig, axs = plt.subplots(2, 1, figsize=(15, 12))

        for i, node_index in enumerate(nodes_to_plot):
            # Plot for the selected node
            axs[i].plot(range(num_steps), data_target_tensor[:num_steps, node_index, 0], label='Real Values', color='blue')
            axs[i].plot(range(num_steps), prediction[:num_steps, node_index, 0], label='Predicted Values', color='red')
            axs[i].set_xlabel('Time Steps')
            axs[i].set_ylabel('Traffic Flow')
            axs[i].legend()
            axs[i].grid(True)
            # Set y-axis limits based on data
            axs[i].set_ylim([min(np.min(data_target_tensor[:, node_index, 0]), np.min(prediction[:, node_index, 0])),
                             max(np.max(data_target_tensor[:, node_index, 0]), np.max(prediction[:, node_index, 0])) + 5])
            
            label = '(a)' if i == 0 else '(b)'
            axs[i].text(0.5, -0.15, f'{label} Node {node_index} on PeMS04', 
                        ha='center', va='center', transform=axs[i].transAxes, fontsize=16)

        plot_filename = os.path.join(params_path, f'plot_nodes_{nodes_to_plot[0]}_{nodes_to_plot[1]}_epoch_{total_step}_{type}.svg')
        plt.savefig(plot_filename, format='svg')
        plt.close()
        print(f'Plot saved to {plot_filename}')