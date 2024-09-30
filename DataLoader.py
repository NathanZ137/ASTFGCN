import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data(dataset_name, num_of_weeks, num_of_days, num_of_hours, device, batch_size, shuffle=True):
    """
    PEMS03:
        train: torch.Size([15711, 358, 1, 12]) torch.Size([15711, 358, 12])
        val: torch.Size([5237, 358, 1, 12]) torch.Size([5237, 358, 12])
        test: torch.Size([5237, 358, 1, 12]) torch.Size([5237, 358, 12])

    PEMS04:
        train: torch.Size([10181, 307, 1, 12]) torch.Size([10181, 307, 12])
        val: torch.Size([3393, 307, 1, 12]) torch.Size([3393, 307, 12])
        test: torch.Size([3395, 307, 1, 12]) torch.Size([3395, 307, 12])

    PEMS07:
        train: torch.Size([16920, 883, 1, 12]) torch.Size([16920, 883, 12])
        val: torch.Size([5640, 883, 1, 12]) torch.Size([5640, 883, 12])
        test: torch.Size([5641, 883, 1, 12]) torch.Size([5641, 883, 12])

    PEMS08:
        train: torch.Size([10699, 170, 1, 12]) torch.Size([10699, 170, 12])
        val: torch.Size([3566, 170, 1, 12]) torch.Size([3566, 170, 12])
        test: torch.Size([3568, 170, 1, 12]) torch.Size([3568, 170, 12])
    """

    dataset_path = f"./data/{dataset_name}/"
    filename = os.path.join(dataset_path, f"{dataset_name}" + 
                           '_h' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks))
    
    print('load file:', dataset_name)

    file_data = np.load(filename + '.npz')

    train_x = file_data['train_x'] 
    train_x = train_x[:, :, 0:1, :]
    train_target = file_data['train_target']

    val_x = file_data['val_x']
    val_x = val_x[:, :, 0:1, :]
    val_target = file_data['val_target']

    test_x = file_data['test_x']
    test_x = test_x[:, :, 0:1, :]
    test_target = file_data['test_target']

    mean = file_data['mean'][:, :, 0:1, :] 
    std = file_data['std'][:, :, 0:1, :] 

    # train DataLoader
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(device) # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(device) # (B, N, T)

    train_dataset = TensorDataset(train_x_tensor, train_target_tensor)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    # val DataLoader
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(device) # (B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(device) # (B, N, T)

    val_dataset = TensorDataset(val_x_tensor, val_target_tensor)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # test DataLoader
    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(device) # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(device) # (B, N, T)

    test_dataset = TensorDataset(test_x_tensor, test_target_tensor)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_x_tensor, train_target_tensor, train_dataloader, val_x_tensor, val_target_tensor, val_dataloader, test_x_tensor, test_target_tensor, test_dataloader, mean, std
