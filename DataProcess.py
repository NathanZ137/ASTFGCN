import os
import numpy as np
import argparse
import configparser


def data_process(dataset_name, num_of_weeks, num_of_days, num_of_hours, num_for_predict, points_per_hours=12, save=False):
    """
    num_of_weeks, num_of_days, num_of_hours: a time span to predict
    """
    dataset_path = f"./data/{dataset_name}/"
    dataset = np.load(os.path.join(dataset_path, f"{dataset_name}.npz"))
    """
    PEMS04: (16992, 307, 3)
    16992 = 59days * 24hours * 12(Collect traffic statistics every 5 minutes)
    307 is the number of detectors
    3 is the number of features
    feature: flow, occupy, speed
    """
    data = dataset['data']
    # return data.shape
    
    all_samples = []

    for index in range(data.shape[0]):
        sample = data_sample(data, num_of_weeks, num_of_days, num_of_hours, num_for_predict, index, points_per_hours)

        if(sample[0] is None) and (sample[1] is None) and (sample[2] is None):
            continue

        week_sample, day_sample, hour_sample, target = sample
        
        sample = []

        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))
            """
            before: (T, N, F)
            after: (1, N, F, T)
            """
            sample.append(week_sample)

            target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:,:,0,:] 
            sample.append(target)

            time_sample = np.expand_dims(np.array([index]), axis=0) 
            sample.append(time_sample)

            all_samples.append(sample) # [(week_sample), (day_sample), (hour_sample), target, time_sample]

        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))
            """
            before: (T, N, F)
            after: (1, N, F, T)
            """
            sample.append(day_sample)

            target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:,:,0,:] 
            sample.append(target)

            time_sample = np.expand_dims(np.array([index]), axis=0) 
            sample.append(time_sample)

            all_samples.append(sample)

        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))
            """
            before: (T, N, F)
            after: (1, N, F, T)
            """
            sample.append(hour_sample)

            target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:,:,0,:] 
            sample.append(target)

            time_sample = np.expand_dims(np.array([index]), axis=0) 
            sample.append(time_sample)

            all_samples.append(sample)


    # Calculating proportion
    train_ratio = 0.6
    val_ratio = 0.2
    # test_ratio = 0.2

    len_train =  int(len(all_samples) * train_ratio)
    len_val = int(len(all_samples) * val_ratio)
    # test_val = len(data) * test_ratio

    train = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[: len_train])]
    val = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[len_train: len_train + len_val])]
    test = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[len_train + len_val: ])]  
    
    train_x = np.concatenate(train[:-2], axis=-1)
    val_x = np.concatenate(val[:-2], axis=-1)
    test_x = np.concatenate(test[:-2], axis=-1)

    train_target = train[-2]
    val_target = val[-2]
    test_target = test[-2]

    train_timestamp = train[-1]
    val_timestamp = val[-1]
    test_timestamp = test[-1]


    (stats, train_x_norm, val_x_norm, test_x_norm) = z_score_normalize(train_x, val_x, test_x)
    """
    x is normalized by z-score and y is the true value
    """

    all_data = {
        'train': {
            'x': train_x_norm,
            'target': train_target,
            'timestamp': train_timestamp,
        },
        'val': {
            'x': val_x_norm,
            'target': val_target,
            'timestamp': val_timestamp,
        },
        'test': {
            'x': test_x_norm,
            'target': test_target,
            'timestamp': test_timestamp,
        },
        'stats': {
            '_mean': stats['_mean'],
            '_std': stats['_std'],
        }
    }

    if save:
        file = os.path.basename(graph_npz_file).split('.')[0]
        dirpath = os.path.dirname(graph_npz_file)
        filename = os.path.join(dirpath, file +  '_h' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks))
        print('save file:', filename)
        np.savez_compressed(filename,
                            train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                            train_timestamp=all_data['train']['timestamp'],
                            val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                            val_timestamp=all_data['val']['timestamp'],
                            test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                            test_timestamp=all_data['test']['timestamp'],
                            mean=all_data['stats']['_mean'], std=all_data['stats']['_std']
                            )
    return all_data


def get_data(num_of_depend, start_idx, num_for_predict, duration, points_per_hours):
    """
    num_of_depend: num_of_weeks, num_of_days, num_of_hours
    start_idx: the index at which the predicted value begins
    duration: hour: 1, day: 24, week: 7 * 24
    """
    slice = []

    for i in range(1, num_of_depend + 1):
        # Select how long to use as training data(week/day/hour)
        start = start_idx - duration * points_per_hours * i
        end = start + num_for_predict

        if start >= 0:
            slice.append((start, end))
        else:
            return None
        
    if len(slice) != num_of_depend:
        return None 
    
    return slice[::-1]
    

def data_sample(data, num_of_weeks, num_of_days, num_of_hours, num_for_predict, start_idx, points_per_hours=12):
    week_sample, day_sample, hour_sample = None, None, None

    if start_idx + num_for_predict > data.shape[0]:
        return week_sample, day_sample, hour_sample, None
    
    if num_of_weeks > 0:
        week_slice = get_data(num_of_weeks, start_idx, num_for_predict, 7 * 24, points_per_hours)

        if not week_slice:
            return None, None, None, None

        week_sample = np.concatenate([data[i: j] for i, j in week_slice], axis=0)

    if num_of_days > 0:
        day_slice = get_data(num_of_days, start_idx, num_for_predict, 24, points_per_hours)

        if not day_slice:
            return None, None, None, None

        day_sample = np.concatenate([data[i: j] for i, j in day_slice], axis=0)

    if num_of_hours > 0:
        hour_slice = get_data(num_of_hours, start_idx, num_for_predict, 1, points_per_hours)

        if not hour_slice:
            return None, None, None, None

        hour_sample = np.concatenate([data[i: j] for i, j in hour_slice], axis=0)

    target = data[start_idx: start_idx + num_for_predict]

    return week_sample, day_sample, hour_sample, target


def z_score_normalize(train, val, test):

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]  # ensure the num of nodes is the same
    mean = train.mean(axis=(0,1,3), keepdims=True)
    std = train.std(axis=(0,1,3), keepdims=True)

    def normalize(x):
        return (x - mean) / std

    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm
        
        
    
def inverse_z_score_normalize(z_score_normalized, mean, std):
    """
    Inverse transform the normalized data
    """
    return z_score_normalized * std + mean

# Config
parser = argparse.ArgumentParser(description='Import parameter')

parser.add_argument('--config', default='config/PEMS04.conf', type=str, help='configuration file')

args = parser.parse_args()
config = configparser.ConfigParser()
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

adj_file = data_config['adj_file']
graph_npz_file = data_config['graph_npz_file']

num_of_weeks = int(training_config['num_of_weeks'])
num_of_days = int(training_config['num_of_days'])
num_of_hours = int(training_config['num_of_hours'])

points_per_hour = int(data_config['points_per_hour'])
num_for_predict = int(data_config['num_for_predict'])

print(data_process('PEMS04', num_of_weeks, num_of_days, num_of_hours, num_for_predict, points_per_hour, save=False))
