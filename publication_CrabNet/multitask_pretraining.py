from pathlib import Path
import sys
from datetime import datetime

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import json
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import hashlib

from sklearn.metrics import roc_auc_score

from crabnet.kingcrab import CrabNet
from crabnet.model import Model

from sklearn.metrics import mean_absolute_error

RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)


# %%
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def get_model(config, tasks, model_name, model_hash, classification=None, batch_size=None,
              transfer=None, verbose=True):
    # Get the TorchedCrabNet architecture loaded
    gpu = f"cuda:{config['gpus'][0]}"
    print(gpu)
    compute_device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    crabnet = CrabNet(config, compute_device=compute_device).to(compute_device)
    if torch.cuda.device_count() > 1:
        device_ids = config['gpus'][1]
        crabnet = nn.DataParallel(crabnet, device_ids=device_ids)
        print("Using", len(device_ids), "GPUs!")

    model = Model(crabnet, config, model_name=f'{model_name}', model_id= model_hash, verbose=verbose)

    # Train network starting at pretrained weights
    if transfer is not None:
        model.load_network(f'{transfer}.pth')
        model.model_name = f'{model_name}'
        
        #for net in model.output_nn:
        #    net.apply(weight_reset)  # thorben added

    # Apply BCEWithLogitsLoss to model output if binary classification is True
    #if classification:
    #    model.classification = True

    train_paths = []
    val_paths = []
    for task in tasks:
        # Get the datafiles you will learn from
        train_paths.append(f'data/{config["task_dir"]}/{task}/train.csv')
        val_paths.append(f'data/{config["task_dir"]}/{task}/val.csv')

    # Load the train and validation data before fitting the network
    batch_size = 2 ** config['pretraining_batchsize']  # 2**round(np.log2(data_size)-4)

    model.load_data(train_paths, classification, batch_size=batch_size, train=True)
    print(f'training with batchsize {model.batch_size} '
        f'(2**{np.log2(model.batch_size):0.3f})')
    model.load_data(val_paths, classification, batch_size=batch_size)

    # Set the number of epochs, decide if you want a loss curve to be plotted
    model.fit(epochs=config['max_epochs'], losscurve=False)

    # Save the network (saved as f"{model_name}.pth")
    model.save_network()
    return model


def to_csv(output, save_name):
    # parse output and save to csv
    act, pred, formulae, uncertainty = output
    df = pd.DataFrame([formulae, act, pred, uncertainty]).T
    df.columns = ['formula', 'actual', 'predicted', 'uncertainty']
    save_path = 'publication_predictions/mat2vec_benchmark__predictions'
    # save_path = 'publication_predictions/onehot_benchmark__predictions'
    # save_path = 'publication_predictions/random_200_benchmark__predictions'
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(f'{save_path}/{save_name}', index_label='Index')


def load_model(model, config, tasks, classification, file_name, verbose=True):
    # Load up a saved network.
    # model = Model(CrabNet(config, compute_device=compute_device).to(compute_device),
    #              config, model_name=f'{model_name}', verbose=verbose)
    # model.load_network(f'{model_name}.pth')

    # Check if classifcation task
    if classification:
        model.classification = True

    test_paths = []
    for task in tasks:
        # Get the datafiles you will learn from
        test_paths.append(f'data/{config["task_dir"]}/{task}/{file_name}.csv')
       
    # Load the data you want to predict with
    #data = f'data/benchmark_data/{task}/{file_name}'
    # data is reloaded to model.data_loader
    model.load_data(test_paths,classification, batch_size=2 ** 9)
    return model


def save_results(model, config, mat_prop, file_name, classification=None, verbose=True):
    model = load_model(model, config, mat_prop, classification, file_name, verbose=verbose)
    pred_tracker = model.predict(model.data_loader) #act_t, pred_t, _, _, mlm_loss, tasks

    mae_t = pred_tracker.loss
    task_samples_t = pred_tracker.task_samples
    
    print(f'\n{mat_prop}: {file_name} mae: {mae_t}')
    # save predictions to a csv
    fname = f'{mat_prop}_{file_name.replace(".csv", "")}_output.csv'
    # to_csv(output, fname)
    return model, mae_t


def generate_prefix():
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    return timestamp


def get_tasks_hash(tasks):
    # Concatenate the tasks and get their MD5 hash
    joined_tasks = '_'.join(tasks)
    return hashlib.md5(joined_tasks.encode()).hexdigest()


def run_model(directory, config):
    # get the models that have been pretrained
    print(config)
    pretrained = config['transfer_model']
    classification = config['task_types']

    results_dict = {}
    results_dict['config'] = config


    tasks = config['task_list']  # os.listdir(data_dir)
    print(f'Multitask training on: {tasks}')

    timestamp = generate_prefix()
    model_hash = get_tasks_hash(tasks)
    #model_name = f"{directory}"
    model_name = directory 
    print(model_name)

    model = get_model(config, tasks, model_name, model_hash, verbose=False,
                      transfer=pretrained)  # 'pretrain:'+pretrained_model+'_second:'+mat_prop,
    print('=====================================================')
    model_test, val_mae = save_results(model, config, tasks,
                                     'val', verbose=False)

    model_test, t_mae = save_results(model, config, tasks,
                                     'test', verbose=False)
    
    results_dict['_'.join(tasks)] = t_mae

    del model  # added
    # print('calculating val mae')
    # model_val, v_mae = save_results(model, mat_prop, classification,
    #                                'val.csv', verbose=False)
    # thorben added
    # with open('results_benchmark_pretrained_models_meanpool.txt', 'a') as f:
    #    f.write('Task:' + mat_prop + ' -- Result = ' + str(np.mean(t_mae))+' std: '+str(np.std(t_mae))+'\n')
    print('=====================================================')

    filename = f'{directory}/pretraining_{model_hash}'
    with open(filename + '.json', 'w') as f:
        json.dump(results_dict, f)

    return val_mae, t_mae
