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


def get_model(config, tasks, model_name, split, classification, batch_size=None,
              transfer=None, verbose=True):
    # Get the TorchedCrabNet architecture loaded
    compute_device = torch.device(f"cuda:{config['gpus'][0]}" if torch.cuda.is_available() else "cpu")
    crabnet = CrabNet(config, compute_device=compute_device).to(compute_device)
    if torch.cuda.device_count() > 1:
        device_ids = [config['gpus'][1]]
        crabnet = nn.DataParallel(crabnet, device_ids=device_ids)
        print("Using", len(device_ids), "GPUs!")

    model = Model(crabnet, config, model_name=f'{model_name}', verbose=verbose)

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
        print(f'data/{config["task_dir"]}/{task}/train{split}.csv')
        train_paths.append(f'data/{config["task_dir"]}/{task}/train{split}.csv')
        val_paths.append(f'data/{config["task_dir"]}/{task}/val{split}.csv')

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


def load_model(model, config, tasks, split, classification, file_name, verbose=True):
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
        test_paths.append(f'data/{config["task_dir"]}/{task}/{file_name}{split}.csv')
       
    # Load the data you want to predict with
    #data = f'data/benchmark_data/{task}/{file_name}'
    # data is reloaded to model.data_loader
    model.load_data(test_paths, classification, batch_size=2 ** 9)
    return model


def get_results(model):
    output = model.predict(model.data_loader)  # predict the data saved here
    return model, output


def save_results(model, config, mat_prop, split, classification, file_name, verbose=True):
    model = load_model(model, config, mat_prop, split, classification, file_name, verbose=verbose)
    model, output = get_results(model)

    # Get appropriate metrics for saving to csv
    if isinstance(model.task_list, list):
        mae = []
        tasks = output[-1]
        for task in range(len(model.task_list)):
            task_target = output[0][np.where(tasks == task)]
            task_pred = output[1][np.where(tasks == task)]
            if model.classification[task]:
                 mae.append(np.around(roc_auc_score(task_target, task_pred), decimals=5))
            else:
                mae.append(np.around(mean_absolute_error(task_target, task_pred), decimals=5))
        print(f'\n{mat_prop}: {file_name} mae: {mae}')
    else:
        if model.classification:
            auc = roc_auc_score(output[0], output[1])
            print(f'\n{mat_prop} ROC AUC: {auc:0.3f}')
        else:
            mae = np.abs(output[0] - output[1]).mean()
            print(f'\n{mat_prop}: {file_name} mae: {mae:0.3g}')

    # save predictions to a csv
    fname = f'{mat_prop}_{file_name.replace(".csv", "")}_output.csv'
    # to_csv(output, fname)
    return model, mae


def generate_prefix():
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    return timestamp


def run_model(model_prefix, config):
    # get the models that have been pretrained
    pretrained = config['transfer_model']

    results_dict = {}
    results_dict['config'] = config
    classification =  config['classification']
    # for pretrained_model in all_models:
    # data_dir = 'data/benchmark_data'
    tasks = config['task_list']  # os.listdir(data_dir)
    classification_list = ['expt_is_metal']
    # print(f'model: {pretrained_model}')
    print(f'Multitask training on: {tasks}')
    # if mat_prop != pretrained_model:
    # todo classification list
    #if mat_prop in classification_list:
    #    classification = True

    test_list = [] 
    for split in range(5):
        timestamp = generate_prefix()
        model_name = model_prefix + '_' + '_'.join(tasks) + '_' + timestamp
        model = get_model(config, tasks, model_name, split, classification, verbose=False,
                        transfer=pretrained)  # 'pretrain:'+pretrained_model+'_second:'+mat_prop,
        print('=====================================================')
        print('calculating test mae')

        model_test, t_mae = save_results(model, config, tasks, split, classification,
                                        'test', verbose=False)

        model_test, val_mae = save_results(model, config, tasks, split, classification,
                                        'val', verbose=False)
        
        results_dict['_'.join(tasks)] = t_mae

        del model  # added
        test_list.append(t_mae)
        print('=====================================================')

        filename = 'results/' + model_prefix + '_' + timestamp
        with open(filename + '.json', 'w') as f:
            json.dump(results_dict, f)

    return test_list
