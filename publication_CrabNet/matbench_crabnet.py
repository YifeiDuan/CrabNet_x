from pathlib import Path
import sys
import json
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime
import re

from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score

from crabnet.kingcrab import CrabNet
from crabnet.model import Model
#from utils.get_compute_device import get_compute_device
from torch import nn

#compute_device = get_compute_device(prefer_last=False)
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def get_model(config, mat_prop, i, classification, batch_size=None,
              transfer=None, data_dir=None, verbose=True):
    # Get the TorchedCrabNet architecture loaded
    gpu = f"cuda:{config['gpus'][0]}"
    print(gpu)
    compute_device = torch.device(gpu if torch.cuda.is_available() else "cpu")
    crabnet = CrabNet(config, compute_device=compute_device).to(compute_device)
    
    model = Model(crabnet, config, model_name=f'{mat_prop}{i}', verbose=False)

    # Train network starting at pretrained weights
    if transfer is not None:
        model.load_network(f'{transfer}.pth')
        model.model_name = f'{mat_prop}'
        #model.model.output_nn.apply(weight_reset) #thorben added

    # Apply BCEWithLogitsLoss to model output if binary classification is True
    if classification:
        model.classification = True

    # Get the datafiles you will learn from
    train_data = f'{data_dir}{mat_prop}/train{i}.csv'
    val_data = f'{data_dir}{mat_prop}/val{i}.csv'

    # Load the train and validation data before fitting the network
    data_size = pd.read_csv(train_data).shape[0]
    if batch_size is None:
        batch_size = 2**round(np.log2(data_size)-4)
        if batch_size < 2**7:
            batch_size = 2**7
        if batch_size > 2**12:
            batch_size = 2**12

    #batch_size = 2**5
    model.load_data(train_data, classification, batch_size=batch_size, train=True)
    print(f'training with batchsize {model.batch_size} '
          f'(2**{np.log2(model.batch_size):0.3f})')
    model.load_data(val_data, classification, batch_size=batch_size)

    # Set the number of epochs, decide if you want a loss curve to be plotted
    model.fit(epochs=config['max_epochs'], losscurve=False)

    # Save the network (saved as f"{model_name}.pth")
    #model.model_id = i
    #model.save_network()
    return model


def to_csv(output, save_name):
    # parse output and save to csv
    act, pred, formulae, uncertainty, _, _, _ = output
    df = pd.DataFrame([formulae, act, pred, uncertainty]).T
    df.columns = ['formula', 'actual', 'predicted', 'uncertainty']
    save_path = 'publication_predictions/mat2vec_matbench__predictions'
    # save_path = 'publication_predictions/onehot_matbench__predictions'
    # save_path = 'publication_predictions/random_200_matbench__predictions'
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(f'{save_path}/{save_name}', index_label='Index')


def load_model(model, mat_prop, i, classification, file_name, data_dir, verbose=True):
    # Load up a saved network.
    #model = Model(CrabNet(config, compute_device=compute_device).to(compute_device),
    #              config, model_name=f'{mat_prop}{i}', verbose=verbose)
    #model.load_network(f'{mat_prop}{i}.pth')

    # Check if classifcation task
    if classification:
        model.classification = True

    # Load the data you want to predict with
    data = f'{data_dir}{mat_prop}/{file_name}'
    # data is reloaded to model.data_loader
    model.load_data(data, classification, batch_size=2**9)
    return model


def get_results(model):
    output = model.predict(model.data_loader)  # predict the data saved here
    return model, output


def save_results(model, mat_prop, i, classification, file_name, data_dir, verbose=True):
    model = load_model(model, mat_prop, i, classification, file_name, data_dir, verbose=verbose)
    model, output = get_results(model)

    # Get appropriate metrics for saving to csv
    if model.classification:
        y_pred_tensor = torch.tensor(output['y_pred'], dtype=torch.float32)
        y_pred_sigmoid = torch.sigmoid(y_pred_tensor)
        output['y_pred'] = y_pred_sigmoid.numpy() #added 25.2
        print(f'\n{mat_prop}{i} y_pred: {output["y_pred"]}')

        mae = roc_auc_score(output['y_true'], output['y_pred'])
        print(f'\n{mat_prop}{i} ROC AUC: {mae:0.3f}')

        y_pred = (output['y_pred'] > 0.5).astype(int)
        precision = precision_score(output['y_true'], y_pred, average='macro')
        print(f'\n{mat_prop}{i} Precision: {precision:0.3f}')
        recall = recall_score(output['y_true'], y_pred, average='macro')
        print(f'\n{mat_prop}{i} Recall: {recall:0.3f}')
        mae= f1 = 2 * precision * recall / (precision + recall) #thorben
        print(f'\n{mat_prop}{i} F1: {f1:0.3f}')
    else:
        mae = np.abs(output['y_true'] - output['y_pred']).mean()
        print(f'\n{mat_prop}{i} mae: {mae:0.3g}')


    # save predictions to a csv
    fname = f'{mat_prop}_{file_name.replace(f"{i}.csv", "")}_output_cv{i}.csv'
    #to_csv(output, fname)
    
    data = model.inference(model.data_loader)
    scores = data[0][:, 0]
    uncertainties = data[0][:, 1]
    formulas = data[1]
    # Create a DataFrame
    df = pd.DataFrame({'formula': formulas, 'score': output['y_pred'], 'uncertainty': uncertainties})
    # Save DataFrame to CSV
    df.to_csv(f'output/a_lab/{mat_prop}_{i}_test_output.csv', index=False)

    return model, mae

def generate_versioned_filename(base_filename):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    versioned_filename = f"{base_filename}/evaluation_{timestamp}"
    return versioned_filename

def find_nb_splits(files_p_task):
    files_p_task = [file for file in files_p_task if file.endswith('.csv')]
    unique_numbers = set()
    for file_name in files_p_task:
        match = re.search(r'\d+', file_name)
        if match:
            number = int(match.group())
            unique_numbers.add(number)
    num_unique_numbers = len(unique_numbers)
    print(f'Number of SPLITS: {num_unique_numbers}')
    return num_unique_numbers


def run_model(base_filename, config):
        # Get data to benchmark on
    batch_size = config['batchsize'] 
    pretrained = config['transfer_model']
    data_dir = 'data/'+config['data_dir']+'/'
    #data_dir = 'data/materials_data/' #materials_data
    mat_props = os.listdir(data_dir)
    mat_props = [file for file in os.listdir(data_dir) if not file.endswith('.ipynb')]

    classification_list = ['pu_ltp', 'synthesizability', 'expt_is_metal', 'glass', 'mp_is_metal', 'gnome', 'synth_LiLaP', 'a_lab',
                           'synthesizability_PU_v1', 'pu_t', 'pu_new']
    print(f'training: {mat_props}')

    results_dict = {}
    results_dict['config'] = config

    for mat_prop in mat_props:
        classification = False
        if mat_prop in classification_list:
            classification = True
        # matbench provides 5 dataset train/val splits
        n_splits = find_nb_splits(os.listdir(data_dir+mat_prop))

        maes = []
        val_maes = []
        for i in range(n_splits):
            print(f'property: {mat_prop}, cv {i}')
            if pretrained:
                print('Pretrained: '+pretrained)
            model = get_model(config, mat_prop, i, classification, 
                              transfer=pretrained, data_dir=data_dir, verbose=True, batch_size=batch_size)
            print('=====================================================')
            print('TEST results:')

            model_test, val_mae = save_results(model, mat_prop, i, classification,
                                             f'val{i}.csv', data_dir, verbose=False)
            model_test, t_mae = save_results(model, mat_prop, i, classification,
                                             f'test{i}.csv', data_dir, verbose=False)
            maes.append(t_mae)
            val_maes.append(val_mae)
            del model #added
        results_dict[mat_prop] = np.mean(maes), np.std(maes)
        print(f'Average test mae: {np.mean(maes)}')
        #thorben added
        #with open('results_matbench.txt', 'a') as f:
        #    f.write('Task:' + mat_prop + ' -- Result = ' + str(np.mean(maes))+' std: '+str(np.std(maes))+'\n')
        print('=====================================================')

        # save res to json
    filename = generate_versioned_filename(base_filename)
    with open(filename + '.json', 'w') as f:
        json.dump(results_dict, f)
    return np.mean(maes), np.mean(val_maes)
