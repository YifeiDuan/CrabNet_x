from pathlib import Path
import sys
from datetime import datetime
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
print(sys.path)

import json
import os
import numpy as np
import pandas as pd
import torch
from torch import nn

from sklearn.metrics import roc_auc_score

from crabnet.kingcrab import CrabNet
from crabnet.model import Model
from utils.get_compute_device import get_compute_device

compute_device = get_compute_device(prefer_last=True)
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)


# %%
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def get_model(config, mat_prop, model_name, classification=False, batch_size=None,
              transfer=None, verbose=True):
    # Get the TorchedCrabNet architecture loaded
    model = Model(CrabNet(config, compute_device=compute_device).to(compute_device),
                  config, model_name=f'{model_name}', verbose=verbose)

    # Train network starting at pretrained weights
    if transfer is not None:
        model.load_network(f'{transfer}.pth')
        model.model_name = f'{model_name}'
        model.model.output_nn.apply(weight_reset) #thorben added

    # Apply BCEWithLogitsLoss to model output if binary classification is True
    if classification:
        model.classification = True

    # Get the datafiles you will learn from
    train_data = f'data/benchmark_data/{mat_prop}/train.csv'
    val_data = f'data/benchmark_data/{mat_prop}/val.csv'

    # Load the train and validation data before fitting the network
    data_size = pd.read_csv(train_data).shape[0]
    batch_size = 2**config['pretraining_batchsize'] #2**round(np.log2(data_size)-4)
    """if batch_size < 2**7:
        batch_size = 2**7
    if batch_size > 2**12:
        batch_size = 2**12
    # batch_size = 2**7"""
    model.load_data(train_data, batch_size=batch_size, train=True)
    print(f'training with batchsize {model.batch_size} '
          f'(2**{np.log2(model.batch_size):0.3f})')
    model.load_data(val_data, batch_size=batch_size)

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


def load_model(model, mat_prop, classification, file_name, verbose=True):
    # Load up a saved network.
    #model = Model(CrabNet(config, compute_device=compute_device).to(compute_device),
    #              config, model_name=f'{model_name}', verbose=verbose)
    #model.load_network(f'{model_name}.pth')

    # Check if classifcation task
    if classification:
        model.classification = True

    # Load the data you want to predict with
    data = f'data/benchmark_data/{mat_prop}/{file_name}'
    # data is reloaded to model.data_loader
    model.load_data(data, batch_size=2**9)
    return model

def get_results(model):
    output = model.predict(model.data_loader)  # predict the data saved here
    return model, output

def save_results(model, mat_prop, classification, file_name, verbose=True):
    model = load_model(model, mat_prop, classification, file_name, verbose=verbose)
    model, output = get_results(model)

    # Get appropriate metrics for saving to csv
    if model.classification:
        auc = roc_auc_score(output[0], output[1])
        print(f'\n{mat_prop} ROC AUC: {auc:0.3f}')
    else:
        mae = np.abs(output[0] - output[1]).mean()
        print(f'\n{mat_prop} mae: {mae:0.3g}')

    # save predictions to a csv
    fname = f'{mat_prop}_{file_name.replace(".csv", "")}_output.csv'
    #to_csv(output, fname)
    return model, mae

def generate_prefix():
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    return timestamp

def run_model(model_prefix, config, all_models):
    #get the models that have been pretrained
    pretrained = config['transfer_model']

    results_dict = {}
    results_dict['config'] = config
    #for pretrained_model in all_models:
    #data_dir = 'data/benchmark_data'
    mat_props = all_models # os.listdir(data_dir)
    classification_list = []
    # print(f'model: {pretrained_model}')
    print(f'training: {mat_props}')
    maes = []
    for mat_prop in mat_props:
        #if mat_prop != pretrained_model:
        classification = False
        if mat_prop in classification_list:
            classification = True
        print(f'property: {mat_prop}')

        timestamp = generate_prefix()
        model_name =  model_prefix + '_' + mat_prop + '_' +  timestamp
        model = get_model(config, mat_prop, model_name, classification, verbose=False, 
                          transfer=pretrained) #'pretrain:'+pretrained_model+'_second:'+mat_prop,
        print('=====================================================')
        print('calculating test mae')
        
        model_test, t_mae = save_results(model, mat_prop, classification,
                                        'test.csv', verbose=False)

        results_dict[mat_prop] = t_mae
        maes.append(t_mae)

        filename = 'models/' + model_name
        with open(filename + '.json', 'w') as f:
            json.dump(results_dict, f)

        del model #added
        #print('calculating val mae')
        #model_val, v_mae = save_results(model, mat_prop, classification,
        #                                'val.csv', verbose=False)
        #thorben added 
        #with open('results_benchmark_pretrained_models_meanpool.txt', 'a') as f:
        #    f.write('Task:' + mat_prop + ' -- Result = ' + str(np.mean(t_mae))+' std: '+str(np.std(t_mae))+'\n')
        print('=====================================================')
    
    filename = 'results/' + model_prefix + '_' +  timestamp
    with open(filename + '.json', 'w') as f:
        json.dump(results_dict, f)
        
    return np.mean(maes)
