from pathlib import Path
import json
from publication_CrabNet.matbench_crabnet import *
import os


def find_pth_file(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pth'):
                return os.path.join(root, file)
    raise AssertionError('No pth file found')


def get_config(model_path):
    parent_dir = Path(model_path).parent.parent
    json_file = 'config.json'

    config_file = os.path.join(parent_dir, json_file)
    config = load_json_from_path(config_file)
    return config


def load_json_from_path(path):
    """Load a JSON file from the given path."""
    with open(path, 'r') as file:
        data = json.load(file)
    return data


def run_eval(config, model_path, gpu):
    pth_file = find_pth_file(model_path)
    print(pth_file)

    base_config = \
        {'model':
            {'decoder': 'cpd', #opt: cpd, meanpool, roost
            'd_model': config['model']['d_model'],
            'N': config['model']['N'],
            'encoder_ff': config['model']['encoder_ff'],
            'heads': config['model']['heads'],
            'residual_nn_dim': config['model']['residual_nn_dim'], #, 256, 128],
            'branched_ffnn': config['model']['branched_ffnn'],
            'dropout': 0,
            'special_tok_zero_fracs': False,  #opt: both, cpd, eos - only set if special tok is used!!
            'numeral_embeddings': config['model']['numeral_embeddings']}, 

        'trainer':
            {'swa_start': 0,
            'n_elements': 'infer',
            'masking': False,
            'fraction_to_mask': 0,
            'cpd_token': config['trainer']['cpd_token'],
            'eos_token': config['trainer']['eos_token'],
            'base_lr': 5e-5,
            'mlm_loss_weight': None, 
            'delay_scheduler': 0},

        'transfer_model': pth_file.split('.pth')[0],
        'max_epochs': 4,
        'sampling_prob': None,
        'task_types': None,
        'save_every': 100000, 
        'data_dir': task_dir,
        'task_list': None,
        'eval': True,
        'wandb': False,
        'batchsize': None,
        'gpus': [gpu, [gpu]]
        }
 
    run_model(model_path, base_config)
    return


if __name__ == '__main__':
    #model_path =  'results/models/20240207_215511_tasks11/trained_models/Epoch_20'
    #model_path =  'results/models/20240207_170355_tasks1/trained_models/Epoch_20'
    model_path = 'results/models/20240211_040852_tasks11/trained_models/Epoch_60'
    #model_path = 'results/models/20240219_101942_tasks11/trained_models/Epoch_100'
    #model_path = 'results/models/20240126_002044_tasks2/trained_models/Epoch_20'
    #model_path = 'results/models/20240124_234201_tasks2/trained_models/Epoch_160'
    #model_path = 'results/models/20240115_230931_tasks2/trained_models/Epoch_160'
    task_dir = 'pu_5_fold' #'PU_collab', 'matbench_cv
    gpu = 0

    config = get_config(model_path)
    run_eval(config, model_path, gpu)