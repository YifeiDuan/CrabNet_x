import sys
sys.path.append("/home/jupyter/YD/MTENCODER/CrabNet__/")

from utils.utils import CONSTANTS
from publication_CrabNet.benchmark_crabnet import *
from parameter_study_command import *

import zarr
import numpy as np
import pandas as pd
import math

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import matplotlib.patheffects as path_effects
import seaborn as sns

cons = CONSTANTS()

main_dir = "/home/jupyter/YD/MTENCODER/CrabNet__/"


"""
Helper functions for data_loader processing
"""
def get_datum(data_loader, idx=0):
    datum = data_loader.dataset[idx]
    return datum

def get_x(data_loader, idx=0):
    x = get_datum(data_loader, idx=idx)["data"]
    return x

def get_atomic_numbers(data_loader, idx=0):
    nums = get_x(data_loader, idx=idx).chunk(2)[0].detach().cpu().numpy()
    nums = nums.astype(int)
    return nums

def get_atomic_fracs(data_loader, idx=0):
    nums = get_x(data_loader, idx=idx).chunk(2)[1].detach().cpu().numpy()
    return nums

def get_target(data_loader, idx=0):
    target = get_datum(data_loader, idx=idx)["target"].detach().cpu().numpy()
    return target

def get_form(data_loader, idx=0):
    form = get_datum(data_loader, idx=idx)["formula"]
    return form




"""
Model loading function
"""
def load_model_for_infer_attn(config, model_path, gpu, data_dir, mat_prop):
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
        'data_dir': data_dir,
        'task_list': None,
        'eval': True,
        'wandb': False,
        'batchsize': None,
        'gpus': [gpu, [gpu]]
        }
    
    pretrained = base_config['transfer_model']

    classification_list = ["expt_is_metal", "glass", "mp_is_metal"]
    classification = False
    if mat_prop in classification_list:
        classification = True

    timestamp = generate_prefix()
    model_name =  model_path + '_' + mat_prop + '_' +  timestamp
    
    model = Model(CrabNet(base_config, compute_device=compute_device).to(compute_device),
                  config, model_name=f'{model_name}',
                  capture_flag=True,  verbose=True)

    # Load network with pretrained weights
    model.load_network(f'{pretrained}.pth')
    model.model_name = f'{model_name}'
    model.model.output_nn.apply(weight_reset) #thorben added

    # Apply BCEWithLogitsLoss to model output if binary classification is True
    if classification:
        model.classification = True
    
    return model




"""
Plotting functions
"""
def plot_attn_ptable(database, mat_prop, elem_sym="CPD", layer=0):
    """
    Plot the average (over all formuae in the dataset) attention weights that the specified elem_sym (default: CPD) token 
    attend to each element. The plot is in the shape of periodic table, with each element patch shaded with attention weights
    (normalized). Each plot corresponds to one head (call this function once to plot all heads) within the specified layer.

    example args: database="matbench", mat_prop="expt_gap". It specifies a particular dataset that has been inferred for attn
    """

    ######### Load Data ##########
    data_dir = main_dir + 'data/' + database

    model_dir = main_dir + 'models/'
    model_path = model_dir + "20240325_162526_tasks12/trained_models/Epoch_40"
    config = get_config(model_path)     # get the correct model config

    model = load_model_for_infer_attn(config, model_path, 0, data_dir, mat_prop)

    data = f'{data_dir}/{mat_prop}.csv'

    data_size = pd.read_csv(data).shape[0]
    batch_size = 2**round(np.log2(data_size)-4)
    if batch_size < 2**7:
        batch_size = 2**7
    if batch_size > 2**10:
        batch_size = 2**10

    model.load_data(data, model.classification, batch_size=batch_size)
    data_loader = model.data_loader

    
    ######### Define the ptable plot function ###########
    def plot(mat_prop, property_tracker, elem_sym="CPD", layer=0, head_option=0, option_text="a)"):
        ptable = pd.read_csv(main_dir + 'data/element_properties/ptable.csv')
        ptable.index = ptable['symbol'].values
        elem_tracker = ptable['count']
        n_row = ptable['row'].max()
        n_column = ptable['column'].max()

        elem_tracker = elem_tracker + pd.Series(property_tracker)

        # log_scale = True
        log_scale = False

        fig, ax = plt.subplots(figsize=(n_column, n_row))
        rows = ptable['row']
        columns = ptable['column']
        symbols = ptable['symbol']
        rw = 0.9  # rectangle width (rw)
        rh = rw  # rectangle height (rh)
        for row, column, symbol in zip(rows, columns, symbols):
        # plot one element after another
            row = ptable['row'].max() - row     # transform so that row=0 is the bottom of canvas
            cmap = sns.cm.rocket_r
            count_min = elem_tracker.min()
            count_max = elem_tracker.max()
            count_min = 0
            count_max = 1
            norm = Normalize(vmin=count_min, vmax=count_max)
            count = elem_tracker[symbol]    # "count" is actually the attn paid to that elem by CPD on average (across formulae)
            if log_scale:
                norm = Normalize(vmin=np.log(1), vmax=np.log(count_max))
                if count != 0:
                    count = np.log(count)
            color = cmap(norm(count))
            if np.isnan(count):
                color = 'silver'
            if row < 3:     # row = 1, 2 are Lanthanides and Actinides
                row += 0.5
            # element box
            rect = patches.Rectangle((column, row), rw, rh,
                                    linewidth=1.5,
                                    edgecolor='gray',
                                    facecolor=color,
                                    alpha=1)
            # plot element text
            text = plt.text(column+rw/2, row+rw/2, symbol,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=22,
                    fontweight='semibold', color='white')

            text.set_path_effects([path_effects.Stroke(linewidth=3,
                                                    foreground='#030303'),
                        path_effects.Normal()])

            ax.add_patch(rect)

        ### This is to plot the color bar in a heatmap manner ###
        granularity = 20
        for i in range(granularity):
            value = (1-i/(granularity-1))*count_min + (i/(granularity-1)) * count_max
            if log_scale:
                if value != 0:
                    value = np.log(value)
            color = cmap(norm(value))
            length = 9
            x_offset = 3.5
            y_offset = 7.8
            x_loc = i/(granularity) * length + x_offset
            width = length / granularity
            height = 0.35
            rect = patches.Rectangle((x_loc, y_offset), width, height,
                                    linewidth=1.5,
                                    edgecolor='gray',
                                    facecolor=color,
                                    alpha=1)

            if i in [0, 4, 9, 14, 19]:
                text = f'{value:0.2f}'
                if log_scale:
                    text = f'{np.exp(value):0.1e}'.replace('+', '')
                plt.text(x_loc+width/2, y_offset-0.4, text,
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontweight='semibold',
                        fontsize=20, color='k')

            ax.add_patch(rect)

        legend_title = f'{elem_sym}, Average Attention ({mat_prop})'
        plt.text(x_offset+length/2, y_offset+0.7,
                f'log({legend_title})' if log_scale else legend_title,
                horizontalalignment='center',
                verticalalignment='center',
                fontweight='semibold',
                fontsize=20, color='k')
        # add annotation for subfigure numbering
        plt.text(0.55, n_row+.1, option_text,
                fontweight='semibold', fontsize=38, color='k')
        ax.set_ylim(-0.15, n_row+.1)
        ax.set_xlim(0.85, n_column+1.1)

        # fig.patch.set_visible(False)
        ax.axis('off')

        plt.draw()
        save_dir = main_dir + f'explainability_mtencoder/figures/{mat_prop}/ptable'
        if save_dir is not None:
            fig_name = f'{save_dir}/{elem_sym}_attn_ptable_layer{layer}_head{head_option}.png'
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(fig_name, bbox_inches='tight', dpi=300)

        plt.pause(0.001)
        plt.close()
    


    ########### Process saved attn data and produce plot #############
    ##### token symbols and mapping to idx #####
    all_symbols_cpd = ['None', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
                'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
                'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
                'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
                'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
                'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
                'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 
                'CPD']

    idx_symbol_dict_cpd = {(i): sym for i, sym in enumerate(all_symbols_cpd)}
    symbol_idx_dict_cpd = {sym: (i) for i, sym in enumerate(all_symbols_cpd)}


    ##### load saved attn data #####
    attn_layerX = zarr.load(main_dir + f"explainability_mtencoder/data_save/{mat_prop}/attn_data_layer{layer}.zip")


    ##### prepare data containers; and config for the plotting #####
    other_dict = {i: [] for i in range(1, 120)}

    head_options = list(range(model.model.heads)) + ['average']
    option_texts = [chr(ord('`') + num+1)+")" for num in head_options[:-1]] + ['average']

    elem_Z = symbol_idx_dict_cpd[elem_sym]

    ##### plot average attn across the dataset for each specified head_option #####
    for idx_plot in range(len(head_options)):
        head_option = head_options[idx_plot]
        option_text = option_texts[idx_plot]

        ##### retrieve necessary info for the material data points, one by one
        for idx in range(len(data_loader.dataset)):
            if isinstance(head_option, int):
                map_data = attn_layerX[f"layer{layer}"][idx,0,head_option,:,:]
            else:   # head_option "average"
                map_data = np.mean(attn_layerX[f"layer{layer}"][idx,0,:,:,:], axis=0)    # average along the head dimension
                # attn_layerX[f"layer{layer}"][idx,0,:,:,:] is itself reduced from 5-dim to 3-dim, so the head dim changes from 2 to 0

            atom_fracs = get_atomic_fracs(data_loader, idx=idx)
            form = get_form(data_loader, idx=idx)
            atomic_numbers = get_atomic_numbers(data_loader, idx=idx).ravel().tolist()
            atoms = [idx_symbol_dict_cpd[num] for num in atomic_numbers]
            atom_presence = np.array(atom_fracs > 0)
            atom_presence[0] = True  # CPD token always considered present
            mask = atom_presence * atom_presence.T
            map_data = map_data * mask
            if elem_Z in atomic_numbers:
                row = atomic_numbers.index(elem_Z)
                for atomic_number in atomic_numbers:
                    if atomic_number == 0:
                        continue
                    col = atomic_numbers.index(atomic_number)
                    # get the raw attention value
                    other_dict[atomic_number].append(map_data[row, col])

        property_tracker = {all_symbols_cpd[key]: np.array(val).mean() for key, val
                            in other_dict.items()
                            if len(val) != 0}

        plot(mat_prop, property_tracker, elem_sym=elem_sym, layer=layer, head_option=head_option, option_text=option_text)




def plot_attn_map_formula(database, mat_prop, formula_id=0, elem_sym="CPD", layer=0):
    """
    Plot the average (over all formuae in the dataset) attention weights that the specified elem_sym (default: CPD) token 
    attend to each element. The plot is in the shape of periodic table, with each element patch shaded with attention weights
    (normalized). Each plot corresponds to one head (call this function once to plot all heads) within the specified layer.

    formula_id (int) is the identifier of any specific formula within the dataset (given by database + mat_prop)

    example args: database="matbench", mat_prop="expt_gap". It specifies a particular dataset that has been inferred for attn
    """

    ######### Load Data ##########
    data_dir = main_dir + 'data/' + database

    model_dir = main_dir + 'models/'
    model_path = model_dir + "20240325_162526_tasks12/trained_models/Epoch_40"
    config = get_config(model_path)     # get the correct model config

    model = load_model_for_infer_attn(config, model_path, 0, data_dir, mat_prop)

    data = f'{data_dir}/{mat_prop}.csv'

    data_size = pd.read_csv(data).shape[0]
    batch_size = 2**round(np.log2(data_size)-4)
    if batch_size < 2**7:
        batch_size = 2**7
    if batch_size > 2**10:
        batch_size = 2**10

    model.load_data(data, model.classification, batch_size=batch_size)
    data_loader = model.data_loader

    
    ######### Define the individual formula attn map plot function ###########
    def plot_map(mat_prop, formula, atoms, attn_map_data, elem_sym="CPD", layer=0, head_option=0, option_text="a)"):
        # remove all "None" from plotted map
        atoms = [atom for atom in atoms if atom!='None']
        attn_map_data = attn_map_data[:len(atoms), :len(atoms)]
        
        # plot the heatmap
        heatmap = sns.heatmap(attn_map_data, cmap=sns.cm.rocket_r, vmin=0.0, vmax=1.0,
                              xticklabels=atoms, yticklabels=atoms)
        plt.title(f"{option_text} Head {head_option}", loc="left")
        
        save_dir = main_dir + f'explainability_mtencoder/figures/{mat_prop}/attn_map/{formula}/'
        os.makedirs(save_dir, exist_ok=True)
        fig = heatmap.get_figure()
        fig.savefig(save_dir + f"layer{layer}_head{head_option}.jpg", bbox_inches="tight")
        
        plt.show()
        plt.close()
    


    ########### Process saved attn data and produce plot #############
    ##### token symbols and mapping to idx #####
    all_symbols_cpd = ['None', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
                'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
                'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
                'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
                'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
                'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
                'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 
                'CPD']

    idx_symbol_dict_cpd = {(i): sym for i, sym in enumerate(all_symbols_cpd)}
    symbol_idx_dict_cpd = {sym: (i) for i, sym in enumerate(all_symbols_cpd)}


    ##### load saved attn data #####
    attn_layerX = zarr.load(main_dir + f"explainability_mtencoder/data_save/{mat_prop}/attn_data_layer{layer}.zip")


    ##### prepare data containers; and config for the plotting #####
    head_options = list(range(model.model.heads)) + ['average']
    option_texts = [chr(ord('`') + num+1)+")" for num in head_options[:-1]] + ['average']

    elem_Z = symbol_idx_dict_cpd[elem_sym]

    ##### plot average attn across the dataset for each specified head_option #####
    for idx_plot in range(len(head_options)):
        head_option = head_options[idx_plot]
        option_text = option_texts[idx_plot]

        ##### retrieve necessary info for the material data points, one by one
        if isinstance(head_option, int):
            map_data = attn_layerX[f"layer{layer}"][formula_id,0,head_option,:,:]
        else:   # head_option "average"
            map_data = np.mean(attn_layerX[f"layer{layer}"][formula_id,0,:,:,:], axis=0)    # average along the head dimension
            # attn_layerX[f"layer{layer}"][idx,0,:,:,:] is itself reduced from 5-dim to 3-dim, so the head dim changes from 2 to 0

        atom_fracs = get_atomic_fracs(data_loader, idx=formula_id)
        form = get_form(data_loader, idx=formula_id)
        atomic_numbers = get_atomic_numbers(data_loader, idx=formula_id).ravel().tolist()
        atoms = [idx_symbol_dict_cpd[num] for num in atomic_numbers]
        atom_presence = np.array(atom_fracs > 0)
        atom_presence[0] = True  # CPD token always considered present
        mask = atom_presence * atom_presence.T
        map_data = map_data * mask
        
        plot_map(mat_prop, form, atoms, map_data, 
                elem_sym=elem_sym, layer=layer, 
                head_option=head_option, option_text=option_text)