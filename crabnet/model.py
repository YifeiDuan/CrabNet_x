import os
from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from sklearn.metrics import mean_absolute_error, roc_auc_score

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CyclicLR

from utils.utils import (Lamb, Lookahead, RobustL1, BCEWithLogitsLoss,
                         EDM_CsvLoader, Scaler, DummyScaler, count_parameters)
from utils.pu_loss import PULoss
from utils.get_compute_device import get_compute_device
from utils.optim import SWA

#from clearml import Logger

from torch.utils.data import DataLoader
from utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from utils.custom_weighted_random_sampler import MultiDatasetWrapper
from utils.prediction_tracker import PredictionTracker
import logging
import wandb


# %%
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32


# %%
class Model():
    def __init__(self,
                 model,
                 config,
                 model_name='UnnamedModel',
                 model_id=None, 
                 capture_every=False,
                 verbose=True,
                 drop_unary=True,
                 scale=True):

        self.model = model
        self.model_id = model_id
        print(config)
        self.save_every = config['save_every']
        self.eval_model = config['eval']
        self.wandb = config['wandb']
        self.decoder = config['model']['decoder']
        self.task_types = config['task_types']
        self.sampling_prob = config['sampling_prob']
        self.config = config['trainer']
        self.task_list = config['task_list']

        self.mlm_loss_weight = self.config['mlm_loss_weight']
        self.n_elements = self.config['n_elements']
        self.cpd_token = self.config['cpd_token']
        self.eos_token = self.config['eos_token']
        self.base_lr = self.config['base_lr']
        self.masking = self.config['masking']
        self.delay_scheduler = self.config['delay_scheduler']
        self.swa_start = self.config['swa_start']  # start at (n/2) cycle (lr minimum)
        if self.masking:
            print('Using MASKING!')
        self.fraction_to_mask = self.config['fraction_to_mask']

        self.model_name = model_name
        self.data_loader = None
        self.train_loader = None
        self.classification = False
        self.compute_device = next(getattr(model, 'module', model).parameters()).device

        self.fudge = 0.02  # expected fractional tolerance (std. dev) ~= 2%
        self.capture_every = capture_every
        self.verbose = verbose
        self.drop_unary = drop_unary
        self.scale = scale
        #if self.compute_device is None:
        #    self.compute_device = get_compute_device()
        self.capture_flag = False
        self.formula_current = None
        self.act_v = None
        self.pred_v = None
        
        if self.verbose:
            print('\nModel architecture: d_model, N, heads')
            print(f'{self.model.d_model}, '
                  f'{self.model.N}, {self.model.heads}')
            #print(f'Running on compute device: {self.compute_device}')
            print(f'Model size: {count_parameters(self.model)} parameters\n')
        if self.capture_every is not None:
            print(f'capturing attention tensors every {self.capture_every}') 

        # assertions
        if self.decoder != 'cpd':
            assert not ("MLM" in self.task_types and self.fraction_to_mask == 0), "fraction_to_mask cannot be 0 when task type contains MLM"
            #self.sigmas = nn.Parameter(torch.rand(len(self.task_list), 1)).to(self.compute_device)
            allowed_values = {'REG', 'MLM', 'BIN', 'MCC', 'MREG'}
            # Assert if any value in my_list is not in allowed_values
            assert all(value in allowed_values for value in self.task_types), "There's a task type specified that is not allowed!"

        if not self.eval_model:
            if self.wandb:
                wandb.init(project='materials representations', name=self.model_name)
            self.init_logger(f'{self.model_name}/{self.model_id}.log')
            self.log_config(config)
    
    def init_logger(self, log_file):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Create a file handler and set the log level
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create a formatter and add it to the file handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add the file handler to the logger
        self.logger.addHandler(file_handler)
        return 

    def log_config(self, config):
        self.logger.info("Logging config:")
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")

    def encode_materials(self):
        # Estimate total number of samples (assuming last batch might be smaller)
        total_samples = len(self.data_loader)
        embeddings = []
        all_formulas = []

        for i, data in enumerate(self.data_loader):
                X, y, formula = data['data'], data['targets'], data['formula']
                if self.capture_flag:
                    self.formula_current = None
                    # HACK for PyTorch v1.8.0
                    # this output used to be a list, but is now a tuple
                    if isinstance(formula, tuple):
                        self.formula_current = list(formula)
                    elif isinstance(formula, list):
                        self.formula_current = formula.copy()

                input = {}
                src, frac = X.squeeze(-1).chunk(2, dim=1)

                
                """src_masked, position_mask, real_mask_ratio = self.mask_seq(src)
                input['position_mask'] = position_mask
                input['src_masked'] = src_masked.to(self.compute_device,
                                        dtype=torch.long,
                                        non_blocking=True)"""
                    
                input['src_masked'] = src.to(self.compute_device,
                             dtype=torch.long,
                             non_blocking=True)
                input['frac'] = frac.to(self.compute_device,
                               dtype=data_type_torch,
                               non_blocking=True)

                res = self.model.forward(input, embeddings=True)

                embeddings.append(res.cpu().detach().numpy())
                all_formulas.append(formula)

        return embeddings, all_formulas

    def load_data(self, file_name, classification, batch_size=2 ** 9, train=False):
        self.classification = classification
        self.batch_size = batch_size
        inference = not train
        data_loaders = EDM_CsvLoader(csv_data=file_name,
                                     cpd_token=self.cpd_token,
                                     eos_token=self.eos_token,
                                     masking=self.masking, 
                                     fraction_to_mask=self.fraction_to_mask, 
                                     batch_size=batch_size,
                                     n_elements=self.n_elements,
                                     inference=inference,
                                     verbose=self.verbose,
                                     drop_unary=self.drop_unary,
                                     scale=self.scale)
        data_loader = data_loaders.get_data_loaders(inference=inference)

        #print(f'loading data with up to {data_loaders.n_elements:0.0f} '
       #       f'elements in the formula')
        # update n_elements after loading dataset
        #self.n_elements = data_loader.dataset[0][0].shape[0]/2

        if train:
            if type(file_name) is str:
                y = data_loader.dataset.data[1]
                self.train_len = len(y)
                if self.classification:
                    self.scaler = DummyScaler(y)
                else:
                    self.scaler = Scaler(y)
                self.train_loader = data_loader
            elif type(file_name) is list:
                self.scaler = []
                for k in range(len(data_loader.dataset.datasets)):
                    y = data_loader.dataset.datasets[k].data[1]
                    if self.task_types[k] == 'REG':
                        self.scaler.append(Scaler(y))
                    if self.task_types[k] == 'MMREG':
                        scalers_ = [Scaler(y[:, i]) for i in range(y.shape[-1])]
                        self.scaler.append(scalers_)
                    else:
                        self.scaler.append(DummyScaler(y))
                self.train_len = len(data_loader.dataset)
                #self.train_loader = data_loader

                dataset_lens = [len(x) for x in data_loader.dataset.datasets]
                num_samples = int(sum([dataset*prob for dataset, prob in zip(dataset_lens, self.sampling_prob)]))

                self.sampler = CustomWeightedRandomSampler(datasets=data_loader.dataset.datasets, weights=self.sampling_prob, num_samples=num_samples)
                self.train_loader = DataLoader(MultiDatasetWrapper(data_loader.dataset.datasets, self.sampling_prob), batch_size=data_loader.batch_size, 
                    sampler=self.sampler, num_workers=1)
                
        else:
            self.data_loader = data_loader

    
    """def mask_seq(self, src):
        src = src.clone()
        position_mask = torch.zeros_like(src)
        # Loop over the first dimension of the array
        for i in range(src.shape[0]):
            # Find the indices of the non-zero elements in the current entry
            src_numpy = src[i, :].cpu().numpy()  # Converting the tensor to numpy array
            non_zero_indices = np.argwhere((src_numpy != 0) & (src_numpy < 119)).flatten()

            # If there's only one non-zero element, don't zero it
            if len(non_zero_indices) == 1:
                continue

            # Compute the maximum number of non-zero elements to zero, ensuring that at least one remains unmasked
            max_indices_to_zero = max(1, len(non_zero_indices) - 1)
            indices_to_zero = [0]*(max_indices_to_zero + 1)

            # Randomly choose a subset of indices to zero
            while len(indices_to_zero) > max_indices_to_zero:
                indices_to_zero = non_zero_indices[np.random.rand(*non_zero_indices.shape) < self.fraction_to_mask]
    
            # Set the chosen indices to zero
            for index in indices_to_zero:
                probability = 1/3
                random_number = np.random.rand()

                # Check the random number to determine which option to choose
                if random_number < probability:
                    # Option 1 keep token
                    position_mask[i, index] = 1
                elif random_number < 2 * probability:
                    # Option 2 choose random token
                    # random between 1 and 118 as upper
                    src[i, index] = np.random.randint(1, 119)
                    position_mask[i, index] = 1
                else:
                    # Option 3 MASK
                    src[i, index] = 121
                    position_mask[i, index] = 1
        
        # track the masking ratio
        # nb_unmasked = np.where((src != 0) & (src < 119))[0].shape[0]
        # TypeError: can't convert cuda:1 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
        nb_unmasked = np.where((src.cpu() != 0) & (src.cpu() < 119))[0].shape[0]
        nb_masked = np.where(position_mask.cpu() == 1)[0].shape[0]

        if src.shape[0] > 0:
            real_mask_ratio = np.where((nb_unmasked+nb_masked) != 0, nb_masked / (nb_unmasked+nb_masked), 0)
        return src, position_mask"""

    def mask_seq(self, src):
        src = src.clone()
        position_mask = torch.zeros_like(src)

        # Assuming src is a 2D tensor [batch_size, seq_length]
        batch_size, seq_length = src.shape
        
        # Mask of valid positions to consider (non-zero and < 119)
        valid_positions = (src != 0) & (src < 119)
        
        # Calculate the number of valid positions per example in the batch
        valid_counts = valid_positions.sum(dim=1)
        
        # Generate random masks based on the fraction to mask, ensuring at least one position remains unmasked
        probabilities = torch.rand_like(src, dtype=torch.float)
        mask = (probabilities < self.fraction_to_mask) & valid_positions
        
        # Ensure at least one token remains unmasked by finding rows with all masked and unmasking one randomly
        for i in range(batch_size):
            if valid_counts[i] > 0 and mask[i].sum() == valid_counts[i]:  # All valid positions are masked
                unmask_idx = torch.where(valid_positions[i])[0][torch.randint(0, valid_counts[i].item(), (1,))]
                mask[i, unmask_idx] = False
        
        # Apply the mask with random choices
        random_choices = torch.rand_like(src, dtype=torch.float)
        # Option 1: Keep token (do nothing, handled by not masking these positions)
        # Option 2: Random token
        random_token_mask = (random_choices < 2/3) & (random_choices >= 1/3) & mask
        src[random_token_mask] = torch.randint(1, 119, size=src[random_token_mask].shape, device=src.device).float()
        # Option 3: MASK token
        mask_token_mask = (random_choices >= 2/3) & mask
        src[mask_token_mask] = 121
        
        # Update position mask where changes were made
        position_mask[mask] = 1
        
        # Compute mask ratio if needed
        # Here you can compute the ratio directly with PyTorch operations if necessary
        
        return src, position_mask

    def scale_multi_task(self, task_indices, y, scale=True):
        for i, task in enumerate(self.task_types):
            if task == 'MMREG':
                mask = (task_indices == i)
                for k in range(y.shape[-1]):
                    # Apply scaling or unscaling for each dimension `k` of task `i`
                    if scale:
                        # Ensure consistent access to scalers for scaling
                        scaled_values = self.scaler[i][k].scale(y[mask, k])
                    else:
                        # Ensure consistent access to scalers for unscaling
                        scaled_values = self.scaler[i][k].unscale(y[mask, k])
                    y[mask, k] = scaled_values
            if task == 'REG':
                mask = task_indices == i
                if scale:
                    y[mask] = self.scaler[i].scale(y[mask])
                else:
                    y[mask] = self.scaler[i].unscale(y[mask])
        return y
    
    def handle_mask(self, src, data, input):
        src_masked = src.clone().to(self.compute_device)  # Move to device at the beginning
        position_mask = torch.zeros_like(src, device=self.compute_device)  # Direct initialization on GPU

        inds = np.where(np.array(self.task_types) == 'MLM')[0]
        # Assuming mask_seq can be optimized or vectorized further
        for ind in list(inds):
            mask = input['tasks'] == ind
            if torch.sum(mask) > 0:
                src_masked_, position_mask_ = self.mask_seq(src_masked[mask])
                src_masked[mask] = src_masked_
                position_mask[mask] = position_mask_

        input['position_mask'] = position_mask
        input['src_masked'] = src_masked.long()  # Assuming src_masked is already on the correct device
        return input


    def train(self):
        self.model.train()
        ti = time()
        minima = []
        if isinstance(self.task_list, list):
            self.losses = np.array([0.0]*len(self.task_list))
        for i, data in enumerate(self.train_loader):
            X, y_true, formula = data['data'], data['targets'], data['formula']

            if isinstance(self.task_list, list):
                task_ind = data['tasks']
                y_true = self.scale_multi_task(task_ind, y_true, scale=True)

            else:
                y_true = self.scaler.scale(y_true)

            src, frac = X.squeeze(-1).chunk(2, dim=1)
            # add a small jitter to the input fractions to improve model
            # robustness and to increase stability
            # frac = frac * (1 + (torch.rand_like(frac)-0.5)*self.fudge)  # uniform

            #define input
            input = {}

            if self.cpd_token:
                frac[:, 0] = 0
            if self.eos_token:
                frac[:, -1] = 0

            if 'tasks' in data:
                input['tasks'] = task_ind.to(self.compute_device,
                           dtype=data_type_torch,
                           non_blocking=True)
            
            input = self.handle_mask(src, data, input)


            frac = frac * (1 + (torch.randn_like(frac)) * self.fudge)  # normal
            frac = torch.clamp(frac, 0, 1)

            frac[src == 0] = 0
            frac = frac / frac.sum(dim=1).unsqueeze(1).repeat(1, frac.shape[-1])
            input['src'] = src.to(self.compute_device,
                         dtype=torch.long,
                         non_blocking=True)

            input['frac'] = frac.to(self.compute_device,
                           dtype=data_type_torch,
                           non_blocking=True)

            y_true = y_true.to(self.compute_device,
                     dtype=data_type_torch,
                     non_blocking=True)

            ##################################
            # Force evaluate dataset so that we can capture it in the hook
            # here we are using the train_loader, but we can also use
            # general data_loader
            if self.capture_every == 'step':
                # print('capturing every step!')
                # print(f'data_loader size: {len(self.data_loader.dataset)}')
                self.capture_flag = True
                # (act, pred, formulae, uncert)
                self.act_v, self.pred_v, _, _, _, tasks = self.predict(self.data_loader)
                self.capture_flag = False
            ##################################
     
            res = self.model.forward(input)

            if isinstance(self.task_list, list):
                #prediction, uncertainty = res['output'].chunk(2, dim=-1)
                #print(mean_absolute_error(res[0]['y_pred'][:,0].cpu().detach().numpy(), y_true.cpu().detach().numpy()))
                loss, loss_list = self.MTLoss(res, y_true, data['tasks'])
                #print(loss)

                #self.losses += np.array([loss_i.item() for loss_i in loss_list])
                #loss = self.criterion(prediction.view(-1),
                #                  uncertainty.view(-1),
                #                  y.view(-1))
            else:
                prediction, uncertainty = res.chunk(2, dim=-1)
                loss = self.criterion(prediction.view(-1),
                                  y_true.view(-1), 
                                  #uncertainty.view(-1), #modified 24.2
                                  )

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.stepping and self.epoch >= self.delay_scheduler:
                self.lr_scheduler.step()
            
            # Assuming `optimizer` is your optimizer instance
            #learning_rate = self.optimizer.param_groups[0]['lr']
            #print("Current Learning Rate:", learning_rate)


            swa_check = (self.epochs_step * self.swa_start - 1)
            epoch_check = (self.epoch + 1) % (2 * self.epochs_step) == 0
            learning_time = epoch_check and self.epoch >= swa_check
            if learning_time:
                with torch.no_grad():
                    pred_res_train = self.predict(self.data_loader)
                if isinstance(self.task_list, list):
                    continue #todo
                else:
                    pred_v = pred_res_train['y_pred'] 
                    act_v = pred_res_train['y_true']
                    if self.classification:
                        # Use BCE loss for early stopping so the model does not break training
                        pred_v_tensor = torch.tensor(pred_v, dtype=torch.float32)
                        act_v_tensor = torch.tensor(act_v, dtype=torch.float32)
                        with torch.no_grad():
                            #mae_v = nn.functional.binary_cross_entropy_with_logits(pred_v_tensor, act_v_tensor)
                            mae_v = self.criterion(pred_v_tensor, act_v_tensor)
                            mae_v = mae_v.cpu()
                            mae_v = mae_v.detach().numpy()
                        mae_v = nn.functional.binary_cross_entropy_with_logits(pred_v_tensor, act_v_tensor)
                        mae_v = self.criterion(pred_v_tensor, act_v_tensor)
                        mae_v = mae_v.cpu()
                        mae_v = mae_v.detach().numpy()
                    else:
                        mae_v = mean_absolute_error(act_v, pred_v)
                
                self.optimizer.update_swa(np.sum(mae_v)) # take loss instead for multi-task
                minima.append(self.optimizer.minimum_found)

        if learning_time and not any(minima):
            self.optimizer.discard_count += 1
            print(f'Epoch {self.epoch} failed to improve.')
            print(f'Discarded: {self.optimizer.discard_count}/'
                  f'{self.discard_n} weight updates ♻🗑️')

        datalen = len(self.train_loader.dataset)
        # print(f'training speed: {datalen/dt:0.3f}')

    def fit(self, epochs=None, checkin=None, losscurve=False):
        assert_train_str = 'Please Load Training Data (self.train_loader)'
        assert_val_str = 'Please Load Validation Data (self.data_loader)'
        assert self.train_loader is not None, assert_train_str
        assert self.data_loader is not None, assert_val_str
        self.loss_curve = {}
        self.loss_curve['train'] = []
        self.loss_curve['val'] = []

        # change epochs_step
        # self.epochs_step = 10
        self.epochs_step = 1
        self.step_size = self.epochs_step * len(self.train_loader)
        print(f'stepping every {self.step_size} training passes,',
              f'cycling lr every {self.epochs_step} epochs')
        if epochs is None:
            n_iterations = 1e4
            epochs = int(n_iterations / len(self.data_loader))
            print(f'running for {epochs} epochs')
        if checkin is None:
            checkin = self.epochs_step * 2
            print(f'checkin at {self.epochs_step * 2} '
                  f'epochs to match lr scheduler')
        if epochs % (self.epochs_step * 2) != 0:
            # updated_epochs = epochs - epochs % (self.epochs_step * 2)
            # print(f'epochs not divisible by {self.epochs_step * 2}, '
            #       f'updating epochs to {updated_epochs} for learning')
            updated_epochs = epochs
            epochs = updated_epochs

        self.step_count = 0
        self.criterion = RobustL1
        
        if self.classification:
            print("Using BCE loss for classification task")
            self.criterion = BCEWithLogitsLoss

        if self.masking:
            print("Using CE loss for mask prediction")
            self.criterion = torch.nn.CrossEntropyLoss()

        self.criterion = PULoss(prior = .2)

        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = SWA(optimizer)
         
        # Apply the fix for the SWA optimizer issue
        self.optimizer.defaults = self.optimizer.optimizer.defaults
        self.optimizer.param_groups = self.optimizer.optimizer.param_groups
        self.optimizer.state = self.optimizer.optimizer.state

        lr_scheduler = CyclicLR(self.optimizer,
                                base_lr=self.base_lr,
                                max_lr=6e-3,
                                cycle_momentum=False,
                                step_size_up=self.step_size)

        desired_lr = 20 * self.base_lr  
        for param_group in base_optim.param_groups:
            param_group['lr'] = desired_lr

        self.lr_scheduler = lr_scheduler
        self.stepping = True #lr change
        self.lr_list = []
        self.xswa = []
        self.yswa = []
        self.discard_n = 3

        for epoch in range(epochs):
            self.epoch_start_time = time()
            ti = time()

            if self.decoder == 'multi-task':
                self.sampler.reshuffle_indices()
            self.epoch = epoch
            self.epochs = epochs
            self.train()

            if (self.epoch != 0) and (self.epoch % self.save_every == 0):
                self.save_network()
            # print(f'epoch time: {(time() - ti):0.3f}')
            self.lr_list.append(self.optimizer.param_groups[0]['lr'])
            #print(self.lr_list)

            ##################################
            # Force evaluate dataset so that we can capture it in the hook
            # here we are using the train_loader, but we can also use
            # general data_loader
            if self.capture_every == 'epoch':
                # print('capturing every epoch!')
                # print(f'data_loader size: {len(self.data_loader.dataset)}')
                self.capture_flag = True
                # (act, pred, formulae, uncert)
                print('due to capture every:')
                pred_res_train = self.predict(self.data_loader)
                self.capture_flag = False
            ##################################

            if (epoch + 1) % checkin == 0 or epoch == epochs - 1 or epoch == 0:
                ti = time()

                # train loss
                with torch.no_grad():
                    pred_res_train = self.predict(self.train_loader)
                
                if isinstance(self.task_list, list):
                    mae_t = pred_res_train.loss
                    task_samples_t = pred_res_train.task_samples
            
                else:
                    pred_t = pred_res_train['y_pred']
                    act_t = pred_res_train['y_true']
                    formula_t = pred_res_train['formula']
                    mae_t = mean_absolute_error(act_t, pred_t)
                self.loss_curve['train'].append(mae_t)

                # val loss
                with torch.no_grad():
                     pred_res_val = self.predict(self.data_loader)

                if isinstance(self.task_list, list):
                    mae_v = pred_res_val.loss
                    task_samples_v = pred_res_val.task_samples
            
                else:
                    pred_v = pred_res_val['y_pred']
                    act_v = pred_res_val['y_true']
                    formula_v = pred_res_val['formula']
                    mae_v = mean_absolute_error(act_v, pred_v)
                self.loss_curve['val'].append(mae_v)
                
                epoch_str = f'Epoch: {epoch}/{epochs} ---'
                dt = time() - ti
                # from clearml import Logger
                # Logger.current_logger().report_scalar(
                   # graph='metric',
                   # series='variant',
                   # value=13.37,
                   # iteration=counter
                # )

                if isinstance(self.task_list, list):
                    val_txt = [round(x, 3) for x in self.loss_curve["val"][-1]]
                    val_str = f'val mae: {val_txt}'
                    """Logger.current_logger().report_scalar(
                        title='val mae 0',
                        series='val',
                        value=self.loss_curve["val"][-1][0],
                        iteration=epoch
                    )"""
                    """Logger.current_logger().report_scalar(
                        title='val mae 0',
                        series='val',
                        value=self.loss_curve["val"][-1][0],
                        iteration=epoch
                    )""" #Clear ML
                    train_txt = [round(x, 3) for x in self.loss_curve["train"][-1]]
                    train_str = f'train mae: {train_txt}'
                    sigma_txt = [round(x,3) for x in self.sigmas.detach().reshape(-1).tolist()]
                    sigma_str = f'loss weights: {sigma_txt}'
                    # Epoch: 23/300 --- train mae: [0.0391, 0.39277] val mae: [0.08814, 0.45242] loss weights: [1.0, 0.0] nb samples/task: [5817, 1038] dt/ep(s): 0.284
                    sampling_str = f'nb samples/task: {task_samples_t}'
                    time_str = f'dt/ep(s): {dt:0.3g}'
                    print(epoch_str, train_str, val_str, sigma_str, sampling_str, time_str)
                else:
                    if self.classification:
                        #train_auc = roc_auc_score(act_t, pred_t)
                        #val_auc = roc_auc_score(act_v, pred_v)
                        with torch.no_grad():
                            train_auc = self.criterion(torch.tensor(pred_t), torch.tensor(act_t))
                            val_auc = self.criterion(torch.tensor(pred_v), torch.tensor(act_v))
                        train_str = f'train auc: {train_auc:0.3f}'
                        val_str = f'val auc: {val_auc:0.3f}'
                        time_str = f'dt/ep(s): {dt:0.3g}'
                        sampling_str = f'nb samples/task: {len(act_t)}'
                        if self.decoder == 'multi-task':
                            sigma_txt = [round(x,3) for x in self.sigmas.detach().reshape(-1).tolist()]
                        else:
                            sigma_txt = 'single'
                        sigma_str = f'loss weights: {sigma_txt}'
                        print(epoch_str, train_str, val_str, sigma_str, sampling_str, time_str)

                    else:
                        val_str = f'val mae: {self.loss_curve["val"][-1]:0.3g}'
                        train_str = f'train mae: {self.loss_curve["train"][-1]:0.3g}'
                        time_str = f'dt/ep(s): {dt:0.3g}'
                        sampling_str = f'nb samples/task: {len(act_t)}'
                        if self.decoder == 'multi-task':
                            sigma_txt = [round(x,3) for x in self.sigmas.detach().reshape(-1).tolist()]
                        else:
                            sigma_txt = 'single'
                        sigma_str = f'loss weights: {sigma_txt}'
                        print(epoch_str, train_str, val_str, sigma_str, sampling_str, time_str)
                
                # log metrics
                if not self.eval_model:
                    self.log_metrics(epoch) 

                if self.epoch >= (self.epochs_step * self.swa_start - 1):
                    if (self.epoch + 1) % (self.epochs_step * 2) == 0:
                        self.xswa.append(self.epoch)
                        self.yswa.append(mae_v)

                if losscurve:
                    plt.figure(figsize=(8, 5))
                    xval = np.arange(len(self.loss_curve['val'])) * checkin - 1
                    xval[0] = 0
                    plt.plot(xval, self.loss_curve['train'],
                             'o-', label='train_mae')
                    plt.plot(xval, self.loss_curve['val'],
                             's--', label='val_mae')
                    plt.plot(self.xswa, self.yswa,
                             'o', ms=12, mfc='none', label='SWA point')
                    plt.ylim(0, 2 * np.mean(self.loss_curve['val']))
                    plt.title(f'{self.model_name}')
                    plt.xlabel('epochs')
                    plt.ylabel('MAE')
                    plt.legend()
                    plt.show()

            if (epoch == epochs - 1 or
                    self.optimizer.discard_count >= self.discard_n):
                # save output df for stats tracking
                xval = np.arange(len(self.loss_curve['val'])) * checkin - 1
                xval[0] = 0
                tval = self.loss_curve['train']
                vval = self.loss_curve['val']
                os.makedirs('figures/lc_data', exist_ok=True)
                df_loss = pd.DataFrame([xval, tval, vval]).T
                df_loss.columns = ['epoch', 'train loss', 'val loss']
                df_loss['swa'] = ['n'] * len(xval)
                df_loss.loc[df_loss['epoch'].isin(self.xswa), 'swa'] = 'y'
                #df_loss.to_csv(f'figures/lc_data/{self.model_name}_lc.csv',
                #               index=False)

                # save output learning curve plot
                plt.figure(figsize=(8, 5))
                xval = np.arange(len(self.loss_curve['val'])) * checkin - 1
                xval[0] = 0
                plt.plot(xval, self.loss_curve['train'],
                         'o-', label='train_mae')
                plt.plot(xval, self.loss_curve['val'], 's--', label='val_mae')
                if self.epoch >= (self.epochs_step * self.swa_start - 1):
                    plt.plot(self.xswa, self.yswa,
                             'o', ms=12, mfc='none', label='SWA point')
                plt.ylim(0, 2 * np.mean(self.loss_curve['val']))
                plt.title(f'{self.model_name}')
                plt.xlabel('epochs')
                plt.ylabel('MAE')
                plt.legend()
                #plt.savefig(f'figures/lc_data/{self.model_name}_lc.png')

            if self.optimizer.discard_count >= self.discard_n:
                print(f'Discarded: {self.optimizer.discard_count}/'
                      f'{self.discard_n} weight updates, '
                      f'early-stopping now 🙅🛑')
                self.optimizer.swap_swa_sgd()
                break

        if not (self.optimizer.discard_count >= self.discard_n):
            self.optimizer.swap_swa_sgd()

    
    def log_metrics(self, epoch):
        # Calculate the duration of the epoch
        epoch_end_time = time()  # Current time
        epoch_duration = epoch_end_time - self.epoch_start_time  # Duration calculation
        
        # Proceed with the rest of the method as before
        train_loss = np.array([round(item, 3) for item in self.loss_curve["train"][-1]])
        val_loss = np.array([round(item, 3) for item in self.loss_curve["val"][-1]])
        tasks = self.task_list

        task_types_array = np.array(self.task_types)  
        ind = np.where(~np.isin(task_types_array, ['REG', 'MREG']))[0]
   
        train_loss[ind] = 1-train_loss[ind]
        val_loss[ind] = 1-val_loss[ind]

        if epoch == 0:
            self.first_epoch_score = np.array([train_loss, val_loss])
        else:
            avg_train_loss = np.mean(train_loss/self.first_epoch_score[0])
            avg_val_loss = np.mean(val_loss/self.first_epoch_score[1])
            if self.wandb:
                wandb.log({
                            f"average/train_loss": avg_train_loss,
                            f"average/val_loss": avg_val_loss,
                            "task": f"average",},
                            step = epoch
                    )

        metrics = []
        for k in range(len(tasks)):
            train_metric = round(train_loss[k], 3)
            val_metric = round(val_loss[k], 3)
            task = tasks[k]
            metric_info = f"{task}: train: {train_metric}, validation: {val_metric}"
            metrics.append(metric_info)

            if self.wandb:
                wandb.log({
                        f"{task}_{self.task_types[k]}/train_loss": train_metric,
                        f"{task}_{self.task_types[k]}/val_loss": val_metric,
                        "task": f"{task}_{self.task_types[k]}",},
                        step = epoch
                )

        # Prepend a newline character to the string
        metrics_str = "\n" + "\n".join(metrics)
        
        self.logger.info(f"Epoch {epoch}:")
        self.logger.info(metrics_str)
        # Log the epoch duration
        if epoch != 0:
            self.logger.info(f"Average train performance: {avg_train_loss:.3f}")
            self.logger.info(f"Average val performance: {avg_val_loss:.3f}")

        self.logger.info(f"Epoch duration: {epoch_duration:.2f} seconds")
        self.logger.info("\n")


    def inference(self, loader):
        outputs = []
        formulas_collected = []   # list to collect formulas

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader):
                X, y, formula = data['data'], data['targets'], data['formula']

                # Collect formula
                if isinstance(formula, (tuple, list)):
                    formulas_collected.extend(formula)
                else:
                    formulas_collected.append(formula)

                if self.capture_flag:
                    self.formula_current = None
                    # HACK for PyTorch v1.8.0
                    # this output used to be a list, but is now a tuple
                    if isinstance(formula, tuple):
                        self.formula_current = list(formula)
                    elif isinstance(formula, list):
                        self.formula_current = formula.copy()

                input_ = {}
                src, frac = X.squeeze(-1).chunk(2, dim=1)
                    
                input_['src_masked'] = src.to(self.compute_device, dtype=torch.long, non_blocking=True)
                input_['frac'] = frac.to(self.compute_device, dtype=data_type_torch, non_blocking=True)
                
                res = self.model.forward(input_)
                res_np = res.cpu().numpy().astype('float32')
                # Append the numpy output to the list
                outputs.append(res_np)

            # Concatenate all the stored outputs
            concatenated_outputs = np.concatenate(outputs, axis=0)
            return concatenated_outputs, formulas_collected  # returning formulas_collected along with outputs

    def predict(self, loader):
        len_dataset = len(loader.dataset)
        #n_atoms = int(len(loader.dataset[0][0]) / 2)
        act = np.zeros(len_dataset)
        pred = np.zeros(len_dataset)
        tasks_ = np.zeros(len_dataset)
        uncert = np.zeros(len_dataset)
        formulae = np.empty(len_dataset, dtype=list)
        if isinstance(self.task_list, list):
            pred_tracker = PredictionTracker(self.task_types, self.scaler)
        #atoms = np.empty((len_dataset, n_atoms))
        #fractions = np.empty((len_dataset, n_atoms))
        self.model.eval()
        with torch.no_grad():
            if isinstance(self.task_list, list):
                running_mlm_loss = np.zeros(len(self.task_types))
                running_mlm_count = np.zeros(len(self.task_types))

            for i, data in enumerate(loader):
                X, y, formula = data['data'], data['targets'], data['formula']
                if self.capture_flag:
                    self.formula_current = None
                    # HACK for PyTorch v1.8.0
                    # this output used to be a list, but is now a tuple
                    if isinstance(formula, tuple):
                        self.formula_current = list(formula)
                    elif isinstance(formula, list):
                        self.formula_current = formula.copy()

                input = {}
                src, frac = X.squeeze(-1).chunk(2, dim=1)

                
                """src_masked, position_mask, real_mask_ratio = self.mask_seq(src)
                input['position_mask'] = position_mask
                input['src_masked'] = src_masked.to(self.compute_device,
                                        dtype=torch.long,
                                        non_blocking=True)"""
                    
                input['src'] = src.to(self.compute_device,
                             dtype=torch.long,
                             non_blocking=True)
                input['frac'] = frac.to(self.compute_device,
                               dtype=data_type_torch,
                               non_blocking=True)
                if 'tasks' in data:
                    input['tasks'] = data['tasks'].to(self.compute_device,
                                                 dtype=data_type_torch,
                                                 non_blocking=True)
                y = y.to(self.compute_device,
                         dtype=data_type_torch,
                         non_blocking=True)
            
                input = self.handle_mask(src, data, input)

                res = self.model.forward(input)
                
                """test_loss = self.criterion(res.view(-1),
                                  y_true.view(-1))"""

                if 'tasks' in data: 
                    for k in range(len(self.task_list)):

                        if self.task_types[k] in ['REG', 'BIN', 'MCC', 'MREG']:
                            y_true, formula = self.retrieve_targets(k, data)
                            pred_tracker.update_predictions(k, self.task_types[k], res[k], formula, y_true)
                            
                        elif self.task_types[k] == 'MLM':
                            pred_tracker.update_predictions(k, self.task_types[k], res[k])

                else:
                    prediction, uncertainty = res.chunk(2, dim=-1)
                    prediction = self.scaler.unscale(prediction)
                    uncertainty = torch.exp(uncertainty) * self.scaler.std

                    data_loc = slice(i * self.batch_size,
                                            i * self.batch_size + len(y),
                                            1)

                    #atoms[data_loc, :] = src.cpu().numpy().astype('int32')
                    #fractions[data_loc, :] = frac.cpu().numpy().astype('float32')
                    act[data_loc] = y.view(-1).cpu().numpy().astype('float32')
                    pred[data_loc] = prediction.view(-1).cpu().detach().numpy().astype('float32')
                    uncert[data_loc] = uncertainty.view(-1).cpu().detach().numpy().astype('float32')
                    formulae[data_loc] = formula
                    mlm_loss = 0
                    running_mlm_count = 0
                    
        self.model.train()
        if 'tasks' in data: 
            pred_tracker.calculate_loss()
            pred_tracker.calculate_task_samples()
            return pred_tracker

        else:
            return {'y_pred':pred, 'y_true':act, 'formula':formulae, 'uncertainty':uncert}
                
    def retrieve_targets(self, task_id, data):
        indices = np.where(data['tasks'] == task_id)[0]
        formulas = np.array(data['formula'])
        return data['targets'][indices], formulas[indices]
    
    
    def safe_divide(self, numerator, denominator, default_value=0.0):
        """Divide two arrays element-wise with safety checks."""
        # Handle division by zero
        # Avoid division where denominator is zero
        mask = denominator != 0
        result = np.full_like(numerator, default_value)
        result[mask] = numerator[mask] / denominator[mask]
        
        # Check for NaN or Inf just in case (not strictly necessary if you're sure of your data)
        if np.isnan(result).any() or np.isinf(result).any():
            print("Warning: Result contains NaN or Inf values!")
        return result

    def save_network(self, model_name=None):
        if model_name is None:
            #model_id = datetime.now().strftime("%Y%m%d%H%M%S")
            model_name = self.model_name
            weights_dir = f'{model_name}/trained_models'
            # create weights dir
            os.makedirs(weights_dir, exist_ok=True)
            # create dir for current epoch
            current_epoch_dir = f'{weights_dir}/Epoch_{self.epoch}'
            os.makedirs(current_epoch_dir, exist_ok=True)
            path = f'{current_epoch_dir}/{self.model_id}_{self.epoch}.pth'
            print(f'Saving network to {path}')
        else:
            path = f'models/trained_models/{model_name}_Epoch{self.epoch}.pth'
            print(f'Saving checkpoint ({model_name}) to {path}')

        if isinstance(self.model, nn.DataParallel):
            save_obj = self.model.module
        else:
            save_obj = self.model

        if isinstance(self.task_list, list):
            save_dict = {'weights': save_obj.state_dict(),
                        'scaler_state': [scaler.state_dict() for scaler in self.scaler],
                        'model_name': model_name}
        else:
            save_dict = {'weights': save_obj.state_dict(),
                        'scaler_state': self.scaler.state_dict(),
                        'model_name': model_name}
        torch.save(save_dict, path)

    def load_network(self, path, load_decoder=False):
        # load the old param dict
        network = torch.load(path, map_location=torch.device(self.compute_device))

        if not load_decoder:
            excluded_layers = 'output_nn'  # List of layers to exclude from loading
            # Filter the state dictionary
            filtered_state_dict = self.filter_state_dict(network['weights'], excluded_layers)
            #filtered_state_dict = network['weights']
        if load_decoder:
            filtered_state_dict = network['weights']
            print('Decoder loaded!')

        base_optim = Lamb(params=self.model.parameters())
        optimizer = Lookahead(base_optimizer=base_optim)
        self.optimizer = SWA(optimizer)

        # Apply the fix for the SWA optimizer issue
        self.optimizer.defaults = self.optimizer.optimizer.defaults
        self.optimizer.param_groups = self.optimizer.optimizer.param_groups
        self.optimizer.state = self.optimizer.optimizer.state

        # Check if the model is wrapped with DataParallel
        if isinstance(self.model, nn.DataParallel):
            # Load the state dict into the original model within the DataParallel wrapper
            self.model.module.load_state_dict(filtered_state_dict, strict=False)
        else:
            self.model.load_state_dict(filtered_state_dict, strict=False)
   
        if load_decoder:
            # Initialize the scalers
            scalers = [Scaler(torch.zeros(3)) for i in range(len(network['scaler_state']))]
     
            # Load the state for each scaler
            for i in range(len(network['scaler_state'])):
                scalers[i].load_state_dict(network['scaler_state'][i])

            self.scaler = scalers
        
        self.model_name = network['model_name']
        # check if state dict got loaded
        loaded_layers = self.count_loaded_subkeys(filtered_state_dict, 
                                                self.model.state_dict())
        print('Layers loaded: ' + str(loaded_layers))

    def count_loaded_subkeys(self, model_state_dict, loaded_state_dict):
        count = 0
        for key in model_state_dict:
            # Handle "module." prefix
            modified_key = 'module.' + key if 'module.' + key in loaded_state_dict else key

            if modified_key in loaded_state_dict:
                if torch.equal(model_state_dict[key], loaded_state_dict[modified_key]):
                    count += 1
        return count

    def filter_state_dict(self, state_dict, substring):
        filtered_state_dict = {}
        for key in state_dict:
            if substring not in key:
                filtered_state_dict[key] = state_dict[key]
        return filtered_state_dict

    def MTLoss(self, res, y_true, mask):
        self.reg_loss = RobustL1
        self.bce_loss = BCEWithLogitsLoss
        self.ce_loss = torch.nn.CrossEntropyLoss()

        # randomly sampled loss weights
        self.sigmas = self.constrained_bernoulli() #torch.nn.functional.softmax(torch.randn(len(self.task_list)), dim=-1)
        losses = []

        for task_id in range(len(self.task_types)):
            mask_task = (mask == task_id)
            if self.task_types[task_id] == 'MLM':
                losses.append(self.ce_loss(res[task_id]['y_pred'], res[task_id]['y_true']))

            if self.task_types[task_id] == 'MCC':
                y_true_long = y_true[mask_task].view(-1).long()
                losses.append(self.ce_loss(res[task_id]['y_pred'], y_true_long))
            
            if self.task_types[task_id] == 'BIN':
                losses.append(self.bce_loss(res[task_id]['y_pred'].view(-1), y_true[mask_task].view(-1)))

            if self.task_types[task_id] == 'REG':
                y_pred, y_uncertainty = res[task_id]['y_pred'].chunk(2, dim=-1)
                losses.append(self.reg_loss(y_pred.view(-1), y_uncertainty.view(-1), y_true[mask_task].view(-1)))
            
            if self.task_types[task_id] == 'MREG':
                y_pred, y_uncertainty = res[task_id]['y_pred'].chunk(2, dim=-1)
                losses.append(self.reg_loss(y_pred, y_uncertainty, y_true[mask_task]))
        
        # use only task-specific loss 
        factors = torch.ones(len(losses)).to(self.compute_device)
        #factors[4] = 1.
        losses = torch.stack(losses)*factors
        loss = torch.sum(losses) #*self.sigmas)

        return loss, losses
    
    def constrained_bernoulli(self):
        while True:
            samples = np.random.choice([0, 1], size=len(self.task_list))
            if np.any(samples):
                break

        non_zero_indices = np.where(samples == 1)[0]
        softmax_values = torch.nn.functional.softmax(torch.randn(len(non_zero_indices)), dim=-1).numpy()

        # Create a results array and fill it
        result = np.zeros_like(samples, dtype=np.float32)
        result[non_zero_indices] = softmax_values #constrained_bernoulli_sample
        
        return torch.tensor(result).to(self.compute_device)




# %%
if __name__ == '__main__':
    pass
