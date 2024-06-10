import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, f1_score

class PredictionTracker:
    def __init__(self, task_list, scalers):
        """
        Initialize the PredictionTracker.
        
        :param scaler: a dictionary of scalers for unscaling predictions for 'REG' tasks
        :param retrieve_targets: a function to retrieve true values and formula from data
        """
        self.infos = {}
        self.infos = [{'task_type': x} for x in task_list]
        self.scalers = scalers

    def update_predictions(self, i_, task_type, res, formula=None, y_true=None):
        """
        Update predictions for a given task type and index, appending data if the method is called multiple times.
        
        :param i_: the index for the task
        :param task_type: the type of task ('REG', 'BIN', 'MCC', 'MLM')
        :param res: the result dictionary containing predictions
        :param data: the data used for retrieving targets
        :param formula: the formula used in the task
        :param y_true: the true values (optional, mainly for 'MLM' task)
        """
        if task_type not in ['REG', 'BIN', 'MCC', 'MLM', 'MREG']:
            raise ValueError(f"Invalid task type: {task_type}")
        if self.infos[i_]['task_type'] != task_type:
            raise ValueError(f"Task type mismatch: {self.infos[i_]['task_type']} != {task_type}")
        
        # Function to append or set the value in the infos dictionary
        def append_or_set(key, value):
            if key in self.infos[i_]:
                if isinstance(self.infos[i_][key], torch.Tensor):
                    self.infos[i_][key] = torch.cat((self.infos[i_][key], value), dim=0)
                else:
                    self.infos[i_][key] = np.append(self.infos[i_][key], value, axis=0)
            else:
                self.infos[i_][key] = value

        if task_type == 'REG':
            if res['y_pred'].shape[0] > 0:
                y_pred, uncertainty = res['y_pred'].chunk(2, dim=-1)
                y_pred = self.scalers[i_].unscale(y_pred)
                y_pred = y_pred.detach().cpu()
                uncertainty = (torch.exp(uncertainty) * self.scalers[i_].std)
                y_true = y_true.detach().cpu()
                append_or_set('uncertainty', uncertainty)
            else:
                y_pred = torch.tensor([])   
        
        if task_type == 'MREG':
            if res['y_pred'].shape[0] > 0:
                y_pred, uncertainty = res['y_pred'].chunk(2, dim=-1)
                
                """for k in range(y_pred.shape[-1]):
                    y_pred[:, k] = self.scalers[i_][k].unscale(y_pred[:, k])
                    #y_true[:, k] = self.scalers[i_][k].unscale(y_true[:, k])
                    uncertainty[:, k] = torch.exp(uncertainty[:, k]) * self.scalers[i_][k].std"""
                uncertainty = torch.tensor([])  

                y_pred = y_pred.detach().cpu()
                y_true = y_true.detach().cpu()
                uncertainty = uncertainty.detach().cpu()
                append_or_set('uncertainty', uncertainty)
            else:
                y_pred = torch.tensor([])  
        
        elif task_type == 'BIN':
            if res['y_pred'].shape[0] > 0:
                y_pred = torch.sigmoid(res['y_pred']).detach().cpu()
                y_pred = torch.where(y_pred > 0.5, torch.tensor(1), torch.tensor(0))
            else:
                y_pred = torch.tensor([])
        
        elif task_type == 'MCC':
            if res['y_pred'].shape[0] > 0:
                y_pred = torch.softmax(res['y_pred'], dim=-1)  
                y_pred = torch.argmax(y_pred, dim=-1).detach().cpu()
            else:
                y_pred = torch.tensor([])
        

        if task_type == 'MLM':
            if res['y_pred'].shape[0] > 0:
                y_pred = torch.softmax(res['y_pred'], dim=-1)  
                y_pred = torch.argmax(y_pred, dim=-1).detach().cpu()
                y_true = res['y_true'].detach().cpu()
            else:
                y_pred = torch.tensor([])
                y_true = torch.tensor([])

        append_or_set('y_pred', y_pred)
        append_or_set('y_true', y_true)
        if formula is not None:
            append_or_set('formula', formula)
    
    def calculate_loss(self):
        """
        Calculate the loss for each case.
        """
        losses = []
        for info in self.infos:
            task_type = info['task_type']
            y_pred = info['y_pred']
            y_true = info['y_true']
            if task_type == 'REG':
                loss = mean_absolute_error(y_true, y_pred)
            elif task_type == 'BIN':
                loss = f1_score(y_true, y_pred)
            elif task_type == 'MCC':
                loss = f1_score(y_true, y_pred, average='macro')
            elif task_type == 'MLM':
                loss = f1_score(y_true, y_pred, average='macro')
            elif task_type == 'MREG':
                loss = mean_absolute_error(y_true, y_pred)
            losses.append(loss)
        
        self.loss = losses
        return 
    
    def calculate_task_samples(self):
        task_samples = []
        for info in self.infos:
            task_type = info['task_type']
            task_samples.append(info['y_pred'].shape[0])
        self.task_samples = task_samples


    def get_predictions(self):
        """
        Get the entire prediction dictionary.
        """
        return self.infos
