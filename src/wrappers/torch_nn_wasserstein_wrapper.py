from .torch_nn_wrapper import TorchNNWrapper
import torch
import tqdm
from callbacks import EarlyStopping, ModelCheckpoint
from requirements import RequirementSet

class EarlyStoppingException(Exception):
    pass

class TorchNNWassersteinWrapper(TorchNNWrapper):
    def __init__(self,*args, **kwargs):
        super(TorchNNWassersteinWrapper,self).__init__(*args, **kwargs)
        self.training_group_name:str = kwargs.get('training_group_name')
        self.requirement_set:RequirementSet = kwargs.get('requirement_set')
        self.surrogate_functions:list[(callable,float)] = kwargs.get('surrogate_functions')
        
        assert self.requirement_set is not None, f'{self.requirement_set} has to be provided'
        assert self.surrogate_functions is not None, f'{self.surrogate_functions} has to be provided'
        assert self.training_group_name is not None, f'{self.training_group_name} has to be provided'
    
    def _training_step(self,batch,batch_idx):
        self.model.train()
        inputs = batch['data'] 
        targets = batch['labels']
        group_ids = batch['groups']
        
        inputs = inputs.float().to(self.device)
        targets = targets.long().to(self.device)
        group_ids = {group_name:group_ids[group_name].to(self.device) for group_name in group_ids.keys()}
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.surrogate_functions.evaluate(logits=outputs,
                                                 labels=targets,
                                                 group_ids=group_ids)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _validation_step(self,batch,batch_idx):
        self.model.eval()
        with torch.no_grad():
            inputs = batch['data'] 
            targets = batch['labels']
            group_ids = batch['groups']
            
            inputs = inputs.float().to(self.device)
            targets = targets.long().to(self.device)
            group_ids = {group_name:group_ids[group_name].to(self.device) for group_name in group_ids.keys()}
            outputs = self.model(inputs)
            loss = self.surrogate_functions.evaluate(logits=outputs,
                                                 labels=targets,
                                                 group_ids=group_ids)
            predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
           
            return loss.item(), outputs, targets, predictions
    
    def fit(self,**kwargs):
        
        num_epochs = kwargs.get('num_epochs',-1)
        disable_log = kwargs.get('disable_log',True)
        n_rounds = self.num_epochs if num_epochs == -1 else num_epochs
        self.model.to(self.device)
        
        try:
            for epoch in tqdm.tqdm(range(n_rounds)):
                train_loss = 0
                val_loss = 0
                train_loader = self.data_module.train_loader()
                val_loader = self.data_module.val_loader()
                train_loader_eval = self.data_module.train_loader_eval()
                for batch_idx, batch in enumerate(tqdm.tqdm(train_loader)):
                    train_loss += self._training_step(batch, batch_idx)
                
                train_loss /= len(train_loader)
                
                train_outputs = []
                train_targets = []
                train_predictions=[]
                train_groups = []
                
                val_outputs = []
                val_targets = []
                val_predictions = []
                val_groups = []
                for batch_idx, batch in enumerate(val_loader):
                    loss, outputs, targets,predictions = self._validation_step(batch, batch_idx)
                    val_loss += loss
                    val_outputs.append(outputs)
                    val_targets.append(targets)
                    val_predictions.append(predictions)
                    val_groups.append(batch['groups'])
                val_loss /= len(val_loader)
                
                for batch_idx, batch in enumerate(train_loader_eval):
                    _, outputs, targets,predictions = self._validation_step(batch, batch_idx)
                    train_outputs.append(outputs)
                    train_targets.append(targets)
                    train_predictions.append(predictions)
                    train_groups.append(batch['groups'])
                
                val_loss /= len(val_loader)
                val_outputs = torch.cat(val_outputs, dim=0)
                val_targets = torch.cat(val_targets, dim=0).detach().cpu()
                val_predictions = torch.cat(val_predictions, dim=0).detach().cpu()
                val_groups_dict = {group_name:torch.cat([batch[group_name] for batch in val_groups],dim=0).detach().cpu() for group_name in val_groups[0].keys()}
                train_outputs = torch.cat(train_outputs, dim=0)
                train_targets = torch.cat(train_targets, dim=0).detach().cpu()
                train_predictions = torch.cat(train_predictions, dim=0).detach().cpu()
                train_groups_dict = {group_name:torch.cat([batch[group_name] for batch in train_groups],dim=0).detach().cpu() for group_name in train_groups[0].keys()}

                val_requirements,_ = self.requirement_set.evaluate(y_pred=val_predictions, 
                                                                 y_true=val_targets, 
                                                                 group_ids=val_groups_dict)
                train_requirements,_ = self.requirement_set.evaluate(y_pred=train_predictions, 
                                                                 y_true=train_targets, 
                                                                 group_ids=train_groups_dict)
                
                metrics = {
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_requirements': val_requirements,
                    'train_requirements': train_requirements
                }

                metrics.update(self._compute_metrics(self.metrics,
                                                    val_predictions,
                                                    val_targets,
                                                    val_groups_dict,
                                                    prefix='val'))
                metrics.update(self._compute_metrics(self.metrics,
                                                    train_predictions,
                                                    train_targets,
                                                    train_groups_dict,
                                                    prefix='train'))
                
                
                for checkpoint in self.checkpoints:
                    if isinstance(checkpoint, EarlyStopping):
                        stop,counter = checkpoint(metrics=metrics)
                        metrics['early_stopping'] = counter
                        if stop:
                            if not disable_log:
                                self.logger.log(metrics)  
                            raise EarlyStoppingException 
                       
                    elif isinstance(checkpoint, ModelCheckpoint):
                        model_checkpoint= checkpoint(
                                save_fn=self.save, 
                                metrics=metrics)
                        metrics['model_checkpoint'] = 1 if model_checkpoint else 0
                if not disable_log:
                    self.logger.log(metrics)  
        except EarlyStoppingException:
                    pass
        
        for checkpoint in self.checkpoints:
                if isinstance(checkpoint, ModelCheckpoint):
                   self.load(checkpoint.get_model_path())

        return self.model

    def score(self, data_loader,metrics,prefix=''):
        assert len(data_loader) == 1, "Data loader should have a single batch"
        assert isinstance(metrics, list), "Metrics should be a list"
        self.model.to(self.device)
        scores = {}
        for batch_idx, batch in enumerate(data_loader):
            loss, _, targets, predictions = self._validation_step(batch, batch_idx)
            
            requirements,_ = self.requirement_set.evaluate(y_pred=predictions, 
                                                         y_true=targets, 
                                                         group_ids=batch['groups'])
            scores = self._compute_metrics(metrics,
                                           predictions,
                                           targets,
                                           batch['groups'],
                                           prefix=prefix)
            if prefix != '':
                scores[f'{prefix}_requirements'] = requirements
                scores[f'{prefix}_loss'] = loss
            else: 
                scores['requirements'] = requirements
                scores['loss'] = loss
        return scores    

    
        