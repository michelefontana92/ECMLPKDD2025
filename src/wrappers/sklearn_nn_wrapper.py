import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.base import BaseEstimator, ClassifierMixin
import os
from metrics import BaseMetric,Performance,GroupFairnessMetric
class EarlyStoppingException(Exception):
    pass

class PytorchSklearnWrapper(BaseEstimator, ClassifierMixin):
    """
    Un wrapper scikit-learn per PyTorch che:
    - Usa DataLoader per training/val
    - Supporta i sample_weight passati da Fairlearn, ricostruendo un train_loader con WeightedRandomSampler
    - Non espone data_module in get_params(), evitando problemi di pickling/cloning
    """

    def __init__(self, 
                 model=None,
                 data_module=None,         # oggetto con val_loader() fisso, ad esempio
                 epochs=1,                 # numero di epoche da fare a ogni .fit()
                 learning_rate=1e-4,
                 device='cpu',
                 disable_log=False,
                 max_grad_norm=1.0,
                 # parametri multi-obiettivo / fairness
                 requirement_set=None,
                 surrogate_functions=None,
                 metrics=None,
                 # callback & logger, da non esporre
                 checkpoints=None,
                 logger=None,
                 **kwargs):

        # Mettiamo data_module e callback come attributi "privati", non li esponiamo
        self._data_module = data_module      
        self._checkpoints = checkpoints or []
        self._logger = logger

        # Parametri principali "esposti"
        self.model = model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = device
        self.disable_log = disable_log
        self.max_grad_norm = max_grad_norm

        # Per la logica multi-obiettivo/fairness (se servono)
        self.requirement_set = requirement_set
        self.surrogate_functions = surrogate_functions
        self.metrics = metrics
       
        # Altri parametri
        self.kwargs = kwargs

        if self.model is None:
            raise ValueError("Serve un model PyTorch in 'model'.")
        
        # Inizializziamo ottimizzatore e simili
        self._initialize_model()

    def _initialize_model(self):
        # Esempio: cross entropy come fallback, se non si usa surrogate function
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.model.to(self.device)

    # =====================================================================
    #           Metodi di training/val ispirati a TorchNNMOWrapper
    # =====================================================================

    def _training_step(self, batch):
        """
        Esegue un singolo step di training su un batch.
        batch: un dizionario o tuple con 'data' e 'labels' (e se serve 'groups', etc.)
        """
        self.model.train()
        inputs = batch['data'].float().to(self.device)
        targets = batch['labels'].long().to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)

        if self.surrogate_functions is not None:
            # Loss multi-obiettivo
            group_ids = batch.get('groups', {})
            # spostiamo su device
            for k in group_ids:
                group_ids[k] = group_ids[k].to(self.device)

            loss = self.surrogate_functions.evaluate(
                logits=outputs,
                labels=targets,
                group_ids=group_ids,
                positive_mask=batch.get('positive_mask'),
                group_ids_list=batch.get('groups_ids_list'),
                group_masks=group_ids
            )
        else:
            # classica cross-entropy
            loss_per_sample = self.criterion(outputs, targets)
            loss = loss_per_sample.mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.item()

    def _validation_step(self, batch):
        """
        Esegue un singolo step di validazione (no backward).
        """
        self.model.eval()
        with torch.no_grad():
            inputs = batch['data'].float().to(self.device)
            targets = batch['labels'].long().to(self.device)
            outputs = self.model(inputs)

            if self.surrogate_functions is not None:
                group_ids = batch.get('groups', {})
                for k in group_ids:
                    group_ids[k] = group_ids[k].to(self.device)
                loss = self.surrogate_functions.evaluate(
                    logits=outputs,
                    labels=targets,
                    group_ids=group_ids,
                    positive_mask=batch.get('positive_mask'),
                    group_ids_list=batch.get('groups_ids_list'),
                    group_masks=group_ids
                )
            else:
                loss_per_sample = self.criterion(outputs, targets)
                loss = loss_per_sample.mean()

            preds = torch.argmax(outputs, dim=1)
        return loss.item(), outputs, targets, preds

    def _evaluate_requirements(self, data_loader):
        """
        Calcola requirements + metriche di fairness su un intero data_loader di validazione
        (simile a TorchNNMOWrapper).
        """
        self.model.eval()
        all_loss = 0.0
        all_outputs, all_targets, all_preds = [], [], []
        all_groups = []
        count_batches = 0

        with torch.no_grad():
            for batch in data_loader:
                loss_val, outputs, targets, preds = self._validation_step(batch)
                all_loss += loss_val
                count_batches += 1

                all_outputs.append(outputs.cpu())
                all_targets.append(targets.cpu())
                all_preds.append(preds.cpu())
                all_groups.append(batch.get('groups', {}))

        avg_loss = all_loss / max(count_batches, 1)
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        # unisci groups
        combined_groups = {}
        if len(all_groups) > 0 and len(all_groups[0]) > 0:
            keys = all_groups[0].keys()
            for k in keys:
                combined_groups[k] = torch.cat([g[k] for g in all_groups], dim=0).cpu()

        # calcola requirement_set, se c'è
        if self.requirement_set is not None:
            requirements, _, _ = self.requirement_set.evaluate(
                y_pred=all_preds,
                y_true=all_targets,
                group_ids=combined_groups
            )
        else:
            requirements = {}

        return requirements, avg_loss, all_outputs, all_targets, all_preds, combined_groups

    def _compute_metrics(self, predictions, targets, groups_dict, logits=None, prefix=''):
        """
        Calcola metriche custom aggiuntive (self.metrics) su predictions e targets.
        """
        tmp_result = {}
        final_result = {}
        y_pred = predictions
        y_true = targets
        group_ids = groups_dict
        for metric in self.metrics:
            metric.reset()
            if issubclass(metric.__class__,GroupFairnessMetric):
                            group_ids_detached = {group_name:group_ids[group_name].detach().cpu() for group_name in group_ids.keys()}
                            metric.calculate(y_pred.detach().cpu(),
                                            y_true.detach().cpu(),
                                            group_ids_detached)
                         
            elif isinstance(metric,Performance):
                metric.calculate(y_pred.detach().cpu(),
                                 y_true.detach().cpu())
            else:
                raise ValueError(f"{metric} is an invalid metric")
            tmp_result.update(metric.get())
            
      
        for key, value in tmp_result.items():
            if prefix == '':
                final_result[key] = value
            else:
                final_result[f'{prefix}_{key}'] = value
        return final_result 

    def _update_metrics(self):
        """
        Calcola metriche su train_loader_eval e val_loader (presi da self._data_module).
        """
        # se il data_module è già costruito e abbiamo metodi val_loader() e train_loader_eval():
        train_requirements, train_loss, t_out, t_y, t_preds, t_groups = self._evaluate_requirements(
            self._data_module.train_loader_eval()
        )
        val_requirements, val_loss, v_out, v_y, v_preds, v_groups = self._evaluate_requirements(
            self._data_module.val_loader()
        )

        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_requirements': train_requirements,
            'val_requirements': val_requirements
        }
        # metriche custom
        metrics.update(
            self._compute_metrics(t_preds, t_y, t_groups, logits=t_out, prefix='train')
        )
        metrics.update(
            self._compute_metrics(v_preds, v_y, v_groups, logits=v_out, prefix='val')
        )
        return metrics

    # =====================================================================
    #        Metodi scikit-learn: fit / predict / predict_proba / score
    # =====================================================================

    def fit(self, X, y, sample_weight=None):
        """
        Allena il modello su (X, y), usando un DataLoader creato "al volo".
        Se sample_weight è fornito (es. da Fairlearn), costruiamo un WeightedRandomSampler.
        Eseguiamo N epoche (self.epochs), come definito nel costruttore.
        """
        # 1) Creiamo un DataLoader in base a X, y, sample_weight
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        if sample_weight is not None:
            # WeightedRandomSampler
            sw_tensor = torch.tensor(sample_weight, dtype=torch.float32)
            sampler = WeightedRandomSampler(
                weights=sw_tensor,
                num_samples=len(sw_tensor),  # di solito = len(X)
                replacement=True
            )
            train_dataset = TensorDataset(X_tensor, y_tensor)
            train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)
        else:
            # standard
            train_dataset = TensorDataset(X_tensor, y_tensor)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # 2) Eseguiamo il training per self.epochs epoche
        for epoch in range(self.epochs):
            total_loss = 0.0
            self.model.train()
            for batch_data, batch_labels in train_loader:
                # Per coerenza con _training_step, costruiamo un "batch" come dict
                batch = {'data': batch_data, 'labels': batch_labels}
                loss_val = self._training_step(batch)
                total_loss += loss_val

            # Se vogliamo loggare metriche su un set "fisso" (val_loader ecc.), lo facciamo qui
            if not self.disable_log and self._data_module is not None:
                metrics = self._update_metrics()
                if self._logger is not None:
                    self._logger.log(metrics)

                # Controllo i checkpoint
                for ck in self._checkpoints:
                    stop = ck(metrics=metrics)
                    if isinstance(stop, tuple):
                        stop, counter = stop
                        metrics['early_stopping'] = counter
                        if stop:
                            if not self.disable_log and self._logger:
                                self._logger.log(metrics)
                            raise EarlyStoppingException
                # eventuali altre azioni

        # 3) Alla fine delle epoche, se ci sono ModelCheckpoint, carichiamo il best
        for ck in self._checkpoints:
            if hasattr(ck, 'get_model_path'):
                path = ck.get_model_path()
                if path and os.path.exists(path):
                    self.load(path)

        return self

    def predict(self, X):
        """
        Prevede le classi per un array X, build di un DataLoader al volo
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=64)

        all_preds = []
        with torch.no_grad():
            for (batch_data,) in loader:
                batch_data = batch_data.to(self.device)
                outputs = self.model(batch_data)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())

        return torch.cat(all_preds).numpy()

    def predict_proba(self, X):
        """
        Restituisce le probabilità stimate (softmax) per un array X
        """
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=64)

        all_probs = []
        with torch.no_grad():
            for (batch_data,) in loader:
                batch_data = batch_data.to(self.device)
                outputs = self.model(batch_data)
                probs = torch.softmax(outputs, dim=1)
                all_probs.append(probs.cpu())

        return torch.cat(all_probs, dim=0).numpy()

    def score(self, X, y):
        """
        Implementazione semplificata per la metrica di default (accuracy).
        """
        preds = self.predict(X)
        return (preds == y).mean()

    # =====================================================================
    #               Salvataggio, caricamento, parametri
    # =====================================================================

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def get_params(self, deep=True):
        """
        Ritorniamo i parametri che scikit-learn userà per il cloning.
        NON includiamo _data_module, _logger, _checkpoints, ecc.
        """
        return {
            'model': self.model,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'device': self.device,
            'disable_log': self.disable_log,
            'max_grad_norm': self.max_grad_norm,
            'requirement_set': self.requirement_set,
            'surrogate_functions': self.surrogate_functions,
            'metrics': self.metrics,
            **self.kwargs
        }

    def set_params(self, **params):
        """
        Se scikit-learn / Fairlearn cambia qualche parametro,
        lo aggiorniamo qui e re-inizializziamo, se necessario.
        """
        for k, v in params.items():
            setattr(self, k, v)
        self._initialize_model()
        return self
    
    def compute_metrics_from_external_predictions(self, y_pred, y_true, groups_dict=None, prefix='final'):
        """
        Usa la stessa logica di _compute_metrics(...) ma con predizioni già fornite.
        Utile per valutare metriche sul predictor finale di ExponentiatedGradient,
        che è un 'mix' di modelli e non self.model.
        """
    
        # Convertiamo se necessario in tensori (se _compute_metrics vuole tensori):
        y_pred_tensor = torch.tensor(y_pred)
        y_true_tensor = torch.tensor(y_true)
        
        # Se groups_dict è un dict di tensori, OK; altrimenti convertili su CPU/gpu come serve
        # Esempio:
        if groups_dict is None:
            groups_dict = {}
        
        # Richiamiamo la funzione interna self._compute_metrics(...)
        # che tipicamente accetta: (predictions, targets, groups_dict, logits=None, prefix='')
        metrics_dict = self._compute_metrics(
            y_pred_tensor,
            y_true_tensor,
            groups_dict,
            logits=None,  # se le metriche non richiedono logits
            prefix=prefix
        )

        return metrics_dict
