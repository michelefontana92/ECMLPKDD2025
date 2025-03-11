import numpy as np
from .torch_nn_wrapper import TorchNNWrapper
import torch
import tqdm
from callbacks import EarlyStopping, ModelCheckpoint
from requirements import RequirementSet
import os
from entmax import entmax_bisect
from metrics import Performance,GroupFairnessMetric
import copy
from surrogates import SurrogateFactory
import functools

class EarlyStoppingException(Exception):
    pass

class TorchNNLagrangianWrapper(TorchNNWrapper):
    def __init__(self, *args, **kwargs):
      
        super(TorchNNLagrangianWrapper, self).__init__(*args, **kwargs)
        self.id = kwargs.get('id','LagrangianWrapper')
        self.compute_only_score =kwargs.get('compute_only_score',False)
        self.optimizer_fn: callable = kwargs.get('optimizer_fn')
        self.lagrangian_checkpoints = kwargs.get('lagrangian_checkpoints', [])
        #self.training_group_name: str = kwargs.get('training_group_name')
        
        self.teacher_model = kwargs.get('teacher_model')
        #self.distillation_loss_fn:callable = kwargs.get('distillation_loss_fn')
        self.batch_objective_function = kwargs.get('batch_objective_fn')
        self.original_objective_fn:callable = kwargs.get('original_objective_fn')
        self.objective_fn: callable = kwargs.get('objective_fn')
        self.inequality_constraints_fn_list: list = kwargs.get('inequality_constraints')
        self.equality_constraints_fn_list: list = kwargs.get('equality_constraints')
        
        self.mu_max = kwargs.get('mu_max', 1e3)
        self.nu_max = kwargs.get('nu_max', 100)
        self.lambda_equality_max = kwargs.get('lambda_equality_max', 100)
        self.lambda_inequality_max = kwargs.get('lambda_inequality_max', 100)

        self.rho = kwargs.get('rho', 2)
        self.mu_0 = kwargs.get('mu_0', 2)
        self.damping_factor = kwargs.get('damping_factor', 1.0)  # Valore di damping per rallentare l'aggiornamento

        self.gamma_objective = kwargs.get('gamma_objective', 0.8)
        self.gamma_constraint = kwargs.get('gamma_constraint', 100)

        self.inequality_lambdas_0_value = kwargs.get('inequality_lambdas_0_value', 0.1)
        self.equality_lambdas_0_value = kwargs.get('equality_lambdas_0_value', 0.)
        self.objective_multiplier_0_value = kwargs.get('objective_multiplier_0_value', 1)
        self.macro_constraints_list= kwargs.get('macro_constraints_list')
        # Assicurati che tutti i tensori siano su device
        self.inequality_lambdas_0 = torch.ones(len(self.inequality_constraints_fn_list), device=self.device) * self.inequality_lambdas_0_value
        self.equality_lambdas_0 = torch.ones(len(self.equality_constraints_fn_list), device=self.device) * self.equality_lambdas_0_value
        self.objective_multiplier_0 = torch.tensor(self.objective_multiplier_0_value, device=self.device)
        self.lambda0_max_value = kwargs.get('lambda0_max_value', 0.1)
        self.dro_lambda = 6 
        self.hard_mode = True
        self.target_groups = set()
        #self.compute_groups_cardinality()
        
        self.active_groups = {}
        self.teacher_model_list = []
        for constraint in self.inequality_constraints_fn_list:
            if constraint.group_name is not None:
                self.target_groups.add(constraint.group_name)
                if constraint.group_name not in self.active_groups:
                    self.active_groups[constraint.group_name] = []
                for c in constraint.target_groups:
                    if c not in self.active_groups[constraint.group_name]:
                        self.active_groups[constraint.group_name].append(c.item())
                  
        #print('Active Groups:',self.active_groups)
        #print('Target groups:',self.target_groups)
        #print('Checkpoints:',self.checkpoints)
        #assert self.training_group_name is not None, f'{self.training_group_name} has to be provided'
        assert self.macro_constraints_list is not None, f'{self.macro_constraints_list} has to be provided'
        self.group_cardinality = None
        self._init_alm_parameters()
        #print('Lagrangian Wrapper initialized')

        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        
        self.model.apply(init_weights)

    def compute_groups_cardinality(self):
        groups = next(iter(self.get_train_loader_eval()))['groups']
        self.group_cardinality = {group_name: {} for group_name in self.target_groups} 
        self.max_cardinality = {group_name: 0 for group_name in self.target_groups}
        #print('Groups:',self.group_cardinality.keys())
        for group_name in self.target_groups:
            for group in groups[group_name].unique():
                self.group_cardinality[group_name].update({group.item(): len(groups[group_name][groups[group_name] == group])})               
                if len(groups[group_name][groups[group_name] == group]) > self.max_cardinality[group_name]:
                    self.max_cardinality[group_name] = len(groups[group_name][groups[group_name] == group])
        #print('Group Cardinality:',self.group_cardinality)
    
    def _compute_constraints_weights(self):
        weights = []
        for i,constraint in enumerate(self.inequality_constraints_fn_list):
            if constraint.group_name is not None:
                weight_list = [self.group_cardinality[constraint.group_name][group.item()] for group in constraint.target_groups]
                weight = np.min(np.array(weight_list))/self.max_cardinality[constraint.group_name]
                weights.append(weight)
            else:
                weights.append(0)
        #for i,constraint in enumerate(self.inequality_constraints_fn_list):
            #if constraint.target_groups is not None:
            #    print(f'Constraint {i} target groups:',constraint.target_groups)
            #print(f'Constraint {i} pre-weight:',weights[i])    
        self.inequality_weights = torch.tensor(weights,device=self.device)
        
    def _init_inequality_lambdas(self):
        self.inequality_lambdas = torch.ones_like(self.inequality_lambdas_0, device=self.device) * self.inequality_lambdas_0_value
    
    def _init_alm_parameters(self):
        
       
        #self._compute_constraints_weights()
        self._init_inequality_lambdas()
        
        #print('Inequality Lambdas:',self.inequality_lambdas)
        self.equality_lambdas = self.equality_lambdas_0
        self.mu = self.mu_0
        self.objective_multiplier = self.objective_multiplier_0

    def update_lambdas_inequality(self, constraints):
       
        if constraints is None:
            return self.inequality_lambdas
        
        old_lambdas = self.inequality_lambdas.clone().detach()
        new_lambdas = torch.max(
            torch.ones_like(self.inequality_lambdas, device=self.device)*self.inequality_lambdas_0_value,
            self.inequality_lambdas + self.mu * torch.max(constraints,torch.zeros_like(constraints, 
                                                                                       device=self.device))
        )
        #new_lambdas = torch.clamp(new_lambdas, min=self.inequality_lambdas_0_value, max=self.lambda_inequality_max)
        assert torch.all(new_lambdas >= self.inequality_lambdas_0_value), 'Negative Lagrange multipliers!'
        #self._monitor_lambda_changes(old_lambdas, new_lambdas, 'inequality', self.inequality_mask)
        return new_lambdas

    def update_lambdas_equality(self, constraints):
        
        if constraints is None:
            return self.equality_lambdas
        old_lambdas = self.equality_lambdas.clone().detach()
        new_lambdas = self.equality_lambdas + self.mu * constraints * self.damping_factor
        new_lambdas = torch.clamp(new_lambdas, min=self.equality_lambdas_0_value, max=self.lambda_equality_max)
        #self._monitor_lambda_changes(old_lambdas, new_lambdas, 'equality', self.equality_mask)
        return new_lambdas

    def _monitor_lambda_changes(self, old_lambdas, new_lambdas, lambda_type, mask):
        # Controlliamo solo dove la maschera è non nulla
        active_mask = mask != 0  # Verifica se la maschera è non nulla
        diff = torch.abs(new_lambdas - old_lambdas)[active_mask]
        
        # Se ci sono cambiamenti significativi, loggali
        if torch.any(diff > 0):
            indices = torch.nonzero(active_mask, as_tuple=False).squeeze()
            new_lambdas_active = new_lambdas[active_mask]  # Estrai solo i nuovi moltiplicatori attivi
            
            # Verifica se abbiamo uno scalare (tensore a 0 dimensioni)
            if indices.dim() == 0:  # Caso scalare
                print(f"Moltiplicatori di Lagrange ({lambda_type}) aggiornati per indice {indices.item()}: {new_lambdas_active.item()}")
            else:
                # Caso non scalare, possiamo iterare su indices e new_lambdas_active
                print(f"Moltiplicatori di Lagrange ({lambda_type}) aggiornati per maschere non nulle:")
                for idx, new_val in zip(indices, new_lambdas_active):
                    print(f"Aggiornato il moltiplicatore {idx.item()}: {new_val.item()}")
          

   
    
    def update_alm_parameters_and_metrics(self, update_alm=True,**kwargs):
        metrics = {}
        self.model.eval()
        with torch.no_grad():
            #print('Loading the val loader')
            #val_loader = next(iter(self.data_module.val_loader()))  # Val loader già calcolato
            val_loader = self.data_module.val_loader(batch_size=None)
            #print('Computing Val Kwargs')
            val_kwargs = self._compute_kwargs_in_batches(val_loader, self.model,use_entmax=False)
            train_loader = self.data_module.train_loader_eval(batch_size=None)
            #print('Computing Val Kwargs')
            train_kwargs = self._compute_kwargs_in_batches(train_loader, self.model,use_entmax=False)
            
            #del val_loader
            #torch.cuda.empty_cache() 
            #print('Loading the train loader')
            #train_loader = self.data_module.train_loader_eval(batch_size=1024)  # Val loader già calcolato
            #print('Computing Train Kwargs')
            #train_kwargs = self._compute_kwargs_in_batches(train_loader, self.model,use_entmax=False)
            #del train_loader 
            #torch.cuda.empty_cache()
            kwargs = {}
           
            #kwargs['train_kwargs'] = train_kwargs 
           
            kwargs['val_kwargs'] = val_kwargs
            kwargs['train_kwargs'] = train_kwargs
            #print('Kwargs computed')

            inequality_constraints = train_kwargs['inequality_constraints']
            equality_constraints = train_kwargs['equality_constraints']
            #inequality_constraints = train_kwargs['inequality_constraints']
            #equality_constraints = train_kwargs['equality_constraints']
            #print('Computing Train Score')
            #train_score = self.compute_score(**train_kwargs)
            #print('Train Score computed')    
            if update_alm:
                self._apply_early_stopping(inequality_constraints, equality_constraints)

                # Applica le maschere ai vincoli una sola volta
                if inequality_constraints is not None:
                    inequality_constraints = inequality_constraints * self.inequality_mask

                if equality_constraints is not None:
                    equality_constraints = equality_constraints * self.equality_mask

                # Aggiornamento dei moltiplicatori di Lagrange solo se i vincoli non sono `None`
                if inequality_constraints is not None:
                    self.inequality_lambdas = self.update_lambdas_inequality(inequality_constraints)
                if equality_constraints is not None:
                    self.equality_lambdas = self.update_lambdas_equality(equality_constraints)

                

            #metrics['train_constraints_score'] = train_score
            #metrics['train_objective_function'] = 1 - train_kwargs['objective_function']
            #real_inequality_constraints = val_kwargs['real_inequality_constraints']
            val_score = self.compute_score(**val_kwargs)    
            metrics['val_constraints_score'] = val_score
            val_loss = self.compute_loss_fn(**val_kwargs)
            metrics['val_loss'] = val_loss
            if not self.compute_only_score:
            #metrics['val_constraint'] = torch.max(real_inequality_constraints + 0.1).item()
                metrics.update(self._compute_metrics(self.metrics,  prefix='val', **val_kwargs))
            return metrics
        
    def _apply_early_stopping(self, inequality_constraints, equality_constraints):
        n_inequality_constraints = len(self.inequality_constraints_fn_list)
        self.inequality_mask = torch.ones_like(self.inequality_lambdas, device=self.device)
        self.equality_mask = torch.ones_like(self.equality_lambdas, device=self.device)
        cached_scores = {}
        # Aggiorna i parametri Lagrangiani
        for i, checkpoint in enumerate(self.lagrangian_checkpoints):
            if isinstance(checkpoint, EarlyStopping):
                if i < n_inequality_constraints:
                    if i not in cached_scores:
                        cached_scores[i] = {'score': inequality_constraints[i]}
                    update, _ = checkpoint(metrics=cached_scores[i])
                    if not update:
                        self.inequality_mask[i] = 0  # Ferma l'aggiornamento per questo vincolo
                    else:
                        checkpoint.reset(keep_best=True)
                else:
                    eq_index = i - n_inequality_constraints
                    if eq_index not in cached_scores:
                        cached_scores[eq_index] = {'score': equality_constraints[eq_index]}
                    update, _ = checkpoint(metrics=cached_scores[eq_index])
                    if not update:
                        self.equality_mask[eq_index] = 0  # Ferma l'aggiornamento per questo vincolo
                    else:
                        checkpoint.reset()
    
    def compute_constraints(self, **kwargs):
        # Usa il dispositivo corrente (GPU se disponibile)
        device = self.device
        
        # Vincoli di disuguaglianza
        if len(self.inequality_constraints_fn_list)>0:
            inequality_constraints = torch.stack(
                [torch.clamp(constraint_fn(**kwargs),min=0) for constraint_fn in self.inequality_constraints_fn_list], dim=0
            ).to(device)
        else:
            inequality_constraints = None

        # Vincoli di uguaglianza
        if len(self.equality_constraints_fn_list)>0:
            equality_constraints = torch.stack(
                [constraint_fn(**kwargs) for constraint_fn in self.equality_constraints_fn_list], dim=0
            ).to(device)
        else:
            equality_constraints = None

        return inequality_constraints, equality_constraints


    def compute_score(self, **kwargs):
        """
        Calcola uno score che combina la funzione obiettivo e le penalità per i vincoli insoddisfatti,
        favorendo il miglior compromesso tra obiettivo e soddisfacimento dei vincoli.
        
        Args:
            kwargs (dict): Contiene 'objective_function', 'inequality_constraints' e 'equality_constraints'.
        
        Returns:
            torch.Tensor: Lo score calcolato.
        """
        # Estrarre la funzione obiettivo e i vincoli
        objective_function = kwargs.get('original_objective_function')
        inequality_constraints = kwargs.get('inequality_constraints')
        equality_constraints = kwargs.get('equality_constraints')
        #print('Original_objective function',objective_function.item())
        # Inizializza lo score con la funzione obiettivo
        score = objective_function.clone()

        # Inizializza una variabile per il conteggio delle violazioni dei vincoli
        total_penalty = 0
        
        # Penalità per vincoli di disuguaglianza
        if inequality_constraints is not None:
            for i,macro_constraint in enumerate(self.macro_constraints_list):
                if len(macro_constraint) > 0:
                    #if self.teacher_model is not None:
                        #print()
                        #print(f'Fairness Constraints #{i}:{torch.max(torch.clamp(inequality_constraints[macro_constraint], min=0))}')
                        #if i == len(self.macro_constraints_list)-1:
                        #    a =inequality_constraints[macro_constraint]
                           # print('Wasserstein Inequalities Constraints:',a)  
                       #print()
                    # Somma delle violazioni positive (vincoli insoddisfatti)
                    inequality_penalty = torch.max(torch.clamp(inequality_constraints[macro_constraint], min=0))   
                    # Incremento della penalità totale per ogni violazione di disuguaglianza
                    total_penalty += inequality_penalty*self.gamma_constraint
                    if self.hard_mode:
                        if self.teacher_model is not None:
                            if i == len(self.macro_constraints_list)-1:       
                                if inequality_penalty > 0:
                                    total_penalty = 10000
                                    print('Wasserstein Inequalities Constraints:',inequality_constraints[macro_constraint])

                    # usa un parametro di penalità specifico

        # Penalità per vincoli di uguaglianza
        if equality_constraints is not None:
            # Penalità basata sulla deviazione dai vincoli di uguaglianza
            equality_penalty = torch.max(torch.abs(equality_constraints))
            # Incremento della penalità totale per le uguaglianze insoddisfatte
            total_penalty += equality_penalty * self.gamma_constraint

        # Peso per il tradeoff tra funzione obiettivo e penalità (maggiore peso alle violazioni di vincolo)
        #tradeoff_weight = kwargs.get('tradeoff_weight', 0.5)

        # Score finale: bilancia obiettivo e penalità con il tradeoff
        score += total_penalty
        print('Total Penalty:',total_penalty)
        return score

       
    def compute_loss_fn(self, **kwargs):
        """
        Calcola la loss usando la funzione obiettivo e le penalità sui vincoli.
        Include modifiche per migliorare la stabilità del gradiente.

        Args:
            kwargs (dict): Contiene la funzione obiettivo, i vincoli di uguaglianza e disuguaglianza.
        
        Returns:
            torch.Tensor: La loss calcolata.
        """
        objective_function = kwargs['objective_function']
        batch_objective_function = kwargs['batch_objective_function']
        equality_constraints = kwargs.get('equality_constraints')
        inequality_constraints = kwargs.get('inequality_constraints')

        group_ids = kwargs.get('group_ids')
        # Inizializza la loss con la funzione obiettivo
        assert group_ids is not None, 'Group ids must be provided'
        if group_ids is not None:
            group_losses = []
            #group_counts = torch.tensor([len(group_ids[group_name]) for group_name in self.target_groups], device=self.device)
            for group_name in self.target_groups:
                group_list = group_ids[group_name]
                unique_groups = torch.unique(group_list)
                total_weight = 0
                for group in unique_groups:
                    
                    mask = group_list == group  # Seleziona i campioni del gruppo
                    if mask.sum() > 0:  # Evita problemi con gruppi vuoti
                        group_loss = batch_objective_function[mask].mean()
                        weight = torch.pow(1.0 - mask.sum() / batch_objective_function.shape[0], self.dro_lambda)
                        group_losses.append(weight*group_loss)
                        total_weight += weight
            # Group DRO: seleziona il gruppo con la perdita massima
            if len(group_losses) > 0:
                #worst_group_loss = torch.logsumexp(torch.stack(group_losses),dim=0)  
                loss = torch.stack(group_losses).sum()#/total_weight
                 # Aggiunge la penalità DRO
        
        else: 
            loss = objective_function.clone()
        
        #loss = objective_function.clone()
        # Penalità sui vincoli di uguaglianza
        if equality_constraints is not None and len(self.equality_constraints_fn_list) > 0:
            # Penalità lineare su uguaglianze per maggiore stabilità
            equality_penalty = torch.mean(torch.abs(equality_constraints))
            equality_penalty *= self.mu  # Opzionale: amplifica con parametro `mu`
            
            # Moltiplicatore di Lagrange per uguaglianza
            equality_lagrange_multipliers = (self.equality_lambdas * equality_constraints).sum()
            
            # Aggiorna la loss con il termine di uguaglianza
            loss += equality_penalty + equality_lagrange_multipliers
        
        if inequality_constraints is not None and len(self.inequality_constraints_fn_list) > 0:
            # Penalità sui vincoli di disuguaglianza
            if torch.any(self.inequality_lambdas > 0):
                inequality_penalty = torch.sum(torch.clamp(inequality_constraints,min=0))
                
                # Penalità smussata per evitare gradiente zero al limite
                inequality_lagrange_multipliers = torch.sum(
                    torch.pow(
                        torch.clamp(self.inequality_lambdas + self.mu * inequality_constraints, min=0), 2
                    )
                )
                inequality_lagrange_multipliers -= torch.pow(self.inequality_lambdas, 2).sum()
                inequality_lagrange_multipliers /= (2 * self.mu)

                # Aggiorna la loss con il termine di disuguaglianza
                loss += inequality_lagrange_multipliers + inequality_penalty

        
        # Verifica e gestione di NaN nella loss
        if torch.isnan(loss).any():
            raise ValueError("NaN trovato nella loss!")

        return loss

    def compute_loss_fn_old(self, **kwargs):
        objective_function = kwargs['objective_function']
        equality_constraints = kwargs.get('equality_constraints')
        inequality_constraints = kwargs.get('inequality_constraints')
        
        # Gestione dei vincoli di uguaglianza
        if len(self.equality_constraints_fn_list) > 0:
            equality_lagrange_multipliers = (self.equality_lambdas * equality_constraints).sum()
            quadratic_penalty = 0.5 * self.mu * equality_constraints.pow(2).sum()
        else:
            equality_constraints = None
            equality_lagrange_multipliers = 0
            quadratic_penalty = 0

        # Gestione dei vincoli di disuguaglianza
        if len(self.inequality_constraints_fn_list) > 0:
            
            inequality_lagrange_multipliers = torch.pow(
                torch.max(torch.zeros_like(inequality_constraints).to(self.device), 
                        self.inequality_lambdas + self.mu * inequality_constraints), 2).sum()
            inequality_lagrange_multipliers -= torch.pow(self.inequality_lambdas, 2).sum()
            inequality_lagrange_multipliers /= (2 * self.mu)
        else:
            inequality_constraints = None
            inequality_lagrange_multipliers = 0

        # Calcolo del loss totale
        loss =  objective_function + equality_lagrange_multipliers + quadratic_penalty + inequality_lagrange_multipliers

        if torch.isnan(loss).any():
            raise ValueError("NaN trovato nella loss!")

        return loss

    def aggregate_kwargs(self, all_kwargs):
        """
        Aggrega i kwargs da tutti i batch calcolati separatamente.

        Args:
            all_kwargs: Lista di kwargs per ciascun batch, ognuno generato da _compute_kwargs.

        Returns:
            aggregated_kwargs: Dizionario combinato che aggrega i kwargs da tutti i batch.
        """
        # Inizializza i contenitori per i kwargs aggregati
        aggregated_kwargs = {}

        # Estrai i nomi delle chiavi dai kwargs del primo batch come riferimento
        reference_keys = all_kwargs[0].keys()

        # Inizializza i contenitori vuoti per ciascun riferimento
        for key in reference_keys:
            # Se i valori associati alle chiavi sono tensori, li mettiamo in una lista per il concatenamento successivo
            aggregated_kwargs[key] = []

        # Itera su ogni batch di kwargs
        for kwargs in all_kwargs:
            for key, value in kwargs.items():
                # Aggiungi i valori del batch alla lista corrispondente per ciascuna chiave
                aggregated_kwargs[key].append(value)

        # Dopo aver accumulato tutti i tensori, concatenali
        for key in aggregated_kwargs:
            if isinstance(aggregated_kwargs[key][0], torch.Tensor):
                # Concatena i tensori lungo la prima dimensione (batch) per formare il kwargs finale
                aggregated_kwargs[key] = torch.cat(aggregated_kwargs[key], dim=0)
            else:
                # Nel caso di valori non tensori, mantieni una lista semplice
                aggregated_kwargs[key] = aggregated_kwargs[key]

        return aggregated_kwargs
     
    def _compute_kwargs_in_batches(self, loader, model, use_entmax=True):
        """
        Calcola i kwargs per i DataLoader in batch più piccoli, accumula i risultati e calcola i vincoli alla fine.
        
        Args:
            loader: Il DataLoader da cui ottenere i dati.
            model: Il modello su cui effettuare le predizioni.
        
        Returns:
            kwargs: I kwargs combinati e aggregati per tutti i batch.
        """
        # Inizializza tutto come una lista
        all_logits = []
        all_labels = []
        all_group_ids = {group_name: [] for group_name in loader.dataset[0]['groups'].keys()}
        all_group_ids_list = {group_name: [] for group_name in loader.dataset[0]['groups_ids_list'].keys()}
        all_group_masks = {group_name: [] for group_name in loader.dataset[0]['groups'].keys()}
        all_positive_masks = []
        all_teacher_logits = []
        
        # Iteriamo sui batch del DataLoader per accumulare i dati necessari
        for batch in loader:
            inputs = batch['data'].float().to(self.device)  # Carica un batch alla volta
            outputs = model(inputs)  # Calcola le predizioni per il batch

            # Aggiungi le predizioni, le etichette e i group_ids per il calcolo successivo
            all_logits.append(outputs)
            all_labels.append(batch['labels'].to(self.device))
            
            for teacher_model_dict in self.teacher_model_list:           
                self.teacher_model = copy.deepcopy(self.model)
                self.teacher_model.load_state_dict(copy.deepcopy(teacher_model_dict))
                self.teacher_model.to(self.device)
                self.teacher_model.eval() 
            
                teacher_outputs = self.teacher_model(inputs)
                all_teacher_logits.append([teacher_outputs])
            
            # Aggiungi i group_ids per ogni gruppo presente nel batch
            for group_name in batch['groups'].keys():
                all_group_ids[group_name].append(batch['groups'][group_name].to(self.device))
                all_group_masks[group_name].append(batch['groups'][group_name].to(self.device))
            
            # Aggiungi i group_ids_list per ogni gruppo presente nel batch
            for group_name in batch['groups_ids_list'].keys():
                all_group_ids_list[group_name].append(batch['groups_ids_list'][group_name].to(self.device))
            
            # Aggiungi la positive_mask
            all_positive_masks.append(batch['positive_mask'].to(self.device))

        # Una volta processati tutti i batch, concatenamo i risultati
        final_logits = torch.cat(all_logits, dim=0)
        final_labels = torch.cat(all_labels, dim=0)
        final_teacher_logits_list = []
        final_teacher_logits = None
        #print(all_teacher_logits)
        for teacher_logits in all_teacher_logits:
            final_teacher_logits_list.append(torch.cat(teacher_logits,dim=0)) if len(all_teacher_logits) > 0 else torch.tensor([],device=self.device)
        if len(final_teacher_logits_list) > 0:  
            final_teacher_logits = torch.stack(final_teacher_logits_list,dim=0)
            #print('Final Teacher Logits Shape:',final_teacher_logits.shape)
        else: 
            final_teacher_logits = torch.tensor([],device=self.device)
        final_group_ids = {group_name: torch.cat(all_group_ids[group_name], dim=0) for group_name in all_group_ids}
        final_group_ids_list = {group_name: torch.cat(all_group_ids_list[group_name], dim=0) for group_name in all_group_ids_list}
        final_group_masks = {group_name: torch.cat(all_group_masks[group_name], dim=0) for group_name in all_group_masks}
        final_positive_masks = torch.cat(all_positive_masks, dim=0)

        # Calcola i kwargs per il dataset completo
        kwargs = {
            'logits': final_logits,
            'labels': final_labels,
            'groups': final_group_ids,
            'groups_ids_list': final_group_ids_list,
            'positive_mask': final_positive_masks,
            'teacher_logits':final_teacher_logits
        }

        # Calcolo dei vincoli su tutto il dataset alla fine
        kwargs = self._compute_kwargs(kwargs, final_logits, use_entmax=use_entmax)
        
        return kwargs

    def _compute_kwargs(self, batch, outputs, use_entmax=True):
        """
        Calcola i kwargs necessari per il calcolo della funzione obiettivo e dei vincoli.

        Args:
            batch: Il batch corrente dal DataLoader, contenente dati, labels, groups, ecc.
            outputs: Le predizioni del modello per il batch corrente.
            use_entmax: Flag per decidere se usare Entmax o Argmax per il calcolo delle probabilità.

        Returns:
            kwargs: Dizionario con i dati necessari per calcolare i vincoli e l'obiettivo.
        """
        # Assicurati che tutto sia già sul dispositivo corretto (GPU)
        device = self.device
        
        # Accedi ai gruppi (assicurati che esistano)
       
        group_ids = {group_name: batch['groups'][group_name].to(device) for group_name in batch['groups'].keys()}
        
        # Accedi a groups_ids_list (assicurati che esistano)
        if 'groups_ids_list' in batch:
            group_ids_list = {group_name: batch['groups_ids_list'][group_name].to(device) for group_name in batch['groups_ids_list'].keys()}
        else:
            print(batch.keys())
            raise ValueError("'groups_ids_list' non è presente nel batch. Verifica la struttura del dataset.")

        # Assicurati che positive_mask e labels siano presenti
        positive_mask = batch.get('positive_mask', None)
        if positive_mask is None:
            raise ValueError("'positive_mask' non è presente nel batch.")
        positive_mask = positive_mask.to(device)

        labels = batch.get('labels', None)
        if labels is None:
            raise ValueError("'labels' non è presente nel batch.")
        labels = labels.to(device)

        predictions = torch.argmax(outputs, dim=-1)
        # Calcola le probabilità usando entmax o argmax
        if use_entmax:
            probabilities = entmax_bisect(outputs, alpha=1.5, dim=-1)
        else:
            probabilities = torch.nn.functional.one_hot(predictions, num_classes=outputs.size(-1)).float()
        
        # Controllo di eventuali NaN nelle probabilità
        if torch.isnan(probabilities).any():
            raise ValueError('Probabilità contiene NaN!')
        teachers_probabilities_list = []
        teachers_predictions_list = []
        
        if 'teacher_logits' not in batch.keys():
            teachers_outputs_list = []
           
            for teacher_model_dict in self.teacher_model_list:           
                self.teacher_model = copy.deepcopy(self.model)
                self.teacher_model.load_state_dict(copy.deepcopy(teacher_model_dict))
                self.teacher_model.to(self.device)
                self.teacher_model.eval() 
                inputs = batch['data'].float().to(self.device)
                teacher_outputs = self.teacher_model(inputs)
                teachers_outputs_list.append(teacher_outputs)
            if len(teachers_outputs_list) > 0:    
                teacher_outputs = torch.stack(teachers_outputs_list,dim=0)
            else:
                teacher_outputs = torch.tensor([],device=self.device)
            batch['teacher_logits'] = teacher_outputs
            #print('Teacher Outputs Shape:',teacher_outputs.shape)
        else:
            teacher_outputs = batch['teacher_logits']
            #print(teacher_outputs)
            #print('Teacher Outputs Shape:',teacher_outputs.shape)
        #print('Teacher Model List:',self.teacher_model_list)
        
        for teacher_outputs in batch['teacher_logits']:
            if len(teacher_outputs) > 0:
                teacher_predictions = torch.argmax(teacher_outputs, dim=-1)
                if use_entmax:
                    teacher_probabilities = entmax_bisect(teacher_outputs, alpha=1.5, dim=-1)
                else:
                    teacher_probabilities = torch.nn.functional.one_hot(teacher_predictions, num_classes=outputs.size(-1)).float()
                
                if torch.isnan(teacher_probabilities).any():
                    raise ValueError('Teacher Probabilities contiene NaN!')
                teachers_probabilities_list.append(teacher_probabilities)
                teachers_predictions_list.append(teacher_predictions)
        
        if len(teachers_probabilities_list) > 0:
            teacher_probabilities = torch.stack(teachers_probabilities_list,dim=0)
            teacher_predictions = torch.stack(teachers_predictions_list,dim=0)
            
        else:
            teacher_probabilities = torch.tensor([],device=self.device)
            teacher_predictions = torch.tensor([],device=self.device)
        
        #print('Teacher Probabilities Shape:',teacher_probabilities.shape)
        # Costruzione del dizionario kwargs che contiene tutte le informazioni necessarie
        kwargs = {
            'group_ids': group_ids,  # ID del gruppo attuale
            'group_ids_list': group_ids_list,  # Lista di ID del gruppo
            'group_masks': group_ids,  # Maschere di gruppo per i vincoli
            'positive_mask': positive_mask,  # Maschera positiva per il batch
            'logits': outputs,  # Uscite del modello (logits)
            'labels': labels,  # Etichette del batch
            'probabilities': probabilities,  # Probabilità calcolate (entmax o argmax)
            'predictions': predictions,
            'teacher_probabilities': teacher_probabilities
        }
        
        # Calcolo della funzione obiettivo e dei vincoli
        objective_fn_value = self.objective_fn(**kwargs)
        inequality_constraints, equality_constraints = self.compute_constraints(**kwargs)
        
        original_objective_fn_value = self.original_objective_fn(**kwargs)
        batch_objective_fn_value = self.batch_objective_function(**kwargs)
        # Aggiungi i vincoli e l'obiettivo ai kwargs
        
        kwargs['inequality_constraints'] = inequality_constraints
        kwargs['equality_constraints'] = equality_constraints
        kwargs['objective_function'] = objective_fn_value
        kwargs['batch_objective_function'] = batch_objective_fn_value
        kwargs['original_objective_function'] = original_objective_fn_value
    
        return kwargs

    def _compute_metrics(self,metrics,prefix='val',**kwargs):
        group_ids = kwargs['group_ids']
        y_pred = kwargs['predictions']
        y_true = kwargs['labels']

        tmp_result = {}
        final_result = {}
        
                
        for metric in metrics:
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

    def _training_step(self, batch, batch_idx):
        self.model.train()
        inputs = batch['data'].float().to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        kwargs = self._compute_kwargs(batch, outputs,use_entmax=True)
        loss = self.compute_loss_fn(**kwargs)

        if torch.isnan(loss).any():
            raise ValueError('Loss contiene NaN!')
        
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.optimizer.step()

        return loss.item()

    def _train_eval_step(self,**kwargs):
        self.model.eval()
        with torch.no_grad():
            train_kwargs = kwargs['train_kwargs']
            outputs = train_kwargs['logits']
            targets = train_kwargs['labels']
            loss = self.compute_loss_fn(**train_kwargs)
            predictions = torch.argmax(outputs, dim=1)

            return loss.item(), outputs, targets, predictions
            
    def _validation_step(self,**kwargs):
        self.model.eval()
        with torch.no_grad():
            val_kwargs = kwargs['val_kwargs']
            outputs = val_kwargs['logits']
            targets = val_kwargs['labels']
            loss = self.compute_loss_fn(**val_kwargs)
            predictions = torch.argmax(outputs, dim=1)

            return loss.item(), outputs, targets, predictions

    def _evaluate_requirements(self, use_validation=True, **kwargs):
        # Assumiamo che i data_loader abbiano un solo batch
        current_kwargs = kwargs['val_kwargs'] if use_validation else kwargs['train_kwargs']
        self.model.eval()
        with torch.no_grad():
            # Eseguiamo il passo di validazione o il passo di training a seconda del flag
            if use_validation:
                loss, outputs, targets, predictions = self._validation_step(**kwargs)
                
            else:
                loss, outputs, targets, predictions = self._train_eval_step(**kwargs)


            # I gruppi possono essere processati direttamente
            groups_dict = {group_name: current_kwargs['group_ids'][group_name]for group_name in current_kwargs['group_ids'].keys()}

            # Valutazione dei requisiti tramite `self.requirement_set.evaluate()`
            requirements, _ ,hard_constraints_satisfied= self.requirement_set.evaluate(y_pred=predictions, y_true=targets, group_ids=groups_dict)

        return requirements, loss, outputs, targets, predictions, groups_dict,hard_constraints_satisfied

    
    def _evaluate_requirements_old(self, data_loader):
        outputs, targets, groups, predictions = [], [], [], []
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                self.model.eval()
                loss, output, target, prediction = self._validation_step(batch, batch_idx)
                outputs.append(output)
                targets.append(target)
                predictions.append(prediction)
                groups.append(batch['groups'])

            # Concatenazione dei risultati
            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0).cpu()
            predictions = torch.cat(predictions, dim=0).cpu()
            groups_dict = {group_name: torch.cat([batch[group_name] for batch in groups], dim=0).cpu() for group_name in groups[0].keys()}

            requirements, _ = self.requirement_set.evaluate(y_pred=predictions, y_true=targets, group_ids=groups_dict)
        
        return requirements, loss, outputs, targets, predictions, groups_dict

    def _update_metrics(self,**kwargs):
        self.model.eval()

        constraint_metrics = kwargs.get('constraints_metrics')
        
        val_requirements, val_loss, val_outputs, val_targets, val_predictions, val_groups_dict,hard_constraints_satisfied = self._evaluate_requirements(use_validation=True, **kwargs)
        train_requirements, train_loss, train_outputs, train_targets, train_predictions, train_groups_dict,_ = self._evaluate_requirements(use_validation=False, **kwargs)
        
        metrics = {
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_requirements': val_requirements,
            'train_requirements': train_requirements,
            'hard_constraints_satisfied': int(hard_constraints_satisfied)
        }
        metrics.update(constraint_metrics)

        # Calcolo delle metriche finali
        metrics.update(self._compute_metrics(self.metrics, val_predictions, val_targets, val_groups_dict, prefix='val', logits=val_outputs))
        metrics.update(self._compute_metrics(self.metrics, train_predictions, train_targets, train_groups_dict, prefix='train', logits=train_outputs))
        
        return metrics

    def set_constraints(self, inequality_constraints_fn_list, equality_constraints_fn_list,macro_constraints_list,inequality_lambdas, equality_lambdas):
        self.inequality_constraints_fn_list = inequality_constraints_fn_list
        self.equality_constraints_fn_list = equality_constraints_fn_list
        self.macro_constraints_list = macro_constraints_list
        self.inequality_lambdas=inequality_lambdas
        self.equality_lambdas=equality_lambdas

    def evaluate(self, model_dict, **kwargs):
       
        # Impostiamo il modello in modalità valutazione
        original_model_dict = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(model_dict)
        self.model.eval()
        self.model.to(self.device)
        metrics = self.update_alm_parameters_and_metrics(update_alm=False) 
        self.model.load_state_dict(original_model_dict)
      
        return metrics 
    
    def compute_val_kwargs(self,model_dict,use_training=False):
        original_model_dict = self.model.state_dict()
        self.model.load_state_dict(model_dict)
        self.model.eval()
        self.model.to(self.device)
        if use_training:
           loader = self.data_module.train_loader_eval(batch_size=None)
        else:
            loader = self.data_module.val_loader(batch_size=None)
        kwargs = self._compute_kwargs_in_batches(loader, self.model,use_entmax=False)
        self.model.load_state_dict(original_model_dict)
        self.model.to(self.device)
        return kwargs
    
    def compute_violations(self,val_kwargs,**kwargs):
        reduce_fn = kwargs.get('reduce_fn')
        inequality_constraints, equality_constraints = self.compute_constraints(**val_kwargs)
        results = {}

        violations = {k:None for k,_ in enumerate(self.macro_constraints_list)}
        
        violations_per_group_list = {}
        violations_per_group = {}
        for key,value_dict in self.group_cardinality.items():
            violations_per_group_list[key] = {k:[] for k in value_dict.keys()}
        
        for i,constraint_violation in enumerate(inequality_constraints):
            constraint = self.inequality_constraints_fn_list[i]
            target_groups = constraint.target_groups
            group_name = constraint.group_name
            if group_name is not None:
                for group in target_groups:
                    violations_per_group_list[group_name][group.item()].append(constraint_violation)
        #print('Violations per group list',violations_per_group_list)
        for key,value_dict in violations_per_group_list.items():
            try:
                violations_per_group[key] = {k:torch.stack(v).max().item() for k,v in value_dict.items()}
            except RuntimeError:
                violations_per_group[key] = 0
            #violations_per_group[key] = {k:torch.stack(v).max().item() for k,v in value_dict.items()}
        
        results['violations_per_group'] = copy.deepcopy(violations_per_group)
        
        for i,macro_constraint in enumerate(self.macro_constraints_list):
            violations[i] = inequality_constraints[macro_constraint].detach().cpu().numpy()
        
        results['inequality_constraints_violations'] = inequality_constraints.detach().cpu().numpy()

        macro_constraints_violation = copy.deepcopy(violations)
        
        for i,macro_constraint in enumerate(self.macro_constraints_list):
            if len(macro_constraints_violation[i]) > 0:
                macro_constraints_violation[i] = [macro_constraints_violation[i].max()]
            else: 
                macro_constraints_violation[i] = []
        results['macro_constraints_violations'] = copy.deepcopy(macro_constraints_violation)
        
        #print('Violations computed',results)

        return results
    
    
    
    def fit(self, **kwargs):
        
        print(f'[{self.id}]:Number of inequality constraints:',len(self.inequality_constraints_fn_list))
        print(f'Macro constraints:',self.macro_constraints_list)
        num_epochs = kwargs.get('num_epochs', -1)
        disable_log = kwargs.get('disable_log', False)
        evaluate_best_model = kwargs.get('evaluate_best_model', True)
        n_rounds = self.num_epochs if num_epochs == -1 else num_epochs
        
        self.teacher_model_list = kwargs.get('teacher_model_list',[])
        start_model_dict = kwargs.get('start_model_dict')
        
        
        if start_model_dict is not None:
            self.model.load_state_dict(copy.deepcopy(start_model_dict))

        self.model.to(self.device)
       
        #print('Starting training...')
        #print('Updating metrics')
        #print('Starting training...')
        #print('Constraints',self.inequality_constraints_fn_list)
        #print('Multipliers',self.inequality_lambdas)
        
        #self.hard_mode = kwargs.get('use_first_model')
        self.hard_mode = False
        metrics = self.update_alm_parameters_and_metrics(update_alm=True) 
        if  self.hard_mode:
            for checkpoint in self.checkpoints:
                if isinstance(checkpoint, EarlyStopping):
                    stop, counter = checkpoint(metrics=metrics)
                    metrics['early_stopping'] = counter
                    if stop:
                        if not disable_log:
                            self.logger.log(metrics)
                        raise EarlyStoppingException

                elif isinstance(checkpoint, ModelCheckpoint):
                    model_checkpoint = checkpoint(save_fn=self.save, metrics=metrics)
                    metrics['model_checkpoint'] = 1 if model_checkpoint else 0
        #print('Starting Metrics: ',metrics)
        if not disable_log:
            self.logger.log(metrics)
        #print('Metrics updated')
        # Inizializzazione dei parametri ALM
        #self._init_alm_parameters()
        self.model.train()
        self.optimizer = self.optimizer_fn(self.model.parameters())
               
        try:
            for epoch in tqdm.tqdm(range(n_rounds), desc=f'Epoch 0/{n_rounds}', total=n_rounds, unit='epoch'):
                train_loader = self.data_module.train_loader()
                batch_iterator = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_rounds}', leave=False)
                for batch_idx, batch in enumerate(batch_iterator):
                    self._training_step(batch, batch_idx)

                with torch.no_grad():                    
                    metrics = self.update_alm_parameters_and_metrics(update_alm=True,**kwargs)
                    
                    # Early stopping e model checkpoint
                    for checkpoint in self.checkpoints:
                        if isinstance(checkpoint, EarlyStopping):
                           
                            stop, counter = checkpoint(metrics=metrics)
                            metrics['early_stopping'] = counter
                            if stop:
                                if not disable_log:
                                    self.logger.log(metrics)
                                raise EarlyStoppingException

                        elif isinstance(checkpoint, ModelCheckpoint):
                            model_checkpoint = checkpoint(save_fn=self.save, metrics=metrics)
                            metrics['model_checkpoint'] = 1 if model_checkpoint else 0
                    #print('Metrics: ',metrics)
                    if not disable_log:
                        self.logger.log(metrics)
                    # Aggiorna la descrizione della barra tqdm con il numero di epoca corrente
                    batch_iterator.set_description(f'Epoch {epoch+1}/{n_rounds}')
        except EarlyStoppingException:
            pass

        # Caricamento del miglior modello se richiesto
        for checkpoint in self.checkpoints:
            if isinstance(checkpoint, ModelCheckpoint):
                if os.path.exists(checkpoint.get_model_path()):
                    self.load(checkpoint.get_model_path())
        #evaluate_best_model = True
        if evaluate_best_model:
            self.model.eval()
           
            metrics = self.update_alm_parameters_and_metrics(update_alm=True,**kwargs)
            #print('Best model evaluated: ', metrics)
            final_metrics = {f'final_{name}': value for name, value in metrics.items()}
            if not disable_log:
               self.logger.log(final_metrics)
            
        return copy.deepcopy(self.model.state_dict())


    
    def score(self, data_loader, metrics, prefix=''):
        assert len(data_loader) == 1, "Data loader should have un solo batch"
        assert isinstance(metrics, list), "Metrics should be a list"
        
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode
        
        # Otteniamo il batch unico
        batch = next(iter(data_loader))

        # Calcoliamo kwargs usando _compute_kwargs per il batch unico
        val_kwargs = self._compute_kwargs(batch, self.model(batch['data'].float().to(self.device)), use_entmax=False)

        # Passiamo val_kwargs direttamente a _validation_step
        loss, outputs, targets, predictions = self._validation_step(val_kwargs=val_kwargs)

        # Estrazione dei gruppi da kwargs
        groups_dict = val_kwargs['group_ids']

        # Valutazione dei requisiti (requirements)
        requirements, _ = self.requirement_set.evaluate(
            y_pred=predictions, y_true=targets, group_ids=groups_dict
        )

        # Calcolo delle metriche
        scores = self._compute_metrics(metrics, predictions, targets, groups_dict, prefix=prefix, logits=outputs)
        
        # Aggiungiamo i risultati di "requirements" e "loss" alle metriche
        scores[f'{prefix}_requirements'] = requirements
        scores[f'{prefix}_loss'] = loss

        return scores


    
        