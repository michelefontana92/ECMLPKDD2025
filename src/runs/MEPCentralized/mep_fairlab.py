import shutil
from .mep_run import CentralizedMEPRun
from ..run_factory import register_run
from metrics import MetricsFactory
from surrogates import SurrogateFactory
from wrappers import OrchestratorWrapper
from dataloaders import DataModule
from torch.optim import Adam
from functools import partial
from callbacks import EarlyStopping, ModelCheckpoint
from loggers import WandbLogger
from torch.nn import CrossEntropyLoss
import torch

@register_run('mep_fairlab')
class MEPHierALMCentralized(CentralizedMEPRun):
    def __init__(self, **kwargs) -> None:
        super(MEPHierALMCentralized, self).__init__(**kwargs)
        

        # Estrazione dei parametri dal dizionario kwargs con valori di default
        self.metrics_list = kwargs.get('metrics_list')
        self.groups_list = kwargs.get('groups_list')
        self.threshold_list = kwargs.get('threshold_list')
        self.use_multiclass = kwargs.get('use_multiclass', False)
        self.use_monolith = kwargs.get('use_monolith', False)
        self.id = kwargs.get('id')
        self.num_clients = kwargs.get('num_clients', 10)
        
        self.lr = kwargs.get('lr', 1e-4)
        self.diff = kwargs.get('diff', 0)
        self.loss = partial(CrossEntropyLoss)
        self.num_lagrangian_epochs = kwargs.get('num_lagrangian_epochs', 1)
        
        self.fine_tune_epochs = kwargs.get('fine_tune_epochs', 0)
        self.batch_size = kwargs.get('batch_size', 128)
        self.project_name = kwargs.get('project_name')
        self.start_index = kwargs.get('start_index', 0)
        
        self.training_group_name = kwargs.get('group_name')
        self.use_hale = kwargs.get('use_hale', False)
        self.metric = kwargs.get('metric_name')
        self.onlyperf = kwargs.get('onlyperf', False)
        self.threshold = kwargs.get('threshold')
        
        self.group_ids = {self.training_group_name: list(range(2))}
        self.checkpoint_dir = kwargs.get('checkpoint_dir', f'checkpoints/{self.project_name}')
        self.checkpoint_name = kwargs.get('checkpoint_name', 'model.h5')
        self.verbose = kwargs.get('verbose', False)
        self.optimizer_fn = partial(Adam, lr=self.lr)
        self.optimizer = Adam(self.model.parameters(),
                              lr=self.lr
                              )
        self.monitor = kwargs.get('monitor', 'val_constraints_score')
        self.mode = kwargs.get('mode', 'min')
        self.log_model = kwargs.get('log_model', False)

        self.num_subproblems = kwargs.get('num_subproblems')
        self.num_global_iterations = kwargs.get('num_global_iterations')
        self.num_epochs = kwargs.get('num_local_iterations')

        self.performance_constraint = kwargs.get('performance_constraint')
        self.delta=kwargs.get('delta', 0.2)
        self.max_constraints_in_subproblem = kwargs.get('max_constraints_in_subproblem')
        self.global_patience = kwargs.get('global_patience')
        print('Groups: ', self.groups_list)
        # Callbacks
        self.callbacks = [
            EarlyStopping(patience=self.global_patience, monitor=self.monitor, mode=self.mode),
            ModelCheckpoint(save_dir=self.checkpoint_dir, save_name=self.checkpoint_name, monitor=self.monitor, mode=self.mode)
        ]

        # Metriche
        self.metrics = [MetricsFactory().create_metric('performance')]

        # Funzione obiettivo e vincoli
        self.objective_function = SurrogateFactory.create(name='performance', surrogate_name='cross_entropy', weight=1, average='weighted')
        self.batch_objective_fn = SurrogateFactory.create(name='performance_batch', surrogate_name='cross_entropy', weight=1, average='weighted')
        self.original_objective_fn = SurrogateFactory.create(name='binary_f1', surrogate_name='binary_f1', weight=1, average='weighted')
        self.equality_constraints = []
        self.shared_macro_constraints = []
        if self.performance_constraint < 1.0:
            print('Performance constraint: ', self.performance_constraint)
            self.inequality_constraints = [SurrogateFactory.create(name='binary_f1', 
                                                                surrogate_name='cross_entropy', 
                                                                weight=1, average='weighted', 
                                                                upper_bound=self.performance_constraint,
                                                                use_max=True)]
            self.lagrangian_callbacks = [EarlyStopping(patience=2, monitor='score', mode='min')]
            self.macro_constraints_list = [[0]]
            self.shared_macro_constraints = [0]
            idx_constraint = 1
        else:
            print('No performance constraint')
            print()
            self.inequality_constraints = []
            self.lagrangian_callbacks = []
            self.macro_constraints_list = []
            idx_constraint = 0
        # Configurazione dei macro vincoli
        
        
        self.all_group_ids = {} 
        for metric, group, threshold in zip(self.metrics_list, self.groups_list, self.threshold_list):
            self.threshold = threshold
            self.metric = metric
            self.training_group_name = group
            self.num_groups = self.compute_group_cardinality(self.training_group_name)
            self.group_ids = {self.training_group_name: list(range(self.num_groups))}
            self.all_group_ids.update(self.group_ids)
            # Aggiunta della metrica
            self.metrics += [MetricsFactory().create_metric(metric, group_ids=self.group_ids, group_name=self.training_group_name, use_multiclass=self.use_multiclass, num_classes=self.num_classes)]
            macro_constraint = []
            for i in range(self.num_groups):
                for j in range(i + 1, self.num_groups):
                    target_groups = torch.tensor([i, j])
                    constraint = SurrogateFactory.create(name=f'diff_{self.metric}', surrogate_name=f'diff_{self.metric}_{group}', surrogate_weight=1, average='weighted', 
                                                         group_name=group, 
                                                         unique_group_ids={group: list(range(self.num_groups))}, 
                                                         lower_bound=self.threshold, 
                                                         use_max=True, 
                                                         multiclass=self.use_multiclass, 
                                                         target_groups=target_groups)
                    self.inequality_constraints.append(constraint)
                    self.lagrangian_callbacks.append(EarlyStopping(patience=2, monitor='score', mode='min'))
                    macro_constraint.append(idx_constraint)
                    idx_constraint += 1
            self.macro_constraints_list.append(macro_constraint)
            # Configurazione dei vincoli e macro constraints per ogni gruppo sensibile
            """
            for group_name, attributes_dict in self.sensitive_attributes:
                if group_name in self.all_group_ids:
                    self.num_groups = self.compute_group_cardinality(group_name)
                    macro_constraint = []
                    for i in range(self.num_groups):
                        for j in range(i + 1, self.num_groups):
                            target_groups = torch.tensor([i, j])
                            constraint = SurrogateFactory.create(name=f'diff_{self.metric}', surrogate_name=f'diff_{self.metric}_{group_name}', surrogate_weight=1, average='weighted', group_name=group_name, unique_group_ids={group_name: list(range(self.num_groups))}, lower_bound=self.threshold, use_max=True, multiclass=self.use_multiclass, target_groups=target_groups)
                            self.inequality_constraints.append(constraint)
                            self.lagrangian_callbacks.append(EarlyStopping(patience=2, monitor='score', mode='min'))
                            macro_constraint.append(idx_constraint)
                            idx_constraint += 1

                    self.macro_constraints_list.append(macro_constraint)
            """
        print('All group ids: ', self.all_group_ids)
        print('Macro constraints: ', self.macro_constraints_list)
        print('Num of macro constraints: ', len(self.macro_constraints_list))
    def setUp(self):
        # Configurazione di setup
        self.config = {
            'hidden1': self.hidden1,
            'hidden2': self.hidden2,
            'dropout': self.dropout,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'dataset': self.dataset,
            'optimizer': 'Adam',
            'num_lagrangian_epochs': self.num_lagrangian_epochs,
            'num_epochs': self.num_epochs,
            'patience': self.global_patience,
            'monitor': self.monitor,
            'mode': self.mode,
            'log_model': self.log_model
        }
        
        self.checkpoints_config = {
            'checkpoint_dir': self.checkpoint_dir,
            'checkpoint_name': self.checkpoint_name,
            'monitor': self.monitor,
            'mode': self.mode,
            'patience': self.global_patience
        }
        # Creazione del DataModule
        path = f'node_{self.start_index}/{self.dataset}' if self.start_index >= 0 else f'{self.dataset}_clean'
        self.data_module = DataModule(dataset=f'{self.dataset}', 
                                      root=self.data_root, 
                                      train_set=f'{path}_train.csv', 
                                      val_set=f'{path}_val.csv', 
                                      test_set=f'{path}_val.csv',
                                      batch_size=self.batch_size, 
                                      num_workers=0, 
                                      sensitive_attributes=self.sensitive_attributes,
                                      clean_data_path=self.clean_data_path)

        # Configurazione del logger
        self.logger = WandbLogger(project=self.project_name, 
                                  config=self.config, 
                                  id=self.id,
                                  checkpoint_dir=self.checkpoint_dir, 
                                  checkpoint_path=self.checkpoint_name,
                                  data_module=self.data_module if self.log_model else None)

        # Configurazione del wrapper HierarchicalLagrangianWrapper
        self.wrapper = OrchestratorWrapper(
            model=self.model,
            inequality_constraints=self.inequality_constraints,
            macro_constraints_list=self.macro_constraints_list,
            #target_groups=self.target_groups,
            batch_objective_fn=self.batch_objective_fn,
            min_subproblems=2,
            max_subproblems=5,
            optimizer_fn=self.optimizer_fn,
            optimizer=self.optimizer,
            objective_function=self.objective_function,
            original_objective_fn=self.original_objective_fn,
            equality_constraints=self.equality_constraints,
            metrics=self.metrics,
            num_epochs=self.num_epochs,
            loss = self.loss,
            data_module=self.data_module,
            logger=self.logger,
            lagrangian_checkpoints=self.lagrangian_callbacks,
            checkpoints=self.callbacks,
            all_group_ids=self.all_group_ids,
            checkpoints_config=self.checkpoints_config,
            shared_macro_constraints=self.shared_macro_constraints,
            delta=self.delta,
            max_constraints_in_subproblem=self.max_constraints_in_subproblem,
            
        )
        
    def run(self):
        # Esecuzione dell'addestramento
        self.wrapper.fit(num_global_iterations=self.num_global_iterations,
                         num_local_epochs=self.num_epochs,
                         num_subproblems=self.num_subproblems)
        
    def tearDown(self) -> None:
        # Pulizia finale dei file di checkpoint, se necessario
        #pass
        shutil.rmtree(f'checkpoints/{self.project_name}')
