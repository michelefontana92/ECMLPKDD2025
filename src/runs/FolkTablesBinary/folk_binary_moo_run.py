from experiments import GlofairCentralizedExperiment
import shutil
from .folk_binary_run import FolkTablesBinaryRun
from ..run_factory import register_run
from requirements import RequirementSet, ConstrainedRequirement, UnconstrainedRequirement
from metrics import MetricsFactory
from surrogates import SurrogateFunctionSet, SurrogateFactory
from wrappers import TorchNNMOWrapper
from dataloaders import DataModule
from torch.optim import Adam
from functools import partial
from callbacks import EarlyStopping, ModelCheckpoint
from loggers import WandbLogger
from torch.nn import CrossEntropyLoss
from itertools import product, combinations
import torch

@register_run('folk_binary_moo')
class FolkTablesMOORun(FolkTablesBinaryRun):
    def __init__(self, **kwargs) -> None:
        super(FolkTablesMOORun, self).__init__(**kwargs)
    

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
        self.monitor = 'val_requirements'
        self.callbacks = [
            EarlyStopping(patience=self.global_patience, monitor=self.monitor, mode=self.mode),
            ModelCheckpoint(save_dir=self.checkpoint_dir, save_name=self.checkpoint_name, monitor=self.monitor, mode=self.mode)
        ]

        
        self.all_group_ids = {} 
        self.metrics = [MetricsFactory().create_metric('performance')]
        self.approximations = [SurrogateFactory().create(name='performance',
                                                         surrogate_name='approx_performance', 
                                                         surrogate_weight=1)]
        self.requirements = [
            UnconstrainedRequirement(name='unconstrained_performance_requirement',
                             metric = MetricsFactory.create_metric(
                                    metric_name='performance'),
                             weight=1,
                             mode='max',
                             bound=1.0,
                             performance_metric='f1'
                             )]
        
     
        
        
        for metric, group, threshold in zip(self.metrics_list, self.groups_list, self.threshold_list):
           
            self.metric = metric
            self.training_group_name = group
            self.num_groups = self.compute_group_cardinality(self.training_group_name)
            self.group_ids = {self.training_group_name: list(range(self.num_groups))}
            
            # Aggiunta della metrica
            current_metric = MetricsFactory().create_metric(metric, group_ids=self.group_ids, group_name=self.training_group_name, 
                                                            )

            approximation = SurrogateFactory.create(name=f'{self.metric}', 
                                                                surrogate_name=f'approx_{self.metric}_{group}', 
                                                                surrogate_weight=1, 
                                                                reduction='mean', 
                                                                group_name=group, 
                                                                unique_group_ids={group: list(range(self.num_groups))}
                                                                )
            
            requirement = ConstrainedRequirement(name=f'{metric}_requirement',
                                                 metric = current_metric,
                                                 weight=1, 
                                                 operator='<=',
                                                 threshold=threshold)       
            self.approximations.append(approximation)
            self.metrics.append(current_metric)
            self.requirements.append(requirement)
            
        self.surrogate_function_set = SurrogateFunctionSet(self.approximations)
        self.requirement_set = RequirementSet(self.requirements)



    
    
    
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
        self.wrapper = TorchNNMOWrapper(
            model=self.model,
            optimizer_fn=self.optimizer_fn,
            optimizer=self.optimizer,

            metrics=self.metrics,
            num_epochs=self.num_epochs,
            loss = self.loss,
            data_module=self.data_module,
            logger=self.logger,
           
            checkpoints=self.callbacks,
            all_group_ids=self.all_group_ids,
            checkpoints_config=self.checkpoints_config,
            requirement_set=self.requirement_set,
            surrogate_functions=self.surrogate_function_set,
            
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
