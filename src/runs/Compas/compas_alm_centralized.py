from experiments import GlofairCentralizedExperiment
import shutil
from .compas_run import CompasRun
from ..run_factory import register_run
from requirements import RequirementSet,ConstrainedRequirement,UnconstrainedRequirement
from surrogates import SurrogateFunctionSet,SurrogateFactory
from metrics import MetricsFactory
from hard_label_estimator import HardLabelsEstimator
from wrappers import TorchNNLagrangianWrapper
from dataloaders import DataModule
from torch.optim import Adam
from functools import partial
from callbacks import EarlyStopping, ModelCheckpoint
from loggers import WandbLogger

@register_run('compas_alm_centralized')
class CompasALMCentralized(CompasRun):
    def __init__(self,**kwargs) -> None:
        super(CompasALMCentralized, self).__init__(**kwargs)
        self.id = kwargs.get('id')
        
        
        self.dataset = 'compas'
        self.num_clients = 10
        self.lr=1e-4
        self.num_lagrangian_epochs = 1
        self.num_epochs = 30
        self.fine_tune_epochs = 0
        self.batch_size = 128
        self.project_name =  kwargs.get('project_name')
        self.start_index = kwargs.get('start_index')

        self.training_group_name = kwargs.get('group_name')
        self.metric = kwargs.get('metric_name')
        self.threshold = kwargs.get('threshold')
        
        self.num_groups = 4 if self.training_group_name == 'GenderRace' else 2
        self.group_ids={self.training_group_name:list(range(self.num_groups))}
        self.checkpoint_dir = kwargs.get('checkpoint_dir', 
                                         f'checkpoints/{self.project_name}')
        self.checkpoint_name = kwargs.get('checkpoint_name',
                                           'model.h5')
        self.verbose = kwargs.get('verbose', False)
        self.optimizer = Adam(self.model.parameters(),
                              lr=self.lr
                              )
        self.early_stopping_patience = kwargs.get('early_stopping_patience',10)
        self.monitor = kwargs.get('monitor','val_constraints_score')
        self.mode = kwargs.get('mode','min')
        self.log_model = kwargs.get('log_model',False)
        self.optimizer_fn = partial(Adam,lr=self.lr)
        
        self.callbacks = [EarlyStopping(patience=self.early_stopping_patience,
                                        monitor=self.monitor,mode=self.mode
                                        ),

                          ModelCheckpoint(save_dir=self.checkpoint_dir,
                                          save_name = self.checkpoint_name,
                                          monitor=self.monitor,mode=self.mode)
                          ]
       
        self.lagrangian_callbacks = [EarlyStopping(patience=1,
                                        monitor=self.monitor,mode=self.mode
                                        )]

                          
        self.equality_constraints = [
           
            SurrogateFactory.create(name='binary_f1',
                                                              surrogate_name='binary_f1',
                                                              weight=1,
                                                              average='weighted',
                                                              distributed_env=True,
                                                              group_name=self.training_group_name,
                                                              class_idx=1
                                                              ) ,

            
                       
        ]

        self.inequality_constraints = [SurrogateFactory.create(name=f'diff_{self.metric}',
                                                              surrogate_name=f'diff_{self.metric}',
                                                              weight=1,
                                                              average='weighted',
                                                              distributed_env=True,
                                                              group_name=self.training_group_name,
                                                              class_idx=1,
                                                              lower_bound=self.threshold
                                                              )]                  
        

        self.requirement_set = RequirementSet([
            UnconstrainedRequirement(name='unconstrained_performance_requirement',
                             metric = MetricsFactory.create_metric(
                                    metric_name='performance'),
                             weight=1,
                             mode='max',
                             bound=1.0,
                             performance_metric='f1'
                             ),
             ConstrainedRequirement(name='dp_requirement',
                                           metric = MetricsFactory.create_metric(
                                                    metric_name=self.metric,
                                                    group_name=self.training_group_name,
                                                    group_ids=self.group_ids),
                                            weight=1,
                                            operator='<=',
                                            threshold=self.threshold),       
                        ])
    
    
    def setUp(self):
        self.config = {
            'hidden1':self.hidden1,
            'hidden2':self.hidden2,
            'dropout':self.dropout,
            'lr':self.lr,
            'batch_size':self.batch_size,
            'dataset':'compas',
            'optimizer':'Adam',
            'num_lagrangian_epochs':self.num_lagrangian_epochs,
            'num_epochs':self.num_epochs,
            'patience':self.early_stopping_patience,
            'monitor':self.monitor,
            'mode':self.mode,
            'log_model':self.log_model
        }
        
        if self.start_index < 0:
            path = f'{self.dataset}_clean'
        else:
            path = f'node_{self.start_index}/{self.dataset}'

        self.data_module = DataModule(dataset=f'{self.dataset}',
                                     root = f'data/{self.dataset.capitalize()}',
                                     train_set=f'{path}_train.csv',
                                     val_set=f'{path}_val.csv',
                                     test_set=f'{path}_val.csv',
                                     batch_size=128,
                                     num_workers=1,
                                     sensitive_attributes=self.sensitive_attributes,
                                     )
        self.logger = WandbLogger(
            project=self.project_name,
            config= self.config,
            id=self.id,
            checkpoint_dir= self.checkpoint_dir,
            checkpoint_path = self.checkpoint_name,
            data_module=self.data_module if self.log_model else None
        )

        self.metrics = [MetricsFactory().create_metric('performance')]
        for group_name in self.group_ids.keys():
            self.metrics += [
                        MetricsFactory().create_metric('demographic_parity',
                                                       group_ids=self.group_ids,
                                                       group_name = group_name),
                        MetricsFactory().create_metric('equal_opportunity',
                                                       group_ids=self.group_ids,
                                                       group_name = group_name),
                        MetricsFactory().create_metric('equalized_odds',
                                                       group_ids=self.group_ids,
                                                       group_name = group_name)
            ]
        self.wrapper = TorchNNLagrangianWrapper(
            model=self.model,
            optimizer=self.optimizer,
            optimizer_fn = self.optimizer_fn,
            checkpoints=self.callbacks,
            lagrangian_checkpoints=self.lagrangian_callbacks,

            requirement_set=self.requirement_set,
            logger=self.logger,
            equality_constraints=self.equality_constraints,
            inequality_constraints=self.inequality_constraints,
            num_lagrangian_epochs=self.num_lagrangian_epochs,
            num_epochs=self.num_epochs,
            fine_tune_epochs=self.fine_tune_epochs,
            verbose=self.verbose,
            data_module=self.data_module,
            training_group_name=self.training_group_name,
            metrics=self.metrics,
        )
        
    def run(self):
       self.setUp()
       self.wrapper.fit(num_epochs=self.num_epochs,
                        num_lagrangian_epochs = self.num_lagrangian_epochs)

    def tearDown(self) -> None:
        pass
        #shutil.rmtree(f'checkpoints/{self.project_name}')