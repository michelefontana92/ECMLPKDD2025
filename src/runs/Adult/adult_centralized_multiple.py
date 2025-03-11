from experiments import GlofairCentralizedExperiment
import shutil
from .adult_run import AdultRun
from ..run_factory import register_run
from requirements import RequirementSet,ConstrainedRequirement,UnconstrainedRequirement
from metrics import MetricsFactory
from surrogates import SurrogateFunctionSet,SurrogateFactory
from wrappers import TorchNNMOWrapper
from dataloaders import DataModule
from torch.optim import Adam
from functools import partial
from callbacks import EarlyStopping, ModelCheckpoint
from loggers import WandbLogger
from torch.nn import CrossEntropyLoss

@register_run('adult_centralized_multiple')
class AdultCentralized(AdultRun):
    def __init__(self,**kwargs) -> None:
        super(AdultCentralized, self).__init__(**kwargs)
        self.dataset = 'adult'

        self.metrics_list = kwargs.get('metrics_list')
        self.groups_list = kwargs.get('groups_list')
        self.threshold_list = kwargs.get('threshold_list')


        self.id = kwargs.get('id')
        self.num_clients = 10
        self.lr=1e-4
        self.loss = partial(CrossEntropyLoss)
        self.num_lagrangian_epochs = 1
        self.num_epochs = 30
        self.fine_tune_epochs = 0
        self.batch_size = 128
        self.project_name =  kwargs.get('project_name')
        self.start_index = kwargs.get('start_index')
        
        self.training_group_name = kwargs.get('group_name')
        self.use_hale = kwargs.get('use_hale')
        self.metric = kwargs.get('metric_name')
        self.onlyperf = kwargs.get('onlyperf')
        self.threshold = kwargs.get('threshold')
        
        self.num_groups = 4 if self.training_group_name == 'GenderRace' else 2
        self.group_ids={self.training_group_name:list(range(2))}
        self.checkpoint_dir = kwargs.get('checkpoint_dir', 
                                         f'checkpoints/{self.project_name}')
        self.checkpoint_name = kwargs.get('checkpoint_name',
                                           'model.h5')
        self.verbose = kwargs.get('verbose', False)
        self.optimizer = Adam(self.model.parameters(),
                              lr=self.lr
                              )
        
        self.early_stopping_patience = kwargs.get('early_stopping_patience',10)
        self.monitor = kwargs.get('monitor','val_requirements')
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
        self.metrics = [MetricsFactory().create_metric('performance')]
        for attributes in  self.sensitive_attributes:
            group_name = attributes[0]   
            self.num_groups = 4 if group_name == 'GenderRace' else 2
            self.group_ids={group_name:list(range(self.num_groups))}
            
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
        perf_surrogate_name = f'binary_f1' if self.use_hale else 'performance'
        if self.onlyperf:
            self.surrogate_set = SurrogateFunctionSet([
                                       SurrogateFactory.create(name=perf_surrogate_name,
                                                              surrogate_weight=1,
                                                              average='weighted',
                                                              distributed_env=True,
                                                              group_name=self.training_group_name
                                                              ),
                                     
                                      ])  
            self.requirement_set = RequirementSet([
                UnconstrainedRequirement(name='unconstrained_performance_requirement',
                                metric = MetricsFactory.create_metric(
                                        metric_name='performance'),
                                weight=1,
                                mode='max',
                                bound=1.0,
                                performance_metric='f1'
                                )])             
        else:
            perf_surrogate_name = f'binary_f1' if self.use_hale else 'performance'
            surrogate_list = [ SurrogateFactory.create(name=perf_surrogate_name,
                                                                surrogate_weight=1,
                                                                average='weighted',
                                                                distributed_env=True,
                                                                group_name=self.training_group_name,
                                                                use_max=True
                                                                )]
            requirement_list = [
                UnconstrainedRequirement(name='unconstrained_performance_requirement',
                                metric = MetricsFactory.create_metric(
                                        metric_name='performance'),
                                weight=1,
                                mode='max',
                                bound=1.0,
                                performance_metric='f1'
                                )]
            
            for metric,group,threshold in zip(self.metrics_list,self.groups_list,self.threshold_list):
                #print(f'Using metric {metric} with threshold {threshold} for group {group}')
                
                self.threshold = threshold
                self.metric = metric
                self.training_group_name = group
                self.num_groups = 4 if self.training_group_name == 'GenderRace' else 2
                self.group_ids={self.training_group_name:list(range(self.num_groups))}
                fairness_surrogate_name = f'diff_{self.metric}' if self.use_hale else self.metric
                surrogate =  SurrogateFactory.create(name=fairness_surrogate_name,
                                                                surrogate_weight=1,
                                                                average='weighted',
                                                                distributed_env=True,
                                                                group_name=self.training_group_name,
                                                                unique_group_ids={
                                                                    self.training_group_name:list(range(self.num_groups))
                                                                    },
                                                                lower_bound=self.threshold,
                                                                use_max=True
                                                                )
                surrogate_list.append(surrogate)
                requirement =  ConstrainedRequirement(name=f'{self.metric}_requirement',
                                           metric = MetricsFactory.create_metric(
                                                    metric_name=self.metric,
                                                    group_name=self.training_group_name,
                                                    group_ids=self.group_ids),
                                            weight=1,
                                            operator='<=',
                                            threshold=self.threshold)
                requirement_list.append(requirement)
            self.surrogate_set = SurrogateFunctionSet(surrogate_list)               
            self.requirement_set = RequirementSet(requirement_list)
        
        #print('Using requirements:',self.requirement_set)
        #print('Using surrogates:',self.surrogate_set)
                        
    
    
    def setUp(self):
        self.config = {
            'hidden1':self.hidden1,
            'hidden2':self.hidden2,
            'dropout':self.dropout,
            'lr':self.lr,
            'batch_size':self.batch_size,
            'dataset':self.dataset,
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

       
        self.wrapper = TorchNNMOWrapper(
            model=self.model,
            optimizer=self.optimizer,
            optimizer_fn = self.optimizer_fn,
            checkpoints=self.callbacks,
            surrogate_functions = self.surrogate_set,
            requirement_set=self.requirement_set,
            logger=self.logger,
            loss = self.loss,
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
        shutil.rmtree(f'checkpoints/{self.project_name}')