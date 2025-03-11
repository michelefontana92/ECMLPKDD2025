from experiments import GlofairCentralizedExperiment
import shutil
from .folk_binary_run import FolkTablesBinaryRun
from ..run_factory import register_run
from requirements import RequirementSet,ConstrainedRequirement,UnconstrainedRequirement
from metrics import MetricsFactory
from surrogates import SurrogateFunctionSet,SurrogateFactory
from wrappers import TorchNNLagrangianWrapper
from dataloaders import DataModule
from torch.optim import Adam
from functools import partial
from callbacks import EarlyStopping, ModelCheckpoint
from loggers import WandbLogger
from torch.nn import CrossEntropyLoss
from itertools import product, combinations
import torch
def generate_intersectional_combinations(data,name):
    """
    Genera tutte le possibili coppie di combinazioni intersezionali di attributi per il calcolo della Demographic Parity (DP),
    rispettando il formalismo richiesto:
    - Un solo attributo avrà due valori distinti.
    - Gli altri attributi avranno lo stesso valore in entrambe le combinazioni.
    
    Args:
        data: Un dizionario in cui le chiavi rappresentano gli attributi (ad esempio, "Race", "Job", "MaritalStatus"),
              e i valori sono liste di categorie associate a ciascun attributo (ad esempio, ["White", "Black"] per "Race").
              
    Returns:
        intersectional_combinations: Una lista di dizionari, ciascuno contenente una chiave come "Group_1"
                                      e un dizionario che rappresenta la coppia di combinazioni intersezionali da confrontare.
    """
    # Ottieni le chiavi (attributi) del dizionario
    keys = list(data.keys())
    
    # Calcola il prodotto cartesiano di tutti i valori associati a ciascun attributo (combinazioni intersezionali)
    combined_values = list(product(*data.values()))  # Tutte le possibili combinazioni intersezionali
    
    # Lista finale che conterrà tutte le coppie intersezionali da confrontare
    intersectional_combinations = []
    
    # Variabile di indice per creare nomi unici per i gruppi
    idx = 1
    
    # Genera tutte le possibili coppie di combinazioni intersezionali
    for pair in combinations(combined_values, 2):
        new_key = f'{name}_Group_{idx}'
        
        # Crea un dizionario che rappresenta le due combinazioni intersezionali da confrontare
        new_entry = {}
        distinct_attribute_count = 0  # Tiene traccia degli attributi distinti
        for i, key in enumerate(keys):
            if pair[0][i] != pair[1][i]:
                distinct_attribute_count += 1
                # Se i valori sono distinti, crea una lista con i due valori distinti
                new_entry[key] = [pair[0][i], pair[1][i]]
            else:
                # Se i valori sono uguali, crea una lista con un solo valore
                new_entry[key] = [pair[0][i]]
        
        # Aggiungi la coppia solo se c'è un solo attributo con due valori distinti
        if distinct_attribute_count == 1:
            intersectional_combinations.append((new_key, new_entry))
            idx += 1  # Incrementa l'indice per il nome del gruppo
    

    return intersectional_combinations

def get_number_of_combinations(data,name):
    keys = list(data.keys())
    # Calcola il prodotto cartesiano di tutti i valori associati a ciascun attributo (combinazioni intersezionali)
    combined_values = list(product(*data.values()))  # Tutte le possibili combinazioni intersezionali
    N = len(combined_values)
    return N*(N-1)/2

@register_run('folk_binary_alm_centralized_multiple')
class FolkTablesBinaryALMCentralized(FolkTablesBinaryRun):
    def __init__(self,**kwargs) -> None:
        super(FolkTablesBinaryALMCentralized, self).__init__(**kwargs)
        self.dataset = 'folktables_binary'

        self.metrics_list = kwargs.get('metrics_list')
        self.groups_list = kwargs.get('groups_list')
        self.threshold_list = kwargs.get('threshold_list')

        self.use_multiclass = False
        self.use_monolith=False
        self.id = kwargs.get('id')
        self.num_clients = 10
        
        self.lr=1e-4
        self.diff=0
        self.loss = partial(CrossEntropyLoss)
        self.num_lagrangian_epochs = 1
        self.num_epochs = 100
        self.fine_tune_epochs = 0
        self.batch_size = 128
        self.project_name =  kwargs.get('project_name')
        self.start_index = kwargs.get('start_index')
        
        self.training_group_name = kwargs.get('group_name')
        self.use_hale = kwargs.get('use_hale')
        self.metric = kwargs.get('metric_name')
        self.onlyperf = kwargs.get('onlyperf')
        self.threshold = kwargs.get('threshold')
        
        
        self.group_ids={self.training_group_name:list(range(2))}
        self.checkpoint_dir = kwargs.get('checkpoint_dir', 
                                         f'checkpoints/{self.project_name}')
        self.checkpoint_name = kwargs.get('checkpoint_name',
                                           'model.h5')
        self.verbose = kwargs.get('verbose', False)
        self.optimizer = Adam(self.model.parameters(),
                              lr=self.lr
                              )
        
        self.early_stopping_patience = kwargs.get('early_stopping_patience',15)
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
        
       
        self.metrics = [MetricsFactory().create_metric('performance')]
        
            
        self.objective_function = SurrogateFactory.create(name='performance',
                                                              surrogate_name='cross_entropy',
                                                              weight=1,
                                                              average='weighted'
                                                              
                                                              )

            
        self.original_objective_fn =  SurrogateFactory.create(name='binary_f1',
                                                              surrogate_name='binary_f1',
                                                              weight=1,
                                                              average='weighted'                                                              
                                                              )
      
        self.equality_constraints = []
        self.inequality_constraints = [SurrogateFactory.create(name='binary_f1',
                                                              surrogate_name='cross_entropy',
                                                              weight=1,
                                                              average='weighted',
                                                              upper_bound=0.60,
                                                              use_max=True,
                                                              )]
        self.lagrangian_callbacks = [EarlyStopping(patience=2,
                                            monitor='score',mode='min'
                                            )]
        
        
        
        self.macro_constraints_list = [[0]]
        idx_constraint = 1
            
        for metric,group,threshold in zip(self.metrics_list,self.groups_list,self.threshold_list):
            print(f'Using metric {metric} with threshold {threshold} for group {group}')
            self.threshold = threshold
            self.metric = metric
            self.training_group_name = group
            self.num_groups = self.compute_group_cardinality(self.training_group_name)
            self.group_ids={self.training_group_name:list(range(self.num_groups))}
            
            self.metrics += [MetricsFactory().create_metric(metric,
                                                        group_ids=self.group_ids,
                                                        group_name =  self.training_group_name,
                                                        use_multiclass=self.use_multiclass,
                                                        num_classes=self.num_classes)]
            if self.use_monolith:   
                fairness_surrogate_name = f'diff_{self.metric}'
                constraint =  SurrogateFactory.create(name=fairness_surrogate_name,
                                                                    surrogate_weight=1,
                                                                    average='weighted',
                                                                    distributed_env=True,
                                                                    group_name=self.training_group_name,
                                                                    unique_group_ids={
                                                                        self.training_group_name:list(range(self.num_groups))
                                                                        },
                                                                    lower_bound=self.threshold - self.diff,
                                                                    use_max=True,
                                                                    multiclass=self.use_multiclass
                                                                    )
                self.inequality_constraints.append(constraint)
                self.lagrangian_callbacks.append(EarlyStopping(patience=2,
                                            monitor='score',mode='min'
                                            )) 
           
            else:
                fairness_surrogate_name = f'diff_{self.metric}'
                for group_name,attributes_dict in self.sensitive_attributes:
                    print(f'Using sensitive attributes {group_name}:\n',attributes_dict)
                    #print(self.sensitive_attributes)
                    self.num_groups = self.compute_group_cardinality(group_name)
                    idx = 1
                    macro_constraint = []
                    for i in range(self.num_groups):
                        for j in range(i+1,self.num_groups):
                            target_groups = torch.tensor([i,j])
                            print(f'Constraint {idx}: Target groups: {target_groups}')    
                            idx+=1
                            constraint =  SurrogateFactory.create(name=fairness_surrogate_name,
                                                                        surrogate_name=f'{fairness_surrogate_name}_{group_name}',
                                                                        surrogate_weight=1,
                                                                        average='weighted',
                                                                        group_name=group_name,
                                                                        unique_group_ids={
                                                                            group_name:list(range(self.num_groups))
                                                                            },
                                                                        lower_bound=self.threshold,
                                                                        use_max=True,
                                                                        multiclass=self.use_multiclass,
                                                                        target_groups = target_groups
                                                                        )
                            self.inequality_constraints.append(constraint)
                            self.lagrangian_callbacks.append(EarlyStopping(patience=2,
                                                    monitor='score',mode='min'
                                                    )) 
        
                        
                            macro_constraint.append(idx_constraint)
                            idx_constraint+=1

                    self.macro_constraints_list.append(macro_constraint)
                    print(f'Macro constraints list: {self.macro_constraints_list}')
    
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

        print('Creating datamodule')
        self.data_module = DataModule(dataset=f'{self.dataset}',
                                     root = f'data/{self.dataset.capitalize()}',
                                     train_set=f'{path}_train.csv',
                                     val_set=f'{path}_val.csv',
                                     test_set=f'{path}_val.csv',
                                     batch_size=128,
                                     num_workers=0,
                                     sensitive_attributes=self.sensitive_attributes,
                                     )
        print('Creating logger')
        self.logger = WandbLogger(
            project=self.project_name,
            config= self.config,
            id=self.id,
            checkpoint_dir= self.checkpoint_dir,
            checkpoint_path = self.checkpoint_name,
            data_module=self.data_module if self.log_model else None,
            resume=False
        )

        print('Creating wrapper')
        self.wrapper = TorchNNLagrangianWrapper(
            model=self.model,
            optimizer=self.optimizer,
            optimizer_fn = self.optimizer_fn,
            checkpoints=self.callbacks,
            lagrangian_checkpoints=self.lagrangian_callbacks,

            macro_constraints_list=self.macro_constraints_list,
            logger=self.logger,

            objective_fn = self.objective_function,
            original_objective_fn = self.original_objective_fn,
            equality_constraints=self.equality_constraints,
            inequality_constraints=self.inequality_constraints,
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
       self.wrapper.fit(num_epochs=self.num_epochs)

    def tearDown(self) -> None:
        pass
        #shutil.rmtree(f'checkpoints/{self.project_name}')