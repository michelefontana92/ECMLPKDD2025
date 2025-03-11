from experiments import GlofairCentralizedExperiment
import shutil
from .credit_run import CreditRun
from ..run_factory import register_run
from requirements import RequirementSet,ConstrainedRequirement,UnconstrainedRequirement
from surrogates import SurrogateFunctionSet,SurrogateFactory
from metrics import MetricsFactory

@register_run('credit_f1')
class CreditF1(CreditRun):
    def __init__(self,**kwargs) -> None:
        super(CreditF1, self).__init__(**kwargs)
        self.project_name = 'CentralizedCreditF1Alpha2'
        self.num_clients = 10
        self.lr=1e-4
        self.num_federated_rounds = 1
        self.num_epochs = 1
        self.fine_tune_epochs = 0
        self.project_name =  kwargs.get('project_name')
        self.start_index = kwargs.get('start_index')
        self.training_group_name = 'Gender'

        self.surrogate_set = SurrogateFunctionSet([
                                       SurrogateFactory.create(name='binary_f1',
                                                              surrogate_weight=1,
                                                              average='weighted',
                                                              distributed_env=False,
                                                              group_name='Gender'
                                                              ),
                                      SurrogateFactory.create(name='diff_demographic_parity',
                                                              surrogate_weight=1,
                                                              average='weighted',
                                                              distributed_env=False,
                                                              group_name='Gender'
                                                              )
                                      ])

        self.requirement_set = RequirementSet([
            UnconstrainedRequirement(name='unconstraned_performance_requirement',
                             metric = MetricsFactory.create_metric(
                                    metric_name='performance'),
                             weight=1,
                             mode='max',
                             bound=1.0,
                             performance_metric='f1'
                             ),
             ConstrainedRequirement(name='dp_requirement',
                                           metric = MetricsFactory.create_metric(
                                                    metric_name='demographic_parity',
                                                    group_name=self.training_group_name,
                                                    group_ids={self.training_group_name:list(range(2))}),
                                            weight=3,
                                            operator='<=',
                                            threshold=0.2),       
                        ])
    
    
    def setUp(self):
       
        self.experiment = GlofairCentralizedExperiment( sensitive_attributes=self.sensitive_attributes,
                                            dataset=self.dataset,
                                            data_root=self.data_root,
                                            model=self.model,
                                            num_clients=self.num_clients,
                                            num_federated_rounds=self.num_federated_rounds,
                                            num_local_epochs=self.num_epochs,
                                            fine_tune_epochs=self.fine_tune_epochs,
                                            lr=self.lr,
                                            project=self.project_name,
                                            training_group_name=self.training_group_name,
                                            surrogate_set=self.surrogate_set,
                                            requirement_set=self.requirement_set,
                                            start_index=self.start_index,
                                            )

    def run(self):
        self.experiment.setup()
        self.experiment.run()

    def tearDown(self) -> None:
        shutil.rmtree(f'checkpoints/{self.project_name}')