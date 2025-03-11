from experiments import GlofairCentralizedExperiment
import shutil
from .adult_run import AdultRun
from ..run_factory import register_run
from requirements import RequirementSet,ConstrainedRequirement,UnconstrainedRequirement
from surrogates import SurrogateFunctionSet,SurrogateFactory
from metrics import MetricsFactory

@register_run('wasserstein_adult_test')
class WassersteinAdultTest(AdultRun):
    def __init__(self,**kwargs) -> None:
        super(WassersteinAdultTest, self).__init__(**kwargs)
        self.project_name = 'WassersteinAdultTest'
        self.num_clients = 10
        self.lr=1e-4
        self.num_federated_rounds = 1
        self.num_epochs = 10
        self.fine_tune_epochs = 30
        
        self.training_group_name = 'Gender'

        self.surrogate_set = SurrogateFunctionSet([SurrogateFactory.create(name='wasserstein_demographic_parity',
                                                               group_name=self.training_group_name,
                                                               unique_group_ids={
                                                                   self.training_group_name:list(range(2))
                                                                   },
                                                               reduction='mean',
                                                               weight=2
                                                               ),

                                      SurrogateFactory.create(name='performance',
                                                              surrogate_weight=1)
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
                    ConstrainedRequirement(name='performance_requirement',
                           metric = MetricsFactory.create_metric(
                                    metric_name='performance'),
                            weight=1,
                            operator='>=',
                            threshold=0.7,
                            performance_metric='f1'),

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
                                            )

    def run(self):
        self.experiment.setup()
        self.experiment.run()

    def tearDown(self) -> None:
        shutil.rmtree('checkpoints')