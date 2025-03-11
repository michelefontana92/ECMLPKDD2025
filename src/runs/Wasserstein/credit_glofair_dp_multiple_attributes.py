from experiments import GlofairExperiment
import shutil
from .credit_run import CreditRun
from ..run_factory import register_run
from requirements import RequirementSet,ConstrainedRequirement,UnconstrainedRequirement
from surrogates import SurrogateFunctionSet,SurrogateFactory
from metrics import MetricsFactory

@register_run('wasserstein_credit_glofair_dp_multiple_attributes')
class WassersteinCreditGlofairDPMultipleAttributesRun(CreditRun):
    def __init__(self,**kwargs) -> None:
        super(WassersteinCreditGlofairDPMultipleAttributesRun, self).__init__(**kwargs)
        self.project_name = 'ConstrainedCentralizedSurrogateCreditGlofairDPMultipleAttributesWeight1'
        self.num_clients = 10
        self.lr=1e-4
        self.num_federated_rounds = 1
        self.training_group_name = 'GenderAge'
        self.num_epochs = 30
        self.fine_tune_epochs = 0
        self.surrogate_set = SurrogateFunctionSet([SurrogateFactory.create(name='demographic_parity',
                                                               group_name='Gender',
                                                               unique_group_ids={
                                                                   'Gender':list(range(2))
                                                                   },
                                                               reduction='mean',
                                                               weight=1
                                                               ),
                                                SurrogateFactory.create(name='demographic_parity',
                                                               group_name='Age',
                                                               unique_group_ids={
                                                                   'Age':list(range(2))
                                                                   },
                                                               reduction='mean',
                                                               weight=1
                                                               ),
                                                SurrogateFactory.create(name='demographic_parity',
                                                               group_name='GenderAge',
                                                               unique_group_ids={
                                                                   'GenderAge':list(range(4))
                                                                   },
                                                               reduction='mean',
                                                               weight=1
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
                   

                    ConstrainedRequirement(name='dp_requirement',
                                           metric = MetricsFactory.create_metric(
                                                    metric_name='demographic_parity',
                                                    group_name='Gender',
                                                    group_ids={'Gender':list(range(2))}),
                                            weight=1,
                                            operator='<=',
                                            threshold=0.2),
                    ConstrainedRequirement(name='dp_requirement',
                                           metric = MetricsFactory.create_metric(
                                                    metric_name='demographic_parity',
                                                    group_name='Age',
                                                    group_ids={'Age':list(range(2))}),
                                            weight=1,
                                            operator='<=',
                                            threshold=0.2),
                    ConstrainedRequirement(name='dp_requirement',
                                           metric = MetricsFactory.create_metric(
                                                    metric_name='demographic_parity',
                                                    group_name='GenderAge',
                                                    group_ids={'GenderAge':list(range(4))}),
                                            weight=1,
                                            operator='<=',
                                            threshold=0.2),
                        ])
    
    def setUp(self):
       
        self.experiment = GlofairExperiment( sensitive_attributes=self.sensitive_attributes,
                                            dataset=self.dataset,
                                            data_root=self.data_root,
                                            model=self.model,
                                            num_clients=self.num_clients,
                                            num_federated_rounds=self.num_federated_rounds,
                                            lr=self.lr,
                                            num_local_epochs=self.num_epochs,
                                            fine_tune_epochs=self.fine_tune_epochs,
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