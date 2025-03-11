from experiments import FedFBExperiment
import shutil
from .credit_run import CreditRun
from ..run_factory import register_run

@register_run('credit_fedfb_gender')
class CreditFedFBGenderRun(CreditRun):
    def __init__(self,**kwargs) -> None:
        super(CreditFedFBGenderRun, self).__init__(**kwargs)
        self.project_name = 'CreditFedFBGender'
        self.num_clients = 10
        self.lr=1e-4
        self.num_federated_rounds = 100
        self.alpha = 0.1
        self.training_group_name = 'Gender'
        self.project_name =  kwargs.get('project_name')
        self.start_index = kwargs.get('start_index')
        self.metric_name = kwargs.get('metric_name','demographic_parity')
    def setUp(self):
       
        self.experiment = FedFBExperiment(  sensitive_attributes=self.sensitive_attributes,
                                            dataset=self.dataset,
                                            data_root=self.data_root,
                                            model=self.model,
                                            num_clients=self.num_clients,
                                            num_federated_rounds=self.num_federated_rounds,
                                            lr=self.lr,
                                            alpha = self.alpha,
                                            project=self.project_name,
                                            training_group_name=self.training_group_name,
                                            start_index=self.start_index,
                                            metric_name=self.metric_name
                                            )

    def run(self):
        self.experiment.setup()
        self.experiment.run()

    def tearDown(self) -> None:
        shutil.rmtree('checkpoints')