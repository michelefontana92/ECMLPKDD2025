from experiments import FedAvgExperiment
import shutil
from .credit_run import CreditRun
from ..run_factory import register_run

@register_run('credit_fedavg')
class CreditFedAvgRun(CreditRun):
    def __init__(self,**kwargs) -> None:
        super(CreditFedAvgRun, self).__init__(**kwargs)
        self.project_name = 'CreditFedAvgAlpha2'
        self.num_clients = 10
        self.lr=1e-4
        self.num_federated_rounds = 100
        self.start_index = 61
        self.project_name =  kwargs.get('project_name')
        self.start_index = kwargs.get('start_index')
    def setUp(self):
       
        self.experiment = FedAvgExperiment( sensitive_attributes=self.sensitive_attributes,
                                            dataset=self.dataset,
                                            data_root=self.data_root,
                                            model=self.model,
                                            num_clients=self.num_clients,
                                            num_federated_rounds=self.num_federated_rounds,
                                            lr=self.lr,
                                            start_index=self.start_index,
                                            project=self.project_name
                                            )

        