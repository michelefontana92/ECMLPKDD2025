from ..base_run import BaseRun
from architectures import ArchitectureFactory

class AdultRun(BaseRun):
    
    def __init__(self,**kwargs):
        super(AdultRun, self).__init__(**kwargs)
        self.model = ArchitectureFactory.create_architecture('mlp2hidden',model_params={'input': 103,
                                                'hidden1': 300,
                                                'hidden2': 100,
                                                'dropout': 0.2,
                                                'output': 2})
        self.dataset = 'adult'
        self.data_root  = 'data/Adult'
        self.sensitive_attributes = kwargs.get('sensitive_attributes',
                                               [
                                                ('Race',
                                                    {'race':['White','Black']}
                                                ),
                                                ('Gender',{'gender':['Male','Female']}),
                                                ('GenderRace',{
                                                    'race':['White','Black'],
                                                    'gender':['Male','Female']
                                                })
                                                ])
        

    def setUp(self):
        pass
    def run(self):
        pass
    def tearDown(self):
        pass