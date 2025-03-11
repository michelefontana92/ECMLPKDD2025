from ..base_run import BaseRun
from architectures import ArchitectureFactory

class CentralizedMEPRun(BaseRun):
    
    def __init__(self,**kwargs):
        super(CentralizedMEPRun, self).__init__(**kwargs)
        self.input = 132
        self.hidden1 = 300
        self.hidden2 = 100
        self.dropout = 0.2
        self.num_classes=2
        self.output = self.num_classes
        self.model = ArchitectureFactory.create_architecture('mlp2hidden',model_params={
                                                'input': self.input,
                                                'hidden1': self.hidden1,
                                                'hidden2': self.hidden2,
                                                'dropout': self.dropout,
                                                'output': self.output})
        self.dataset = 'mep'
        self.data_root  = 'data/Centralized_MEP'
        self.clean_data_path = 'data/Centralized_MEP/mep1_clean.csv'
        race_values = ['Hispanic','Black','White','Other']
        marry_values = ['Married','Never Married','Other']
        gender_values = ['Male','Female']
        region_values = ['Midwest', 'Northeast', 'South', 'West']
        race_var = 'RACE'
        gender_var = 'SEX'
        marry_var = 'MARRY'
        region_var = 'REGION'
        self.sensitive_attributes = kwargs.get('sensitive_attributes',
                                               [
                                                ('Race',
                                                    {race_var:race_values}
                                                ),
                                                ('Gender',
                                                 {gender_var:gender_values}
                                                 ),
                                                 ('Region',
                                                 {region_var:region_values}
                                                 ),
                                                 ('Marriage',
                                                  {marry_var:marry_values}),
                                                ('GenderRace',
                                                 {gender_var:gender_values,
                                                  race_var:race_values}),
                                                ('GenderMarriage',
                                                 {gender_var:gender_values,
                                                  marry_var:marry_values}),
                                                ('RaceMarriage',
                                                 {marry_var:marry_values,
                                                  race_var:race_values}),
                                                ('GenderRaceMarriage',
                                                 {gender_var:gender_values,
                                                  marry_var:marry_values,
                                                  race_var:race_values}),
                                                
                                                 ('GenderRaceMarriageRegion',
                                                 {gender_var:gender_values,
                                                  marry_var:marry_values,
                                                  race_var:race_values,
                                                  region_var:region_values}),

                                               
                                                ])
        
        

    def setUp(self):
        pass
    def run(self):
        pass
    def tearDown(self):
        pass