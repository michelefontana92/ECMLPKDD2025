from ..base_run import BaseRun
from architectures import ArchitectureFactory

class FolkTablesMultiRun(BaseRun):
    
    def __init__(self,**kwargs):
        super(FolkTablesMultiRun, self).__init__(**kwargs)
        self.num_classes=3
        self.input = 60
        self.hidden1 = 400
        self.hidden2 = 200
        self.hidden3 = 100
        self.hidden4 = 100
        self.dropout = 0.2
        self.output = self.num_classes
        
        self.model = ArchitectureFactory.create_architecture('mlp3hidden',model_params={
                                                'input': self.input,
                                                'hidden1': self.hidden1,
                                                'hidden2': self.hidden2,
                                                'hidden3': self.hidden3,
                                                'hidden4': self.hidden4,
                                                'dropout': self.dropout,
                                                'output': self.output})
        self.dataset = 'folktables_multi'
        self.data_root  = 'data/FolkTablesMulti'
        self.sensitive_attributes_2 = kwargs.get('sensitive_attributes',
                                               [
                                                ('Race',
                                                    {'Race':['White','Black','Asian','Other','Indigenous']}
                                                ),
                                                ('Gender',{'Gender':['Male','Female']}),
                                                ('Job',{
                                                    'Job':['Public Employee','Self Employed','Private Employee']
                                                }),
                                                ('Marital',{
                                                    'Marital':['Married','Never Married','Divorced','Other']
                                                }),
                                                ('GenderRace',{
                                                    'Race':['White','Black','Asian','Other','Indigenous'],
                                                    'Gender':['Male','Female']
                                                }),
                                                ('GenderJob',{
                                                    'Job':['Public Employee','Self Employed','Private Employee'],
                                                    'Gender':['Male','Female']
                                                }
                                                ),
                                                ('GenderMarital',{
                                                    'Marital':['Married','Never Married','Divorced','Other'],
                                                    'Gender':['Male','Female']
                                                }),
                                                ('RaceJob',{
                                                    'Job':['Public Employee','Self Employed','Private Employee'],
                                                  'Race':['White','Black','Asian','Other','Indigenous']
                                                    }),
                                                ('RaceMarital',{
                                                    'Race':['White','Black','Asian','Other','Indigenous'],
                                                    'Marital':['Married','Never Married','Divorced','Other'],
                                                    }),
                                                ('JobMarital',{
                                                    'Job':['Public Employee','Self Employed','Private Employee'],
                                                    'Marital':['Married','Never Married','Divorced','Other'],
                                                    }),
                                                ('MaritalRaceJob',{
                                                    'Job':['Public Employee','Self Employed','Private Employee'],
                                                    'Race':['White','Black','Asian','Other','Indigenous'],
                                                    'Marital':['Married','Never Married','Divorced','Other'],
                                                    })

                                                ])
        
        self.sensitive_attributes = kwargs.get('sensitive_attributes',[('Marital',{
                                                    'Marital':['Married','Never Married','Divorced','Other']}),
                                                    ('Race',
                                                    {'Race':['White','Black','Asian','Other','Indigenous']}),
                                                    ('Job',{
                                                    'Job':['Public Employee','Self Employed','Private Employee']
                                                }),
                                                ('RaceJob',{
                                                    'Job':['Public Employee','Self Employed','Private Employee'],
                                                  'Race':['White','Black','Asian','Other','Indigenous']
                                                    }),
                                                ('RaceMarital',{
                                                    'Race':['White','Black','Asian','Other','Indigenous'],
                                                    'Marital':['Married','Never Married','Divorced','Other'],
                                                    }),
                                                ])
    def setUp(self):
        pass
    def run(self):
        pass
    def tearDown(self):
        pass