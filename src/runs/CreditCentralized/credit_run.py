from ..base_run import BaseRun
from architectures import ArchitectureFactory

class CentralizedCreditRun(BaseRun):
    
    def __init__(self,**kwargs):
        super(CentralizedCreditRun, self).__init__(**kwargs)
        self.input = 94
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
        self.dataset = 'credit_centralized'
        self.data_root  = 'data/Centralized_Credit'
        self.clean_data_path = 'data/Centralized_Credit/credit_clean.csv'
        self.sensitive_attributes = kwargs.get('sensitive_attributes',
                                               [
                                                ('Education',
                                                    {'EDUCATION':[1,2,3]}
                                                ),
                                                ('Gender',{'SEX':[1,2]}),
                                                ('AgeCat',{'AGE_CAT':['Young', 'Mid_Age', 'Senior', 'Elderly']}),
                                                ('Marriage',{'MARRIAGE':[1,2]}),
                                                ('GenderEducation',{
                                                    'EDUCATION':[1,2,3],
                                                    'SEX':[1,2]
                                                }),
                                                
                                                ('GenderAgeCat',{
                                                    'AGE_CAT':['Young', 'Mid_Age', 'Senior', 'Elderly'],
                                                    'SEX':[1,2]
                                                }),

                                                ('GenderMarriage',{
                                                    'MARRIAGE':[1,2],
                                                     'SEX':[1,2]
                                                }),



                                                ('EducationAgeCat',{
                                                    'AGE_CAT':['Young', 'Mid_Age', 'Senior', 'Elderly'],
                                                    'EDUCATION':[1,2,3],
                                                }),


                                                ('GenderEducationAgeCat',{
                                                    'AGE_CAT':['Young', 'Mid_Age', 'Senior', 'Elderly'],
                                                    'EDUCATION':[1,2,3],
                                                    'SEX':[1,2]
                                                }),

                                                ('MarriageEducationAgeCat',{
                                                    'AGE_CAT':['Young', 'Mid_Age', 'Senior', 'Elderly'],
                                                    'EDUCATION':[1,2,3],
                                                    'MARRIAGE':[1,2]
                                                }),
                                                
                                                ('GenderMarriageEducationAgeCat',{
                                                    'AGE_CAT':['Young', 'Mid_Age', 'Senior', 'Elderly'],
                                                    'EDUCATION':[1,2,3],
                                                    'MARRIAGE':[1,2],
                                                    'SEX':[1,2]
                                                }),
                                                ])
        
        

    def setUp(self):
        pass
    def run(self):
        pass
    def tearDown(self):
        pass