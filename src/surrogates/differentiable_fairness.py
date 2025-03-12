from .soft_confusion_matrix.fairness import *
from .soft_confusion_matrix.performance import *
from .surrogate_factory import register_surrogate


@register_surrogate('diff_demographic_parity')
class DifferentiableDemographicParitySurrogate:
    def __init__(self, **kwargs) -> None:
        self.surrogate_name = kwargs.get('surrogate_name', 'surrogate')
        self.weight = kwargs.get('weight', 1.0)
        self.group_name = kwargs.get('group_name')
        self.lower_bound = kwargs.get('lower_bound')
        self.use_max = kwargs.get('use_max', False)
        self.multiclass = kwargs.get('multiclass', False)
        self.target_groups = kwargs.get('target_groups')
        self.temperature = 1e-3
        self.alpha = 1.5
        self.device= 'cuda' if torch.cuda.is_available() else 'cpu'
        assert self.lower_bound is not None, 'lower_bound must be provided'
        assert self.target_groups is not None, 'target_groups must be provided'
        #print('Demographic Parity Surrogate on group:', self.group_name)
    
    def __call__(self, **kwargs):
        probabilities = kwargs.get('probabilities')
        assert probabilities is not None, 'probabilities must be provided'
        # Controllo NaN nei probabilities
        if torch.isnan(probabilities).any():
            print('Probabilities contengono NaN!')
            raise ValueError('Probabilities contiene NaN!')
        
        
        group_masks = kwargs.get('group_masks')
        assert group_masks is not None, 'group_masks must be provided'
        #print('Target groups: ',self.target_groups)
        dp = demographic_parity(
            probabilities,
            group_masks=group_masks[self.group_name],
            target_groups=self.target_groups.to(self.device),
            multiclass=self.multiclass
        )
        
        # Controlla NaN in dp
        if torch.isnan(dp).any():
            print('Demographic parity contiene NaN!')
            raise ValueError('Demographic parity contiene NaN!')
        
        
        if self.use_max:
            # Gestisci NaN in torch.max
            return torch.max(torch.zeros_like(dp), dp - self.lower_bound)
        #print(f'Demographic Parity on group {self.group_name} = {dp}')
        return dp - self.lower_bound


@register_surrogate('diff_equal_opportunity')
class DifferentiableEqualOpportunitySurrogate:
    def __init__(self,**kwargs) -> None:
        self.surrogate_name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.group_name = kwargs.get('group_name')
        self.lower_bound = kwargs.get('lower_bound',0.0)
        self.use_max = kwargs.get('use_max',False)
        self.alpha = 1.5
        self.target_groups = kwargs.get('target_groups')
        self.device= 'cuda' if torch.cuda.is_available() else 'cpu'
        assert self.target_groups is not None, 'target_groups must be provided'

    def __call__(self,**kwargs):
        positive_mask = kwargs.get('positive_mask')
        
        group_masks = kwargs.get('group_masks')

        probabilities = kwargs.get('probabilities')
        assert probabilities is not None, 'probabilities must be provided'
        # Controllo NaN nei probabilities
        if torch.isnan(probabilities).any():
            print('Probabilities contengono NaN!')
            raise ValueError('Probabilities contiene NaN!')
        
        
        assert group_masks is not None, 'group_masks must be provided'
        dp = equal_opportunity(probabilities,
                                group_masks= group_masks[self.group_name],
                                group_ids = self.target_groups.to(self.device),
                                positive_mask = positive_mask,
                                labels=kwargs.get('labels')
                                )
        
        # Controlla NaN in dp
        if torch.isnan(dp).any():
            print('Equal Opportunity contiene NaN!')
            raise ValueError('Equal Opportunity contiene NaN!')
        
        if self.use_max:
            return torch.max(torch.zeros_like(dp),dp - self.lower_bound)
        return dp - self.lower_bound
    

@register_surrogate('diff_predictive_equality')
class DifferentiablePredictiveEqualitySurrogate:
    def __init__(self,**kwargs) -> None:
        self.surrogate_name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.group_name = kwargs.get('group_name')
        self.lower_bound = kwargs.get('lower_bound',0.0)
        self.use_max = kwargs.get('use_max',False)
        self.alpha=1.5
        self.target_groups = kwargs.get('target_groups')
        self.device= 'cuda' if torch.cuda.is_available() else 'cpu'
        assert self.target_groups is not None, 'target_groups must be provided'
        
    def __call__(self,**kwargs):
        positive_mask = kwargs.get('positive_mask')
        group_masks = kwargs.get('group_masks')

        probabilities = kwargs.get('probabilities')
        assert probabilities is not None, 'probabilities must be provided'
        # Controllo NaN nei probabilities
        if torch.isnan(probabilities).any():
            print('Probabilities contengono NaN!')
            raise ValueError('Probabilities contiene NaN!')
        
       
        assert group_masks is not None, 'group_masks must be provided'
        dp = predictive_equality(
                                probabilities,
                                group_masks= group_masks[self.group_name],
                                group_ids = self.target_groups.to(self.device),
                                positive_mask = positive_mask,
                                labels=kwargs.get('labels')
                                )
        
        # Controlla NaN in dp
        if torch.isnan(dp).any():
            print('Predictive Equality contiene NaN!')
            raise ValueError('Predictive Equality contiene NaN!')
        if self.use_max:
            return torch.max(torch.zeros_like(dp),dp - self.lower_bound)
        return torch.max(dp - self.lower_bound,0)
    


@register_surrogate('diff_equalized_odds')
class DifferentiableEqualizedOddsSurrogate:
    def __init__(self,**kwargs) -> None:
        self.surrogate_name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.group_name = kwargs.get('group_name')
        self.lower_bound = kwargs.get('lower_bound',0.0)
        self.use_max = kwargs.get('use_max',False)
        self.alpha = 1.5
        self.target_groups = kwargs.get('target_groups')
        self.device= 'cuda' if torch.cuda.is_available() else 'cpu'
        assert self.target_groups is not None, 'target_groups must be provided'
        
    def __call__(self,**kwargs):
        positive_mask = kwargs.get('positive_mask')
        
        group_masks = kwargs.get('group_masks')

        probabilities = kwargs.get('probabilities')
        assert probabilities is not None, 'probabilities must be provided'
        # Controllo NaN nei probabilities
        if torch.isnan(probabilities).any():
            print('Probabilities contengono NaN!')
            raise ValueError('Probabilities contiene NaN!')
        
        assert group_masks is not None, 'group_masks must be provided'
        dp = equalized_odds(probabilities,
                                group_masks= group_masks[self.group_name],
                                group_ids = self.target_groups.to(self.device),
                                positive_mask = positive_mask,
                                labels=kwargs.get('labels')
                                )
        if torch.isnan(dp).any():
            print('Equalized Odds contiene NaN!')
            raise ValueError('Equalized Odds contiene NaN!')
        if self.use_max:
            return torch.max(torch.zeros_like(dp),dp - self.lower_bound)
        return dp - self.lower_bound