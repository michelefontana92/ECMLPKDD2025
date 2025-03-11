from hard_label_estimator.metrics.fairness import *
from hard_label_estimator.estimator import HardLabelsEstimator
from torch.nn import functional as F
from .surrogate_factory import register_surrogate


@register_surrogate('diff_demographic_parity')
class DifferentiableDemographicParitySurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.distributed_env = kwargs.get('distributed_env',False)
        self.estimator = HardLabelsEstimator(distributed_env=self.distributed_env)
        self.group_name = kwargs.get('group_name')
        self.lower_bound = kwargs.get('lower_bound',0.0)
        self.use_max = kwargs.get('use_max',False)
        self.multiclass = kwargs.get('multiclass',False)
        self.temperature = 1e-3

    def __call__(self,**kwargs):
        group_ids = kwargs.get('group_ids_list')
        logits = kwargs.get('logits')
        probabilities = F.softmax(logits/self.temperature,dim=1)
        group_masks = kwargs.get('group_masks')
        assert group_ids is not None, 'group_ids must be provided'
        assert group_masks is not None, 'group_masks must be provided'
        dp = demographic_parity(probabilities,
                                group_masks= group_masks[self.group_name],
                                group_ids = group_ids[self.group_name][0][0],
                                multiclass = self.multiclass
                                )
        if self.use_max:
            return torch.max(torch.zeros_like(dp),dp - self.lower_bound)
        return dp - self.lower_bound


@register_surrogate('diff_equal_opportunity')
class DifferentiableEqualOpportunitySurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.distributed_env = kwargs.get('distributed_env',False)
        self.estimator = HardLabelsEstimator(distributed_env=self.distributed_env)
        self.group_name = kwargs.get('group_name')
        self.lower_bound = kwargs.get('lower_bound',0.0)
        self.use_max = kwargs.get('use_max',False)
    def __call__(self,**kwargs):
        positive_mask = kwargs.get('positive_mask')
        group_ids = kwargs.get('group_ids_list')
        logits = kwargs.get('logits')
        probabilities = F.softmax(logits,dim=1)
        group_masks = kwargs.get('group_masks')
        assert group_ids is not None, 'group_ids must be provided'
        assert group_masks is not None, 'group_masks must be provided'
        dp = equal_opportunity(self.estimator,
                                probabilities[:,1],
                                group_masks= group_masks[self.group_name],
                                group_ids = group_ids[self.group_name][0][0],
                                positive_mask = positive_mask
                                )
        if self.use_max:
            return torch.max(torch.zeros_like(dp),dp - self.lower_bound)
        return dp - self.lower_bound
    

@register_surrogate('diff_predictive_equality')
class DifferentiablePredictiveEqualitySurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.distributed_env = kwargs.get('distributed_env',False)
        self.estimator = HardLabelsEstimator(distributed_env=self.distributed_env)
        self.group_name = kwargs.get('group_name')
        self.lower_bound = kwargs.get('lower_bound',0.0)
        self.use_max = kwargs.get('use_max',False)
    def __call__(self,**kwargs):
        positive_mask = kwargs.get('positive_mask')
        group_ids = kwargs.get('group_ids_list')
        logits = kwargs.get('logits')
        probabilities = F.softmax(logits,dim=1)
        group_masks = kwargs.get('group_masks')
        assert group_ids is not None, 'group_ids must be provided'
        assert group_masks is not None, 'group_masks must be provided'
        dp = predictive_equality(self.estimator,
                                probabilities[:,1],
                                group_masks= group_masks[self.group_name],
                                group_ids = group_ids[self.group_name][0][0],
                                positive_mask = positive_mask
                                )
        if self.use_max:
            return torch.max(torch.zeros_like(dp),dp - self.lower_bound)
        return torch.max(dp - self.lower_bound,0)
    


@register_surrogate('diff_equalized_odds')
class DifferentiableEqualizedOddsSurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.distributed_env = kwargs.get('distributed_env',False)
        self.estimator = HardLabelsEstimator(distributed_env=self.distributed_env)
        self.group_name = kwargs.get('group_name')
        self.lower_bound = kwargs.get('lower_bound',0.0)
        self.use_max = kwargs.get('use_max',False)
    
    def __call__(self,**kwargs):
        positive_mask = kwargs.get('positive_mask')
        group_ids = kwargs.get('group_ids_list')
        logits = kwargs.get('logits')
        probabilities = F.softmax(logits,dim=1)
        group_masks = kwargs.get('group_masks')
        assert group_ids is not None, 'group_ids must be provided'
        assert group_masks is not None, 'group_masks must be provided'
        dp = equalized_odds(self.estimator,
                                probabilities[:,1],
                                group_masks= group_masks[self.group_name],
                                group_ids = group_ids[self.group_name][0][0],
                                positive_mask = positive_mask
                                )
        if self.use_max:
            return torch.max(torch.zeros_like(dp),dp - self.lower_bound)
        return dp - self.lower_bound