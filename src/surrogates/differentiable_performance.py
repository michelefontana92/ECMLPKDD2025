from .surrogate_factory import register_surrogate
from hard_label_estimator import HardLabelsEstimator
from hard_label_estimator.metrics import binary_accuracy,binary_f1_score,binary_precision,binary_recall,true_negative,true_positive,false_negative,false_positive
from torch.nn import functional as F
from hard_label_estimator.metrics import multiclass_accuracy,multiclass_f1_score,multiclass_precision,multiclass_recall
import torch
from entmax import entmax_bisect
from torch.nn import CrossEntropyLoss

def straight_through_softmax(logits, dim=1):
    # Calcola la softmax per il backward pass (differenziabile)
    soft_output = F.softmax(logits, dim=dim)
    # Calcola l'argmax per il forward pass (non differenziabile)
    hard_output = torch.zeros_like(logits)
    hard_output.scatter_(dim, logits.argmax(dim=dim, keepdim=True), 1.0)
    # Combina la versione hard per il forward con il soft per il backward
    return (hard_output - soft_output).detach() + soft_output

def gumbel_softmax(logits, tau=1.0, hard=False):
    # Genera il rumore Gumbel
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    # Aggiunge il rumore Gumbel ai logits
    y = logits + gumbel_noise
    # Applica la softmax con temperatura tau
    y_soft = F.softmax(y / tau, dim=-1)
    
    if hard:
        # Se richiedi una hard label, fai il campionamento argmax
        y_hard = torch.zeros_like(logits)
        y_hard.scatter_(1, y_soft.argmax(dim=-1, keepdim=True), 1.0)
        # Passa il gradiente attraverso la softmax ma usa il risultato hard
        return (y_hard - y_soft).detach() + y_soft
    else:
        return y_soft
@register_surrogate('binary_accuracy')
class BinaryAccuracySurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.multiclass = kwargs.get('multiclass',False)
        self.tau = 0.2
        self.decay= 0.99
        self.estimator = HardLabelsEstimator()
        self.lambda_entropy = 0.01
    def __call__(self,**kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        #print(kwargs.get('positive_mask'))
        #probabilities = gumbel_softmax(logits,tau=0.1,hard=True)
        #print(probabilities[:10,:])
        #probabilities = self.estimator.get_prediction(F.softmax(logits,dim=1),
         #                                             multiclass=True)
        #print('Probabilities:',probabilities[:10,:])
        probabilities = entmax_bisect(logits, alpha=1.5, dim=-1)
        #probabilities = F.softmax(F.softmax(logits,dim=1)/1e-2,dim=1)
        probabilities_softmax = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probabilities_softmax * torch.log(probabilities_softmax + 1e-9), dim=-1).mean()

        if not self.multiclass:
            y_hat = probabilities[:,1]
            positive_mask = labels==1
            positive_mask = positive_mask.view(y_hat.shape)
            accuracy = binary_accuracy(y_hat,
                                       positive_mask=positive_mask)#kwargs.get('positive_mask'))    
        else:
           accuracy = multiclass_accuracy(probabilities,
                                          labels=labels)
        #self.tau= max(0.1,self.tau*self.decay)
        return (1- accuracy) #+ (self.lambda_entropy*entropy)

@register_surrogate('binary_precision')
class BinaryPrecisionSurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.average = kwargs.get('average',None)
        self.multiclass = kwargs.get('multiclass',False)
    def __call__(self,**kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        #probabilities = F.softmax(logits/0.2,dim=1)
        probabilities = entmax_bisect(logits, alpha=1.5, dim=-1)
        if not self.multiclass:
            positive_mask = labels==1
            precision = binary_precision(probabilities[:,1],
                                       positive_mask=positive_mask,
                                       average=self.average)
        else:
            precision = multiclass_precision(probabilities,
                                             labels=labels,
                                             average=self.average)
        return 1 - precision



@register_surrogate('binary_recall')
class BinaryRecallSurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.multiclass = kwargs.get('multiclass',False)    
        self.average = kwargs.get('average',None)
        
    def __call__(self,**kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        #probabilities = F.softmax(logits/0.2,dim=1)
        probabilities = entmax_bisect(logits, alpha=1.5, dim=-1)
        if not self.multiclass:
            positive_mask = labels==1
            recall = binary_recall(probabilities[:,1],
                                       positive_mask=positive_mask,
                                       average=self.average)
        else: 
            recall = multiclass_recall(probabilities,
                                      labels=labels,
                                      average=self.average)
        return 1 - recall



@register_surrogate('binary_f1')
class BinaryF1Surrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.average = kwargs.get('average',None)
        self.multiclass = kwargs.get('multiclass',False)
        self.upper_bound = kwargs.get('upper_bound',1.0)
        self.use_max = kwargs.get('use_max',False)
        self.target_groups = None
        self.group_name = None
        
    def __call__(self,**kwargs):
        labels = kwargs.get('labels')
        probabilities = kwargs.get('probabilities')
        assert probabilities is not None, 'probabilities must be provided'
        # Controllo NaN nei probabilities
        if torch.isnan(probabilities).any():
            print('Probabilities contengono NaN!')
            probabilities = torch.nan_to_num(probabilities, nan=0.0)  # Sostituisci NaN nei logits
        
        
        #print('Predicted labels:',probabilities[:10,:])
        #print('True labels:',labels[:10])
        #p = probabilities[:,1]
        #print(p[labels==1])
        if not self.multiclass:
            positive_mask = labels==1
            f1 = binary_f1_score(probabilities[:,1],
                                       positive_mask=positive_mask,
                                       average=self.average)
        else:
            f1 = multiclass_f1_score(probabilities,
                                    labels=labels,
                                    average=self.average)
        
        if self.use_max:
            return torch.max(torch.zeros_like(f1),self.upper_bound-f1)
        return self.upper_bound-f1



@register_surrogate('binary_true_positive')
class BinaryTruePositiveSurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.average = kwargs.get('average',None)
        self.multiclass = kwargs.get('multiclass',False)
    
    def __call__(self,**kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        #probabilities = F.softmax(logits/0.2,dim=1)
        probabilities = entmax_bisect(logits, alpha=1.5, dim=-1)
        if not self.multiclass:
            positive_mask = labels==1
            tp = true_positive(probabilities[:,1],
                                       positive_mask=positive_mask,
                                       average=self.average,
                                       get_probability=True)
        else: 
            tp = multiclass_precision(probabilities,
                                    labels=labels,
                                    average=self.average,
                                    get_probability=True)
       
        return 1 - tp


@register_surrogate('binary_true_negative')
class BinaryTrueNegativeSurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.average = kwargs.get('average',None)
        self.multiclass = kwargs.get('multiclass',False)
    
    def __call__(self,**kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        #probabilities = F.softmax(logits/0.2,dim=1)
        probabilities = entmax_bisect(logits, alpha=1.5, dim=-1)
        if not self.multiclass:
            positive_mask = labels==1
            tn = true_negative(probabilities[:,1],
                                       positive_mask=positive_mask,
                                       average=self.average,
                                       get_probability=True)
        else: 
            tn = multiclass_precision(probabilities,
                                    labels=labels,
                                    average=self.average,
                                    get_probability=True)
  
        return 1-tn


@register_surrogate('binary_false_positive')
class BinaryFalsePositiveSurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.average = kwargs.get('average',None)
        self.multiclass = kwargs.get('multiclass',False)

    def __call__(self,**kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        #probabilities = F.softmax(logits/0.2,dim=1)
        probabilities = entmax_bisect(logits, alpha=1.5, dim=-1)
        if not self.multiclass:
            positive_mask = labels==1
            fp = false_positive(probabilities[:,1],
                                       positive_mask=positive_mask,
                                       average=self.average,
                                       get_probability=True)
        else: 
            fp = multiclass_precision(probabilities,
                                    labels=labels,
                                    average=self.average,
                                    get_probability=True)
       
        return fp


@register_surrogate('binary_false_negative')
class BinaryFalseNegativeSurrogate:
    def __init__(self,**kwargs) -> None:
        self.name = kwargs.get('surrogate_name','surrogate')
        self.weight = kwargs.get('weight',1.0)
        self.average = kwargs.get('average',None)
        self.multiclass = kwargs.get('multiclass',False)

    def __call__(self,**kwargs):
        logits = kwargs.get('logits')
        labels = kwargs.get('labels')
        #probabilities = F.softmax(logits/0.2,dim=1)
        probabilities = entmax_bisect(logits, alpha=1.5, dim=-1)
        if not self.multiclass:
            positive_mask = labels==1
            fn = false_negative(probabilities[:,1],
                                       positive_mask=positive_mask,
                                       average=self.average,
                                       get_probability=True)
        else: 
            fn = multiclass_precision(probabilities,
                                    labels=labels,
                                    average=self.average,
                                    get_probability=True)
       
        return fn