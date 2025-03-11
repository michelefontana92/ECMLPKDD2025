from experiments import GlofairCentralizedExperiment
import shutil
from .compas_run import CentralizedCompasRun
from ..run_factory import register_run
from requirements import RequirementSet, ConstrainedRequirement, UnconstrainedRequirement
from metrics import MetricsFactory
from surrogates import SurrogateFunctionSet, SurrogateFactory
from wrappers import PytorchSklearnWrapper  # import del wrapper definito sopra
from dataloaders import DataModule
from torch.optim import Adam
from functools import partial
from callbacks import EarlyStopping, ModelCheckpoint
from loggers import WandbLogger
from torch.nn import CrossEntropyLoss
import torch
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

class FairlearnDatasetAdapter:
    def __init__(self, pytorch_dataset):
        self.features = pytorch_dataset.x.cpu()
        self.labels = pytorch_dataset.y.cpu()
        self.sensitive_features = {
            name: group.cpu() for name, group in pytorch_dataset.groups_tensor.items()
        }

@register_run('compas_expgrad')
class CompasExpGradCentralized(CentralizedCompasRun):
    def __init__(self, **kwargs) -> None:
        super(CompasExpGradCentralized, self).__init__(**kwargs)
        self.dataset = 'compas'
        # estrai i parametri da kwargs
        self.metrics_list = kwargs.get('metrics_list')
        self.groups_list = kwargs.get('groups_list')
        self.threshold_list = kwargs.get('threshold_list')
        self.id = kwargs.get('id')
        self.num_clients = kwargs.get('num_clients', 10)
        self.lr = kwargs.get('lr', 1e-4)
        self.num_lagrangian_epochs = kwargs.get('num_lagrangian_epochs', 1)
        self.fine_tune_epochs = kwargs.get('fine_tune_epochs', 0)
        self.batch_size = kwargs.get('batch_size', 128)
        self.project_name = kwargs.get('project_name')
        self.start_index = kwargs.get('start_index', 0)
        self.training_group_name = kwargs.get('group_name')
        self.use_hale = kwargs.get('use_hale', False)
        self.metric = kwargs.get('metric_name')
        self.onlyperf = kwargs.get('onlyperf', False)
        self.threshold = kwargs.get('threshold')
        self.checkpoint_dir = kwargs.get('checkpoint_dir', f'checkpoints/{self.project_name}')
        self.checkpoint_name = kwargs.get('checkpoint_name', 'model.h5')
        self.verbose = kwargs.get('verbose', False)
        self.monitor = kwargs.get('monitor', 'val_requirements')
        self.mode = kwargs.get('mode', 'min')
        self.log_model = kwargs.get('log_model', False)
        self.num_subproblems = kwargs.get('num_subproblems')
        self.num_global_iterations = kwargs.get('num_global_iterations')
        self.num_epochs = kwargs.get('num_local_iterations', 1)  # se vuoi 1 epoca di default
        self.performance_constraint = kwargs.get('performance_constraint')
        self.delta = kwargs.get('delta', 0.2)
        self.max_constraints_in_subproblem = kwargs.get('max_constraints_in_subproblem')
        self.global_patience = kwargs.get('global_patience', 10)

        # Altri setup di metriche, requirements, ecc. (come nel tuo codice)
        self.metrics = [MetricsFactory().create_metric('performance')]
        self.approximations = [
            SurrogateFactory().create(name='performance',
                                      surrogate_name='approx_performance',
                                      surrogate_weight=1)
        ]
        self.requirements = [
            UnconstrainedRequirement(
                name='unconstrained_performance_requirement',
                metric=MetricsFactory.create_metric(metric_name='performance'),
                weight=1,
                mode='max',
                bound=1.0,
                performance_metric='f1'
            )
        ]
        # Se hai metriche fairness aggiuntive, le inserisci come nel tuo codice
        # ...

        self.surrogate_function_set = SurrogateFunctionSet(self.approximations)
        self.requirement_set = RequirementSet(self.requirements)

        # Costruisci un modello PyTorch (self.model), definito altrove
        self.model = self.model.to('cpu')  # ad es. MLP, etc.

        # Costruisci l'optimizer
        self.optimizer_fn = partial(Adam, lr=self.lr)
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

        # Esempio di callback
        self.callbacks = [
            EarlyStopping(patience=self.global_patience, monitor=self.monitor, mode=self.mode),
            ModelCheckpoint(save_dir=self.checkpoint_dir, save_name=self.checkpoint_name,
                            monitor=self.monitor, mode=self.mode)
        ]

        for metric, group, threshold in zip(self.metrics_list, self.groups_list, self.threshold_list):
           
            self.metric = metric
            self.training_group_name = group
            self.num_groups = self.compute_group_cardinality(self.training_group_name)
            self.group_ids = {self.training_group_name: list(range(self.num_groups))}
            
            # Aggiunta della metrica
            current_metric = MetricsFactory().create_metric(metric, group_ids=self.group_ids, group_name=self.training_group_name, 
                                                            )

            
            self.metrics.append(current_metric)
          

    def setUp(self):
        # Creazione del DataModule
        path = f'node_{self.start_index}/{self.dataset}' if self.start_index >= 0 else f'{self.dataset}_clean'
        self.data_module = DataModule(dataset=self.dataset,
                                      root=self.data_root,
                                      train_set=f'{path}_train.csv',
                                      val_set=f'{path}_val.csv',
                                      test_set=f'{path}_val.csv',
                                      batch_size=self.batch_size,
                                      num_workers=0,
                                      sensitive_attributes=self.sensitive_attributes,
                                      clean_data_path=self.clean_data_path)

        # Logger
        self.logger = WandbLogger(project=self.project_name,
                                  config={},
                                  id=self.id,
                                  checkpoint_dir=self.checkpoint_dir,
                                  checkpoint_path=self.checkpoint_name,
                                  data_module=self.data_module if self.log_model else None)

        # Ora costruiamo IL WRAPPER scikit-learn
        self.wrapper = PytorchSklearnWrapper(
            model=self.model,
            optimizer_fn=self.optimizer_fn,
            optimizer=self.optimizer,
            loss=partial(CrossEntropyLoss, reduction='none'),  # important for sample_weight
            metrics=self.metrics,  # se vuoi
            num_epochs=self.num_epochs,   # es. 1 epoca
            device='cpu',
            disable_log=False
        )

        # Se vuoi usare data_module dentro il wrapper (ad es. per validazione),
        # assegni a un attributo privato. NON esporre in get_params()!
        self.wrapper._data_module = self.data_module
        # Se hai callback:
        self.wrapper._checkpoints = self.callbacks
        self.wrapper._logger = self.logger
    
    
    def _update_metrics_final(self, expgrad, dataset):
        """
        Calcola le metriche (self.metrics) sul dataset 'dataset'
        usando il predittore finale di ExponentiatedGradient 'expgrad'.
        Ritorna un dizionario {nome_met: valore}.
        """
        # 1) Estraggo X, y in numpy
        X_np = dataset.x.cpu().numpy()
        y_np = dataset.y.cpu().numpy()
        
        # Se hai bisogno dei gruppi (per metriche di fairness), estrai groups_dict
        groups_dict = dataset.groups_tensor  # dict con {group_name: tensor([...])}
        
        # 2) Ottengo le predizioni finali dal mix
        y_pred_np = expgrad.predict(X_np)
        
        # 3) Converto in tensori PyTorch, se la tua _compute_metrics(...) lo richiede
   
        y_pred_tensor = torch.tensor(y_pred_np)
        y_true_tensor = torch.tensor(y_np)
        
        # 4) Calcolo le metriche con la stessa funzione che usi nel training
        #    (assumendo che la firma sia: 
        #       _compute_metrics(self, metric_list, preds, targets, groups, prefix='', logits=None)
        #    e ritorni un dict con i valori)
        
        metrics_dict = self._compute_metrics(
            self.metrics,
            y_pred_tensor,
            y_true_tensor,
            groups_dict,
            prefix='final'
        )
        
        return metrics_dict
    def _compute_metrics(self, metrics_list, y_pred, y_true, groups_dict, prefix='', logits=None):
        """
        Calcola i valori per un elenco di metriche 'metrics_list', 
        e ritorna un dizionario {f'{prefix}_metric_name': value}.
        """
        results = {}
        for metric_obj in metrics_list:
            # Esempio: metric_obj(y_true, y_pred, groups=groups_dict, logits=logits)
            # Dipende dalla implementazione del tuo MetricsFactory
            val = metric_obj(
                y_true=y_true,
                y_pred=y_pred,
                groups=groups_dict,
                logits=logits
            )
            # Nome della metrica (se metric_obj ha un attributo .name)
            name = getattr(metric_obj, 'name', 'unnamed_metric')
            full_name = f"{prefix}_{name}" if prefix else name
            results[full_name] = val
        return results

    def run(self):
        # Prepara il train dataset
        pytorch_dataset = self.data_module.datasets['train']
        fairlearn_dataset = FairlearnDatasetAdapter(pytorch_dataset)

        X = fairlearn_dataset.features.cpu().numpy()
        y = fairlearn_dataset.labels.cpu().numpy()
        sensitive = fairlearn_dataset.sensitive_features[self.training_group_name]

        # Constraint di fairness
        constraint = DemographicParity(difference_bound=0.10)

        expgrad = ExponentiatedGradient(
            estimator=self.wrapper,
            constraints=constraint,
            eps=0.01,
            max_iter=100  # puoi regolare
        )

        print("Inizio fitting con ExponentiatedGradient...")
        expgrad.fit(X, y, sensitive_features=sensitive)

        # Ora expgrad è il predittore misto. 
        # Se vuoi predire su un set di test:
        test_dataset = self.data_module.datasets['val']
       
        X_test = test_dataset.x.cpu().numpy()
        y_test = test_dataset.y.cpu().numpy()

        # Se ti servono i gruppi del test set
        groups_test = test_dataset.groups_tensor  # dict { 'groupname': tensor([...]) }

        # Ottieni la classe predetta dal mix
        y_pred_test = expgrad.predict(X_test)

        # 4) Calcola le metriche usando la nuova funzione
        final_metrics = self.wrapper.compute_metrics_from_external_predictions(
            y_pred=y_pred_test,
            y_true=y_test,
            groups_dict=groups_test,
            prefix='final'
        )

        print("=== Metriche finali su test set (ExpGrad) ===")
        for met_name, met_val in final_metrics.items():
            print(f"{met_name}: {met_val:.4f}")


    def tearDown(self) -> None:
        shutil.rmtree(f'checkpoints/{self.project_name}')
