# src/MMDLearning/training/trainer.py
from dataclasses import dataclass
from pathlib import Path
import torch, json, time
from torch import distributed as dist
from torch import nn, optim
import os

from .policies import StandardPolicy

from ml_tools.training.metrics import get_test_metrics, RunningStats
from ml_tools.training.reporting import display_epoch_summary, make_logits_plt, make_train_plt, display_status, finish_roc_plot
from ml_tools.training.schedulers import SchedConfig, make_scheduler
from ml_tools.training.training_utils import Initialization, TrainEpochStart, ValidEpochStart, TestEpochStart, MetricHistory
from ml_tools.utils.distributed import globalize_epoch_totals, epoch_metrics_from_globals
from ml_tools.utils.buffers import EpochLogitBuffer  
from ml_tools.utils.distributed import DistInfo


@dataclass
class Trainer:
    # core state you currently use as globals
    cfg: any
    model: nn.Module
    optimizer: optim.Optimizer
    dist_info: DistInfo
    device: torch.device
    sched_config: SchedConfig
    loss_fn: nn.Module 
    dataloaders: dict   # {'train': ..., 'valid': ..., 'test': ...}
    metrics: MetricHistory
    start_epoch: int = 0
    mode: str = ''  
    dtype: torch.dtype = torch.float32
    def __post_init__(self):
        self.final_epoch = self.cfg.epochs + self.start_epoch
        self.state = {}
        policy_kwargs = {}
        #TODO: come back to these once my policy is created
        self._handlers = {
            Initialization: self._initialize,
            TrainEpochStart: self._start_train_epoch,
            ValidEpochStart: self._start_valid_epoch,
            TestEpochStart: self._start_test_epoch,
        }
        self.update_state(Initialization()) #guess values
        self.scheduler = make_scheduler(self.optimizer, self.sched_config)
        self.buffer = EpochLogitBuffer(keep_indices=False, 
                                            keep_domains=False,
                                            assume_equal_lengths=True)
        self.policy = StandardPolicy(loss_fn=self.loss_fn, 
                                  bufs=self.buffer, 
                                  device=self.device, 
                                  dtype=self.dtype,
                                  **policy_kwargs)

    def _save_ckp(self, state, is_best, epoch, save_all=False):
        p = Path(self.cfg.logdir) / self.cfg.exp_name
        p.mkdir(parents=True, exist_ok=True)
        if save_all and self.dist_info.is_primary:
            torch.save(state, p / f"checkpoint-epoch-{epoch}.pt")
        if is_best and self.dist_info.is_primary:
            for _ in range(3):
                try:
                    torch.save(state, p / "best-val-model.pt"); break
                except OSError:
                    time.sleep(5)
        
    def _initialize(self, event):
        self.state['get_buffers'] = False
        self.state['epochs_completed'] = -1

    def _start_train_epoch(self, event):
        self.state['phase'] = 'train'
        self.state['get_buffers'] = False
        self.state['epochs_completed'] += 1
        
    def _start_valid_epoch(self, event):
        self.state['phase'] = 'valid'
        if self.state['epochs_completed'] == -1:
            self.state['get_buffers'] = True
            
    def _start_test_epoch(self, event):
        self.state['phase'] = 'test'
        self.state['get_buffers'] = True
            
    def update_state(self, event):
        handler = self._handlers.get(type(event), None)
        if handler is not None:
            handler(event)

    def _run_epoch(self, epoch: int, loader=None):
        if self.state['phase'] == 'train':
            self.model.train()
            loader.sampler.set_epoch(epoch)
        else:
            self.model.eval()

        tracker = RunningStats(window=self.cfg.log_interval) #checked

        loader_length = len(loader) if loader is not None else 0
        #need to make prediction for source and target data
        for batch_idx, data in enumerate(loader):
            if self.state['phase'] == 'train':
                self.optimizer.zero_grad()
            batch_metrics, tot_loss = self.policy.compute_batch_metrics(data=data, 
                                                                  model=self.model, 
                                                                  state=self.state)
            if self.state['phase'] == 'train':
                tot_loss.backward()
                self.optimizer.step()

            
            tracker.update(**batch_metrics)

            if (batch_idx+1) % self.cfg.log_interval == 0:
                display_status(phase=self.state['phase'], epoch=epoch, 
                                tot_epochs=0 if self.state['phase']=='test' else self.final_epoch-1, #TODO: check if this gives correct tot_epochs
                                batch_idx=batch_idx+1, num_batches=loader_length,
                                running_acc=tracker.running_acc, avg_batch_time=tracker.avg_batch_time(),
                                running_loss=tracker.running_loss, logger=None)
    
        torch.cuda.empty_cache() #can put this in the batch loop to free memory at the end of each batch but it slows things down
        # ---------- reduce -----------
        #globalize epoch metrics
        device = next(self.model.module.parameters()).device
        g_corr, g_cnt, _, _, g_loss = globalize_epoch_totals(
            local_correct=tracker.epoch_correct,
            local_count=tracker.epoch_count,
            local_loss_sum=tracker.epoch_loss_sum,
            device=device,
        )
        metrics = epoch_metrics_from_globals(g_correct=g_corr, g_count=g_cnt, g_loss_sum=g_loss)
        metrics['time'] = tracker.epoch_time()
        #gather logits and labels if buffers are requested
        gathered_buffers = self.buffers.gather_to_rank0(cast_fp16=False) if self.state['get_buffers'] else None
        if self.buffers is not None:
            self.buffers.clear()
        return metrics, gathered_buffers if self.state['get_buffers'] else None

    def train(self):
        if self.cfg.pretrained !='':
            self.update_state(ValidEpochStart())
            with torch.no_grad():
                # first validation run to get initial MMD and BCE
                #TODO: change the way secondary loader is passed and conditions are checked
                val_metrics, val_buffers = self._run_epoch(self.start_epoch-1, loader=self.dataloaders['valid'])
            #save logits and labels for validation
            if val_buffers is not None: 
                torch.save(val_buffers, f'{self.cfg.logdir}/{self.cfg.exp_name}/init_val_buffers.pt')

            self.metrics.append(
                epochs = self.start_epoch-1,
                val_loss = val_metrics['loss'], #TODO: what do I do about total vs BCE loss
                val_acc = val_metrics['acc'],
                val_time = val_metrics['time'],
            )

            self.metrics.update(best_val=self.metrics.get('val_loss')[-1],
                                best_epoch=self.start_epoch-1
                                )
            
            if self.dist_info.is_primary:
                display_epoch_summary(partition="validation", epoch=self.start_epoch-1, tot_epochs=self.final_epoch-1,
                                acc=self.metrics.get("val_acc")[-1], time_s=self.metrics.get("val_time")[-1], loss=self.metrics.get("val_loss")[-1],
                                logger=getattr(self, "logger", None))
        else:
            self.metrics.update(best_val=float('inf'), best_epoch=-1)
        ### training and validation
        if 'train' in self.dataloaders:
            self.dataloaders['train'].sampler.set_epoch(self.start_epoch-1)
        for epoch in range(self.start_epoch, self.final_epoch):
            self.update_state(TrainEpochStart())
            is_best=False
            #----------display learning rates------------
            lr_message =  'Learning rates\n'   
            for g in self.optimizer.param_groups:      
                lr_message += g['name'] + f": {g['lr']:.3e}  "
            lr_message += '\n' + 124*'-'
            if self.dist_info.is_primary:
                print(lr_message)
            #----------training------------
            train_metrics, _ = self._run_epoch(epoch, loader=self.dataloaders['train'])

            self.metrics.append(
                    epochs = epoch,
                    train_BCE = train_metrics['BCE_loss'],
                    train_loss = train_metrics['BCE_loss'], #TODO: what do I do about total vs BCE loss
                    train_acc = train_metrics['acc'],
                    train_time = train_metrics['time'],
                    lr = self.optimizer.param_groups[0]['lr'] #TODO: change this keyword in metrics
                )
            if self.dist_info.is_primary:
                display_epoch_summary(partition="train", epoch=epoch, tot_epochs=self.final_epoch-1,
                                bce=self.metrics.get("train_BCE")[-1], acc=self.metrics.get("train_acc")[-1], time_s=self.metrics.get("train_time")[-1],
                                logger=getattr(self, "logger", None))
            #----------validation------------
            if epoch % self.cfg.val_interval == 0:
                self.update_state(ValidEpochStart())
                with torch.no_grad():
                    val_metrics, _ = self._run_epoch(epoch,
                                                     loader=self.dataloaders['valid'])
                val_loss = val_metrics['loss']
                self.metrics.append(
                        val_loss = val_loss,
                        val_acc = val_metrics['acc'],
                        val_time = val_metrics['time'],
                    )
                
                if val_loss < self.metrics.get('best_val'):
                    is_best=True
                    self.metrics.update(best_val=val_loss,
                                        best_epoch=epoch
                                        )
                    
                checkpoint = {'epoch': epoch + 1, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
                self._save_ckp(checkpoint, is_best, epoch)

                json_object = json.dumps(self.metrics.to_dict(), indent=4)
                with open(f"{self.cfg.logdir}/{self.cfg.exp_name}/train-result.json", "w") as outfile:
                    outfile.write(json_object)
                if self.dist_info.is_primary:      
                    display_epoch_summary(partition="validation", epoch=epoch, tot_epochs=self.final_epoch-1,
                                acc=self.metrics.get("val_acc")[-1], time_s=self.metrics.get("val_time")[-1], 
                                best_epoch=self.metrics.get('best_epoch'), best_val=self.metrics.get('best_val'), 
                                loss=self.metrics.get("val_loss")[-1], logger=getattr(self, "logger", None))
            self.scheduler.step_epoch(val_metric=val_loss)
            dist.barrier() # syncronize
        # keys needed: ['epochs', 'train_BCE', 'val_BCE'] if do_MMD: += ['train_MMD', 'val_MMD', 'train_loss', 'val_loss']
        # otherwise use argument rename_map = {'old_key': 'new_key', ...}
        if self.dist_info.is_primary:
            make_train_plt(self.metrics.to_dict(), 
                            f"{self.cfg.logdir}/{self.cfg.exp_name}", 
                            pretrained=(self.cfg.pretrained !=''),
                            main_loss_name='loss')
    
    def test(self):
        ### test on best model
        best_model = torch.load(f"{self.cfg.logdir}/{self.cfg.exp_name}/best-val-model.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(best_model['state_dict'])
        self.update_state(TestEpochStart())
        
        with torch.no_grad():
            test_metrics, test_buffers = self._run_epoch(0, loader=self.dataloaders['valid'])
    
        #print(f'rank: {self.dist_info.rank}, test_buffers: {test_buffers}')
        if test_buffers is not None:
            torch.save(test_buffers, f'{self.cfg.logdir}/{self.cfg.exp_name}/best_val_buffers.pt')
            # plot final logits
            make_logits_plt({'Source': test_buffers['logit_diffs']},
                            f"{self.cfg.logdir}/{self.cfg.exp_name}", 
                            domains=None)
            ax = None
            metrics, ax = get_test_metrics(test_buffers['labels'].numpy(), test_buffers['logit_diffs'].numpy(), ax=ax) #TODO check default behavior of domain
            test_metrics.update(metrics)
            finish_roc_plot(f"{self.cfg.logdir}/{self.cfg.exp_name}", ax, is_primary=self.dist_info.is_primary)

        if os.path.exists(f'{self.cfg.logdir}/{self.cfg.exp_name}/init_val_buffers.pt'):
            # plot initial logits
            if self.dist_info.is_primary:
                #load initial validation buffers
                init_val_buffers = torch.load(f'{self.cfg.logdir}/{self.cfg.exp_name}/init_val_buffers.pt', weights_only=True)
                #TODO: check if I need this anymore or if I can generaize it
                make_logits_plt({"Source": init_val_buffers['logit_diffs']},
                                f"{self.cfg.logdir}/{self.cfg.exp_name}", name='initial',
                                domains=None)

        #TODO: check default behavior of domain
        if self.dist_info.is_primary:
            display_epoch_summary(partition="test", epoch=1, tot_epochs=0,
                            acc=test_metrics.get('acc', None), time_s=test_metrics.get('time', None),
                            logger=getattr(self, "logger", None), domain=None, auc=test_metrics.get('auc', None), 
                            r30=test_metrics.get('1/eB ~ 0.3', None), loss=test_metrics.get('loss', None))
        if self.dist_info.is_primary:
            json_object = json.dumps(test_metrics, indent=4)
            with open(f"{self.cfg.logdir}/{self.cfg.exp_name}/test-result.json", "w") as outfile:
                outfile.write(json_object)