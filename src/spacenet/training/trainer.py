# src/MMDLearning/training/trainer.py
from dataclasses import dataclass
from pathlib import Path
import torch, json, time
from torch import distributed as dist
from torch import nn, optim
import os

from .policies import StandardPolicy
from spacenet.configs import DistRuntime, TrainerConfig

from ml_tools.metrics.core import get_test_metrics, RunningStats
from ml_tools.metrics.segmentation import segmentation_agg_metrics
from ml_tools.training.reporting import display_epoch_summary, make_logits_plt, make_train_plt, display_status, finish_roc_plot
from ml_tools.training.training_utils import (
    Initialization, 
    TrainEpochStart, 
    ValidEpochStart,
    TestEpochStart, 
    MetricHistory,
    prefix_metrics
)
from ml_tools.utils.distributed import globalize_epoch_totals, epoch_metrics_from_globals
from ml_tools.utils.buffers import EpochLogitBuffer
from ml_tools.utils.random import make_and_set_seed

@dataclass
class Trainer:
    # core state you currently use as globals
    cfg: TrainerConfig
    model: nn.Module
    optimizer: optim.Optimizer
    dist_rt: DistRuntime
    scheduler: any
    loss_fn: nn.Module 
    dataloaders: dict   # {'train': ..., 'valid': ..., 'test': ...}
    start_epoch: int = 0
    base_seed: int = 0
    
    def __post_init__(self):
        self.final_epoch = self.cfg.epochs + self.start_epoch
        self.state = {}
        policy_kwargs = {}
        self.device = self.dist_rt.device
        self.is_primary = self.dist_rt.is_primary
        self.amp_dtype = getattr(torch, self.cfg.amp_dtype_str)
        self.scaler = torch.amp.GradScaler(self.device.type, 
                                           enabled=(self.cfg.use_amp and self.cfg.amp_dtype_str=='float16'))
        self.metrics = MetricHistory()
        lr_by_group = {group.get('name', 'all'): group['lr'] for group in self.optimizer.param_groups}
        assert len(lr_by_group) == len(self.optimizer.param_groups), "parameter groups must have names key if there are more than one group"
        self.metrics.update(lr_by_group=lr_by_group)
        self._handlers = {
            Initialization: self._initialize,
            TrainEpochStart: self._start_train_epoch,
            ValidEpochStart: self._start_valid_epoch,
            TestEpochStart: self._start_test_epoch,
        }
        self.update_state(Initialization()) #guess values
        self.buffer = EpochLogitBuffer(keep_indices=False, 
                                            keep_domains=False,
                                            assume_equal_lengths=True)
        self.policy = StandardPolicy(loss_fn=self.loss_fn, 
                                  buffers=self.buffer, 
                                  device=self.device,
                                  use_amp=self.cfg.use_amp,
                                  amp_dtype=self.amp_dtype,
                                  **policy_kwargs)
        self.trackers = {phase: RunningStats(
            phase=phase, 
            window=self.cfg.log_interval,
            weight_key='num_pixels',
            mean_keys=('loss',),
            sum_keys=("tp","fp","fn","tn"),
            ddp_sync=self.dist_rt.cfg.world_size>1,
        ) for phase in ('train', 'valid', 'test')}

    def _save_ckp(self, state, is_best, epoch, save_all=False):
        p = Path(self.cfg.logdir) / self.cfg.exp_name
        p.mkdir(parents=True, exist_ok=True)
        if save_all:
            torch.save(state, p / f"checkpoint-epoch-{epoch+1}.pt")
        if is_best:
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
        make_and_set_seed(self.base_seed, phase=self.state['phase'], epoch=epoch, rank=self.dist_rt.rank)
        if self.state['phase'] == 'train':
            self.model.train()
            loader.sampler.set_epoch(epoch)
        else:
            self.model.eval()

        tracker = self.trackers[self.state['phase']]
        tracker.reset_epoch()

        loader_length = len(loader) if loader is not None else 0
        #need to make prediction for source and target data
        for batch_idx, data in enumerate(loader):
            
            batch_metrics = self.policy.compute_batch_metrics(
                data=data, 
                model=self.model, 
                state=self.state
            )
            if self.state['phase'] == 'train':
                self.scaler.scale(batch_metrics['loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.optimizer.zero_grad(set_to_none=True)
                
            tracker.update(batch_metrics)

            if (batch_idx+1) % self.cfg.log_interval == 0:
                snap = tracker.rolling_snapshot()
                rolling_metrics = segmentation_agg_metrics(snap['tp'], snap['fp'], snap['fn'])
                rolling_metrics['loss'] = snap['loss']
                display_status(phase=self.state['phase'], epoch=epoch+1, 
                                tot_epochs=1 if self.state['phase']=='test' else self.final_epoch, #TODO: check if this gives correct tot_epochs
                                batch_idx=batch_idx+1, num_batches=loader_length,
                                metrics=rolling_metrics, avg_batch_time=tracker.avg_batch_time(),
                                is_master=self.is_primary, logger=getattr(self, "logger", None))
    
        torch.cuda.empty_cache() #can put this in the batch loop to free memory at the end of each batch but it slows things down
        # ---------- reduce -----------
        #globalize epoch metrics
        device = next(self.model.module.parameters()).device
        snap = tracker.epoch_snapshot_ddp(device=device)
        #gather logits and labels if buffers are requested
        gathered_buffers = self.buffer.gather_to_rank0(cast_fp16=False) if self.state['get_buffers'] else None
        if self.buffer is not None:
            self.buffer.clear()
        return snap, gathered_buffers if self.state['get_buffers'] else None, tracker.epoch_time()
    
    def _finalize_epoch(self,
                        *,
                        phase: str,
                        epoch: int,
                        snapshot: dict,
                        epoch_time: float):
        # will add 1 to epoch for display purposes, but keep it 0-indexed internally for checkpointing and scheduling
        if self.is_primary:
            phase_metrics = segmentation_agg_metrics(snapshot['tp'], snapshot['fp'], snapshot['fn'])
            phase_metrics['loss'] = snapshot['loss']
            phase_metrics['time'] = epoch_time
            if phase == 'train':
                phase_metrics['lr'] = self.optimizer.param_groups[0]['lr']
            self.metrics.append(epoch=epoch+1, **prefix_metrics(phase, phase_metrics))
            is_best = False
            if phase == 'valid' and phase_metrics['loss'] < self.metrics.get('best_val', float('inf')):
                is_best = True
                self.metrics.set(
                    best_val=phase_metrics['loss'],
                    best_epoch=epoch+1
                )
        
            display_epoch_summary(
                partition=phase,
                epoch=epoch+1,
                tot_epochs=self.final_epoch,
                time=epoch_time,
                metrics=phase_metrics,
                best_epoch=self.metrics.get("best_epoch"),
                best_val=self.metrics.get("best_val"),
                logger=getattr(self, "logger", None),
                is_master=self.is_primary,
            )
            return is_best if phase=='valid' else None
        
    def train(self):
        if self.start_epoch != 0:
            self.update_state(ValidEpochStart())
            with torch.no_grad():
                # first validation run to get initial MMD and BCE
                snapshot, val_buffers, epoch_time = self._run_epoch(self.start_epoch-1, loader=self.dataloaders['valid'])
            #save logits and labels for validation
            if val_buffers is not None: 
                torch.save(val_buffers, f'{self.cfg.logdir}/{self.cfg.exp_name}/init_val_buffers.pt')
            self._finalize_epoch(
                phase='valid',
                epoch=self.start_epoch-1,
                snapshot=snapshot,
                epoch_time=epoch_time
            ) 
        else:
            self.metrics.set(best_val=float('inf'), best_epoch=-1)
        ### training and validation
        if 'train' in self.dataloaders:
            self.dataloaders['train'].sampler.set_epoch(self.start_epoch-1)
        for epoch in range(self.start_epoch, self.final_epoch):
            self.update_state(TrainEpochStart())
            is_best=False
            #----------display learning rates------------
            lr_message =  'Learning rates\n'   
            for g in self.optimizer.param_groups:      
                lr_message += g.get('name', 'all') + f": {g['lr']:.3e}  "
            lr_message += '\n' + 124*'-'
            if self.is_primary:
                print(lr_message)
            #----------training------------
            snapshot, _, epoch_time = self._run_epoch(epoch, loader=self.dataloaders['train'])
            self._finalize_epoch(
                phase='train',
                epoch=epoch,
                snapshot=snapshot,
                epoch_time=epoch_time
            ) 
            #----------validation------------
            if epoch % self.cfg.val_interval == 0:
                self.update_state(ValidEpochStart())
                with torch.no_grad():
                    snapshot, _, epoch_time= self._run_epoch(epoch, loader=self.dataloaders['valid'])
                is_best = self._finalize_epoch(
                    phase='valid',
                    epoch=epoch,
                    snapshot=snapshot,
                    epoch_time=epoch_time
                )
                if self.is_primary:
                    checkpoint = {'epoch': epoch, 'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
                    self._save_ckp(checkpoint, is_best, epoch)

                json_object = json.dumps(self.metrics.to_dict(), indent=4)
                with open(f"{self.cfg.logdir}/{self.cfg.exp_name}/train-result.json", "w") as outfile:
                    outfile.write(json_object)
            dist.barrier() # syncronize
        # keys needed: ['epochs', 'train_BCE', 'val_BCE'] if do_MMD: += ['train_MMD', 'val_MMD', 'train_loss', 'val_loss']
        # otherwise use argument rename_map = {'old_key': 'new_key', ...}
        if self.is_primary:
            make_train_plt(self.metrics.to_dict(), 
                            f"{self.cfg.logdir}/{self.cfg.exp_name}", 
                            pretrained=(self.start_epoch != 0),
                            main_loss_name='loss')
    
    def test(self):
        ### test on best model
        best_model = torch.load(f"{self.cfg.logdir}/{self.cfg.exp_name}/best-val-model.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(best_model['state_dict'])
        self.update_state(TestEpochStart())
        
        with torch.no_grad():
            test_metrics, test_buffers = self._run_epoch(0, loader=self.dataloaders['valid'])
    
        if test_buffers is not None:
            torch.save(test_buffers, f'{self.cfg.logdir}/{self.cfg.exp_name}/best_val_buffers.pt')
            # plot final logits
            make_logits_plt({'Source': test_buffers['logit_diffs']},
                            f"{self.cfg.logdir}/{self.cfg.exp_name}", 
                            domains=None)
            ax = None
            metrics, ax = get_test_metrics(test_buffers['labels'].numpy(), test_buffers['logit_diffs'].numpy(), ax=ax)
            test_metrics.update(metrics)
            finish_roc_plot(f"{self.cfg.logdir}/{self.cfg.exp_name}", ax, is_primary=self.is_primary)

        if os.path.exists(f'{self.cfg.logdir}/{self.cfg.exp_name}/init_val_buffers.pt'):
            # plot initial logits
            if self.is_primary:
                #load initial validation buffers
                init_val_buffers = torch.load(f'{self.cfg.logdir}/{self.cfg.exp_name}/init_val_buffers.pt', weights_only=True)
                #TODO: check if I need this anymore or if I can generaize it
                make_logits_plt({"Source": init_val_buffers['logit_diffs']},
                                f"{self.cfg.logdir}/{self.cfg.exp_name}", name='initial',
                                domains=None)

        if self.is_primary:
            display_epoch_summary(partition="test", epoch=1, tot_epochs=0,
                            acc=test_metrics.get('acc', None), time_s=test_metrics.get('time', None),
                            logger=getattr(self, "logger", None), domain=None, auc=test_metrics.get('auc', None), 
                            r30=test_metrics.get('1/eB ~ 0.3', None), loss=test_metrics.get('loss', None))
        if self.is_primary:
            json_object = json.dumps(test_metrics, indent=4)
            with open(f"{self.cfg.logdir}/{self.cfg.exp_name}/test-result.json", "w") as outfile:
                outfile.write(json_object)