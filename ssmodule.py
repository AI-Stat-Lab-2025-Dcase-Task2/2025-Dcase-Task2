import os 
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.utils.compute_result import compute_result
from model.utils.optimizer import create_optimizer_scheduler
from model.data.dataset import custom_collate_fn, CustomDataset

import lightning.pytorch as pl

import loralib as lora
from pytorch_metric_learning.losses import ArcFaceLoss
from model.ssmodel.models import audio_feature_extractor
from model.utils.pool import attentive_statistics_pooling
from model.utils.center_loss import CenterLoss

import warnings
warnings.filterwarnings('ignore')
        
class ssmodule(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        
        self.audio_feature_extractor = audio_feature_extractor(
            model_name=kwargs['audio_model'], 
            use_lora=kwargs['lora'], 
            aggregation=kwargs['aggregation'], 
            adaptor_lora=kwargs['adaptor_lora']
            )
        self.att_pool = attentive_statistics_pooling(
            input_dim=768, 
            embed_dim=768, 
            ds_rate=kwargs['pool_ds_rate'], 
            output_dim=kwargs['pool_output_dim'], 
            dropout=kwargs['pool_dropout'], 
            use_lora=kwargs['pool_lora'])
        
        if kwargs['lora']: 
            print('Marking only LoRA as trainable')
            lora.mark_only_lora_as_trainable(self.audio_feature_extractor.model, bias=kwargs['lora_bias'])

        if kwargs['adaptor_lora']:
            lora.mark_only_lora_as_trainable(self.audio_feature_extractor.adaptor, bias=kwargs['lora_bias'])
        
        if kwargs['pool_lora']:
            lora.mark_only_lora_as_trainable(self.att_pool, bias=kwargs['lora_bias'])

        n_classes = pd.read_csv(kwargs['train_path'])['label'].nunique()

        if kwargs['loss_type']=='arcface':
            self.arcface = ArcFaceLoss(n_classes, 512, margin=kwargs['arcface_m'], scale=kwargs['arcface_s'])
        elif kwargs['loss_type']=='center_loss':
            self.center_loss = CenterLoss(n_classes, embedding_dim=512)
            self.center_loss_lambda = kwargs['center_loss_lambda']
            self.CE_loss = nn.CrossEntropyLoss()
        
        self.validation_outputs = []
        self.test_outputs = []
        
        # Initialize best final score and best checkpoints list
        self.best_final_score = float('-inf')
        self.best_checkpoints = []  # List to keep track of best checkpoints
        self.save_top_k = kwargs['save_top_k']

        self.distributed_mode = True if len([int(d) for d in kwargs['devices'].split(",")])>1 else False
        print(f"Distributed mode: {self.distributed_mode}")

        # Freeze the audio_feature_extractor
        if kwargs['freeze_extractor']:
            for param in self.audio_feature_extractor.model.parameters():
                param.requires_grad = False
            print('Freeze Model')

        self.kwargs = kwargs
        
    def make_dataloader(self, data_path, shuffle=False, augment=False, drop_last=False):
        dataset = CustomDataset(data_path, augment=augment, freqm=self.kwargs['freqm'], timem=self.kwargs['timem'], audio_model=self.kwargs['audio_model'])
        return DataLoader(dataset, batch_size=self.kwargs['batch_size'], num_workers=self.kwargs['num_workers'], shuffle=shuffle, collate_fn=custom_collate_fn, drop_last=drop_last)

    def train_dataloader(self):
        return self.make_dataloader(self.kwargs['train_path'], shuffle=True, augment=True, drop_last=True)
        
    def val_dataloader(self):
        return self.make_dataloader(self.kwargs['val_path'])
        
    def test_dataloader(self):
        return self.make_dataloader(self.kwargs['test_path'])
        
    def forward(self, batch):
        xs = torch.stack(batch['source'], dim=0).to(self.device)
        if self.kwargs['audio_model']=='beats':
            xs = xs.squeeze()
        padding_mask = torch.stack(batch['padding_mask'], dim=0).to(self.device)
        padding_mask = padding_mask.squeeze()
        if self.kwargs['audio_model']=='beats':

            if self.training:
                spec_aug = True
            else:
                spec_aug = False

            with torch.cuda.amp.autocast(enabled=False):
                feats = self.audio_feature_extractor(xs, padding_mask, spec_aug=spec_aug)
        else:
            feats = self.audio_feature_extractor(xs, padding_mask)
        feats = self.att_pool(feats)
        return feats
        
    def training_step(self, batch, batch_idx):
        feats = self(batch)
        y = torch.tensor(batch['label']).to(self.device)

        if self.kwargs['loss_type']=='arcface':
            loss = self.arcface(feats, y)

        elif self.kwargs['loss_type']=='center_loss':
            with torch.cuda.amp.autocast(enabled=False):
                center_loss = self.center_loss(feats, y)
                ce_loss = self.CE_loss(feats, y)
                loss = ce_loss + (self.center_loss_lambda * center_loss)

        self.log('train_loss', loss, batch_size=len(feats), sync_dist=True, prog_bar=True)
        return loss
    
    def get_embeddings(self, dataset):
        # Compute embeddings of the dataset
        if dataset=='train':
            data_loader = self.make_dataloader(self.kwargs['dev_train_path'])
        elif dataset=='test':
            data_loader = self.make_dataloader(self.kwargs['val_path'])
        embeddings = []
        self.audio_feature_extractor.eval()
        self.att_pool.eval()
        with torch.no_grad():
            for batch in data_loader:
                feats = self(batch)
                embeddings.extend(feats.cpu())
        return torch.stack(embeddings)

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            feats = self(batch)        
        self.validation_outputs.extend(feats.cpu())
        
    def on_validation_epoch_end(self):
        result_train = self.get_embeddings("train").numpy()
        result_test = torch.stack(self.validation_outputs).numpy()

        if self.distributed_mode:
            all_result_train = self.all_gather(result_train)
            all_result_test = self.all_gather(result_test)

            result_train = all_result_train.reshape(-1, result_train.shape[-1])
            result_test = all_result_test.reshape(-1, result_test.shape[-1])
        

        machine_results, mean_auc_source, mean_auc_target, mean_p_auc, final_score, sampling_strategy = compute_result(
            result_train, result_test
            )

        self.log('mean_AUC_source', mean_auc_source, add_dataloader_idx=False, sync_dist=True)
        self.log('mean_AUC_target', mean_auc_target, add_dataloader_idx=False, sync_dist=True)
        self.log('mean_pAUC', mean_p_auc, add_dataloader_idx=False, sync_dist=True)
        self.log('final_score', final_score, add_dataloader_idx=False, sync_dist=True)
        self.log('sampling_strategy', sampling_strategy, add_dataloader_idx=False, sync_dist=True)
            
        if self.trainer.is_global_zero:
            self.update_best_checkpoints(final_score)
        
        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            feats = self(batch)  
        self.test_outputs.extend(feats.cpu())
        
    def on_test_epoch_end(self):
        result_train = self.get_embeddings("train").numpy()
        result_eval = torch.stack(self.test_outputs).numpy()
        
        # compute result
        anomaly_scores = compute_result(result_train, result_eval, self.ss, test=True)

        eval_eval = pd.read_csv(self.kwargs['test_path'])
        eval_eval['anomaly_score'] = anomaly_scores
        sub_eval = pd.DataFrame(eval_eval['audio_path'].apply(lambda x: x.split('/')[-1]))
        sub_eval['anomaly score'] = eval_eval['anomaly_score']
        sub_eval['machine'] = eval_eval['machine']
            
        for machine in sub_eval['machine'].unique():
            temp = sub_eval[sub_eval['machine']==machine]
            temp.drop(columns='machine', inplace=True)
            save_path = os.path.join("exp", self.kwargs['exp'], "submission")
            os.makedirs(save_path, exist_ok=True)
            temp.to_csv(f'{save_path}/anomaly_score_{machine}_section_00_test.csv', index=None, header=False)
            temp.to_csv(f'{save_path}/decision_result_{machine}_section_00_test.csv', index=None, header=False)
    
        # Clear the outputs for the next epoch
        self.test_outputs.clear()
        
    def update_best_checkpoints(self, current_score):
        checkpoint_dir = f"exp/{self.kwargs['exp']}/checkpoints/"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f'checkpoint_epoch{self.current_epoch}_score{current_score:.4f}.pth'
        )

        ckpt = {
            'att_pool': self.att_pool.state_dict(), 
            'sampling_strategy': self.ss
        }

        if self.kwargs['aggregation']: 
            ckpt['adaptor'] = self.audio_feature_extractor.adaptor.state_dict()
        
        if self.kwargs['lora']: 
            ckpt['model'] = lora.lora_state_dict(self.audio_feature_extractor.model, bias=self.kwargs['lora_bias'])
        else:
            ckpt['model'] = self.audio_feature_extractor.model.state_dict(),
        
        torch.save(ckpt, checkpoint_path)
        self.best_checkpoints.append((current_score, checkpoint_path))
        
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        if len(self.best_checkpoints) > self.save_top_k:
            worst_checkpoint = self.best_checkpoints.pop(-1)
            worst_checkpoint_path = worst_checkpoint[1]
            if os.path.exists(worst_checkpoint_path):
                os.remove(worst_checkpoint_path)
                print(f"Removed checkpoint {worst_checkpoint_path} with score {worst_checkpoint[0]:.4f}")
            
    def configure_optimizers(self):
        num_samples = len(self.train_dataloader().dataset)
        optimizer, scheduler = create_optimizer_scheduler(self.parameters(), num_samples, **self.kwargs)
        return [optimizer], [scheduler]