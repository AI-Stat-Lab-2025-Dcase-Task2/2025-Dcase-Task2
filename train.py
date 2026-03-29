import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", category=FutureWarning, module="hear21passt.models.preprocess")

import os
import glob
from typing import Union, List, Mapping
import wandb
import argparse
import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy

from ssmodule import ssmodule


def train(
        model: ssmodule,
        logger: Union[None, WandbLogger],
        args: dict
):
    lr_callback = LearningRateMonitor(logging_interval='step')
    # trainer
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=[int(d) for d in args['devices'].split(",")], 
        # strategy=DDPStrategy(find_unused_parameters=True),
        logger=logger,
        callbacks=[lr_callback],
        max_epochs=args['epochs'],
        accumulate_grad_batches=args['accumulation_steps'], 
        precision="16-mixed",
        num_sanity_val_steps=0,
        fast_dev_run=False, 
        enable_checkpointing=False
    )

    ### train on training set; monitor performance on val
    trainer.fit(
        model
    )

    return model

def test(
        model: ssmodule,
        logger: Union[None, WandbLogger],
        args: dict
) -> List[Mapping[str, float]]:
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=[int(d) for d in args['devices'].split(",")], 
        # strategy=DDPStrategy(find_unused_parameters=True), 
        logger=logger,
        callbacks=None,
        max_epochs=args['max_epochs'],
        precision="16-mixed",
        num_sanity_val_steps=0,
        fast_dev_run=False, 
        enable_checkpointing=False
    )

    ### test on the eval set
    result = trainer.test(
        model
    )

    return result

def load_best_model(exp_name):
    checkpoint_dir = f'exp/{exp_name}/checkpoints'
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch*_score*.pth'))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    best_checkpoint = None
    best_score = float('-inf')
    for ckpt_file in checkpoint_files:
        basename = os.path.basename(ckpt_file)
        score_str = basename.split('score')[-1].replace('.pth', '')
        try:
            score = float(score_str)
            if score > best_score:
                best_score = score
                best_checkpoint = ckpt_file
        except ValueError:
            continue

    print(f"Loading best model from {best_checkpoint} with final score {best_score:.4f}")
    return best_checkpoint

def get_args() -> dict:
    parser = argparse.ArgumentParser(description="Argument parser for training configuration.")

    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--word_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--logging', default=True, action=argparse.BooleanOptionalAction)

    # 🔥 핵심 수정 (상대경로)
    parser.add_argument('--train_path', type=str, default='data/pretrain_0.csv')
    parser.add_argument('--val_path', type=str, default='data/dev_test.csv')
    parser.add_argument('--test_path', type=str, default='data/eval.csv')
    parser.add_argument('--dev_train_path', type=str, default='data/dev_train.csv')

    parser.add_argument('--finetune_from', type=str, default=None)

    # 이하 동일
    parser.add_argument('--seed', type=int, default=21208)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--lr_decay_rate', type=float, default=0.8)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--scheduler', type=str, default='cosine_restart')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--warmup_epochs', type=int, default=1)
    parser.add_argument('--max_lr', type=float, default=1e-5)
    parser.add_argument('--min_lr', type=float, default=1e-7)
    parser.add_argument('--restart_period', type=int, default=5)
    parser.add_argument('--save_top_k', type=int, default=3)

    parser.add_argument('--audio_model', type=str, default='eat')
    parser.add_argument('--lora', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--lora_bias', type=str, default='all')
    parser.add_argument('--pool_lora', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--adaptor_lora', default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument('--freqm', type=float, default=80)
    parser.add_argument('--timem', type=float, default=80)

    parser.add_argument('--arcface_m', type=float, default=0.5)
    parser.add_argument('--arcface_s', type=float, default=64.0)

    parser.add_argument('--pool_ds_rate', type=int, default=3)
    parser.add_argument('--pool_output_dim', type=int, default=512)
    parser.add_argument('--pool_dropout', type=float, default=0.2)

    parser.add_argument('--aggregation', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--freeze_extractor', default=False, action=argparse.BooleanOptionalAction)

    parser.add_argument('--loss_type', type=str, default='arcface')
    parser.add_argument('--center_loss_lambda', type=float, default=1.0)

    parser.add_argument('--train', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--test', default=True, action=argparse.BooleanOptionalAction)

    parser.add_argument('--exp', default=None)

    return vars(parser.parse_args())


if __name__ == '__main__':
    """
    Entry point for training and testing the model.
    - Initializes logging and model.
    - Runs training and/or testing based on arguments.
    """
    torch.set_float32_matmul_precision('high')

    args = get_args()

    # set a seed to make experiments reproducible
    if args['seed'] > 0:
        seed_everything(args['seed'], workers=True)
    else:
        print("Not seeding experiment.")

    # initialize wandb, i.e., the logging framework
    logger = WandbLogger(
        project='asd',
        name=args['exp'],
    )
    # initialize the model
    model = ssmodule(**args)

    # train
    if args['train']:
        device = 'cuda:' + args['devices']
        if args['finetune_from'] is not None:
            ckpt_path = load_best_model(args['finetune_from'])
            ckpt = torch.load(ckpt_path, map_location=device)
            model.audio_feature_extractor.model.load_state_dict(ckpt['model'], strict=False)
            model.ss = ckpt['sampling_strategy']

            if args['pool_lora']:
                print('LoRA in pooling activated')
                model.att_pool.load_state_dict(ckpt['att_pool'], strict=False)
            else:
                model.att_pool.load_state_dict(ckpt['att_pool'])

            if args['aggregation']:
                if args['adaptor_lora']:
                    print('LoRA in adaptor activated')
                    model.audio_feature_extractor.adaptor.load_state_dict(ckpt['adaptor'], strict=False)
                else:
                    model.audio_feature_extractor.adaptor.load_state_dict(ckpt['adaptor'])

            model = train(model, logger, args)
        else:
            model = train(model, logger, args)

    # test
    if args['test']:
        # load weights
        ckpt_path = load_best_model(args['exp'])
        ckpt = torch.load(ckpt_path)
        model.audio_feature_extractor.model.load_state_dict(ckpt['model'], strict=False)
        model.att_pool.load_state_dict(ckpt['att_pool'])
        model.ss = ckpt['sampling_strategy']

        if args['aggregation']:
            model.audio_feature_extractor.adaptor.load_state_dict(ckpt['adaptor'])

        results = test(model, logger, args)
        print(results)
