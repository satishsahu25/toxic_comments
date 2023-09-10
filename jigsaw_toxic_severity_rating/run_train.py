import os
import gc
import warnings
import wandb
import time
import copy
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AdamW
from collections import defaultdict
from medal_challenger.utils import (
    id_generator, set_seed, get_dataframe, get_folded_dataframe, ConfigManager
)
from medal_challenger.configs import BERT_MODEL_LIST
from medal_challenger.dataset import prepare_loaders
from medal_challenger.model import JigsawModel, fetch_scheduler
from medal_challenger.train import train_one_epoch, valid_one_epoch, score_one_epoch
from colorama import Fore, Style

red_font = Fore.RED
blue_font = Fore.BLUE
yellow_font = Fore.YELLOW
reset_all = Style.RESET_ALL

# 경고 억제
warnings.filterwarnings("ignore")

# CUDA가 구체적인 에러를 보고하도록 설정
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def run_training(
        model, 
        optimizer, 
        scheduler, 
        fold,
        save_dir,
        train_loader,
        valid_loader,
        run,
        cfg
    ):

    # 자동으로 Gradients를 로깅
    wandb.watch(model, log_freq=100)
    
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_epoch_loss = np.inf
    history = defaultdict(list)
    best_file = None
    
    for epoch in range(1, cfg.train_param.epochs + 1): 
        gc.collect()
        train_epoch_loss = train_one_epoch(
                                model, 
                                optimizer, 
                                scheduler, 
                                dataloader=train_loader, 
                                epoch=epoch,
                                cfg=cfg,
                            )
        
        val_epoch_loss = valid_one_epoch(
                                model, 
                                valid_loader, 
                                device=cfg.model_param.device, 
                                epoch=epoch,
                                margin=cfg.train_param.ranking_margin
                            )

        if cfg.data_param.do_score_check:
            preds, score = score_one_epoch(
                                model, 
                                cfg
                            )
            print(f"Validation Score: {score}")
            
    
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        
        # Loss 로깅
        wandb.log({"Train Loss": train_epoch_loss})
        wandb.log({"Valid Loss": val_epoch_loss})
        if cfg.data_param.do_score_check:
            wandb.log({"Valid Score": score})
            wandb.log({"Valid Pred": preds})
        
        # 베스트 모델 저장
        if val_epoch_loss <= best_epoch_loss:
            print(f"{blue_font}Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            # 이전 베스트 모델 삭제
            if best_file is None:
                best_file = f'{save_dir}/[{cfg.training_keyword.upper()}]_SCHEDULER_{cfg.model_param.scheduler}_FOLD_{fold}_EPOCH_{epoch}_LOSS_{best_epoch_loss:.4f}.pt'
            else:
                os.remove(best_file)
                best_file = f'{save_dir}/[{cfg.training_keyword.upper()}]_SCHEDULER_{cfg.model_param.scheduler}_FOLD_{fold}_EPOCH_{epoch}_LOSS_{best_epoch_loss:.4f}.pt'
            
            run.summary["Best Loss"] = best_epoch_loss
            PATH = f"{save_dir}/[{cfg.training_keyword.upper()}]_SCHEDULER_{cfg.model_param.scheduler}_FOLD_{fold}_EPOCH_{epoch}_LOSS_{best_epoch_loss:.4f}.pt"
            # 모델 저장
            torch.save(model,PATH)
            print(f"{red_font}Model Saved{reset_all}")
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Loss: {:.4f}".format(best_epoch_loss))
    
    return history


def main(cfg):
 
    cfg.tokenizer = AutoTokenizer.from_pretrained(
        f"../models/{BERT_MODEL_LIST[cfg.model_param.model_name]}"
    )
    cfg.group = f'{cfg.program_param.project_name}.{cfg.model_param.model_name}.{cfg.training_keyword}'

    wandb.login(key=cfg.program_param.wandb_key)

    set_seed(cfg.program_param.seed)

    HASH_NAME = id_generator(size=12)

    # 모델 저장 경로 
    root_save_dir = '/jigsaw/checkpoint'
    save_dir = os.path.join(root_save_dir,cfg.model_param.model_name)
    os.makedirs(save_dir,exist_ok=True)

    # 데이터 경로 
    root_data_dir = '../input/'+cfg.data_param.dir_name.replace("_","-")
    train_csv = os.path.join(root_data_dir,cfg.data_param.train_file_name)   
    
    # 데이터프레임
    train_df = get_dataframe(train_csv)

    # K Fold
    train_df = get_folded_dataframe(
                    train_df, 
                    cfg.train_param.num_folds, 
                    cfg.program_param.seed, 
                    cfg.train_param.shuffle,
                    cfg.train_param.is_skf
                )

    # 학습 진행
    for fold in range(0, cfg.train_param.num_folds):

        print(f"{yellow_font}====== Fold: {fold} ======{reset_all}")

        run = wandb.init(
            project=cfg.program_param.project_name, 
            config=cfg,
            job_type='Train',
            group=cfg.group,
            tags=[
                cfg.model_param.model_name, HASH_NAME, cfg.train_param.loss
            ],
            name=f'{HASH_NAME}-fold-{fold}',
            anonymous='must'
        )
        
        train_loader, valid_loader = prepare_loaders(
            train_df,
            cfg,
            fold,
            is_train=True
        )

        model = JigsawModel(
            f"../models/{BERT_MODEL_LIST[cfg.model_param.model_name]}",
            cfg.model_param.num_classes,
            cfg.model_param.drop_p,
            cfg.model_param.is_extra_attn,
            cfg.model_param.is_deeper_attn,
            cfg.model_param.device,
            cfg.model_param.level_list,
        )
        model.to(cfg.model_param.device)
        
        optimizer = AdamW(
            model.parameters(), 
            lr=float(cfg.train_param.lr), 
            weight_decay=float(cfg.train_param.weight_decay)
        )
        scheduler = fetch_scheduler(optimizer,cfg)
        
        history = run_training(
                    model, 
                    optimizer, 
                    scheduler,
                    fold=fold,
                    save_dir=save_dir,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    run=run,
                    cfg=cfg,
                )
        
        run.finish()
        
        del model, history, train_loader, valid_loader
        _ = gc.collect()
        print()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config-file", 
        type=str, 
        required=True,
        help="Type Name Of Config File To Use."
    )
    parser.add_argument(
        "--train", 
        action='store_true',
        help="Toggle On If Model Is On Training."
    )
    parser.add_argument(
        "--training-keyword", 
        type=str, 
        default='hyperparameter_tuning',
        help="Type Keyword Of This Training."
    )

    args = parser.parse_args()

    cfg = ConfigManager(args).cfg
    main(cfg)