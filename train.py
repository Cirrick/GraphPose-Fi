import os
import argparse
import json
import math
import yaml
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from feeder.mmfi import make_dataset, make_dataloader
from utils import *
from model.model import GraphPoseFiNet, _weights_init

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pose Decoding Stage")
    parser.add_argument("--config_file", type=str, help="Configuration YAML file", default="config/mmfi/pose_config_p1s1.yaml")
    parser.add_argument("--experiment_name", type=str, help="Experiment name", default="graphposefi")
    parser.add_argument("--tag", type=str, help="", default="")
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default=3e-4)
    parser.add_argument("--weight_decay", type=float, help="Weight decay", default=0.02)
    parser.add_argument("--total_epoch", type=int, help="Total epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=256)
    parser.add_argument("--max_device_batch_size", type=int, help="Max device batch size", default=256)
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument("--num_frames", type=int, help="Number of frames", default=1)
    parser.add_argument("--dropout", type=float, help="Dropout rate", default=0.1)
    parser.add_argument("--num_workers", type=int, default=16, help="Number of workers for DataLoader")
    parser.add_argument("--pretrained_weights", action="store_true", help="Use pretrained weights")
    parser.add_argument("--graattention_layers", type=int, default=4, help="Number of layers in the graph attention network")
    parser.add_argument("--agg_mode", type=str, default="attn2", help="Aggregation mode")
    parser.add_argument("--description", type=str, default="", help="Description of the experiment")

    args = parser.parse_args()
    exp_tag = f"{args.experiment_name}_{args.tag}" if args.tag else args.experiment_name
    
    with open(args.config_file, 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    
    setup_seed(args.seed)
    dataset_root = os.path.expanduser(config['dataset_root'])
    if args.tag == 'debug':
        logs_path = os.path.join('logs_debug', config['dataset_name'], config['setting'], 'pose_scratch', exp_tag)
    else:
        logs_path = os.path.join('logs', config['dataset_name'], config['setting'], 'pose_scratch', exp_tag)

    total_epochs = args.total_epoch
    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)
    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    print(f"Effective Batch Size: {batch_size}")
    print(f"GPU Load Batch Size: {load_batch_size}")
    print(f"Gradient Accumulation Steps: {steps_per_update}")

    # load dataset
    if config['dataset_name'] == 'mmfi-csi':
        print(f"Loading MM-FI dataset from: {dataset_root}")
        train_dataset, val_dataset = make_dataset(config['training_semi'], dataset_root, config)
        rng_generator = torch.manual_seed(config['init_rand_seed'])
        train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, batch_size=load_batch_size)
        val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, batch_size=load_batch_size)
    else:
        print('No dataset!')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    writer = SummaryWriter(logs_path)

    print('*'*20+'   Training from Scratch  '+'*'*20)
    print('*'*20+'  '+config['dataset_name']+','+config['setting']+','+args.experiment_name+'   '+'*'*20)
    if config['dataset_name'] == 'mmfi-csi':
        model = GraphPoseFiNet(num_keypoints=17, num_coor=3, num_person=config['num_person'],dataset=config['dataset_name'], pretrained_weights = args.pretrained_weights, num_layers=args.graattention_layers, agg_mode=args.agg_mode).to(device)
        model.apply(_weights_init)
        warmup_epochs = 5
        def lr_lambda(cur_epoch):
            if cur_epoch < warmup_epochs:
                return (cur_epoch + 1) / warmup_epochs
            progress = (cur_epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
        optim = torch.optim.AdamW(param_groups_simple(model, lr=args.learning_rate, wd=args.weight_decay), lr=args.learning_rate, weight_decay=0.0, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
        print("Initializing GraphPoseFi model...")

        # save model
        weights_path = os.path.join(config['save_path'], config['dataset_name'], config['setting'], 'pose_scratch', config['experiment_name'], exp_tag)
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)

    optim.zero_grad()
    step_count = 0
    best_val_mpjpe = float('inf')
    best_val_pampjpe = float('inf')
    best_val_mpjpe_align = 1
    best_val_pampjpe_align = 1
    best_val_pck = [0 for _ in range(5)]
    best_val_pck_align = [0 for _ in range(5)]
    pck_order = [50, 40, 30, 20, 10]
    best_weights = {
        "mpjpe": None,
        "pampjpe": None,
        "pck@50": None,
        "pck@40": None,
        "pck@30": None,
        "pck@20": None,
        "pck@10": None,
    }
    sum_pose, sum_bone, sum_total = 0.0, 0.0, 0.0
    for epoch in range(1, args.total_epoch+1):
        model.train()
        epoch_train_losses = []
        losses = []
        optim.zero_grad()
        start_time = time.time()
        
        mpjpe_list = []
       
        pck_iter = [[] for _ in range(5)]
        pck_align_iter = [[] for _ in range(5)]
        for batch_idx, batch_data in enumerate(train_loader):
            if args.experiment_name == 'graphposefi':
                csi_data = batch_data['input_wifi-csi'].unsqueeze(-1)
            csi_data = csi_data.to(device) 
            pose_gt = batch_data['output'].to(device)
            if args.experiment_name == 'graphposefi':
                predicted_pose, _ = model(csi_data)
            loss = torch.mean(torch.norm(predicted_pose-pose_gt, dim=-1))
            sum_pose += float(loss.detach().cpu())
            mpjpe, _, _, _ = calulate_error(predicted_pose.data.cpu().numpy(), pose_gt.data.cpu().numpy(), align=False)
            mpjpe_list += mpjpe.tolist()

            loss.backward()
            losses.append(loss.item())
            step_count += 1

            if step_count % steps_per_update == 0:
                torch.nn.utils.clip_grad_norm_( (p for p in model.parameters() if p.requires_grad), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
        scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_mpjpe = sum(mpjpe_list) / len(mpjpe_list)
        epoch_time = time.time() - start_time
        current_lr = optim.param_groups[0]['lr']
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        print(f'Epoch {epoch}/{total_epochs} | Train Loss: {avg_train_loss:.4f} | Train mpjpe: {avg_mpjpe:.4f} | LR: {current_lr:.6f} | Time: {epoch_time:.2f}s')

        # Validation
        model.eval()
        epoch_val_losses = []
        all_val_preds = []
        all_val_gts = []
        with torch.no_grad():
            losses = []
            mpjpe_list = []
            pampjpe_list = []
            
            pck_iter = [[] for _ in range(5)]
            pck_align_iter = [[] for _ in range(5)]
            subject_mpjpe = {}
            for batch_idx, batch_data in enumerate(val_loader):
                if args.experiment_name == 'graphposefi':
                    val_csi_data = batch_data['input_wifi-csi'].unsqueeze(-1)
                val_csi_data = val_csi_data.to(device)  
                val_pose_gt = batch_data['output'].to(device)
                if args.experiment_name == 'graphposefi':
                    predicted_val_pose, _ = model(val_csi_data)               
                loss = torch.mean(torch.norm(predicted_val_pose-val_pose_gt, dim=-1))
                # calculate the pck, mpjpe, pampjpe
                for idx, percentage in enumerate([0.5, 0.4, 0.3, 0.2, 0.1]):
                    pck_iter[idx].append(compute_pck_pckh(predicted_val_pose.permute(0,2,1).data.cpu().numpy(), val_pose_gt.permute(0,2,1).data.cpu().numpy(), percentage, align=False, dataset=config['dataset_name']))
                mpjpe, pampjpe, mpjpe_joints, pampjpe_joints = calulate_error(predicted_val_pose.data.cpu().numpy(), val_pose_gt.data.cpu().numpy(), align=False)
                mpjpe_list += mpjpe.tolist()
                pampjpe_list += pampjpe.tolist()
                losses.append(loss.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_mpjpe = sum(mpjpe_list) / len(mpjpe_list)
            avg_val_pampjpe = sum(pampjpe_list) / len(pampjpe_list)
            if config['dataset_name'] == 'mmfi-csi':
                pck_overall = [np.mean(pck_value, 0)[17] for pck_value in pck_iter]           
        print(f'In epoch {epoch}, test losss: {avg_val_loss}')
        print(f'test mpjpe: {avg_val_mpjpe}, test pa-mpjpe: {avg_val_pampjpe}, test pck50: {pck_overall[0]}, test pck40: {pck_overall[1]}, test pck30: {pck_overall[2]}, test pck20: {pck_overall[3]}, test pck10: {pck_overall[4]}.')

        ''' save model '''
        # save mpjpe pampjpe
        if avg_val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = avg_val_mpjpe
            best_weights["mpjpe"] = f'{weights_path}/pose_mpjpe.pt'
            torch.save(model.state_dict(), '{}/pose_mpjpe.pt'.format(weights_path)) 
        if avg_val_pampjpe < best_val_pampjpe:
            best_val_pampjpe = avg_val_pampjpe
            best_weights["pampjpe"] = f'{weights_path}/pose_pampjpe.pt'
            torch.save(model.state_dict(), '{}/pose_pampjpe.pt'.format(weights_path)) 
        for idx, pck_value in enumerate(pck_overall):
            if pck_value > best_val_pck[idx]:
                best_val_pck[idx] = pck_value
                pck_tag = pck_order[idx]
                best_weights[f"pck@{pck_tag}"] = f'{weights_path}/pose_pck{pck_tag}.pt'
                torch.save(model.state_dict(), '{}/pose_pck{}.pt'.format(weights_path, pck_order[idx]))
        
        writer.add_scalars('cls/loss', {'train' : avg_train_loss, 'val' : avg_val_loss}, global_step=epoch)
        torch.cuda.empty_cache()

    print('*'*100)
    print(f'Best mpjpe: {best_val_mpjpe}') 
    print(f'Best pa-mpjpe: {best_val_pampjpe}')  
    for idx, pck_value in enumerate(best_val_pck):
        print(f'Best pck{pck_order[idx]}: {pck_value}')  
    print('*'*100)
    # ─── Write TensorBoard Scalar ────────────────────────
    writer.add_scalar('Best/MPJPE', best_val_mpjpe, global_step=0)
    writer.add_scalar('Best/PA-MPJPE', best_val_pampjpe, global_step=0)
    for idx, pck_value in enumerate(best_val_pck):
        writer.add_scalar(f'Best/PCK@{pck_order[idx]}', pck_value, global_step=0)

    # ─── Write TensorBoard Text (Summary) ─────────────────────
    summary_text = f"Best MPJPE: {best_val_mpjpe:.5f} mm\nBest PA-MPJPE: {best_val_pampjpe:.5f} mm\n"
    summary_text += "\n".join([f"PCK@{pck_order[idx]}: {pck_value:.4f}" for idx, pck_value in enumerate(best_val_pck)])
    writer.add_text("Eval/Best_Results", summary_text, global_step=0)
    args_save_path = os.path.join(logs_path, "args.json")
    args_dict = vars(args)
    log_config(config, logs_path, writer)
    os.makedirs(os.path.dirname(args_save_path), exist_ok=True)
    with open(args_save_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    best_json = {
        "best_values": {
            "mpjpe": best_val_mpjpe,
            "pampjpe": best_val_pampjpe,
            "pck@50": best_val_pck[0],
            "pck@40": best_val_pck[1],
            "pck@30": best_val_pck[2],
            "pck@20": best_val_pck[3],
            "pck@10": best_val_pck[4],
        },
        "weights": best_weights,
        "paths": {
            "logs_path": logs_path,
            "weights_path": weights_path,
        }
    }
    with open(os.path.join(logs_path, "best_metrics.json"), "w") as f:
        json.dump(best_json, f, indent=4)

    writer.close()
    
    
