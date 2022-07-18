import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from canonical3d import Canonical3D, LinReg

import os
from tqdm import tqdm
import argparse
import time
import numpy as np
import random

### importing OGB-LSC
from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator

reg_criterion = torch.nn.L1Loss()

def train(canonic_model, pred_model, device, loader, optimizer, args, task = "canonical"):
    if task == "canonical":
        canonic_model.train()
    elif task == "predict":
        pred_model.train()
    
    loss_accum = 0
    
    print("---------- start training {} ----------".format(task))
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        optimizer.zero_grad()
        if task == "canonical":
            # if training canonical, then model should be Canonical3d
            # perform batch normalization, computing covariance matrices and calculate
            # loss for Canonical3d
            z1, z2 = canonic_model(batch)
            
            c = torch.mm(z1.T, z2)
            c1 = torch.mm(z1.T, z1)
            c2 = torch.mm(z2.T, z2)
            
            c = c / batch.size(0)
            c1 = c1 / batch.size(0)
            c2 = c2 / batch.size(0)
            
            
            loss_inv = -torch.diagonal(c).sum()
            iden = torch.tensor(np.eye(c.shape[0])).to(args.device)
            loss_dec1 = (iden - c1).pow(2).sum()
            loss_dec2 = (iden - c2).pow(2).sum()

            loss = loss_inv + args.lambd * (loss_dec1 + loss_dec2)

            loss.backward()
            optimizer.step()
            loss_accum += loss.detach().cpu().item()
            
            print('Batch={:03d}, loss={:.4f}'.format(step, loss.item()))

        elif task == "predict":
            # predict property by regression
            embed = canonic_model(batch)
            pred = pred_model(embed).view(-1,)
            loss = reg_criterion(pred, batch.y)
            loss.backward()
            optimizer.step()
            loss_accum += loss.detach().cpu().item()
            
            print('Batch={:03d}, loss={:.4f}'.format(step, loss.item()))
        
        else:
            raise ValueError("Unvalid task")

    return loss_accum / (step + 1)

def eval(canonic_model, pred_model, device, loader, evaluator):
    canonic_model.eval()
    pred_model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():  
            embed = canonic_model(batch)
            pred = pred_model(embed).view(-1,)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]

def test(canonic_model, pred_model, device, loader):
    canonic_model.eval()
    pred_model.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            embed = canonic_model(batch)
            pred = pred_model(embed).view(-1,)

        y_pred.append(pred.detach().cpu())

    y_pred = torch.cat(y_pred, dim = 0)

    return y_pred


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on pcqm4m with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--graph_pooling', type=str, default='sum',
                        help='graph pooling strategy mean or sum (default: sum)')
    parser.add_argument('--drop_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=600,
                        help='dimensionality of hidden units in GNNs (default: 600)')
    parser.add_argument('--train_subset', action='store_true')
    parser.add_argument('--lambd', type=float, default=1e-3, help='trade-off ratio.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default = '', help='directory to save checkpoint')
    parser.add_argument('--save_test_dir', type=str, default = '', help='directory to save test submission file')
    parser.add_argument('--lr1', type=float, default=1e-3, help='Learning rate of canonical3d')
    parser.add_argument('--lr2', type=float, default=1e-2, help='Learning rate of linear regressor.')
    parser.add_argument('--wd1', type=float, default=0, help='Weight decay of canonical3d.')
    parser.add_argument('--wd2', type=float, default=1e-4, help='Weight decay of linear regressor.')

    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygPCQM4Mv2Dataset(root = '../data/dataset/')

    split_idx = dataset.get_idx_split()

    ### automatic evaluator. takes dataset name as input
    evaluator = PCQM4Mv2Evaluator()

    if args.train_subset:
        subset_ratio = 0.1
        subset_idx = torch.randperm(len(split_idx["train"]))[:int(subset_ratio*len(split_idx["train"]))]
        train_loader = DataLoader(dataset[split_idx["train"][subset_idx]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    else:
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    if args.save_test_dir != '':
        testdev_loader = DataLoader(dataset[split_idx["test-dev"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        testchallenge_loader = DataLoader(dataset[split_idx["test-challenge"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    if args.checkpoint_dir != '':
        os.makedirs(args.checkpoint_dir, exist_ok = True)

    shared_params = {
        'num_layers': args.num_layers,
        'emb_dim': args.emb_dim,
        'drop_ratio': args.drop_ratio,
    }

    
    canonical_model = Canonical3D(residual = True, gnn_type = 'gin', virtual = False, **shared_params).to(device)
    pred_model = LinReg(args.emb_dim, args.emb_dim).to(device)

    canonical_optimizer = optim.Adam(canonical_model.parameters(), lr=args.lr1, weight_decay=args.wd1)
    pred_optimizer = optim.Adam(pred_model.parameters(), lr=args.lr2, weight_decay=args.wd2)

    if args.log_dir != '':
        writer = SummaryWriter(log_dir=args.log_dir)

    best_valid_mae = 1000

    if args.train_subset:
        scheduler = StepLR(pred_optimizer, step_size=300, gamma=0.25)
        args.epochs = 1000
    else:
        scheduler = StepLR(pred_optimizer, step_size=30, gamma=0.25)

    # 1. First train self-supervised Canonical
    for epoch in range(1, args.epochs + 1):
       
        print("------ Training Canonical ------")
       
        train_loss = train(canonic_model = canonical_model, pred_model = pred_model, device = device, loader = train_loader, 
                                args = args, optimizer = canonical_optimizer, task = "canonical")

        print("Epoch:{}, loss: {:.4f}".format(epoch, train_loss))


    # 2. Second train Regressor
    for epoch in range(1, args.epochs + 1):
        train_mae = train(canonic_model = canonical_model, pred_model = pred_model, device = device, 
                            loader = train_loader, args = args, optimizer = pred_optimizer, task = "predict")

        print('Evaluating...')
        valid_mae = eval(canonical_model, pred_model, device, valid_loader, evaluator)

        print({'Train': train_mae, 'Validation': valid_mae})

        if args.log_dir != '':
            writer.add_scalar('valid/mae', valid_mae, epoch)
            writer.add_scalar('train/mae', train_mae, epoch)

        if valid_mae < best_valid_mae:
            best_valid_mae = valid_mae
            if args.checkpoint_dir != '':
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': pred_model.state_dict(), 'optimizer_state_dict': pred_optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_val_mae': best_valid_mae}
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'checkpoint.pt'))

            if args.save_test_dir != '':
                testdev_pred = test(canonical_model, pred_model, device, testdev_loader)
                testdev_pred = testdev_pred.cpu().detach().numpy()

                testchallenge_pred = test(canonical_model, pred_model, device, testchallenge_loader)
                testchallenge_pred = testchallenge_pred.cpu().detach().numpy()

                print('Saving test submission file...')
                evaluator.save_test_submission({'y_pred': testdev_pred}, args.save_test_dir, mode = 'test-dev')
                evaluator.save_test_submission({'y_pred': testchallenge_pred}, args.save_test_dir, mode = 'test-challenge')

        scheduler.step()
            
        print(f'Best validation MAE so far: {best_valid_mae}')

    if args.log_dir != '':
        writer.close()


if __name__ == "__main__":
    main()