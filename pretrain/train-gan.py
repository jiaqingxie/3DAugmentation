import torch

from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from gnn import GNN

import os
from tqdm import tqdm
import argparse
import time
import numpy as np
import random

from gan import Discriminatorfor3D
import sys
sys.path.append("../")
### importing OGB-LSC
from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator
from data.PCQM4Mv2_xyz import *

reg_criterion = torch.nn.L1Loss()
D_criterion=torch.nn.BCELoss()
def train(netG, device, loader, optimizer):
    netG.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        pred = netG(batch).view(-1,)
        optimizer.zero_grad()
        loss = reg_criterion(pred, batch.y)
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)
def train_gan(netG,netD,device, loader, optimizerG,optimizerD):
    netG.train()
    netD.train()
    G_losses = []
    D_losses = []
    real_label=1
    fake_label=0
            ###########################
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
                ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        batch = batch.to(device)
        # Forward pass real batch through D
        import ipdb
        ipdb.set_trace()
        output = netD(batch.xyz).view(-1)
        b_size = output.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Calculate loss on all-real batch
        errD_real = D_criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()


        ## Train with all-fake batch
        # Generate batch of latent vectors
        # Generate fake image batch with G
        _,fake = netG(batch)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        label.fill_(fake_label)
        errD_fake = D_criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = D_criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()
        G_losses.append(errG.item())
        D_losses.append(errD.item())

    return sum(D_losses)/(step+1),sum(G_losses)/(step+1)
        # Output training stats
def eval(netG, device, loader, evaluator):
    netG.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = netG(batch).view(-1,)

        y_true.append(batch.y.view(pred.shape).detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict)["mae"]

def test(netG, device, loader):
    netG.eval()
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = netG(batch).view(-1,)

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
    args = parser.parse_args()

    print(args)

    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ### automatic dataloading and splitting
    dataset = PygPCQM4Mv2Dataset_xyz(root = '/remote-home/yxwang/Graph/dataset')

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
        'graph_pooling': args.graph_pooling
    }

    if args.gnn == 'gin':
        netG = GNN(gnn_type = 'gin', virtual_node = False, **shared_params).to(device)
    elif args.gnn == 'gin-virtual':
        netG = GNN(gnn_type = 'gin', virtual_node = True, **shared_params).to(device)
    elif args.gnn == 'gcn':
        netG = GNN(gnn_type = 'gcn', virtual_node = False, **shared_params).to(device)
    elif args.gnn == 'gcn-virtual':
        netG = GNN(gnn_type = 'gcn', virtual_node = True, **shared_params).to(device)
    else:
        raise ValueError('Invalid GNN type')
    netD=Discriminatorfor3D().to(device)

    num_params = sum(p.numel() for p in netG.parameters())
    print(f'#Params: {num_params}')

    optimizerG = optim.Adam(netG.parameters(), lr=0.001)
    optimizerD = optim.Adam(netD.parameters(), lr=0.001)
    if args.log_dir != '':
        writer = SummaryWriter(log_dir=args.log_dir)

    best_G_loss = 1000
    best_D_loss = 1000
    if args.train_subset:
        schedulerG = StepLR(optimizerG, step_size=300, gamma=0.25)
        schedulerD = StepLR(optimizerD, step_size=300, gamma=0.25)
        args.epochs = 1000
    else:
        schedulerG = StepLR(optimizerG, step_size=30, gamma=0.25)
        schedulerD = StepLR(optimizerD, step_size=30, gamma=0.25)

    best_epoch=1000
    

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        
        D_loss, G_loss = train_gan(netG,netD, device, train_loader, optimizerG,optimizerD)

        # print('Evaluating...')
        # valid_mae = eval(netG, device, valid_loader, evaluator)

        print({'D_loss': D_loss, 'G_loss': G_loss})

        if args.log_dir != '':
            writer.add_scalar('D_loss', D_loss, epoch)
            writer.add_scalar('G_loss', G_loss, epoch)

        # if G_loss < best_G_loss:
        #     best_G_loss = G_loss
        #     if args.checkpoint_dir != '':
        #         print('Saving checkpoint...')
        #         checkpoint = {'epoch': epoch, 'netG_state_dict': netG.state_dict(), 'optimizerG_state_dict': optimizerG.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_G_loss': best_G_loss, 'num_params': num_params}
        #         torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'checkpoint.pt'))

        if G_loss < best_G_loss:
            best_G_loss = G_loss
        if D_loss < best_D_loss:
            best_D_loss = D_loss
        if args.checkpoint_dir != '':
            print('Saving checkpoint...')
            checkpoint = {'epoch': epoch, 'netG_state_dict': netG.state_dict(), 'optimizerG_state_dict': optimizerG.state_dict(), 'schedulerG_state_dict': schedulerG.state_dict(), 'best_G_loss': best_G_loss, 'num_params': num_params}
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'checkpointG{epoch}.pt'))
            checkpoint = {'epoch': epoch, 'netD_state_dict': netD.state_dict(), 'optimizerD_state_dict': optimizerD.state_dict(), 'schedulerD_state_dict': schedulerD.state_dict(), 'best_D_loss': best_G_loss, 'num_params': num_params}
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'checkpointG{epoch}.pt'))
            # if args.save_test_dir != '':
            #     testdev_pred = test(netG, device, testdev_loader)
            #     testdev_pred = testdev_pred.cpu().detach().numpy()

            #     testchallenge_pred = test(netG, device, testchallenge_loader)
            #     testchallenge_pred = testchallenge_pred.cpu().detach().numpy()

            #     print('Saving test submission file...')
            #     evaluator.save_test_submission({'y_pred': testdev_pred}, args.save_test_dir, mode = 'test-dev')
            #     evaluator.save_test_submission({'y_pred': testchallenge_pred}, args.save_test_dir, mode = 'test-challenge')

        schedulerG.step()
        schedulerD.step()
            
        print(f'Best G_loss so far: {G_loss}, best epoch: {best_epoch}')

    if args.log_dir != '':
        writer.close()


if __name__ == "__main__":
    main()
######

        
 