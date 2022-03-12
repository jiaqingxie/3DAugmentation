import torch

from torch_geometric.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

import itertools
import os
from tqdm import tqdm
import argparse
import time
import numpy as np
import random

from cyclegan import GNN_Disciminator_CycleGAN,GNN_Generator_CycleGAN
import sys
sys.path.append("../")
### importing OGB-LSC
from ogb.lsc import PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator
from data.PCQM4Mv2_xyz import *
import wandb



reg_criterion = torch.nn.L1Loss()
D_criterion=torch.nn.BCELoss()
criterionCycle = torch.nn.L1Loss()
criterionIdt = torch.nn.L1Loss()
def criterionGAN(pred,target):
    loss_fc=torch.nn.BCELoss()
    label = torch.Tensor([target]).float().to(pred.device).expand_as(pred)
    return loss_fc(pred,label)

def backward_D(netD,real,fake,edge_index,edge_attr,batch):
    pred_real = netD(real,edge_index,edge_attr,batch)
    loss_D_real = criterionGAN(pred_real, True)
    # Fake
    pred_fake = netD(fake.detach(),edge_index,edge_attr,batch)
    loss_D_fake = criterionGAN(pred_fake, False)
    # Combined loss and calculate gradients
    loss_D = (loss_D_real + loss_D_fake) * 0.5
    loss_D.backward()
    return loss_D

def train_cyclegan(netG_A,netG_B,netD_A,netD_B,device, loader, optimizerG,optimizerD):
    netG_A.train()
    netD_A.train()
    netG_B.train()
    netD_B.train()
    G_losses = []
    D_losses = []
    real_label=1
    fake_label=0
    lambda_A=1
    lambda_B=1
    lambda_idt=0

            ###########################
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch.xyz_edge_attr=batch.xyz_edge_attr.long()
        batch.xyz_edge_index=batch.xyz_edge_index.long()
        batch=batch.to(device)

        #####forward######
        _,fake_xyz=netG_A(batch.x,batch.edge_index,batch.edge_attr,batch.batch)
        _,rec_x=netG_B(fake_xyz,batch.xyz_edge_index,batch.xyz_edge_attr,batch.batch)
        _,fake_x=netG_B(batch.xyz,batch.edge_index,batch.edge_attr,batch.batch)
        _,rec_xyz=netG_A(fake_x,batch.xyz_edge_index,batch.xyz_edge_attr,batch.batch)
        ##################Ds don't require grad at this time
        for net in [netD_A,netD_B]:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad=False
        #############
        optimizerG.zero_grad()
        """Calculate the loss for generators G_A and G_B"""
        ###identity loss######
        if lambda_idt > 0:
            # # G_A should be identity if real_B is fed: ||G_A(B) - B||
            # idt_A = netG_A(real_B)
            # loss_idt_A = criterionIdt(idt_A, real_B) * lambda_B * lambda_idt
            # # G_B should be identity if real_A is fed: ||G_B(A) - A||
            # idt_B = netG_B(real_A)
            # loss_idt_B = criterionIdt(idt_B, real_A) * lambda_A * lambda_idt
            pass
        else:
            loss_idt_A = 0
            loss_idt_B = 0
        #########
        # GAN loss D_A(G_A(A))
        loss_G_A = criterionGAN(netD_A(fake_xyz,batch.xyz_edge_index,batch.xyz_edge_attr,batch.batch), 1.0)
        # GAN loss D_B(G_B(B))
        loss_G_B = criterionGAN(netD_B(fake_x,batch.xyz_edge_index,batch.xyz_edge_attr,batch.batch), 1.0)
        # Forward cycle loss || G_B(G_A(A)) - A||
        loss_cycle_A = criterionCycle(rec_x, netG_A.gnn_node.atom_encoder(batch.x)) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        loss_cycle_B = criterionCycle(rec_xyz, batch.xyz) * lambda_B
        # combined loss and calculate gradients
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        loss_G.backward()
        G_losses.append(loss_G)
        optimizerG.step()
        ###########D_A and D_B##########
                ##################Ds require grad at this time
        for net in [netD_A,netD_B]:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad=True
        #############
        optimizerD.zero_grad()
        D_A_loss=backward_D(netD_A,batch.xyz,fake_xyz,batch.xyz_edge_index,batch.xyz_edge_attr,batch.batch)
        D_B_loss=backward_D(netD_B,batch.x,fake_x,batch.edge_index,batch.edge_attr,batch.batch)
        D_loss=D_A_loss+D_B_loss
        D_losses.append(D_loss)
        optimizerD.step()

        ##########


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
    parser.add_argument('--D_lr', type=float, default=0.001,
                        help='lr for disciminator)')
    parser.add_argument('--G_lr', type=float, default=0.001,
                        help='lr for generator)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--log_dir', type=str, default="",
                        help='tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default = '', help='directory to save checkpoint')
    parser.add_argument('--save_test_dir', type=str, default = '', help='directory to save test submission file')
    args = parser.parse_args()
    wandb.init(project="3DInjection", entity="yxwang123",name="cycleGAN"+str(args),config={
  "G_learning_rate": args.G_lr,
  "D_learning_rate": args.D_lr,
  "epochs": args.epochs,
  "batch_size": args.batch_size,
  "log_dir":args.log_dir,
  "checkpoint_dir":args.checkpoint_dir,
  "save_test_dir":args.save_test_dir,
})
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
    # import ipdb
    # ipdb.set_trace()
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
        netG_A = GNN_Generator_CycleGAN(gnn_type = 'gin', virtual_node = False, **shared_params).to(device)
        netG_B = GNN_Generator_CycleGAN(gnn_type = 'gin', virtual_node = False, **shared_params).to(device)
        netD_A=GNN_Disciminator_CycleGAN(gnn_type = 'gin', virtual_node = False, **shared_params).to(device)
        netD_B=GNN_Disciminator_CycleGAN(gnn_type = 'gin', virtual_node = False, **shared_params).to(device)
    elif args.gnn == 'gin-virtual':
        netG = GNN_Generator_CycleGAN(gnn_type = 'gin', virtual_node = True, **shared_params).to(device)
    elif args.gnn == 'gcn':
        netG = GNN_Generator_CycleGAN(gnn_type = 'gcn', virtual_node = False, **shared_params).to(device)
    elif args.gnn == 'gcn-virtual':
        netG = GNN_Generator_CycleGAN(gnn_type = 'gcn', virtual_node = True, **shared_params).to(device)
    else:
        raise ValueError('Invalid GNN type')


    num_params = sum(p.numel() for p in netG_A.parameters())
    print(f'#Params: {num_params}')

    optimizerG = optim.Adam(itertools.chain(netG_A.parameters(),netG_B.parameters()), lr=args.G_lr)
    optimizerD = optim.Adam(itertools.chain(netD_A.parameters(),netD_B.parameters()), lr=args.D_lr)
    if args.log_dir != '':
        writer = SummaryWriter(log_dir=args.log_dir)

    best_G_loss = 1000
    best_D_loss = 1000
    if args.train_subset:
        schedulerG = StepLR(optimizerG, step_size=300, gamma=0.25)
        schedulerD = StepLR(optimizerD, step_size=300, gamma=0.25)
        args.epochs = 1000
    else:
        schedulerG = StepLR(optimizerG, step_size=1000, gamma=0.25)
        schedulerD = StepLR(optimizerD, step_size=1000, gamma=0.25)

    best_epoch=1000
    

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        
        D_loss, G_loss = train_cyclegan(netG_A,netG_B,netD_A,netD_B, device, train_loader, optimizerG,optimizerD)

        # print('Evaluating...')
        # valid_mae = eval(netG, device, valid_loader, evaluator)

        print({'D_loss': D_loss, 'G_loss': G_loss})

        wandb.log({'D_loss': D_loss, 'G_loss': G_loss})
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
            best_epoch=epoch
        if D_loss < best_D_loss:
            best_D_loss = D_loss
        if args.checkpoint_dir != '':
            print('Saving checkpoint...')
            checkpoint = {'epoch': epoch, 'netG_state_dict': netG_A.state_dict(), 'optimizerG_state_dict': optimizerG.state_dict(), 'schedulerG_state_dict': schedulerG.state_dict(), 'best_G_loss': best_G_loss, 'num_params': num_params}
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'checkpointG_A{epoch}.pt'))
            checkpoint = {'epoch': epoch, 'netD_state_dict': netD_A.state_dict(), 'optimizerD_state_dict': optimizerD.state_dict(), 'schedulerD_state_dict': schedulerD.state_dict(), 'best_D_loss': best_G_loss, 'num_params': num_params}
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'checkpointD_A{epoch}.pt'))
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
            
        print(f'Best G_loss so far: {best_G_loss}, best epoch: {best_epoch}')

    if args.log_dir != '':
        writer.close()


if __name__ == "__main__":
    main()
######

        
 