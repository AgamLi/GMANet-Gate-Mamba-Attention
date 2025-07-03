import argparse
import random
import os
import yaml

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss, NLLLoss

import numpy as np
from pathlib import Path
from tqdm import tqdm

from utils import DiceLoss,MyDC,DCloss
from mydataset import JointTransform2D, Fetal_dataset
from my_model.Model import Model



def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()

    # basic parameters
    parser.add_argument('--name', default=None, help='model name')
    parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='number of data loading workers')
    parser.add_argument('--seed', default=42, type=int, help='random seed')
    

    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='gmanet')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int, help='input channels')
    parser.add_argument('--num_classes', default=3, type=int, help='number of classes')
    parser.add_argument('--input_w', default=256, type=int, help='image width')
    parser.add_argument('--input_h', default=256, type=int, help='image height')
    parser.add_argument('--pretrained', default=None, type=str, help='path to pre-trained model (default: none)')
    
    
    # dataset
    parser.add_argument('--dataset', default='psfh', help='dataset name')
    parser.add_argument('--data_dir', default='./../dataset', help='path to dataset')
    parser.add_argument('--output_dir', default='./output', help='path to output')


    # optimizer
    parser.add_argument('--optimizer', default='Adam',choices=['Adam', 'SGD'],
                        help='loss: ' + ' | '.join(['Adam', 'SGD']) + ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool, help='nesterov')


    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-6, type=float, help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int, metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )

    
    config = parser.parse_args()

    return config


def seed_torch(seed_value=42):
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True  # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True    
    # torch.use_deterministic_algorithms(True)
    print(f"Random seed set as {seed_value}")

import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    # parse arguments
    config = vars(parse_args())
    
    # empty the cache
    torch.cuda.empty_cache()


    # set random seed
    seed_torch(config['seed'])


    # create model directory
    if config['name'] is None:
        config['name'] = '%s_%s' % (config['arch'], config['dataset'])
    
    os.makedirs('config/%s' % config['name'], exist_ok=True)

    print('-' * 30)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 30)

    with open('config/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)




    # create model
    model = Model(num_classes=config['num_classes']).cuda()


    # make checkpoint directory
    os.makedirs('./checkpoints', exist_ok=True)


    # load pre-trained model
    if config['pretrained'] is not None:
        if os.path.isfile(config['pretrained']):
            print("=> loading pre-trained model '{}'".format(config['pretrained']))
            model.load_state_dict(torch.load(config['pretrained']), strict=False)
        else:
            print("=> no pre-trained model found at '{}'".format(config['pretrained']))
    

    # define optimizer and scheduler
    params = filter(lambda p: p.requires_grad, model.parameters())
    
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError


    # Data loading code
    print("Setting up data...")
    root_path = Path(config['data_dir'])
    image_files = np.array([(root_path / Path("image_mha") / Path(str(i).zfill(5) + '.mha')) for i in range(1, 5102)])
    label_files = np.array([(root_path / Path("label_mha") / Path(str(i).zfill(5) + '.mha')) for i in range(1, 5102)])
    with open(os.path.join(config['data_dir'], 'train.txt'), "r") as file:
            lines = file.readlines()
            train_index = [int(line.strip().split("/")[-1]) - 1 for line in lines]
    with open(os.path.join(config['data_dir'], 'val.txt'), "r") as file:
            lines = file.readlines()
            test_index = [int(line.strip().split("/")[-1]) - 1 for line in lines]
    print('the number of train images:', len(train_index), 'the number of val images:', len(test_index))


    # transform for data augmentation
    tf_train = JointTransform2D(img_size=256, low_img_size=128, ori_size=256, crop=None, p_flip=0.5, p_rota=0.5,
                                p_scale=0.0, p_gaussn=0.0,
                                p_contr=0.0, p_gama=0.0, p_distor=0.0, color_jitter_params=None,
                                long_mask=True)  # image reprocessing
    tf_val = JointTransform2D(img_size=256, low_img_size=128, ori_size=256, crop=None, p_flip=0.0,
                                color_jitter_params=None, long_mask=True)
    
    
    # create dataset and dataloader
    db_train = Fetal_dataset(transform=tf_train, list_dir=(image_files[np.array(train_index)], label_files[np.array(train_index)]))
    db_val = Fetal_dataset(transform=tf_val, list_dir=(image_files[np.array(test_index)], label_files[np.array(test_index)]))
    trainloader = DataLoader(db_train, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    valloader = DataLoader(db_val, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)
    
    
    # define loss function (criterion)
    dice_loss_func = DiceLoss(config['num_classes'])
    ce_loss_func = CrossEntropyLoss()
    
    muti_dice_loss_func = DCloss()


    # train function
    print("start training...")
    best = 1
    start_epoch = 0
    max_epoch = config['epochs']
    max_iterations = max_epoch * len(trainloader)
    trigger = 0

    
    
    for epoch in range(start_epoch + 1, max_epoch + 1):
        # train for one epoch
        model.train()
        total_batches = len(trainloader)
        pbar = tqdm(enumerate(trainloader), total=total_batches, ncols=120, desc='Training Progress', 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed}, remaining: {remaining}]')
        for i_batch, sampled_batch in pbar:
            batch_images, batch_labels = sampled_batch['image'], sampled_batch['label']
            batch_images, batch_labels = batch_images.cuda(), batch_labels.cuda().squeeze(dim=1)
            out = model(batch_images)
            
            ce_loss = ce_loss_func(out, batch_labels.long())
            dice_loss = muti_dice_loss_func(out, batch_labels.long())
            loss = 0.5 * ce_loss + dice_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch[{epoch}|{max_epoch}] Loss:{loss.detach().cpu().numpy():.4f} lr:{scheduler.get_last_lr()[0]:.6f}")

        # update learning rate
        scheduler.step()

        # evaluate on validation set
        if epoch >= 1:
            model.eval()
            val_loss = []
            with torch.no_grad():
                total_batches = len(valloader)
                pbar = tqdm(enumerate(valloader), total=total_batches, ncols=120, desc='Validating Progress', 
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed}, remaining: {remaining}]')
                for i_batch, sampled_batch in pbar:
                # for i_batch, sampled_batch in enumerate(valloader):
                    batch_images, batch_labels = sampled_batch['image'], sampled_batch['label']
                    batch_images, batch_labels = batch_images.cuda(), batch_labels.cuda().squeeze(dim=1)
                    pred = model(batch_images)
                    pred = torch.softmax(pred, dim=1)
                    loss_dice = dice_loss_func(pred, batch_labels.long(), softmax=True)
                    val_loss.append(loss_dice.detach().cpu().numpy())
                    pbar.set_description(f"Epoch[{epoch}|{max_epoch}] VALLoss:{loss_dice.detach().cpu().numpy():.4f}")
            
                val_loss_dice_mean = np.mean(val_loss)
                if val_loss_dice_mean < best:
                    best = val_loss_dice_mean
                    print(f"Epoch[{epoch}|{max_epoch}] Val Loss: {val_loss_dice_mean:.4f} | Saving model...")
                    save_mode_path = os.path.join('./checkpoints', 'epoch_' + str(epoch).zfill(5) + '_dice_' + str(best) + '.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    trigger = 0  # reset trigger
                else:
                    print(f"Epoch[{epoch}|{max_epoch}] Val Loss: {val_loss_dice_mean:.4f} |...")
                    trigger += 1

                # early stopping
                if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
                    print("=> early stopping")
                    break            


if __name__ == '__main__':
    main()
