from __future__ import print_function

import argparse
import enum
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.auto import tqdm

from dataloders import load_neuro_face
from models.resnet_big import SupCCEResNet
from utils import (
    AverageMeter,
    EarlyStopping,
    accuracy,
    adjust_learning_rate,
    make_scheduler,
    save_model,
    set_optimizer,
    warmup_learning_rate,
)


def parse_option():
    parser = argparse.ArgumentParser("argument for training")

    parser.add_argument("--print_freq", type=int, default=10, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=50, help="save frequency")
    parser.add_argument("--batch_size", type=int, default=512, help="batch_size")
    parser.add_argument("--num_workers", type=int, default=16, help="num of workers to use")
    parser.add_argument("--epochs", type=int, default=100, help="number of training epochs")

    # optimization
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
    parser.add_argument("--momentum", type=float, default=0.9, help="momentum")
    parser.add_argument("--optim_type", type=str, default="Adam", help="optimizer")

    # model dataset
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument(
        "--dataset",
        type=str,
        default="neuro_face",
        choices=["neuro_face"],
        help="dataset",
    )

    parser.add_argument(
        "--loss_type",
        type=str,
        default="CCE",
        help="loss type for traning CCE",
    )

    # other setting
    parser.add_argument("--cosine", action="store_true", help="using cosine annealing")
    parser.add_argument(
        "--syncBN", action="store_true", help="using synchronized batch normalization"
    )
    parser.add_argument("--warm", action="store_true", help="warm-up for large batch training")
    parser.add_argument("--trial", type=str, default="0", help="id for recording multiple runs")
    parser.add_argument(
        "--decay_type",
        default="step",
        choices=("step", "step_warmup", "cosine_warmup"),
        help="optimizer to use (step | step_warmup | cosine_warmup)",
    )

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = "./datasets/"
    opt.model_path = "./save/model_weights/supervised/{}_models".format(opt.dataset)
    opt.csv_path = "./save/csv/supervised/{}_models".format(opt.dataset)

    opt.model_name = "{}_{}_{}_{}_lr_{}_proj_{}_radius_{}".format(
        opt.dataset,
        opt.model,
        opt.loss_type,
        opt.optim_type,
        opt.learning_rate,
    )

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    opt.csv_folder = opt.csv_path
    if not os.path.isdir(opt.csv_folder):
        os.makedirs(opt.csv_folder)

    if opt.dataset == "neuro_face":
        opt.n_cls = 3
    else:
        raise ValueError("dataset not supported: {}".format(opt.dataset))

    return opt


def set_model(opt):
    model = SupCCEResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()
    model = model.to(opt.device)
    criterion = criterion.to(opt.device)
    cudnn.benchmark = True
    return model, criterion


def set_loader(opt):
    if opt.dataset == "neuro_face":
        train_loader, val_loader = load_neuro_face(opt)

    return train_loader, val_loader

def top1_accuracy(
    output,
    labels,
):
    _, preds = torch.max(output, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


@torch.no_grad()
def evaluate(opt, model, val_loader, criterion):
    model.eval()
    temp_loss = AverageMeter()
    temp_acc = AverageMeter()
    for batch in val_loader:
        images, labels = batch
        images = images.to(opt.device, non_blocking=True)
        labels = labels.to(opt.device, non_blocking=True)
        bsz = labels.shape[0]
        out = model(images)
        loss = criterion(out, labels)
        temp_loss.update(round(loss.item(), 5), bsz)
        top1 = top1_accuracy(out, labels)
        temp_acc.update(round(top1.item(), 5), bsz)

    return temp_loss.avg, temp_acc.avg


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


# This is Trning Loop for class OurModelConvAngularPen and class ConvAngularPen
def fit(opt, train_dl, test_dl, model, criterion, optimizer, grad_clip=0.1):

    history = []

    # Decay LR by a factor of 0.1 every 7 epochs
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # scheduler = make_scheduler(opt, optimizer, train_dl)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=0.1, patience=4, verbose=True
    # )

    # early_stopping = EarlyStopping(patience=7, verbose=False)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=0.1, patience=6, verbose=True
    # )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[30, 90],
        gamma=0.1,
    )

    model.train()
    for epoch in range(opt.epochs):
        # Training Phase
        batch_time = AverageMeter()
        data_time = AverageMeter()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        result = {}

        end = time.time()
        for idx, (images, labels) in enumerate(train_dl):
            data_time.update(time.time() - end)
            images = images.to(opt.device, non_blocking=True)
            labels = labels.to(opt.device, non_blocking=True)
            bsz = labels.shape[0]
            out = model(images)
            loss = criterion(out, labels)
            acc = top1_accuracy(out, labels)
            train_acc.update(round(acc.item(), 5), bsz)
            train_loss.update(round(loss.item(), 5), bsz)

            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if (idx + 1) % opt.print_freq == 0:
                print(
                    "Train: [{0}][{1}/{2}]\t"
                    "BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "DT {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "acc {acc.val:.3f} ({acc.avg:.3f})\t"
                    "loss {loss.val:.3f} ({loss.avg:.3f})".format(
                        epoch,
                        idx + 1,
                        len(train_dl),
                        batch_time=batch_time,
                        data_time=data_time,
                        acc=train_acc,
                        loss=train_loss,
                    )
                )
                sys.stdout.flush()

        # evalution pahase
        val_loss, val_acc = evaluate(
            opt,
            model,
            test_dl,
            criterion,
        )

        lrs = optimizer.param_groups[0]["lr"]

        scheduler.step()
        # scheduler.step(round(val_loss, 2))

        # tensor_board_logger(logger, epoch, train_loss.avg, train_acc.avg, val_loss, val_acc, lr)

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, "ckpt_epoch_{epoch}.pth".format(epoch=epoch)
            )
            save_model(model, optimizer, opt, epoch, save_file)

        result["train_loss"] = round(train_loss.avg, 5)
        result["train_acc"] = round(train_acc.avg, 5)
        result["val_loss"] = round(val_loss, 5)
        result["val_acc"] = round(val_acc, 5)
        result["lrs"] = lrs
        print("Epoch", epoch, result)
        history.append(result)

        # early_stopping(val_loss, model)

        # if early_stopping.early_stop:
        #     print("Early stopping")
        #     break

    return history


def main():
    opt = parse_option()

    # build data loader
    train_loader, val_loader = set_loader(opt)

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    history_data = fit(opt, train_loader, val_loader, model, criterion, optimizer)

    df = pd.DataFrame.from_dict(history_data)
    print("Saveing history data to csv file...")
    csv_file_name = "{}_{}_{}_{}_lr_{}_proj_{}_radius_{}.csv".format(
        opt.dataset,
        opt.model,
        opt.loss_type,
        opt.optim_type,
        opt.learning_rate,
    )
    save_csv_file_path = os.path.join(opt.csv_folder, csv_file_name)

    df.to_csv(save_csv_file_path, index=False)

    # save the last model
    save_file = os.path.join(opt.save_folder, "last.pth")
    save_model(model, optimizer, opt, opt.epochs, save_file)


if __name__ == "__main__":
    main()
