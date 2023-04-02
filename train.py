import argparse
from tqdm import trange
from pathlib import Path

import monai
import torch
import torchio as tio
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler

from data import MedicalDecathlonDataModule

def get_logdir(args):
    ''' Get path to logging directory '''
    if args.checkpoint is not None:
        return args.checkpoint.parent
    
    runs_dir = Path('runs')
    if not runs_dir.is_dir():
        runs_dir.mkdir()
    
    previous_runs = runs_dir.iterdir()
    previous_max = max((int(str(s).rsplit('_', maxsplit=1)[-1]) for s in previous_runs), default=0)
    new_run_number = previous_max + 1
    return runs_dir.joinpath(f'hippocampus_{new_run_number:02d}')

def get_writer(path):
    ''' Get log writer '''
    writer = SummaryWriter(str(path))
    return writer

def get_criterion():
    ''' Currently always returns the Dice+CE loss'''
    return monai.losses.DiceCELoss(softmax=True)

def get_net():
    ''' Initializes an untrained UNet '''
    net = monai.networks.nets.UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(8, 16, 32, 64),
        strides=(2, 2, 2)
    )
    return net

def get_optimizer(net, args):
    ''' Currently returns an AdamW optimizer over all trainable model parameters '''
    return torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr)


def train_epoch(dataloader, net, criterion, optimizer,
                device, writer, current_step,
                mixed_precision_context, scaler):
    ''' Train the net for one epoch and log the loss '''
    net.train()

    running_loss = 0
    log_every = 13 # wait this many batches to log, set low because we have a small dataset
    for i, batch in enumerate(dataloader):
        image = batch['image'][tio.DATA]
        label = batch['label'][tio.DATA]

        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        with mixed_precision_context:
            pred = net(image)
            loss = criterion(pred, label)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        if i % log_every == (log_every - 1):
            writer.add_scalar('loss/Training', (running_loss / log_every), (current_step+i))
            running_loss = 0
    net.eval()
    return


def validate(dataloader, net, criterion, device, writer, current_step):
    ''' Log the validation loss '''
    net.eval()
    # for validation we simply log once per epoch
    # note that validation loader does not have drop_last, so we need to weight
    # the last batch appropriately
    running_loss = 0
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            image = batch['image'][tio.DATA]
            label = batch['label'][tio.DATA]

            image = image.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            pred = net(image)
            loss = criterion(pred, label)

            batch_size = len(label)
            running_loss += loss.item() * batch_size
            count += batch_size
    val_loss = running_loss / count
    writer.add_scalar('loss/Validation', val_loss, current_step)
    return val_loss

def finetune_mode(net):
    ''' Freeze all model layers except the last layer. '''
    # freeze all
    for parameter in net.parameters():
        parameter.requires_grad = False
    # unfreeze last layer
    # NOTE: if we switch to a different model, make sure that the last named module
    # is really the last layer
    _, last_layer = list(net.named_modules())[-1]
    for parameter in last_layer.parameters():
        parameter.requires_grad = True
    assert any(p.requires_grad for p in net.parameters()), "No trainable parameters left"
    return


def run(args):
    ''' Main function to train a hippocampus segmentation model '''

    monai.utils.set_determinism()
    device = torch.device("cuda" if args.use_gpu else "cpu")

    # init dataloaders
    data = MedicalDecathlonDataModule(task='Task04_Hippocampus', 
                                    google_id='1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C',
                                    batch_size=args.batch_size,
                                    train_val_ratio=0.8)
    print(f"{len(data.train_set)=},\t{len(data.val_set)=}")
    train_dl = data.train_dataloader(args.steps_per_epoch, num_workers=args.workers)
    val_dl = data.val_dataloader(num_workers=args.workers)

    # init model, criterion, optimizer
    net = get_net()
    net.to(device)
    criterion = get_criterion()
    optimizer = get_optimizer(net, args)

    # logging stuff
    logdir = get_logdir(args)
    writer = get_writer(logdir)
    batches_per_epoch = len(train_dl)
    current_step = 0
    starting_epoch = 0

    # automatic mixed precision: autocast and scaler
    mp_context = torch.autocast(device_type=str(device), dtype=torch.float16, enabled=args.mixed_precision)
    scaler = GradScaler(enabled=args.mixed_precision)

    # load stuff from model checkpoint
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_step = checkpoint['step']
        starting_epoch = checkpoint['epoch'] + 1
        # only load scaler if it was used before & we want to use it again
        if 'scaler' in checkpoint and args.mixed_precision is True:
            scaler.load_state_dict(checkpoint['scaler'])

    # finetuning
    if args.finetuning is True:
        finetune_mode(net)
    
    # start training
    pbar = trange(starting_epoch, (starting_epoch + args.epochs), colour='green', unit='epoch')
    for e in pbar:
        pbar.set_description(f"Epoch {e}")
        pbar.set_postfix(val_loss='....')

        train_epoch(train_dl, net, criterion, optimizer,
                    device, writer, current_step, mp_context, scaler)
        current_step += batches_per_epoch

        # we keep full precision for validation, since val set is small anyway
        val_loss = validate(val_dl, net, criterion, device, writer, current_step)
        writer.add_scalar('epoch', e, current_step)
        pbar.set_postfix(val_loss=val_loss)
        # print newline to keep validation losses of previous epochs in terminal
        print('')

    # save checkpoint of trained model
    checkpoint = {
        'epoch' : e,
        'step' : current_step,
        'model_state_dict' : net.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
    }
    # only save a scaler if we actually used it
    if args.mixed_precision is True:
        checkpoint['scaler'] = scaler.state_dict()
    torch.save(checkpoint, logdir.joinpath(f'model_step{current_step:05d}.ckpt'))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a hyppocampus segmentation model")
    parser.add_argument('-lr',  '--lr',         type=float, default=1e-2,
                        help='Adam learning rate.')
    
    parser.add_argument('-b',   '--batch_size', type=int,   default=16,
                        help='Batch size.')
    
    parser.add_argument('-e',   '--epochs',     type=int,   default=5,
                        help='Number of epochs to train.')
    
    parser.add_argument('-w',   '--workers',    type=int,   default=2,
                        help='Number of dataloader cpus.')
    
    parser.add_argument('-c',   '--checkpoint', type=Path,  default=None,
                        help='Path to model checkpoint, default=None to train from scratch.')
    
    parser.add_argument('-g',   '--use_gpu', action=argparse.BooleanOptionalAction, default=torch.cuda.is_available(),
                        help='Whether to use gpu. By default, uses gpu when available.')
    
    parser.add_argument('-m',   '--mixed_precision', action=argparse.BooleanOptionalAction, default=True,
                        help='Whether to use (float16) mixed precision for training. Default is True.')
    
    parser.add_argument('-s',   '--steps_per_epoch', type=int, default=None,
                        help='Number of batches/steps per epoch. Default=None to use the number of batches in the train set.')
    
    parser.add_argument('-ft',  '--finetuning', action='store_true',
                        help='Whether to only finetune last layer of model. Default=False.')
    
    args = parser.parse_args()
    print(f"{args=}")
    run(args)