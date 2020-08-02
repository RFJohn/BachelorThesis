import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger, VisdomSaver
from tqdm import tqdm

from ShiftDataLoader import ShiftDataLoader, ShiftXYDataLoader
from model import Net, NetXY
from psnrmeter import PSNRMeter


def processor(sample):
    data, target, training = sample
    data = Variable(data)
    target = Variable(target)
    if torch.cuda.is_available():
        data = data.cuda()
        target = target.cuda()

    output = model(data)
    loss = criterion(output, target)

    return loss, output


def on_sample(state):
    state['sample'].append(state['train'])


def reset_meters():
    meter_psnr.reset()
    meter_loss.reset()


def on_forward(state):
    meter_psnr.add(state['output'].data, state['sample'][1])
    meter_loss.add(state['loss'].item())


def on_start_epoch(state):
    reset_meters()
    scheduler.step()
    state['iterator'] = tqdm(state['iterator'])


def on_end_epoch(state):
    print('[%s][Epoch %d] Train Loss: %.4f (PSNR: %.2f db)' % (
        current_mode, state['epoch'], meter_loss.value()[0], meter_psnr.value()))

    train_loss_logger.log(state['epoch'], meter_loss.value()[0])
    train_psnr_logger.log(state['epoch'], meter_psnr.value())

    reset_meters()

    engine.test(processor, val_loader)
    val_loss_logger.log(state['epoch'], meter_loss.value()[0])
    val_psnr_logger.log(state['epoch'], meter_psnr.value())

    print('[%s][Epoch %d] Val Loss: %.4f (PSNR: %.2f db)' % (
        current_mode, state['epoch'], meter_loss.value()[0], meter_psnr.value()))

    torch.save(model.state_dict(), f"epochs/epoch_{UPSCALE_FACTOR}_{current_mode}_{state['epoch']}.pt")


def data_loader(xy_mode):
    if xy_mode:
        train_set = ShiftXYDataLoader("train-data", current_mode, UPSCALE_FACTOR)
        val_set = ShiftXYDataLoader("val-data", current_mode, UPSCALE_FACTOR)
    else:
        train_set = ShiftDataLoader("train-data", current_mode, UPSCALE_FACTOR)
        val_set = ShiftDataLoader("val-data", current_mode, UPSCALE_FACTOR)

    _train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)
    _val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=64, shuffle=False)
    return _train_loader, _val_loader


def setup_train(xy_mode=True):
    global train_loader, val_loader, model, criterion, optimizer, scheduler, meter_loss, meter_psnr

    train_loader, val_loader = data_loader(xy_mode)

    model = NetXY(UPSCALE_FACTOR) if xy_mode else Net(UPSCALE_FACTOR)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    print('# parameters:', sum(param.numel() for param in model.parameters()))

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    meter_loss = tnt.meter.AverageValueMeter()
    meter_psnr = PSNRMeter()


def setup_logger():
    global train_loss_logger, train_psnr_logger, val_loss_logger, val_psnr_logger
    train_loss_logger = VisdomPlotLogger('line', opts={'title': f"{current_mode} Train Loss"})
    train_psnr_logger = VisdomPlotLogger('line', opts={'title': f"{current_mode} Train PSNR"})
    val_loss_logger = VisdomPlotLogger('line', opts={'title': f"{current_mode} Val Loss"})
    val_psnr_logger = VisdomPlotLogger('line', opts={'title': f"{current_mode} Val PSNR"})


def train_engine():
    global engine
    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.train(processor, train_loader, maxepoch=NUM_EPOCHS, optimizer=optimizer)


def train(mode):
    global current_mode
    current_mode = mode
    setup_train()
    setup_logger()
    train_engine()
    VisdomSaver(envs=["main"]).save()


if __name__ == "__main__":

    # Defines global variables here. Easier Workaround to not change code too much from original github repo.
    train_loader = val_loader = model = criterion = optimizer = scheduler = None
    meter_loss = meter_psnr = engine = None
    train_loss_logger = train_psnr_logger = val_loss_logger = val_psnr_logger = None

    # Command line inputs from original github repo
    parser = argparse.ArgumentParser(description='Train Super Resolution')
    parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
    parser.add_argument('--num_epochs', default=1, type=int, help='super resolution epochs number')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs

    # train different modes
    current_mode = None
    train("control")
    train("half")
    train("quarter")
    train("random0")
