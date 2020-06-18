import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchnet.engine import Engine
from torchnet.logger import VisdomPlotLogger
from torchnet.logger import VisdomSaver
from tqdm import tqdm

from ShiftDataLoader import ShiftDataLoader
from model import Net
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


def train_data_loader(data_folder, upscale_factor):
    train_set = ShiftDataLoader("train-data", data_folder, upscale_factor)
    return DataLoader(dataset=train_set, num_workers=4, batch_size=64, shuffle=True)


def val_data_loader(data_folder, upscale_factor):
    val_set = ShiftDataLoader("val-data", data_folder, upscale_factor)
    return DataLoader(dataset=val_set, num_workers=4, batch_size=64, shuffle=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train Super Resolution')
    parser.add_argument('--upscale_factor', default=4, type=int, help='super resolution upscale factor')
    parser.add_argument('--num_epochs', default=100, type=int, help='super resolution epochs number')
    opt = parser.parse_args()

    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs

    # mode 1
    current_mode = "control"

    train_loader = train_data_loader(current_mode, UPSCALE_FACTOR)
    val_loader = val_data_loader(current_mode, UPSCALE_FACTOR)

    model = Net(upscale_factor=UPSCALE_FACTOR)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    print('# parameters:', sum(param.numel() for param in model.parameters()))

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_psnr = PSNRMeter()

    train_loss_logger = VisdomPlotLogger('line', opts={'title': f"{current_mode} Train Loss"})
    train_psnr_logger = VisdomPlotLogger('line', opts={'title': f"{current_mode} Train PSNR"})
    val_loss_logger = VisdomPlotLogger('line', opts={'title': f"{current_mode} Val Loss"})
    val_psnr_logger = VisdomPlotLogger('line', opts={'title': f"{current_mode} Val PSNR"})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, train_loader, maxepoch=NUM_EPOCHS, optimizer=optimizer)

    VisdomSaver(envs=["main"]).save()

    # mode 2
    current_mode = "half"

    train_loader = train_data_loader(current_mode, UPSCALE_FACTOR)
    val_loader = val_data_loader(current_mode, UPSCALE_FACTOR)

    model = Net(upscale_factor=UPSCALE_FACTOR)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    print('# parameters:', sum(param.numel() for param in model.parameters()))

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_psnr = PSNRMeter()

    train_loss_logger = VisdomPlotLogger('line', opts={'title': f"{current_mode} Train Loss"})
    train_psnr_logger = VisdomPlotLogger('line', opts={'title': f"{current_mode} Train PSNR"})
    val_loss_logger = VisdomPlotLogger('line', opts={'title': f"{current_mode} Val Loss"})
    val_psnr_logger = VisdomPlotLogger('line', opts={'title': f"{current_mode} Val PSNR"})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, train_loader, maxepoch=NUM_EPOCHS, optimizer=optimizer)

    VisdomSaver(envs=["main"]).save()

    # mode 3
    current_mode = "quarter"

    train_loader = train_data_loader(current_mode, UPSCALE_FACTOR)
    val_loader = val_data_loader(current_mode, UPSCALE_FACTOR)

    model = Net(upscale_factor=UPSCALE_FACTOR)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    print('# parameters:', sum(param.numel() for param in model.parameters()))

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_psnr = PSNRMeter()

    train_loss_logger = VisdomPlotLogger('line', opts={'title': f"{current_mode} Train Loss"})
    train_psnr_logger = VisdomPlotLogger('line', opts={'title': f"{current_mode} Train PSNR"})
    val_loss_logger = VisdomPlotLogger('line', opts={'title': f"{current_mode} Val Loss"})
    val_psnr_logger = VisdomPlotLogger('line', opts={'title': f"{current_mode} Val PSNR"})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, train_loader, maxepoch=NUM_EPOCHS, optimizer=optimizer)

    VisdomSaver(envs=["main"]).save()

    # mode 4
    current_mode = "random0"

    train_loader = train_data_loader(current_mode, UPSCALE_FACTOR)
    val_loader = val_data_loader(current_mode, UPSCALE_FACTOR)

    model = Net(upscale_factor=UPSCALE_FACTOR)
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    print('# parameters:', sum(param.numel() for param in model.parameters()))

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)

    engine = Engine()
    meter_loss = tnt.meter.AverageValueMeter()
    meter_psnr = PSNRMeter()

    train_loss_logger = VisdomPlotLogger('line', opts={'title': f"{current_mode} Train Loss"})
    train_psnr_logger = VisdomPlotLogger('line', opts={'title': f"{current_mode} Train PSNR"})
    val_loss_logger = VisdomPlotLogger('line', opts={'title': f"{current_mode} Val Loss"})
    val_psnr_logger = VisdomPlotLogger('line', opts={'title': f"{current_mode} Val PSNR"})

    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch

    engine.train(processor, train_loader, maxepoch=NUM_EPOCHS, optimizer=optimizer)

    VisdomSaver(envs=["main"]).save()
