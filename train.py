from opts import get_opts
import numpy as np
from parse_config import ConfigParser

from models.losses import loss_dict
from utils.util import *
from torch.utils.data import DataLoader
from utils.nerfsystem import NeRFSystem
import models.metrics as module_metric
from trainer.trainer import Trainer

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config, args):
    logger = config.get_logger('train')

    train_dataset, val_dataset = prepare_data(args)

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=True,
                                  num_workers=4,
                                  batch_size=args.batch_size,
                                  pin_memory=True)

    val_dataloader = DataLoader(val_dataset,
                                shuffle=False,
                                num_workers=4,
                                batch_size=1,
                                pin_memory=True)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])

    # build model architecture, then print to console
    system = NeRFSystem(args, device)
    model = system.models
    logger.info(model)

    if len(device_ids) > 1:

        models = [torch.nn.DataParallel(model[0], device_ids=device_ids)]
        models += [torch.nn.DataParallel(model[1], device_ids=device_ids)]
    else:
        models = model

    # get function handles of loss and metrics
    metrics = [getattr(module_metric, met) for met in config['metrics']]
    loss = loss_dict[args.loss_type]()

    # # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    optimizer = get_optimizer(args, models)
    lr_scheduler = get_scheduler(args, optimizer)


    trainer = Trainer(models, loss, metrics, optimizer, system,
                      config=config,
                      device=device,
                      data_loader=train_dataloader,
                      valid_data_loader=val_dataloader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':

    arg = get_opts()
    parser = get_opts()
    args = parser.parse_args()

    config = ConfigParser.from_args(arg)
    main(config,args)
