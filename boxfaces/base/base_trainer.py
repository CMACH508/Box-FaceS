import torch
from abc import abstractmethod
from functools import partial
from utils.util import set_seed
from importlib import import_module

from logger import TensorboardWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.trainer_utils import accumulate
import warnings
from model.inception_score import InceptionScore
import lpips


class BaseTrainer:
    """+-------
    Base class for all trainers
    """

    def __init__(self, config):
        self.config = config
        self.start_epoch = 1
        self.iteration = 0

        self.main_process = False
        if (not config['distributed']) or config['global_rank'] == 0:
            self.main_process = True
        # setup data_set instances
        data_set = import_module('data_loader.' + config['dataset'])
        self.train_dataset = config.init_obj('data_set', data_set, 'train')
        self.train_sampler = None

        # setup writer and logger
        self.writer = None

        # build model architecture
        model_arch = import_module('model.' + config['model_arch'])
        self.netG = self._set_device(config.init_obj('generator', model_arch))
        self.netD = self._set_device(config.init_obj('discriminator', model_arch))
        self.netE = self._set_device(config.init_obj('encoder', model_arch))
        # prepare pre-trained vgg for calculating lpips loss
        self.percept = self._set_device(lpips.PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=True, gpu_ids=config['global_rank']
        ))

        self.e_ema = self._set_device(config.init_obj('encoder', model_arch))
        self.g_ema = self._set_device(config.init_obj('generator', model_arch))
        self.e_ema.eval()
        self.g_ema.eval()
        accumulate(self.e_ema, self.netE, 0)
        accumulate(self.g_ema, self.netG, 0)

        # build optimizer
        d_reg_ratio = self.config['loss']['d_reg_every'] / (self.config['loss']['d_reg_every'] + 1)
        self.optimizer_G = torch.optim.Adam(
            list(self.netE.parameters()) + list(self.netG.parameters()),
            lr=self.config['optimizer_G']['lr'],
            betas=(0, 0.99),
        )
        self.optimizer_D = torch.optim.Adam(
            list(self.netD.parameters()),
            lr=self.config['optimizer_D']['lr'] * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )

        if config.resume is not None:
            self._resume_checkpoint(config.resume)

        # prepare for distributed training
        if config['distributed']:
            self.train_sampler = DistributedSampler(self.train_dataset,
                                                    num_replicas=config['world_size'],
                                                    rank=config['global_rank'])
            self.netG = self._set_ddp(self.netG)
            self.netD = self._set_ddp(self.netD)
            self.netE = self._set_ddp(self.netE)
            self.percept = self._set_ddp(self.percept)

        # prepare data loader
        worker_init_fn = partial(set_seed, base=config['seed'])
        init_kwargs = {
            'dataset': self.train_dataset,
            'batch_size': config['data_loader']['batch_size'],
            'shuffle': (self.train_sampler is None),
            'sampler': self.train_sampler,
            'num_workers': config['data_loader']['num_workers'],
            'pin_memory': True,
            'worker_init_fn': worker_init_fn
        }
        self.data_loader = DataLoader(**init_kwargs)

        self.valid_data_loader = None

        if self.main_process:
            self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
            self.writer = TensorboardWriter(config.log_dir, self.logger, config['trainer']['tensorboard'])
            self.valid_dataset = config.init_obj('val_data_set', data_set, 'val')
            init_kwargs = {
                'dataset': self.valid_dataset,
                'batch_size': config['data_loader']['batch_size'],
                'shuffle': False,
                'num_workers': config['data_loader']['num_workers'],
                'pin_memory': True
            }
            self.valid_data_loader = DataLoader(**init_kwargs)

            self.inception_score = InceptionScore(cuda=True,
                                                  batch_size=config['data_loader']['batch_size'],
                                                  resize=True)

            # print necessary information to console
            print("training images = %d" % len(self.train_dataset))
            print("val images = %d" % len(self.valid_dataset))

        cfg_trainer = config['trainer']
        self.save_period = cfg_trainer['save_period']
        self.valid_period = cfg_trainer['val_period']
        self.log_period = cfg_trainer['log_period']
        self.update_ckpt = cfg_trainer['update_ckpt']
        self.checkpoint_dir = config.save_dir
        self.epochs = cfg_trainer['epochs']

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    @abstractmethod
    def _valid_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self, ):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.epochs + 1):
            if self.config['distributed']:
                self.train_sampler.set_epoch(epoch)
            # train epoch
            self._train_epoch(epoch)
            # validation and save model
            if self.main_process:
                if epoch % self.valid_period == 0:
                    val_metrics = self._valid_epoch(epoch)
                    with open('log.txt', 'a') as f:
                        f.write('*****************EPOCH ' + str(epoch) + '*****************')
                        f.write('\n')
                        for k in val_metrics.keys():
                            f.write(str(k) + ': ' + str(val_metrics[k]))
                            f.write('\n')
                if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch, latest=False)

    def _set_device(self, args):
        if torch.cuda.is_available():
            if isinstance(args, list):
                return (item.cuda() for item in args)
            if isinstance(args, dict):
                return (item.cuda() for item in args.values())
            else:
                return args.cuda()
        return args

    def _set_ddp(self, arch):

        return DDP(arch, device_ids=[self.config['global_rank']], output_device=self.config['global_rank'],
                   broadcast_buffers=False, find_unused_parameters=True)

    def _save_checkpoint(self, epoch, latest):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param latest: if True, rename the saved checkpoint to 'checkpoint_latest.pth'
        """

        state = {
            'epoch': epoch,
            'iteration': self.iteration,
            'state_dict_D': self.netD,
            'state_dict_G': self.netG,
            'state_dict_E': self.netE,
            "e_ema": self.e_ema.state_dict(),
            "g_ema": self.g_ema.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
            'config': self.config
        }

        if isinstance(self.netG, torch.nn.DataParallel) or isinstance(self.netG, DDP):
            for k, v in state.items():
                if k.split('_')[0] == 'state':
                    state[k] = v.module.state_dict()
        else:
            for k, v in state.items():
                if k.split('_')[0] == 'state':
                    state[k] = v.state_dict()
        if latest:
            filename = str(self.checkpoint_dir / 'checkpoint-latest.pth')
        else:
            filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        strict = True
        if self.main_process:
            print("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, map_location=lambda storage, loc: self._set_device(storage))
        self.start_epoch = checkpoint['epoch'] + 1
        self.iteration = checkpoint['iteration'] + 1

        # load architecture params from checkpoint.
        if checkpoint['config']['model_arch'] != self.config['model_arch']:
            if self.main_process:
                warnings.warn(
                    "Warning: Architecture configuration given in config file is different from that of "
                    "checkpoint. This may yield an exception while state_dict is being loaded.")
            strict = False

        self.netG.load_state_dict(checkpoint['state_dict_G'], strict=strict)
        self.netD.load_state_dict(checkpoint['state_dict_D'], strict=strict)
        self.netE.load_state_dict(checkpoint['state_dict_E'], strict=strict)
        self.e_ema.load_state_dict(checkpoint['e_ema'], strict=strict)
        self.g_ema.load_state_dict(checkpoint['g_ema'], strict=strict)

        try:
            # print(checkpoint['optimizer_G']['state'])
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        except:
            if self.main_process:
                warnings.warn(
                    "Warning: Failed to load optimiser, start from epoch 1.")
            self.start_epoch = 1
            self.iteration = 0
            pass
        if self.main_process:
            print("Checkpoint loaded. Resume training from epoch {}, iteration {}".
                  format(self.start_epoch, self.iteration))
