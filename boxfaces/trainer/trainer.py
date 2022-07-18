import torch
from torchvision.utils import make_grid
import torch.nn.functional as F
from base.base_trainer import BaseTrainer
from model.losses import LossManager, g_nonsaturating_loss, d_logistic_loss, d_r1_loss
from utils.trainer_utils import requires_grad, accumulate

CRITIC_ITER = 5

ACCUM = 0.5 ** (32 / (10 * 1000))


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, config):

        super().__init__(config)
        self.len_epoch = len(self.data_loader)
        self.loss_args = self.config['loss']

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: .
        """
        self.netE.train()
        self.netG.train()
        for batch_idx, batch in enumerate(self.data_loader):
            self.iteration += 1

            ##########################
            # Load data
            ##########################
            real_img, mask1, mask2 = [tensor.cuda() for tensor in batch[0:3]]
            bbox = batch[-1]
            ##########################
            # Update D network
            ##########################
            requires_grad(self.netE, False)
            requires_grad(self.netG, False)
            requires_grad(self.netD, True)

            real_pred = self.netD(real_img)
            out, style = self.netE(real_img, mask1, mask2, bbox)
            fake_img = self.netG(out, style, bbox)
            fake_pred = self.netD(fake_img.detach())
            # cal loss
            d_losses = LossManager()
            d_loss = d_logistic_loss(real_pred, fake_pred)
            d_losses.add_loss(d_loss, 'd_loss', self.loss_args['d_loss_weight'])
            # update discriminator
            self.optimizer_D.zero_grad()
            d_losses.total_loss.backward()
            self.optimizer_D.step()

            d_regularize = self.iteration % self.loss_args['d_reg_every'] == 0
            if d_regularize:
                real_img.requires_grad = True
                real_pred = self.netD(real_img)
                r1_loss = d_r1_loss(real_pred, real_img)
                self.optimizer_D.zero_grad()
                r1_loss_sum = self.loss_args['r1'] / 2 * r1_loss * self.loss_args['d_reg_every']
                r1_loss_sum += 0 * real_pred[0, 0]
                r1_loss_sum.backward()
                self.optimizer_D.step()

            requires_grad(self.netE, True)
            requires_grad(self.netG, True)
            requires_grad(self.netD, False)

            ##########################
            # Update G network
            ##########################
            g_losses = LossManager()

            out, style = self.netE(real_img, mask1, mask2, bbox)
            fake_img = self.netG(out, style, bbox)
            fake_pred = self.netD(fake_img)

            # cal loss
            g_loss = g_nonsaturating_loss(fake_pred)
            p_loss = self.percept(fake_img, real_img).mean()
            l1_loss = F.l1_loss(fake_img, real_img)

            g_losses.add_loss(p_loss, 'perception_loss', self.loss_args['p_loss_weight'])
            g_losses.add_loss(l1_loss, 'l1_loss', self.loss_args['l1_loss_weight'])
            g_losses.add_loss(g_loss, 'g_loss', self.loss_args['g_loss_weight'])

            # update generator
            self.optimizer_G.zero_grad()
            g_losses.total_loss.backward()
            self.optimizer_G.step()

            if self.config['distributed']:
                accumulate(self.e_ema, self.netE.module, ACCUM)
                accumulate(self.g_ema, self.netG.module, ACCUM)
            else:
                accumulate(self.e_ema, self.netE, ACCUM)
                accumulate(self.g_ema, self.netG, ACCUM)

            ###################################################
            # print and write losses, update checkpoint
            ###################################################
            if self.main_process:
                # print loss to screen
                if batch_idx % self.log_period == 0:
                    print('******************************\n' + \
                          'Train Epoch: {} {} '.format(epoch, self._progress(batch_idx)))
                    [print('d_loss-[%s]: %.4f' % (k, v)) for (k, v) in d_losses.items()]
                    [print('g_loss-[%s]: %.4f' % (k, v)) for (k, v) in g_losses.items()]
                # writer loss to tensorboard
                if self.iteration % 500 == 0:
                    self.writer.set_step(self.iteration)
                    [self.writer.add_scalar(k, v) for k, v in d_losses.items()]
                    [self.writer.add_scalar(k, v) for k, v in g_losses.items()]
                    self.writer.add_scalar('lr_g', self.optimizer_G.state_dict()['param_groups'][0]['lr'])
                    self.writer.add_scalar('lr_d', self.optimizer_D.state_dict()['param_groups'][0]['lr'])
                # writer images to tensorboard
                if batch_idx % self.update_ckpt == 0:
                    self.writer.set_step(self.iteration)
                    samples = dict()
                    samples['gt_img'] = real_img
                    samples['mask1'] = real_img * mask1
                    samples['mask2'] = real_img * mask2
                    samples['recon'] = fake_img
                    self.vis_images(samples)
                # update checkpoint
                if self.iteration % self.update_ckpt == 0:
                    self._save_checkpoint(epoch, latest=True)

        return

    def vis_images(self, images):
        for k, v in images.items():
            vis_images = v[:self.config['trainer']['vis_img_num'], :, :, :]
            self.writer.add_image('{}'.format(k),
                                  make_grid(vis_images.cpu(), nrow=4, padding=4, normalize=True))

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        print('*******checking on val*********')
        # test model on valida dataset
        val_results = self.check_model(self.valid_data_loader)
        val_samples, val_mse = val_results
        val_metrics = {'avg_mse': val_mse}
        # print validation results
        print('Valid Epoch: {} val mse {:6f}'.format(epoch, val_mse))
        # writer results (metrics and images) to tensorboard
        self.writer.set_step(self.iteration, mode='val')
        [self.writer.add_scalar(k, v) for k, v in val_metrics.items()]
        self.vis_images(val_samples)
        return val_metrics

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def check_model(self, loader):
        num_samples = 0
        mse = 0
        samples = dict()
        self.inception_score.clean()
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):

                real_img, mask1, mask2 = [tensor.cuda() for tensor in batch[0:3]]
                bbox = batch[-1]
                ##########################
                # forward pass
                ##########################
                out, style = self.e_ema(real_img, mask1, mask2, bbox)
                recon = self.g_ema(out, style, bbox)

                ##########################
                # cal metrics
                ##########################
                mse += F.mse_loss(recon, real_img) * real_img.size(0)
                num_samples += real_img.size(0)

                # get vis samples
                if batch_idx == 0:
                    samples = dict()
                    samples['gt_img'] = real_img
                    samples['mask1'] = real_img * mask1
                    samples['mask2'] = real_img * mask2
                    samples['recon'] = recon

            mean_mse = mse / num_samples
        out = [samples, mean_mse]

        return tuple(out)
