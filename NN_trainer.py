import torch.optim
from NN_model import *
from nn_dataloader import NNDataLoader
import math
import numpy as np


class NNTrainer:

    def __init__(self, data, two_layer=False, device='cpu', batch_size=64,
                 num_epochs=1000, log_step=None, valid_log_step=None, hidden=40, verbose=True):

        self.verbose = verbose
        self.hidden = hidden
        self.data = data
        self.dimension = self.data.train['input'].shape[1]
        self.loss_list = list()
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.log_step = log_step
        self.valid_log_step = valid_log_step
        self.global_step = 0

        self.data_loader = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.init_data_loader()

        self.device = device
        self.init_device()

        self.num_samples, self.num_batches = {}, {}
        self.save_train_test_info()

        self.net = None
        self.optimizer = None
        self.scheduler = None

        self.build_network(self.dimension, two_layer)
        self.set_optimizer()

        self.criterion = nn.MSELoss(reduction='mean')
        self.test_criterion = nn.MSELoss(reduction='sum')

        self.criterion.to(self.device)
        self.test_criterion.to(self.device)

    def init_device(self):
        if self.device == 'cuda:0':
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                if self.verbose:
                    print('\nUsing GPU.')
            else:
                self.device = 'cpu'
                if self.verbose:
                    print('\nCUDA not available. Using CPU.')
        else:
            self.device = 'cpu'
            if self.verbose:
                print('\nUsing CPU.')

    def init_data_loader(self):

        if self.verbose:
            print('\n==> Preparing data..')

        self.data_loader = NNDataLoader(self.data, batch_size=self.batch_size)
        self.train_loader, self.valid_loader = self.data_loader.get_train_valid_loader()
        self.test_loader = self.data_loader.get_test_loader()

    def save_train_test_info(self):

        for mode, loader in zip(['train', 'valid', 'test'],
                                [self.train_loader, self.valid_loader, self.test_loader]):

            self.num_samples[mode] = sum(inputs.size(0) for inputs, _ in loader)
            self.num_batches[mode] = math.ceil(self.num_samples[mode] / self.data_loader.batch_size)

            if self.verbose:
                print(f'no.  of {mode} samples : {self.num_samples[mode]}')
                print(f'\nNumber of {mode} batches per epoch : {self.num_batches[mode]}')

    def build_network(self, dimension, two_layer):

        self.net = Net(dimension, two_layer=two_layer, hidden=self.hidden)
        self.net = self.net.to(self.device)
        if self.verbose:
            print(f'Number of model parameters is {self.net.num_params}')

    def set_optimizer(self, weight_decay=0.1, milestone=None):

        if milestone is None:
            milestone = [int(0.6*self.num_epochs), int(0.75*self.num_epochs), int(0.9*self.num_epochs)]

        if self.verbose:
            print(f'milestone: {milestone}')
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001, weight_decay=weight_decay)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestone)

        if self.verbose:
            print('\nOptimizer :', self.optimizer, sep='\n')
            print('\nScheduler :', self.scheduler, sep='\n')

    def train(self):

        self.net.train()

        if self.log_step is None:
            # log at every epoch
            self.log_step = self.num_batches['train']
            if self.valid_log_step is None:
                # validation done halfway and at the end of training
                self.valid_log_step = \
                    self.num_batches['train'] * int(self.num_epochs / 2)

        if self.verbose:
            print('\n\nStarting training...')
        self.global_step = 0

        for i in range(self.num_epochs):
            self.train_epoch(i)
            self.train_loader = self.data_loader.get_shuffled_train_loader_in_memory()

        if self.verbose:
            print('\nFinished training.')

    def train_epoch(self, epoch):
        print(f'\nEpoch: {epoch}\n')

        running_loss = 0.

        for batch_idx, (inputs, labels) in enumerate(self.train_loader):

            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            if batch_idx % self.log_step == (self.log_step - 1):
                mse = (running_loss / self.log_step)
                self.loss_list = self.loss_list + [mse]
                losses_dict = {'mse': mse}
                self.train_log_step(epoch, batch_idx, losses_dict)
                running_loss = 0.  # reset running loss

            if self.global_step % self.valid_log_step == (self.valid_log_step - 1):
                self.validation_step()
                self.net.train()

            self.global_step += 1

        if self.scheduler is not None:
            self.scheduler.step()
            if self.verbose:
                lr = [group['lr'] for group in self.optimizer.param_groups]
                print(f'scheduler: epoch - {self.scheduler.last_epoch}; '
                      f'learning rate - {lr}')

    def train_log_step(self, epoch, batch_idx, losses_dict):
        """
        Args:
            losses_dict - a dictionary {loss name : loss value}
        """
        assert isinstance(losses_dict, dict)

        print('[{:3d}, {:6d} / {:6d}] '.format(epoch + 1, batch_idx + 1, self.num_batches['train']), end='')
        for key, value in losses_dict.items():
            print('{}: {:7.7f} | '.format(key, value), end='')

    def validation_step(self):
        loss = self.evaluate_results(mode='valid')

        print(f'\nvalidation mse : {loss}')
        losses_dict = {'mse': loss}

        self.valid_log_step_f(losses_dict)

    @staticmethod
    def valid_log_step_f(losses_dict):
        assert isinstance(losses_dict, dict)

        print('\nvalidation_step : ', end='')
        for key, value in losses_dict.items():
            print('{}: {:7.3f} | '.format(key, value), end='')

    def evaluate_results(self, mode='valid'):
        assert mode in ['valid', 'test']

        if mode == 'valid':
            dataloader = self.valid_loader
            data_dict = self.data.valid
        else:
            dataloader = self.test_loader
            data_dict = self.data.test

        self.net.eval()
        running_loss = 0.
        total = 0
        predictions = torch.tensor([])

        with torch.no_grad():

            for batch_idx, (inputs, labels) in enumerate(dataloader):

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.net(inputs)
                predictions = torch.cat((predictions, outputs.cpu()), dim=0)

                loss = self.test_criterion(outputs, labels)
                running_loss += loss.item()
                total += labels.size(0)

        data_dict['predictions'] = predictions

        loss = running_loss / total
        # sanity check
        mse, _ = self.compute_mse_psnr(data_dict['values'], predictions)
        assert np.allclose(mse, loss), '(mse: {:.7f}, loss: {:.7f})'.format(mse, loss)

        return loss

    def compute_mse_psnr(self, x_values, x_values_hat, pixel_max=255):
        """ """
        mse = ((x_values - x_values_hat) ** 2).mean().item()
        psnr = self.compute_psnr_from_mse(mse, pixel_max)

        return mse, psnr

    @staticmethod
    def compute_psnr_from_mse(mse, pixel_max=255):
        if mse == 0:
            psnr = 100
        else:
            psnr = 20 * math.log10(pixel_max / math.sqrt(mse))

        return psnr
