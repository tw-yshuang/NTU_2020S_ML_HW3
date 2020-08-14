from torch.autograd.grad_mode import no_grad
import time
import torch


class HW3_Model(object):
    def __init__(self, device, net, loss_func, optimizer):
        self.device = device
        self.net = net
        self.loss_func = loss_func
        self.optimizer = optimizer

        # self.train_acc_list = []
        # self.train_loss_list = []
        # self.val_acc_list = []
        # self.val_loss_list = []
        self.performance_history = {
            'train_acc': [],
            'train_loss': [],
            'val_acc': [],
            'val_loss': []
        }

    def training(self, loader, val_loader=None, NUM_EPOCH=1, printPerformance=True, saveDir=None, checkpoint=10):
        for epoch in range(1, NUM_EPOCH+1):
            time_start = time.time()
            train_acc = 0.
            train_loss = 0.

            self.net.train()  # change model to the train mode.
            for i, dataset in enumerate(loader):
                data = dataset[0].to(self.device)
                label = dataset[1].cpu()

                self.optimizer.zero_grad()
                pred = self.net(data).cpu()
                loss = self.loss_func(pred, label)
                loss.backward()
                self.optimizer.step()

                # train_pred.size = (batch_size, 11)
                pred_label = torch.argmax(pred, dim=1).numpy()
                train_acc += sum(pred_label == label.numpy())
                train_loss += loss.item()

            self.train_acc = train_acc / loader.sampler.num_samples
            self.train_loss = train_loss / loader.sampler.num_samples
            self.performance_history['train_acc'].append(self.train_acc)
            self.performance_history['train_loss'].append(self.train_loss)

            if val_loader is not None:
                self.valiating(val_loader)
                if printPerformance:
                    print('[{:>2d}/{}] {:.3f} sec, Train acc: {:.5f}, loss: {:.5f} | Val acc: {:.5f}, loss: {:.5f}'.format(
                        epoch, NUM_EPOCH, time.time()-time_start,
                        self.train_acc, self.train_loss, self.val_acc, self.val_loss))

            else:
                if printPerformance:
                    print('[{:>2d}/{}] {:.3f}sec, Train acc: {:.5f}, loss: {:.5f}'.format(
                        epoch, NUM_EPOCH, time.time()-time_start,
                        self.train_acc, self.train_loss))

            if saveDir is not None:
                self.saveDir = saveDir
                self.checkpoint = checkpoint
                if epoch % self.checkpoint == 0:
                    path = ''
                    # path name rule is: mmdd-hhmm_epoch_loss-value, e.g. 0814-1600_e010_0.00138.pkl
                    if val_loader is not None:
                        # loss-value = val_loss
                        path = '{}/{}_e{:03d}_{}.pkl'.format(self.saveDir, time.strftime(
                            '%m%d-%H%M'), epoch, str(self.val_loss)[:6])
                    else:
                        # loss-value = train_loss
                        path = '{}/{}_e{:03d}_{}.pkl'.format(self.saveDir, time.strftime(
                            '%m%d-%H%M'), epoch, str(self.train_loss)[:6])

                    self.save_model(path, onlyParameters=True)

    def valiating(self, loader):
        self.net.eval()  # change model to the evaluation(val or test) mode.
        with no_grad():
            val_acc = 0.
            val_loss = 0.

            for i, dataset in enumerate(loader):
                data = dataset[0].to(self.device)
                label = dataset[1].cpu()

                pred = self.net(data).cpu()
                loss = self.loss_func(pred, label)
                pred_label = torch.argmax(pred, dim=1).numpy()

                val_acc += sum(pred_label == label.numpy())
                val_loss += loss.item()

            self.val_acc = val_acc / loader.sampler.num_samples
            self.val_loss = val_loss / loader.sampler.num_samples

            self.performance_history['val_acc'].append(self.val_acc)
            self.performance_history['val_loss'].append(self.val_loss)

    def testing(self, loader):
        self.net.eval()  # change model to the evaluation(val or test) mode.
        prediction = []
        with no_grad():
            for i, dataset in enumerate(loader):
                data = dataset.to(self.device)

                pred = self.net(data).cpu()
                pred_label = torch.argmax(pred, dim=1).numpy()
                prediction.append(pred_label)

            return prediction

    def save_model(self, path, onlyParameters=True):
        if onlyParameters:
            torch.save(self.net.state_dict(), path)
        else:
            torch.save(self.net, path)
