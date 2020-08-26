from torch.autograd.grad_mode import no_grad
import os
import numpy as np
import time
import torch
try:
    from src.Model.find_file_name import get_filenames
    from src.Model.Model_Perform_Tool import draw_plot
except ModuleNotFoundError:
    from Model.find_file_name import get_filenames
    from Model.Model_Perform_Tool import draw_plot


class HW3_Model(object):
    def __init__(self, device, net, loss_func, optimizer):
        self.device = device
        self.net = net
        self.loss_func = loss_func
        self.optimizer = optimizer

        self.performance_history = {
            'train_acc': [],
            'train_loss': [],
            'val_acc': [],
            'val_loss': []
        }

    def training(self, loader, val_loader=None, NUM_EPOCH=1, printPerformance=True, saveDir=None, checkpoint=0, bestModelSave=False):
        epoch_start = len(self.performance_history['train_acc'])
        NUM_EPOCH = epoch_start + NUM_EPOCH
        if saveDir is not None:
            self.saveDir = '{}/{}'.format(saveDir, time.strftime('%m%d-%H%M'))
            try:
                os.mkdir(self.saveDir)
            except OSError:
                print("Fail to create the directory {} !".format(self.saveDir))
            else:
                print("Successfully created the directory: {}".format(self.saveDir))
        try:
            if self.saveDir is not None:
                if checkpoint > 0:
                    self.checkpoint = checkpoint
                else:
                    self.checkpoint = NUM_EPOCH
        except NameError:
            self.saveDir = None
            print("No save Dir path & checkpoint !")
        for epoch in range(epoch_start+1, NUM_EPOCH+1):
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

            # valiation test
            info_word = ""
            if val_loader is not None:
                self.valiating(val_loader)
                info_word = " | Val acc: {:.7f}, loss: {:.7f}".format(
                    self.val_acc, self.val_loss)
                if epoch == 1:
                    self.best_model_epoch = 1

            # training_info
            self.train_acc = train_acc / loader.sampler.num_samples
            self.train_loss = train_loss / loader.sampler.num_samples
            self.performance_history['train_acc'].append(self.train_acc)
            self.performance_history['train_loss'].append(self.train_loss)

            # print train performance
            if printPerformance:
                print("[{:>2d}/{}] {:.3f}sec, Train acc: {:.7f}, loss: {:.7f}{}".format(
                    epoch, NUM_EPOCH, time.time()-time_start,
                    self.train_acc, self.train_loss, info_word))

            # only valiating is work, self.best_model_epoch and bestModelSave has meaning
            if val_loader is not None and self.performance_history['val_loss'][self.best_model_epoch-1] >= self.val_loss:
                if bestModelSave is True:
                    filenames = get_filenames(
                        self.saveDir, 'best_e{:03d}*.pickle'.format(self.best_model_epoch))
                    [os.remove(filename) for filename in filenames]
                    self.best_model_epoch = epoch
                    path = '{}/best_e{:03d}_{}.pickle'.format(
                        self.saveDir, epoch, str(self.val_loss)[:6])
                    self.save_model(path, onlyParameters=True)
                else:
                    self.best_model_epoch = epoch

            # checkpoint save model
            mark_word = ''
            MODEL_SAVE = True
            if epoch == NUM_EPOCH:
                mark_word = 'final_'
            elif epoch % self.checkpoint == 0:
                mark_word = ''
            else:
                MODEL_SAVE = False
            # path name rule is: (mark_name)_e(epoch)_(loss-value).pickle, e.g. best_e010_0.00138.pickle
            if MODEL_SAVE is True and self.saveDir is not None:
                if val_loader is not None:
                    LOSS_VALUE = self.val_loss
                else:
                    LOSS_VALUE = self.train_loss
                path = '{}/{}e{:03d}_{}.pickle'.format(
                    self.saveDir, mark_word, epoch, str(LOSS_VALUE)[:6])

                if mark_word == 'final_':
                    self.save_model(path, onlyParameters=False)
                else:
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

            # valiating_info
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

                prediction = np.concatenate(
                    (prediction, pred_label), axis=0).astype('int8')
                # prediction.append(pred_label)

            return prediction

    def save_model(self, path, onlyParameters=True):
        if onlyParameters:
            # torch.save(self.net.state_dict(), path)
            net = self.net
            optimizer = self.optimizer
            self.net = self.net.state_dict()
            self.optimizer = self.optimizer.state_dict()
            # with open(path, 'wb') as target:
            #     pickle.dump(self, target)
            # self.net = net
            torch.save(self, path)
            self.net = net
            self.optimizer = optimizer
        else:
            # with open(path, 'wb') as target:
            #     pickle.dump(self, path)
            torch.save(self, path)

    def load_model(self, path):
        model = torch.load(path)
        try:
            self.best_model_epoch = model.best_model_epoch
        except:
            pass
        self.saveDir = model.saveDir
        self.performance_history = model.performance_history
        self.net.load_state_dict(model.net)
        self.optimizer.load_state_dict(model.optimizer)
        self.net.eval()

    def get_performance_plt(self):
        draw_plot(self.performance_history['train_acc'], self.performance_history['train_loss'],
                  self.performance_history['val_acc'], self.performance_history['val_loss'], self.saveDir)
