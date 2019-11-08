import csv
import datetime
import os
import numpy as np
import time
import torch
import torchvision
from tqdm import tqdm
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torchvision.models as models
from util import write_multi_label

class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                    type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class StepLR(_LRScheduler):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        self.step_size = step_size
        self.gamma = gamma
        super(StepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** (self.last_epoch // self.step_size)
                for base_lr in self.base_lrs]

class Solver(object):
	def __init__(self, config, train_loader, valid_loader, test_loader):

		# Data loader
		self.train_loader = train_loader
		self.valid_loader = valid_loader
		self.test_loader = test_loader

		# Models
		self.model_type = config['model_type']
		self.num_classes = config['num_classes']
		self.net = None     
		self.optimizer = None   
		self.scheduler = None
		self.criterion = torch.nn.CrossEntropyLoss()

		# Hyper-parameters
		self.lr = config['lr']
		self.momentum = config['momentum']
		self.weight_decay = config['weight_decay']
		self.step_size = config['step_size']
		self.gamma = config['gamma']
		
		self.threashold = config['threashold']

		# Training settings
		self.batch_size = config['batch_size']
		self.num_epochs = config['num_epochs']

		# Path
		self.model_path = config['model_path']
		self.net_path = config['net_path']
		self.save_freq = config['save_freq']
		self.train_checkpoint_file = config['train_checkpoint_file']
		self.valid_checkpoint_file = config['valid_checkpoint_file']
		self.test_ans_file = config['test_ans_file']
		
		self.mode = config['mode']

		self.cuda = torch.cuda.is_available()
		torch.backends.cudnn.benchmark = True
		self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu') 
		self.build_model()

	def build_model(self):
		"""Build generator and discriminator."""
		print(self.model_type)
		if self.model_type =='resnet50':
			if self.mode == 'train':
				self.net = models.resnet50(pretrained=True)
			elif self.mode == 'test':
				self.net = models.resnet50(pretrained=False)
			self.net.fc = torch.nn.Linear(2048, self.num_classes)
			if self.mode == 'test':
				checkpoint = torch.load(self.net_path, map_location=lambda storage, loc: storage)
				pretrained_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
				self.net.load_state_dict(pretrained_dict)	
			self.net.eval()  #????	

		self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
		self.scheduler = StepLR(self.optimizer, self.step_size, self.gamma)
		self.net = torch.nn.DataParallel(self.net, device_ids=[1,2,3])
		self.net.to(self.device)
		#self.print_network(self.net, self.model_type)

    
	def print_network(self, model, name):
		"""Print out the network information."""
		num_params = 0
		for p in model.parameters():
			num_params += p.numel()
		print(model)
		print(name)
		print("The number of parameters: {}".format(num_params))

	def save(self, epoch, **kwargs):
		if self.model_path is not None:
			if not os.path.exists(self.model_path):
				os.makedirs(self.model_path)
			state = self.net
			print(self.model_path + self.model_type + "_epoch_{}.pth".format(epoch))
			torch.save(state.state_dict(), self.model_path+ "/"+self.model_type+"_epoch_{}.pth".format(epoch))

	def train(self):
		for epoch in range(self.num_epochs):
			if self.scheduler is not None:
				self.scheduler.step()
			print("epochs: {}".format(epoch))
			#=== train
			self.net.train(True)
			with torch.enable_grad():#???
				epoch_loss = []
				accuracy = []
				TP = np.zeros(self.num_classes)
				FP = np.zeros(self.num_classes)
				FN = np.zeros(self.num_classes)
				TN = np.zeros(self.num_classes)
				precision = np.zeros(self.num_classes)
				recall = np.zeros(self.num_classes)
				F1 = 0.
				Acc = 0.
				for image, GT, image_file in tqdm(self.train_loader, ncols=80):#???
					image = image.to(self.device)
					GT = GT.to(self.device)
					SR = self.net(image) 
					loss = self.criterion(SR, GT)
					epoch_loss.append(loss.data.item() / len(self.train_loader))
					print(SR.data.max(1)[1])
					print(GT.data)
					
					maxinum = SR.max(-1)[0].view(-1,1)
					SR_multi = SR > (maxinum - self.threashold)
					GT_multi = np.zeros([self.batch_size, self.num_classes])
					for i in range(len(image_file)):
						GT_multi[i][GT[i].item()] = 1
						for j in range(self.num_classes):
							if SR_multi[i][j].item() == 1 and GT_multi[i][j].item() == 1:
								TP[j] = TP[j] + 1
							if SR_multi[i][j].item() == 1 and GT_multi[i][j].item() == 0:
								FP[j] = FP[j] + 1
							if SR_multi[i][j].item() == 0 and GT_multi[i][j].item() == 1:
								FN[j] = FN[j] + 1
							if SR_multi[i][j].item() == 0 and GT_multi[i][j].item() == 0:
								TN[j] = TN[j] + 1
					accuracy.append((SR.data.max(1)[1] == GT.data).sum().item())
					# Backprop + optimize
					self.optimizer.zero_grad()
					loss.backward()
					self.optimizer.step()

				epoch_loss = sum(epoch_loss)
				for i in range(self.num_classes):
					precision[i] = float(TP[i])/(float(TP[i]+FP[i]) + 1e-6)
					recall[i] = float(TP[i])/(float(TP[i]+FN[i]) + 1e-6)
					F1 = F1 + 2*recall[i]*precision[i]/(recall[i]+precision[i] + 1e-6)
					Acc = Acc + float(TP[i]+TN[i])/(float(TP[i]+TN[i]+FP[i]+FN[i]) + 1e-6)
				F1 = F1/self.num_classes
				Acc = Acc/self.num_classes
				accuracy = sum(accuracy) / float(len(self.train_loader.dataset))
				print('Epoch [%d/%d], Loss: %.4f \n[Training] Acc: %.4f, F1: %.4f, Acc_multi: %.4f\n' % (epoch+1, self.num_epochs, epoch_loss, accuracy, F1, Acc))
				with open(self.train_checkpoint_file,'a+') as f:
					csv_write = csv.writer(f)
					data_row = [epoch+1, epoch_loss, accuracy, F1, Acc]
					csv_write.writerow(data_row)

		    ## valid
			self.net.train(False)
			self.net.eval()
			with torch.no_grad():#???
				epoch_loss = []
				accuracy = []
				TP = np.zeros(self.num_classes)
				FP = np.zeros(self.num_classes)
				FN = np.zeros(self.num_classes)
				TN = np.zeros(self.num_classes)
				precision = np.zeros(self.num_classes)
				recall = np.zeros(self.num_classes)
				F1 = 0.
				Acc = 0.
				for image, GT, image_file in tqdm(self.valid_loader, ncols=80):
					image = image.to(self.device)
					GT = GT.to(self.device)
					SR = self.net(image) 
					loss = self.criterion(SR, GT)
					epoch_loss.append(loss.item()/len(self.valid_loader))
					print(SR.data.max(1)[1])
					print(GT.data)
					maxinum = SR.max(-1)[0].view(-1,1)
					SR_multi = SR > (maxinum - self.threashold)
					GT_multi = np.zeros([self.batch_size, self.num_classes])
					for i in range(len(image_file)):
						GT_multi[i][GT[i].item()] = 1
						for j in range(self.num_classes):
							if SR_multi[i][j].item() == 1 and GT_multi[i][j].item() == 1:
								TP[j] = TP[j] + 1
							if SR_multi[i][j].item() == 1 and GT_multi[i][j].item() == 0:
								FP[j] = FP[j] + 1
							if SR_multi[i][j].item() == 0 and GT_multi[i][j].item() == 1:
								FN[j] = FN[j] + 1
							if SR_multi[i][j].item() == 0 and GT_multi[i][j].item() == 0:
								TN[j] = TN[j] + 1
					accuracy.append((SR.data.max(1)[1] == GT.data).sum().item())

				epoch_loss = sum(epoch_loss)
				for i in range(self.num_classes):
					precision[i] = float(TP[i])/(float(TP[i]+FP[i]) + 1e-6)
					recall[i] = float(TP[i])/(float(TP[i]+FN[i]) + 1e-6)
					F1 = F1 + 2*recall[i]*precision[i]/(recall[i]+precision[i] + 1e-6)
					Acc = Acc + float(TP[i]+TN[i])/(float(TP[i]+TN[i]+FP[i]+FN[i]) + 1e-6)
				F1 = F1/self.num_classes
				Acc = Acc/self.num_classes
				accuracy = sum(accuracy) / float(len(self.valid_loader.dataset))
				print('Epoch [%d/%d], Loss: %.4f \n[Validation] Acc: %.4f, F1: %.4f, Acc_multi: %.4f\n' % (epoch+1, self.num_epochs, epoch_loss, accuracy, F1, Acc))
				with open(self.valid_checkpoint_file,'a+') as f:
					csv_write = csv.writer(f)
					data_row = [epoch+1, epoch_loss, accuracy, F1, Acc]
					csv_write.writerow(data_row)

			## save	
			if (epoch+1) % self.save_freq == 0:
				self.save(epoch+1)

	def test(self):
		self.net.train(False)
		self.net.eval()
		with torch.no_grad():#???
			epoch_loss = []
			accuracy = []
			TP = np.zeros(self.num_classes)
			FP = np.zeros(self.num_classes)
			FN = np.zeros(self.num_classes)
			TN = np.zeros(self.num_classes)
			precision = np.zeros(self.num_classes)
			recall = np.zeros(self.num_classes)
			F1 = 0.
			Acc = 0.
			for image, GT, image_file in tqdm(self.test_loader, ncols=80):
				image = image.to(self.device) 
				GT = GT.to(self.device)
				SR = self.net(image) 
				loss = self.criterion(SR, GT)
				epoch_loss.append(loss.item()/len(self.test_loader))
				print(SR.data.max(1)[1])
				print(GT.data)
				maxinum = SR.max(-1)[0].view(-1,1)#64->64*1
				SR_multi = SR > (maxinum - self.threashold)
				GT_multi = np.zeros([self.batch_size, self.num_classes])
				#print(SR.data.max(1)[1])
				#print(SR_multi.data)
				for i in range(len(image_file)):
					GT_multi[i][GT[i].item()] = 1
					for j in range(self.num_classes):
						if SR_multi[i][j].item() == 1 and GT_multi[i][j].item() == 1:
							TP[j] = TP[j] + 1
						if SR_multi[i][j].item() == 1 and GT_multi[i][j].item() == 0:
							FP[j] = FP[j] + 1
						if SR_multi[i][j].item() == 0 and GT_multi[i][j].item() == 1:
							FN[j] = FN[j] + 1
						if SR_multi[i][j].item() == 0 and GT_multi[i][j].item() == 0:
							TN[j] = TN[j] + 1
				accuracy.append((SR.data.max(1)[1] == GT.data).sum().item())
				write_multi_label(image_file, SR_multi)
				with open(self.test_ans_file,'a+') as f:
					csv_write = csv.writer(f)
					for i in range(len(image_file)):
						data_row = [image_file[i], SR[i].data.max(0)[1].item()+1]
						csv_write.writerow(data_row)
			epoch_loss = sum(epoch_loss)
			for i in range(self.num_classes):
				precision[i] = float(TP[i])/(float(TP[i]+FP[i]) + 1e-6)
				recall[i] = float(TP[i])/(float(TP[i]+FN[i]) + 1e-6)
				F1 = F1 + 2*recall[i]*precision[i]/(recall[i]+precision[i] + 1e-6)
				Acc = Acc + float(TP[i]+TN[i])/(float(TP[i]+TN[i]+FP[i]+FN[i]) + 1e-6)
			F1 = F1/self.num_classes
			Acc = Acc/self.num_classes
			accuracy = sum(accuracy) / float(len(self.test_loader.dataset))
			print('Loss: %.4f, \n[Test] Acc: %.4f, F1: %.4f, Acc_multi: %.4f\n' % (epoch_loss, accuracy, F1, Acc))
