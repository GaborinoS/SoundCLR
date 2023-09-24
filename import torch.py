import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import os
import datetime
import sys
import config
from torch.utils import data
import torchvision


import imageio
import random
import collections
import csv
import librosa


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2048, config.class_numbers)
        
        
        
        
    def forward(self, x):
        x = self.fc(x)
        
        return x
    


def scale(old_value, old_min, old_max, new_min, new_max):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    return new_value


class ToTensor1D(tv.transforms.ToTensor):

    def __call__(self, tensor: np.ndarray):
        tensor_2d = super(ToTensor1D, self).__call__(tensor[..., np.newaxis])
        
        return tensor_2d.squeeze_(0)

class RandomNoise():
    def __init__(self, min_noise=0.0, max_noise=0.05): #0.002, 0.01
        super(RandomNoise, self).__init__()
        
        self.min_noise = min_noise
        self.max_noise = max_noise
        
    def addNoise(self, wave):
        noise_val = random.uniform(self.min_noise, self.max_noise)
        noise = torch.from_numpy(np.random.normal(0, noise_val, wave.shape[0]))
        noisy_wave = wave + noise
        
        return noisy_wave
    
    def __call__(self, x):
        return self.addNoise(x)



class RandomScale():

    def __init__(self, max_scale: float = 1.25):
        super(RandomScale, self).__init__()

        self.max_scale = max_scale

    @staticmethod
    def random_scale(max_scale: float, signal: torch.Tensor) -> torch.Tensor:
        scaling = np.power(max_scale, np.random.uniform(-1, 1)) #between 1.25**(-1) and 1.25**(1)
        output_size = int(signal.shape[-1] * scaling)
        ref = torch.arange(output_size, device=signal.device, dtype=signal.dtype).div_(scaling)
        
        # ref1 is of size output_size
        ref1 = ref.clone().type(torch.int64)
        ref2 = torch.min(ref1 + 1, torch.full_like(ref1, signal.shape[-1] - 1, dtype=torch.int64))
        
        r = ref - ref1.type(ref.type())
        
        scaled_signal = signal[..., ref1] * (1 - r) + signal[..., ref2] * r
        
        
        return scaled_signal

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.random_scale(self.max_scale, x)

    
    

class RandomCrop():

    def __init__(self, out_len: int = 44100, train: bool = True):
        super(RandomCrop, self).__init__()

        self.out_len = out_len
        self.train = train

    def random_crop(self, signal: torch.Tensor) -> torch.Tensor:
        if self.train:
            left = np.random.randint(0, signal.shape[-1] - self.out_len)
        else:
            left = int(round(0.5 * (signal.shape[-1] - self.out_len)))

        orig_std = signal.float().std() * 0.5
        output = signal[..., left:left + self.out_len]

        out_std = output.float().std()
        if out_std < orig_std:
            output = signal[..., :self.out_len]

        new_out_std = output.float().std()
        if orig_std > new_out_std > out_std:
            output = signal[..., -self.out_len:]

        return output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.random_crop(x) if x.shape[-1] > self.out_len else x


class RandomPadding():

    def __init__(self, out_len: int = 88200, train: bool = True):
        super(RandomPadding, self).__init__()

        self.out_len = out_len
        self.train = train

    def random_pad(self, signal: torch.Tensor) -> torch.Tensor:
        
        if self.train:
            left = np.random.randint(0, self.out_len - signal.shape[-1])
        else:
            left = int(round(0.5 * (self.out_len - signal.shape[-1])))

        right = self.out_len - (left + signal.shape[-1])

        pad_value_left = signal[..., 0].float().mean().to(signal.dtype)
        pad_value_right = signal[..., -1].float().mean().to(signal.dtype)
        output = torch.cat((
            torch.zeros(signal.shape[:-1] + (left,), dtype=signal.dtype, device=signal.device).fill_(pad_value_left),
            signal,
            torch.zeros(signal.shape[:-1] + (right,), dtype=signal.dtype, device=signal.device).fill_(pad_value_right)
        ), dim=-1)

        return output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.random_pad(x) if x.shape[-1] < self.out_len else x
    
            
    
    
class FrequencyMask():
    def __init__(self, max_width, numbers): 
        super(FrequencyMask, self).__init__()
        
        self.max_width = max_width
        self.numbers = numbers
    
    def addFreqMask(self, wave):
        #print(wave.shape)
        for _ in range(self.numbers):
            #choose the length of mask
            mask_len = random.randint(0, self.max_width)
            start = random.randint(0, wave.shape[1] - mask_len) #start of the mask
            end = start + mask_len
            wave[:, start:end, : ] = 0
            
        return wave
    
    def __call__(self, wave):
        return self.addFreqMask(wave)
    
        

class TimeMask():
    def __init__(self, max_width, numbers): 
        super(TimeMask, self).__init__()
        
        self.max_width = max_width
        self.numbers = numbers
    
    
    def addTimeMask(self, wave):
        
        for _ in range(self.numbers):
            #choose the length of mask
            mask_len = random.randint(0, self.max_width)
            start = random.randint(0, wave.shape[2] - mask_len) #start of the mask
            end = start + mask_len
            wave[ : , : , start:end] = 0
            
        return wave
    
    def __call__(self, wave):
        return self.addTimeMask(wave)

class EarlyStopping:
	"""Early stops the training if validation loss doesn't improve after a given patience."""
	def __init__(self, patience=7, verbose=False, delta=0, log_path='', output_file = './results.txt'):
		"""
		Args:
		patience (int): How long to wait after last time validation loss improved.
                            Default: 7
		verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
		delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
		"""
		self.patience = patience
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.val_loss_min = np.Inf
		self.delta = delta
		self.log_path = log_path
		self.output_file = output_file
        

	def __call__(self, val_loss, model, epoch):

		score = -val_loss
		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(val_loss, model, epoch)
		elif score < self.best_score - self.delta:
			self.counter += 1
			print(f'EarlyStopping counter: {self.counter} out of {self.patience}', file=self.output_file)
			if self.counter >= self.patience:
				self.early_stop = True
		else:
			self.best_score = score
			self.save_checkpoint(val_loss, model, epoch)
			self.counter = 0

	def save_checkpoint(self, val_loss, model, epoch):
		'''Saves model when validation loss decrease.'''
		if self.verbose:
			print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...', file=self.output_file)
        
		torch.save(model.state_dict(), os.path.join(self.log_path, 'checkpoint.pt'))
		self.val_loss_min = val_loss
        
        

        
class WarmUpStepLR(torch.optim.lr_scheduler._LRScheduler):

	def __init__(self, optimizer: torch.optim.Optimizer, cold_epochs: int, warm_epochs: int, step_size: int, 
			gamma: float = 0.1, last_epoch: int = -1):
		
		super(WarmUpStepLR, self).__init__(optimizer=optimizer, last_epoch=last_epoch)
		self.cold_epochs = cold_epochs
		self.warm_epochs = warm_epochs
		self.step_size = step_size
		self.gamma = gamma

		

	def get_lr(self):
		if self.last_epoch < self.cold_epochs:
			return [base_lr * 0.1 for base_lr in self.base_lrs]
		elif self.last_epoch < self.cold_epochs + self.warm_epochs:
			return [
				base_lr * 0.1 + (1 + self.last_epoch - self.cold_epochs) * 0.9 * base_lr / self.warm_epochs
				for base_lr in self.base_lrs
				]
		else:
			return [
				base_lr * self.gamma ** ((self.last_epoch - self.cold_epochs - self.warm_epochs) // self.step_size)
				for base_lr in self.base_lrs
				]


class WarmUpExponentialLR(WarmUpStepLR):

	def __init__(self, optimizer: torch.optim.Optimizer, cold_epochs: int, warm_epochs: int,
                 	gamma: float = 0.1, last_epoch: int = -1):

		self.cold_epochs = cold_epochs
		self.warm_epochs = warm_epochs
		self.step_size = 1
		self.gamma = gamma

		super(WarmUpStepLR, self).__init__(optimizer=optimizer, last_epoch=last_epoch)
        
        
        
        
def calculateClassInfo(class_to_representations, class_to_projections, epoch):
	class_to_repMeans = {} # key is the class_id and values are mean vector for each class
	class_to_projMeans = {}
    
	for class_id in class_to_representations:
		class_to_repMeans[class_id] = torch.mean(class_to_representations[class_id], dim=0)
		class_to_projMeans[class_id] = torch.mean(class_to_projections[class_id], dim=0)
    
	rep_distances = torch.zeros(50,50)
	proj_distances = torch.zeros(50,50)
	for i in range(50):
		for j in range(50):
			rep_distances[i][j] = torch.dist(class_to_repMeans[i], class_to_repMeans[j])
			proj_distances[i][j] = torch.dist(class_to_projMeans[i], class_to_projMeans[j])
    
    
	#calculating std for each class
	rep_std = torch.zeros(50)
	proj_std = torch.zeros(50)
	for i in range(50):
		rep_std_vec = torch.std(class_to_representations[i], dim=0)
		rep_std[i] = torch.norm(rep_std_vec, p=2, dim=0)
        
		proj_std_vec = torch.std(class_to_projections[i], dim=0)
		proj_std[i] = torch.norm(proj_std_vec, p=2, dim=0)
    
    
	fig = plt.figure(figsize=(8, 6))

	fig.add_subplot(221)
	plt.title('distance between means of {} features in representation space with average of {:.4f}'.format(
		class_to_representations[0][0].shape[0], float(rep_distances.mean())), fontsize=6)
	plt.imshow(rep_distances.numpy(), cmap='Blues')
	plt.colorbar()

	fig.add_subplot(222)
	plt.title('std of {} features in representation space with average of {:.4f}'.format(
		class_to_representations[0][0].shape[0], float(rep_std.mean())), fontsize=6)
	plt.bar(range(50), rep_std.numpy(), 0.5 )
    
      
	fig.add_subplot(223)
	plt.title('distance between means of {} features in projection space with average of {:.4f}'.format(
		class_to_projections[0][0].shape[0], float(proj_distances.mean())), fontsize=6)
	plt.imshow(proj_distances.numpy(), cmap='Blues')
	plt.colorbar()
    
	fig.add_subplot(224)
	plt.title('std of {} features in projection spacewith average of {:.4f}'.format(
		class_to_projections[0][0].shape[0], float(proj_std.mean())), fontsize=6)
	plt.bar(range(50), proj_std.numpy(), 0.5 )
    
    
	plt.savefig(fig_path + 'epoch_' + str(epoch)  + '.png', dpi=175)
    
	plt.clf()
	plt.close()
        

import os
import numpy as np
import imageio
import random
import collections
import csv
import librosa




# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")



class MyDataset(data.Dataset):
    
    def __init__(self, train=True):
        self.root = './data/ESC/ESC-50-master/audio/'
        self.train = train
        
        #getting name of all files inside the all of the train_folds
        temp = os.listdir(self.root)
        temp.sort()
        self.file_names = []
        if train:
            for i in range(len(temp)):
                if int(temp[i].split('-')[0]) in config.train_folds:
                    self.file_names.append(temp[i])
        else:
            for i in range(len(temp)):
                if int(temp[i].split('-')[0]) in config.test_fold:
                    self.file_names.append(temp[i])
        
        if self.train:
            self.wave_transforms = torchvision.transforms.Compose([ transforms.ToTensor1D(), 
                                                              transforms.RandomScale(max_scale = 1.25), 
                                                              transforms.RandomPadding(out_len = 220500),
                                                              transforms.RandomCrop(out_len = 220500)])
             
            
            self.spec_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor() , 
									transforms.FrequencyMask(max_width = config.freq_masks_width, numbers = config.freq_masks), 
									transforms.TimeMask(max_width = config.time_masks_width, numbers = config.time_masks)])
            
        else: #for test
            self.wave_transforms = torchvision.transforms.Compose([ transforms.ToTensor1D(),
                                                              transforms.RandomPadding(out_len = 220500),
                                                             transforms.RandomCrop(out_len = 220500)])
        
            self.spec_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor() ])

    
    def __len__(self):
        return len(self.file_names)
    
    

    def __getitem__(self, index):
        file_name = self.file_names[index ]  
        path = self.root + file_name
        wave, rate = librosa.load(path, sr=44100)
        
        #identifying the label of the sample from its name
        temp = file_name.split('.')[0]
        class_id = int(temp.split('-')[-1])
        
        if wave.ndim == 1:
            wave = wave[:, np.newaxis]
		
	# normalizing waves to [-1, 1]
        if np.abs(wave.max()) > 1.0:
            wave = transforms.scale(wave, wave.min(), wave.max(), -1.0, 1.0)
        wave = wave.T * 32768.0
        
        # Remove silent sections
        start = wave.nonzero()[1].min()
        end = wave.nonzero()[1].max()
        wave = wave[:, start: end + 1]  
        
        wave_copy = np.copy(wave)
        wave_copy = self.wave_transforms(wave_copy)
        wave_copy.squeeze_(0)
        
        s = librosa.feature.melspectrogram(wave_copy.numpy(), sr=44100, n_mels=128, n_fft=1024, hop_length=512) 
        log_s = librosa.power_to_db(s, ref=np.max)
        
	# masking the spectrograms
        log_s = self.spec_transforms(log_s)
        
        
        #creating 3 channels by copying log_s1 3 times 
        spec = torch.cat((log_s, log_s, log_s), dim=0)
        
        return file_name, spec, class_id
        

    
    
    
 

def create_generators():
    train_dataset = MyDataset(train=True)
    test_dataset = MyDataset(train=False)
    
    train_loader = data.DataLoader(train_dataset, batch_size = config.batch_size, shuffle=True, num_workers=10 ,drop_last=False)
    
    test_loader = data.DataLoader(test_dataset, batch_size = config.batch_size, shuffle=True, num_workers=10 ,drop_last=False)
    
    return train_loader, test_loader
    
   




use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Update the model loading line
model = torchvision.models.resnet50(pretrained=True).to(device)
model.fc = nn.Sequential(nn.Identity())

model = nn.DataParallel(model, device_ids=[0])
model = model.to(device)

classifier = model_classifier.Classifier().to(device)

train_loader, val_loader = dataset.create_generators()

root = './results/'
main_path = root + str(datetime.datetime.now().strftime('%Y-%m')) + "_crossEntropyLoss"
if not os.path.exists(main_path):
    os.mkdir(main_path)

classifier_path = main_path + '/' + 'classifier'

# Modify the code that creates directories to handle existing directories
if not os.path.exists(classifier_path):
    os.mkdir(classifier_path)
else:
    print(f"Directory {classifier_path} already exists.")

optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()),
                             lr=config.lr, weight_decay=1e-3)

scheduler = WarmUpExponentialLR(optimizer, cold_epochs=0, warm_epochs=config.warm_epochs, gamma=config.gamma)

def hotEncoder(v):
    ret_vec = torch.zeros(v.shape[0], config.class_numbers).to(device)
    for s in range(v.shape[0]):
        ret_vec[s][v[s]] = 1
    return ret_vec

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss(weight=class_weights)(input, labels)

###########################################################################################
class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).to(device)
main_model = model.module if hasattr(model, 'module') else model
###########################################################################################

def train_crossEntropy():
	num_epochs = 800
	with open(main_path + '/results.txt','w', 1) as output_file:
		mainModel_stopping = EarlyStopping(patience=300, verbose=True, log_path=main_path, output_file=output_file)
		classifier_stopping = EarlyStopping(patience=300, verbose=False, log_path=classifier_path, output_file=output_file)

		print('*****', file=output_file)
		print('BASELINE', file=output_file)
		print('transfer - augmentation on both waves and specs - 3 channels', file=output_file)
		if config.ESC_10:
			print('ESC_10', file=output_file)
			print('train folds are {} and test fold is {}'.format(config.train_folds, config.test_fold), file=output_file)
		elif config.ESC_50:
			print('ESC_50', file=output_file)
			print('train folds are {} and test fold is {}'.format(config.train_folds, config.test_fold), file=output_file)
		elif config.US8K:
			print('US8K', file=output_file)
			print('train folds are {} and test fold is {}'.format(config.us8k_train_folds, config.us8k_test_fold), file=output_file)


		print('number of freq masks are {} and their max length is {}'.format(config.freq_masks, config.freq_masks_width), file=output_file)
		print('number of time masks are {} and their max length is {}'.format(config.time_masks, config.time_masks_width), file=output_file)
		print('*****', file=output_file)
	


		for epoch in range(num_epochs):
			model.train()
			classifier.train()
        
			train_loss = []
			train_corrects = 0
			train_samples_count = 0
        
			for _, x, label in train_loader:
				loss = 0
				optimizer.zero_grad()
            
				inp = x.float().to(device)
				label = label.to(device).unsqueeze(1)
				label_vec = hotEncoder(label)
            
				y_rep = model(inp)
				y_rep = F.normalize(y_rep, dim=0)
            
				y_pred = classifier(y_rep)
            
				loss += cross_entropy_one_hot(y_pred, label_vec)
				loss.backward()
				train_loss.append(loss.item() )
				optimizer.step()
            
				train_corrects += (torch.argmax(y_pred, dim=1) == torch.argmax(label_vec, dim=1)).sum().item()
				train_samples_count += x.shape[0]
        
        
			val_loss = []
			val_corrects = 0
			val_samples_count = 0
        
			model.eval()
			classifier.eval()
        
			with torch.no_grad():
				for _, val_x, val_label in val_loader:
					inp = val_x.float().to(device)
					label = val_label.to(device)
					label_vec = hotEncoder(label)
                
					y_rep = model(inp)
					y_rep = F.normalize(y_rep, dim=0)

					y_pred = classifier(y_rep)
                
					temp = cross_entropy_one_hot(y_pred, label_vec)
					val_loss.append(temp.item() )
                
					val_corrects += (torch.argmax(y_pred, dim=1) == torch.argmax(label_vec, dim=1)).sum().item() 
					val_samples_count += val_x.shape[0]
        
		
        
			scheduler.step()
        
			train_acc = train_corrects / train_samples_count
			val_acc = val_corrects / val_samples_count
			print('\n', file=output_file)
			print("Epoch: {}/{}...".format(epoch+1, num_epochs), "Loss: {:.4f}...".format(np.mean(train_loss)),
				"Val Loss: {:.4f}".format(np.mean(val_loss)), file=output_file)
			print('train_acc is {:.4f} and val_acc is {:.4f}'.format(train_acc, val_acc), file=output_file)
			mainModel_stopping(-val_acc, main_model, epoch+1)
			classifier_stopping(-val_acc, classifier, epoch+1)
			if mainModel_stopping.early_stop:
				print("Early stopping", file=output_file)
				return


if __name__ == "__main__":
	train_crossEntropy()


# Correct the instantiation of the classifier object
#classifier = Classifier().to(device) # Ensure Classifier class is defined before this line

# Get the main model from the DataParallel module
main_model = model.module if hasattr(model, 'module') else model

# Load the checkpoints
main_model.load_state_dict(torch.load('results/2023-09_crossEntropyLoss10_f/checkpoint.pt'))
classifier.load_state_dict(torch.load('results/2023-09_crossEntropyLoss10_f/classifier/checkpoint.pt'))


import dataset_US8K as downstream_dataset