from .Provisioner import *
from .src.utils import *
from .src.opt import *
from simulator.environment.AzureFog import *

class CILPProvisioner(Provisioner):
	def __init__(self, datacenter, CONTAINERS):
		super().__init__(datacenter, CONTAINERS)
		self.model_name = 'CILP'
		self.search = CILPSearch
		self.model_loaded = False
		self.window_buffer = []
		self.host_util = None
		self.window = None
		self.training = False
		self.load_model()

	def load_model(self):
		# Load model
		self.model, self.optimizer, self.scheduler, epoch, self.loss_list = load_model(self.model_name, self.containers)
		# Load dataset
		trainO, testO, self.minv, self.maxv = load_dataset('apparentips_with_interval.csv')
		trainD, testD = convert_to_windows(trainO, self.model), convert_to_windows(testO, self.model)
		# Train model for ips data prediction
		if epoch == -1 or self.training:
			for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
				lossT, lr = backprop(e, self.model, trainD, trainO, self.optimizer, self.scheduler)
				lossTest, _ = backprop(e, self.model, testD, testO, self.optimizer, self.scheduler, False)
				self.loss_list.append((lossT, lossTest, lr))
				tqdm.write(f'Epoch {e},\tTrain Loss = {lossT},\tTest loss = {lossTest}')
				plot_accuracies(self.loss_list, base_url, self.model)
			save_model(self.model, self.optimizer, self.scheduler, e, self.loss_list)
		# Freeze encoder
		if not self.training: freeze(self.model)
		self.model_loaded = True

	def updateBuffer(self):
		ips_data = [c.getApparentIPS() if c else self.minv for c in self.env.containerlist]
		self.window_buffer.append(ips_data)
		temp = np.array(self.window_buffer)
		temp = normalize(temp, self.minv, self.maxv)
		self.window = convert_to_windows(temp, self.model)[-1]

	def prediction(self):
		self.updateBuffer()
		self.host_util = [h.getCPU() for h in self.env.hostlist]
		feats = self.window.shape[1]
		d = self.window[None, :]
		window = d.permute(1, 0, 2)
		elem = window[-1, :, :].view(1, 1, feats)
		pred, _ = self.model(window, elem, torch.tensor([sum(self.host_util)]))
		pred = pred.view(-1)
		pred = denormalize(pred, self.minv, self.maxv)
		return pred.tolist()

	def provision(self):
		predips = self.prediction()
		opt = self.search(predips, self.env, self.maxv, self.window_buffer, self.host_util, self.model, self.optimizer, self.scheduler, self.loss_list, self.training)
		decision = opt.search()
		print(decision)
		for add in decision['add']:
			typeID = add.replace('PM', '')
			IPS = self.datacenter.types[typeID]['IPS']
			Ram = RAM(self.datacenter.types[typeID]['RAMSize'], self.datacenter.types[typeID]['RAMRead']*5, self.datacenter.types[typeID]['RAMWrite']*5)
			Disk_ = Disk(self.datacenter.types[typeID]['DiskSize'], self.datacenter.types[typeID]['DiskRead']*5, self.datacenter.types[typeID]['DiskWrite']*10)
			Bw = Bandwidth(self.datacenter.types[typeID]['BwUp'], self.datacenter.types[typeID]['BwDown'])
			Power = eval(self.datacenter.types[typeID]['Power']+'()')
			Latency = 0.003 if typeID == 'B2s' else 0.076
			newhost = (IPS, Ram, Disk_, Bw, Latency, Power)
			self.addHost(newhost)
		orphaned = []; indices = []
		for remove in decision['remove']:
			if remove == []: continue
			removeID = remove[0]
			indices.append(removeID - len(indices))
		for removeID in indices:
			orphaned = self.removeHost(removeID)
		self.migrateOrphaned(orphaned)
		return decision, orphaned