from .Provisioner import *
from .src.utils import *
from .src.opt import *
from simulator.environment.AzureFog import *

class SemiDirectProvisioner(Provisioner):
	def __init__(self, datacenter, CONTAINERS):
		super().__init__(datacenter, CONTAINERS)
		self.model_name = 'NPN'
		self.search = StochasticSearch
		self.model_loaded = False
		self.window_buffer = []
		self.window = None
		self.load_model()

	def load_model(self):
		# Load model
		self.model, optimizer, scheduler, epoch, loss_list = load_model(self.model_name, self.containers)
		# Load dataset
		trainO, testO, self.minv, self.maxv = load_dataset('apparentips_with_interval.csv')
		trainD, testD = convert_to_windows(trainO, self.model), convert_to_windows(testO, self.model)
		# Train model
		if epoch == -1:
			for e in tqdm(list(range(epoch+1, epoch+num_epochs+1))):
				lossT, lr = backprop(e, self.model, trainD, trainO, optimizer, scheduler)
				lossTest, _ = backprop(e, self.model, testD, testO, optimizer, scheduler, False)
				loss_list.append((lossT, lossTest, lr))
				tqdm.write(f'Epoch {e},\tTrain Loss = {lossT},\tTest loss = {lossTest}')
				plot_accuracies(loss_list, base_url, self.model)
			save_model(self.model, optimizer, scheduler, e, loss_list)
		# Freeze encoder
		freeze(self.model); self.model_loaded = True

	def updateBuffer(self):
		ips_data = [c.getApparentIPS() if c else self.minv for c in self.env.containerlist]
		self.window_buffer.append(ips_data)
		temp = np.array(self.window_buffer)
		temp = normalize(temp, self.minv, self.maxv)
		self.window = convert_to_windows(temp, self.model)[-1]

	def prediction(self):
		self.updateBuffer()
		pred, std = self.model(self.window)
		pred = denormalize(pred, self.minv, self.maxv)
		return pred.tolist()[0], std.tolist()[0]

	def provision(self):
		predips, stdips = self.prediction()
		opt = self.search(predips, stdips, self.env, self.maxv)
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