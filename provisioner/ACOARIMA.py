from .Provisioner import *
from .src.utils import *
from .src.opt import *
from simulator.environment.AzureFog import *

class ACOARIMAProvisioner(Provisioner):
	def __init__(self, datacenter, CONTAINERS):
		super().__init__(datacenter, CONTAINERS)
		self.search = ACO
		self.model_loaded = False
		self.window_buffer = []
		self.window = None
		self.load_model()

	def load_model(self):
		dataset, _, self.minv, self.maxv = load_dataset('apparentips_with_interval.csv')
		self.model_loaded = True

	def updateBuffer(self):
		ips_data = [c.getApparentIPS() if c else self.minv for c in self.env.containerlist]
		temp = np.array(ips_data)
		temp = normalize(temp, self.minv, self.maxv)
		self.window = temp

	def prediction(self):
		self.updateBuffer()
		pred = self.window
		pred = denormalize(pred, self.minv, self.maxv)
		return pred.tolist()

	def provision(self):
		predips = self.prediction()
		opt = self.search(predips, self.env, self.maxv)
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