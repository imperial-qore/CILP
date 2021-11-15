from .Provisioner import *
from .src.utils import *
from .src.opt import *
from scheduler.HGP.train import *
from simulator.environment.AzureFog import *

class UAHSProvisioner(Provisioner):
	def __init__(self, datacenter, CONTAINERS):
		super().__init__(datacenter, CONTAINERS)
		self.model_name = 'UAHS'
		self.search = StochasticSearch
		self.model_loaded = False
		self.window_buffer = []
		self.window = None
		self.load_model()

	def load_model(self):
		# Load dataset
		dataset, _, self.minv, self.maxv = load_dataset('apparentips_with_interval.csv')
		# Load model
		X = np.array([np.array(i).reshape(-1) for i in dataset])
		y = np.roll(X, 1, axis=0); print(X.shape, y.shape)
		kernel_hetero = C(1.0, (1e-10, 1000)) * RBF(0.5, (0.00, 100.0)) 
		self.model = [GaussianProcessRegressor(kernel=kernel_hetero, alpha=0.01) for _ in range(X.shape[1])]
		file_path = base_url + f'checkpoints/UAHS.pt'
		if os.path.exists(file_path):
			print(color.GREEN+"Loading pre-trained model: UAHS"+color.ENDC)
			with open(file_path, 'rb') as f:
				self.model = pickle.load(f)
		else:
			print(color.GREEN+"Creating new model: UAHS"+color.ENDC)
			for i in range(X.shape[1]):
				dx, dy = X[:, i].reshape(X.shape[0], -1), y[:, i].reshape(X.shape[0], -1)
				self.model[i] = self.model[i].fit(dx, dy)
			with open(file_path, 'wb') as f:
				pickle.dump(self.model, f)
		self.model_loaded = True

	def updateBuffer(self):
		ips_data = [c.getApparentIPS() if c else self.minv for c in self.env.containerlist]
		temp = np.array(ips_data)
		temp = normalize(temp, self.minv, self.maxv)
		self.window = temp

	def prediction(self):
		self.updateBuffer()
		pred, std = [], []
		for i in range(self.window.shape[1]):
			dx = np.array([self.window[i]])
			p, s = self.model.predict(dx.reshape(1, -1), return_std=True)
			pred.append(p); std.append(s)
		pred = denormalize(pred, self.minv, self.maxv)[0]
		return pred, std[0]

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