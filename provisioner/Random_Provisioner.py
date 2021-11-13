from .Provisioner import *
import random

class RandomProvisioner(Provisioner):
	def __init__(self, datacenter, CONTAINERS):
		super().__init__(datacenter, CONTAINERS)

	def prediction(self):
		pass

	def provision(self):
		orphaned = []
		if random.choice([0, 1]):
			removeID = random.choice(list(range(len(self.env.hostlist))))
			orphaned = self.removeHost(removeID)
			print('Removed host', removeID)
		if random.choice([0, 1]):
			hostlist = self.datacenter.generateHosts()
			newhost = random.choice(hostlist)
			self.addHost(newhost)
			print('Added host')
		self.migrateOrphaned(orphaned)
		return orphaned