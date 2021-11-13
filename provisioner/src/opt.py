import numpy as np
from copy import deepcopy
import random
from .constants import *

class Opt:
	def __init__(self, ipsdata, env, maxv):
		self.env = env
		self.maxv = maxv
		self.allpowermodels = ['PMB2s', 'PMB4ms', 'PMB8ms']
		costs = np.array([0.08, 0.17, 0.33]) / 12
		ipscaps = [2019, 4029, 16111]
		self.costs = dict(zip(self.allpowermodels, costs))
		self.ipscaps = dict(zip(self.allpowermodels, ipscaps))
		self.ipsdata = ipsdata
		self.decision = {'remove': [], 'add': []}

	def checkdecision(self, decision):
		return not (\
			len(decision['remove']) > remove_limit or \
			len(decision['add']) > add_limit or \
			len(self.env.hostlist) - len(decision['remove']) < 40 or \
			len(self.env.hostlist) - len(decision['remove']) > 60)

	def neighbours(self, decision):
		neighbours = []; numadds = 0
		# add host
		for pmodel in self.allpowermodels:
			dec = deepcopy(decision)
			dec['add'].append(pmodel)
			if self.checkdecision(dec):
				neighbours.append(dec); numadds += 1
		# remove host
		for hostID in range(len(self.env.hostlist)):
			dec = deepcopy(decision)
			if hostID in [i[0] for i in dec['remove']]: continue
			orphaned = self.env.getContainersOfHost(hostID)
			alloc = self.migrateOrphaned(orphaned, hostID, len(dec['add']))
			dec['remove'].append((hostID, alloc))
			if self.checkdecision(dec):
				neighbours.append(dec)
		neighbours.append(deepcopy(decision))
		return neighbours, numadds

	def migrateOrphaned(self, orphaned, inithostid, numnewnodes):
		indices = list(range(len(self.env.hostlist) + numnewnodes))
		indices.remove(inithostid)
		alloc = []
		for o in orphaned:
			random.shuffle(indices)
			for hostID in indices:
				if hostID >= len(self.env.hostlist) or self.env.getPlacementPossible(o, hostID):
					alloc.append((o, hostID))
					break
		return alloc

	def evaluatedecision(self, decision):
		host_alloc = []
		for hostID in range(len(self.env.hostlist)):
			host_alloc.append([])
		for c in self.env.containerlist:
			if c and c.getHostID() != -1: 
				host_alloc[c.getHostID()].append(c.id) 
		new_hids = []; old_hids = list(range(len(host_alloc)))
		# Add new hosts
		for h in decision['add']:
			new_hids.append(len(host_alloc))
			host_alloc.append([])
		new_hids = np.array(new_hids)
		# Migrate orphans
		for dec in decision['remove']:
			for cid, hid in dec[1]:
				host_alloc[hid].append(cid)
		# Remove hosts
		indices = []
		for dec in decision['remove']:
			inithid = dec[0]
			old_hids.remove(inithid)
			indices.append(inithid - len(indices))
			new_hids -= 1
		for i in indices: del host_alloc[i]
		# Balance IPS by migrating to new hosts
		allcaps = [self.env.hostlist[hid].ipsCap for hid in old_hids] + [self.ipscaps[nid] for nid in decision['add']]
		for hid, cids in enumerate(host_alloc):
			if sum([self.ipsdata[cid] for cid in cids]) > allcaps[hid]:
				cid = host_alloc[hid][np.argmax([self.ipsdata[c] for c in host_alloc[hid]])]
				host_alloc[hid].remove(cid)
				fromlist = new_hids if new_hids.shape[0] > 0 else list(range(len(host_alloc))) 
				hid = np.random.choice(fromlist)
				host_alloc[hid].append(cid)
		# Calculate cost
		allpmodels = [host.powermodel.__class__.__name__ for host in self.env.hostlist] + decision['add']
		cost = sum([self.costs[pmodel] for pmodel in allpmodels]) 
		r = sum(self.ipsdata) / sum(allcaps)
		capsr = sum(allcaps) / self.maxv 
		overhead = sum([(1 if 'B2s' in m else 2 if 'B4ms' in m else 4) for m in decision['add']])
		# print(decision, ', Cost', cost, ', R', r, ', IPS', capsr, 'Overhead', overhead)
		return r - 0.5 * cost + 0.4 * capsr - 0.5 * overhead

	def getweights(self, fitness, adds):
		removes = len(fitness) - adds - 1
		weights = np.array([0.5 / (adds+1e-4)] * adds + [0.5 / (removes+1e-4)] * removes + [0.5])
		return weights / np.sum(weights)

class LocalSearch(Opt):
	def __init__(self, ipsdata, env, maxv):
		super().__init__(ipsdata, env, maxv)

	def search(self):
		oldfitness, newfitness = 0, 1
		for _ in range(50):
			if newfitness < oldfitness: break
			oldfitness = newfitness
			neighbourhood, numadds = self.neighbours(self.decision)
			if neighbourhood == []: break
			fitness = [self.evaluatedecision(n) for n in neighbourhood]
			index = np.random.choice(list(range(len(fitness))), p=self.getweights(fitness, numadds)) \
				if np.random.random() < 0.4 else np.argmax(fitness)
			self.decision = neighbourhood[index]
			newfitness = fitness[index]
		return self.decision

class ACO(Opt):
	def __init__(self, ipsdata, env, maxv):
		super().__init__(ipsdata, env, maxv)
		self.n = 5

	def search(self):
		oldfitness = [0] * self.n; newfitness = [1] * self.n
		self.decisions = [{'remove': [], 'add': []} for _ in range(self.n)]
		for _ in range(50):
			for ant in range(self.n):
				if newfitness[ant] < oldfitness[ant]: continue
				oldfitness[ant] = newfitness[ant]
				neighbourhood, numadds = self.neighbours(self.decisions[ant])
				if neighbourhood == []: continue
				fitness = [self.evaluatedecision(n) for n in neighbourhood]
				if random.choice([0, 1]): continue
				index = np.random.choice(list(range(len(fitness))), p=self.getweights(fitness, numadds)) \
					if np.random.random() < 0.4 else np.argmax(fitness)
				self.decisions[ant] = neighbourhood[index]
				newfitness[ant] = fitness[index]
		return self.decisions[np.argmax(newfitness)]