import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scheduler.GOBI import GOBIScheduler
from tabulate import tabulate

plt.style.use(['science'])
plt.rcParams["text.usetex"] = False

class Stats():
	def __init__(self, Environment, WorkloadModel, Datacenter, Scheduler):
		self.env = Environment
		self.env.stats = self
		self.workload = WorkloadModel
		self.datacenter = Datacenter
		self.scheduler = Scheduler
		powermodels, costs = ['PMB2s', 'PMB4ms', 'PMB8ms'], np.array([0.08, 0.17, 0.33]) / 12
		self.costs = dict(zip(powermodels, costs))
		self.initStats()

	def initStats(self):	
		self.hostinfo = []
		self.workloadinfo = []
		self.activecontainerinfo = []
		self.allcontainerinfo = []
		self.metrics = []
		self.schedulerinfo = []

	def saveHostInfo(self):
		hostinfo = dict()
		hostinfo['interval'] = self.env.interval
		hostinfo['cpu'] = [host.getCPU() for host in self.env.hostlist]
		hostinfo['numcontainers'] = [len(self.env.getContainersOfHost(i)) for i,host in enumerate(self.env.hostlist)]
		hostinfo['power'] = [host.getPower() for host in self.env.hostlist]
		hostinfo['baseips'] = [host.getBaseIPS() for host in self.env.hostlist]
		hostinfo['ipsavailable'] = [host.getIPSAvailable() for host in self.env.hostlist]
		hostinfo['ipscap'] = [host.ipsCap for host in self.env.hostlist]
		hostinfo['apparentips'] = [host.getApparentIPS() for host in self.env.hostlist]
		hostinfo['ram'] = [host.getCurrentRAM() for host in self.env.hostlist]
		hostinfo['ramavailable'] = [host.getRAMAvailable() for host in self.env.hostlist]
		hostinfo['disk'] = [host.getCurrentDisk() for host in self.env.hostlist]
		hostinfo['diskavailable'] = [host.getDiskAvailable() for host in self.env.hostlist]
		self.hostinfo.append(hostinfo)

	def saveWorkloadInfo(self, deployed, migrations):
		workloadinfo = dict()
		workloadinfo['interval'] = self.env.interval
		workloadinfo['totalcontainers'] = len(self.workload.createdContainers)
		if self.workloadinfo != []:
			workloadinfo['newcontainers'] = workloadinfo['totalcontainers'] - self.workloadinfo[-1]['totalcontainers'] 
		else:
			workloadinfo['newcontainers'] = workloadinfo['totalcontainers']
		workloadinfo['deployed'] = len(deployed)
		workloadinfo['migrations'] = len(migrations)
		workloadinfo['inqueue'] = len(self.workload.getUndeployedContainers())
		self.workloadinfo.append(workloadinfo)

	def saveContainerInfo(self):
		containerinfo = dict()
		containerinfo['interval'] = self.env.interval
		containerinfo['activecontainers'] = self.env.getNumActiveContainers()
		containerinfo['ips'] = [(c.getBaseIPS() if c else 0) for c in self.env.containerlist]
		containerinfo['apparentips'] = [(c.getApparentIPS() if c else 0) for c in self.env.containerlist]
		containerinfo['ram'] = [(c.getRAM() if c else 0) for c in self.env.containerlist]
		containerinfo['disk'] = [(c.getDisk() if c else 0) for c in self.env.containerlist]
		containerinfo['creationids'] = [(c.creationID if c else -1) for c in self.env.containerlist]
		containerinfo['hostalloc'] = [(c.getHostID() if c else -1) for c in self.env.containerlist]
		containerinfo['active'] = [(c.active if c else False) for c in self.env.containerlist]
		self.activecontainerinfo.append(containerinfo)

	def saveAllContainerInfo(self):
		containerinfo = dict()
		allCreatedContainers = [self.env.getContainerByCID(cid) for cid in list(np.where(self.workload.deployedContainers)[0])]
		containerinfo['interval'] = self.env.interval
		if self.datacenter.__class__.__name__ == 'Datacenter':
			containerinfo['application'] = [self.env.getContainerByCID(cid).application for cid in list(np.where(self.workload.deployedContainers)[0])]
		containerinfo['ips'] = [(c.getBaseIPS() if c.active else 0) for c in allCreatedContainers]
		containerinfo['create'] = [(c.createAt) for c in allCreatedContainers]
		containerinfo['start'] = [(c.startAt) for c in allCreatedContainers]
		containerinfo['destroy'] = [(c.destroyAt) for c in allCreatedContainers]
		containerinfo['apparentips'] = [(c.getApparentIPS() if c.active else 0) for c in allCreatedContainers]
		containerinfo['ram'] = [(c.getRAM() if c.active else 0) for c in allCreatedContainers]
		containerinfo['disk'] = [(c.getDisk() if c.active else 0) for c in allCreatedContainers]
		containerinfo['hostalloc'] = [(c.getHostID() if c.active else -1) for c in allCreatedContainers]
		containerinfo['active'] = [(c.active) for c in allCreatedContainers]
		self.allcontainerinfo.append(containerinfo)

	def saveMetrics(self, destroyed, migrations, pdecision):
		metrics = dict()
		metrics['interval'] = self.env.interval
		metrics['numdestroyed'] = len(destroyed)
		metrics['nummigrations'] = len(migrations)
		metrics['energy'] = [host.getPower()*self.env.intervaltime for host in self.env.hostlist]
		metrics['energytotalinterval'] = np.sum(metrics['energy'])
		metrics['energypercontainerinterval'] = np.sum(metrics['energy'])/self.env.getNumActiveContainers()
		metrics['responsetime'] = [c.totalExecTime + c.totalMigrationTime for c in destroyed]
		metrics['avgresponsetime'] = np.average(metrics['responsetime']) if len(destroyed) > 0 else 0
		metrics['migrationtime'] = [c.totalMigrationTime for c in destroyed]
		metrics['avgmigrationtime'] = np.average(metrics['migrationtime']) if len(destroyed) > 0 else 0
		metrics['slaviolations'] = len(np.where([c.destroyAt > c.sla for c in destroyed]))
		metrics['slaviolationspercentage'] = metrics['slaviolations'] * 100.0 / len(destroyed) if len(destroyed) > 0 else 0
		metrics['waittime'] = [c.startAt - c.createAt for c in destroyed]
		metrics['util'] = [host.getCPU()/100 for host in self.env.hostlist]
		metrics['cost'] = [self.costs[host.powermodel.__class__.__name__] for host in self.env.hostlist]
		metrics['added'] = len(pdecision['add'])
		metrics['time'] = sum([(1 if 'B2s' in m else 2 if 'B4ms' in m else 4) for m in pdecision['add']])
		self.metrics.append(metrics)

	def saveSchedulerInfo(self, selectedcontainers, decision, schedulingtime):
		schedulerinfo = dict()
		schedulerinfo['interval'] = self.env.interval
		schedulerinfo['selection'] = selectedcontainers
		schedulerinfo['decision'] = decision
		schedulerinfo['schedule'] = [(c.id, c.getHostID()) if c else (None, None) for c in self.env.containerlist]
		schedulerinfo['schedulingtime'] = schedulingtime
		if self.datacenter.__class__.__name__ == 'Datacenter':
			schedulerinfo['migrationTime'] = self.env.intervalAllocTimings[-1]
		self.schedulerinfo.append(schedulerinfo)

	def saveStats(self, deployed, migrations, destroyed, selectedcontainers, decision, pdecision, schedulingtime):	
		self.saveHostInfo()
		self.saveWorkloadInfo(deployed, migrations)
		self.saveContainerInfo()
		self.saveAllContainerInfo()
		self.saveMetrics(destroyed, migrations, pdecision)
		self.saveSchedulerInfo(selectedcontainers, decision, schedulingtime)

	########################################################################################################

	def generateGraphsWithInterval(self, dirname, listinfo, obj, metric, metric2=None):
		fig, axes = plt.subplots(len(listinfo[0][metric]), 1, sharex=True,figsize=(4, 0.5*len(listinfo[0][metric])))
		title = obj + '_' + metric + '_with_interval' 
		totalIntervals = len(listinfo)
		x = list(range(totalIntervals))
		metric_with_interval = []; metric2_with_interval = []
		ylimit = 0; ylimit2 = 0
		for hostID in range(len(listinfo[0][metric])):
			metric_with_interval.append([(0 if hostID >= len(listinfo[interval][metric]) else listinfo[interval][metric][hostID]) for interval in range(totalIntervals)])
			ylimit = max(ylimit, max(metric_with_interval[-1]))
			if metric2:
				metric2_with_interval.append([(0 if hostID >= len(listinfo[interval][metric2]) else listinfo[interval][metric][hostID]) for interval in range(totalIntervals)])
				ylimit2 = max(ylimit2, max(metric2_with_interval[-1]))
		for hostID in range(len(listinfo[0][metric])):
			axes[hostID].set_ylim(0, max(ylimit, ylimit2))
			axes[hostID].plot(x, metric_with_interval[hostID])
			if metric2:
				axes[hostID].plot(x, metric2_with_interval[hostID])
			axes[hostID].set_ylabel(obj[0].capitalize()+" "+str(hostID))
			axes[hostID].grid(b=True, which='both', color='#eeeeee', linestyle='-')
		plt.tight_layout(pad=0)
		plt.savefig(dirname + '/' + title + '.pdf')

	def generateMetricsWithInterval(self, dirname):
		x = list(range(len(self.metrics)))
		res = {}; table = []
		labels = ['Total Containers Run', 'Number of Migrations', 'Energy per interval', 'Response Time per interval',\
			'Migration Time per interval', 'SLA Violations per interval', 'SLA Violations (%) per interval', \
			'Waiting Time per interval', 'Energy per container per interval', 'Cost (USD) per interval', \
			'Number of Provisions', 'Provisioning Overhead (B2s time)', 'Utilization Ratio']
		fig, axes = plt.subplots(len(labels), 1, sharex=True, figsize=(4, 6))
		for i,metric in enumerate(['numdestroyed', 'nummigrations', 'energytotalinterval', 'avgresponsetime',\
			 'avgmigrationtime', 'slaviolations', 'slaviolationspercentage', 'waittime', 'energypercontainerinterval', 'cost',\
			 'added', 'time']):
			metric_with_interval = [self.metrics[i][metric] for i in range(len(self.metrics))] if metric != 'waittime' and metric != 'cost' else \
				[sum(self.metrics[i][metric]) for i in range(len(self.metrics))]
			axes[i].plot(x, metric_with_interval)
			axes[i].set_ylabel(metric, fontsize=5)
			axes[i].grid(b=True, which='both', color='#eeeeee', linestyle='-')
			res[metric] = sum(metric_with_interval)
			avg = np.average(metric_with_interval)
			std = np.std(metric_with_interval)
			if 'energy' in metric:
				avg /= 10e4; std /= 10e4
			if 'numdestroyed' in metric:
				avg *= 100
			table.append([labels[i], "{:.4f}".format(avg) + u" \u00B1" + f' {"{:.4f}".format(std)}'])
		metric_with_interval = [sum(self.hostinfo[i]['apparentips']) / sum(self.hostinfo[i]['ipscap']) for i in range(len(self.hostinfo))]
		avg = np.average(metric_with_interval); std = np.std(metric_with_interval)
		table.append([labels[-1], "{:.4f}".format(avg) + u"\u00B1" + f' {"{:.4f}".format(std)}'])
		print(tabulate(table, headers=['Metric', 'Value'], tablefmt='orgtbl'))
		print('Average energy (sum energy interval / sum numdestroyed) :', res['energytotalinterval']/res['numdestroyed']/10e6)
		plt.tight_layout(pad=0)
		plt.savefig(dirname + '/' + 'Metrics' + '.pdf')

	def generateWorkloadWithInterval(self, dirname):
		fig, axes = plt.subplots(5, 1, sharex=True, figsize=(4, 5))
		x = list(range(len(self.workloadinfo)))
		for i,metric in enumerate(['totalcontainers', 'newcontainers', 'deployed', 'migrations', 'inqueue']):
			metric_with_interval = [self.workloadinfo[i][metric] for i in range(len(self.workloadinfo))]
			axes[i].plot(x, metric_with_interval)
			axes[i].set_ylabel(metric)
			axes[i].grid(b=True, which='both', color='#eeeeee', linestyle='-')
		plt.tight_layout(pad=0)
		plt.savefig(dirname + '/' + 'Workload' + '.pdf')

	########################################################################################################

	def generateCompleteDataset(self, dirname, data, name):
		title = name + '_with_interval' 
		metric_with_interval = []
		headers = list(data[0].keys())
		for datum in data:
			metric_with_interval.append([datum[value] for value in datum.keys()])
		df = pd.DataFrame(metric_with_interval, columns=headers)
		df.to_csv(dirname + '/' + title + '.csv', index=False)

	def generateDatasetWithInterval(self, dirname, metric, objfunc, metric2=None, objfunc2=None):
		title = metric + '_' + (metric2 + '_' if metric2 else "") + (objfunc + '_' if objfunc else "") + (objfunc2 + '_' if objfunc2 else "") + 'with_interval' 
		totalIntervals = len(self.hostinfo)
		metric_with_interval = []; metric2_with_interval = [] # metric1 is of host and metric2 is of containers
		host_alloc_with_interval = []; objfunc2_with_interval = []
		objfunc_with_interval = []
		for interval in range(totalIntervals-1):
			metric_with_interval.append([self.hostinfo[interval][metric][hostID] for hostID in range(len(self.hostinfo[0][metric]))])
			host_alloc_with_interval.append([self.activecontainerinfo[interval]['hostalloc'][cID] for cID in range(len(self.activecontainerinfo[0]['hostalloc']))])
			objfunc_with_interval.append(self.metrics[interval+1][objfunc])
			if metric2:
				metric2_with_interval.append(self.activecontainerinfo[interval][metric2])
			if objfunc2:
				objfunc2_with_interval.append(self.metrics[interval+1][objfunc2])
		df = pd.DataFrame(metric_with_interval)
		if metric2: df = pd.concat([df, pd.DataFrame(metric2_with_interval)], axis=1)
		df = pd.concat([df, pd.DataFrame(host_alloc_with_interval)], axis=1)
		df = pd.concat([df, pd.DataFrame(objfunc_with_interval)], axis=1)
		if objfunc2: df = pd.concat([df, pd.DataFrame(objfunc2_with_interval)], axis=1)
		df.to_csv(dirname + '/' + title + '.csv' , header=False, index=False)

	def generateDatasetWithInterval2(self, dirname, metric):
		title = metric + '_' + 'with_interval' 
		totalIntervals = len(self.activecontainerinfo)
		metric_with_interval = []
		for interval in range(totalIntervals-1):
			metric_with_interval.append([self.activecontainerinfo[interval][metric][cid] for cid in range(len(self.activecontainerinfo[0][metric]))])
		df = pd.DataFrame(metric_with_interval)
		df.to_csv(dirname + '/' + title + '.csv' , header=False, index=False)

	def generateGraphs(self, dirname):
		self.generateGraphsWithInterval(dirname, self.hostinfo, 'host', 'cpu')
		# self.generateGraphsWithInterval(dirname, self.hostinfo, 'host', 'numcontainers')
		# self.generateGraphsWithInterval(dirname, self.hostinfo, 'host', 'power')
		# self.generateGraphsWithInterval(dirname, self.hostinfo, 'host', 'baseips', 'apparentips')
		# self.generateGraphsWithInterval(dirname, self.hostinfo, 'host', 'ipscap', 'apparentips')
		# self.generateGraphsWithInterval(dirname, self.activecontainerinfo, 'container', 'ips', 'apparentips')
		# self.generateGraphsWithInterval(dirname, self.activecontainerinfo, 'container', 'hostalloc')
		self.generateMetricsWithInterval(dirname)
		# self.generateWorkloadWithInterval(dirname)

	def generateDatasets(self, dirname):
		# self.generateDatasetWithInterval(dirname, 'cpu', objfunc='energytotalinterval')
		# self.generateDatasetWithInterval(dirname, 'cpu', metric2='apparentips', objfunc='energytotalinterval', objfunc2='avgresponsetime')
		self.generateDatasetWithInterval2(dirname, 'apparentips')

	def generateCompleteDatasets(self, dirname):
		# self.generateCompleteDataset(dirname, self.hostinfo, 'hostinfo')
		# self.generateCompleteDataset(dirname, self.workloadinfo, 'workloadinfo')
		# self.generateCompleteDataset(dirname, self.metrics, 'metrics')
		# self.generateCompleteDataset(dirname, self.activecontainerinfo, 'activecontainerinfo')
		# self.generateCompleteDataset(dirname, self.allcontainerinfo, 'allcontainerinfo')
		# self.generateCompleteDataset(dirname, self.schedulerinfo, 'schedulerinfo')
		pass