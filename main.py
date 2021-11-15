import os, sys, stat
import sys
import optparse
import logging as logger
import configparser
import pickle
import shutil
import sqlite3
import platform
from time import time
from subprocess import call
from os import system, rename

# Simulator imports
from simulator.Simulator import *
from simulator.environment.AzureFog import *
from simulator.environment.BitbrainFog import *
from simulator.workload.BitbrainWorkload_GaussianDistribution import *
from simulator.workload.BitbrainWorkload2 import *
from simulator.workload.Azure2017Workload import *
from simulator.workload.Azure2019Workload import *

# Scheduler imports
from scheduler.IQR_MMT_Random import IQRMMTRScheduler
from scheduler.MAD_MMT_Random import MADMMTRScheduler
from scheduler.MAD_MC_Random import MADMCRScheduler
from scheduler.LR_MMT_Random import LRMMTRScheduler
from scheduler.Random_Random_FirstFit import RFScheduler
from scheduler.Random_Random_LeastFull import RLScheduler
from scheduler.Threshold_MMT_Random import TMMTRScheduler
from scheduler.Threshold_MMT_LeastFull import TMMTLScheduler
from scheduler.RLR_MMT_Random import RLRMMTRScheduler
from scheduler.Threshold_MC_Random import TMCRScheduler
from scheduler.Random_Random_Random import RandomScheduler
from scheduler.HGP_LBFGS import HGPScheduler

# Provisioner imports
from provisioner.Provisioner import Provisioner
from provisioner.Random_Provisioner import RandomProvisioner
from provisioner.DecisionNN import DecisionNNProvisioner
from provisioner.ACOLSTM import ACOLSTMProvisioner
from provisioner.ACOARIMA import ACOARIMAProvisioner
from provisioner.UAHS import UAHSProvisioner
from provisioner.CAHS import CAHSProvisioner
from provisioner.SemiDirect import SemiDirectProvisioner
from provisioner.Narya import NaryaProvisioner
from provisioner.CILP import CILPProvisioner
from provisioner.CILP_IL import CILP_ILProvisioner
from provisioner.CILP_Trans import CILP_TransProvisioner

# Auxiliary imports
from stats.Stats import *
from utils.Utils import *
from pdb import set_trace as bp
from sys import argv
import argparse

usage = "usage: python main.py -provisioner <provisioner> -workload <workload>"
parser = argparse.ArgumentParser(usage=usage)
parser.add_argument('--provisioner', 
                    help='Name of provisioner. One of ACOARIMA, ACOLSTM, DecisionNN, SemiDirect, UAHS, Narya, CAHS, or CILP, CILP_IL, CILP_Trans.')
parser.add_argument('--workload', 
                    help='Name of workload. One of Azure2017, Azure2019 or Bitbrain.')
args = parser.parse_args()

# Global constants
NUM_SIM_STEPS = 200
HOSTS = 10 * 5
CONTAINERS = HOSTS
TOTAL_POWER = 1000
ROUTER_BW = 10000
INTERVAL_TIME = 300 # seconds
NEW_CONTAINERS = 7

# Proposed: CILP. Ablations: CILP_IL, CILP_Trans
# Baselines: ACOARIMA, ACOLSTM, DecisionNN, SemiDirect, UAHS, Narya, CAHS

def initalizeEnvironment(environment, logger):
	# Initialize simple fog datacenter
	''' Can be SimpleFog, BitbrainFog, AzureFog '''
	datacenter = AzureFog(HOSTS)

	# Initialize workload
	''' Can be Bitbrain, Azure2017, Azure2019 '''
	workload = eval(args.workload + 'Workload')(NEW_CONTAINERS, 1.5)
	
	# Initialize scheduler
	''' Can be LRMMTR, RF, RL, RM, Random, RLRMMTR, TMCR, TMMR, TMMTR, GA, GOBI (arg = 'energy_latency_'+str(HOSTS)) '''
	scheduler = RLScheduler() 

	# Initialize provisioner
	''' Can be CILP, ACOARIMA, ACOLSTM, DecisionNN, SemiDirect, UAHS, Narya, CAHS '''
	provisioner = eval(args.provisioner + 'Provisioner')(datacenter, CONTAINERS)

	# Initialize Environment
	hostlist = datacenter.generateHosts()
	env = Simulator(TOTAL_POWER, ROUTER_BW, scheduler, provisioner, CONTAINERS, INTERVAL_TIME, hostlist)

	# Execute first step
	newcontainerinfos = workload.generateNewContainers(env.interval) # New containers info
	deployed = env.addContainersInit(newcontainerinfos) # Deploy new containers and get container IDs
	start = time()
	decision = scheduler.placement(deployed) # Decide placement using container ids
	schedulingTime = time() - start
	migrations = env.allocateInit(decision) # Schedule containers
	workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed)) # Update workload allocated using creation IDs
	print("Deployed containers' creation IDs:", env.getCreationIDs(migrations, deployed))
	print("Containers in host:", env.getContainersInHosts())
	print("Schedule:", env.getActiveContainerList())
	printDecisionAndMigrations(decision, migrations)

	# Initialize stats
	stats = Stats(env, workload, datacenter, scheduler)
	stats.saveStats(deployed, migrations, [], deployed, decision, provisioner.decision, schedulingTime)
	return datacenter, workload, scheduler, provisioner, env, stats

def stepSimulation(workload, scheduler, provisioner, env, stats):
	newcontainerinfos = workload.generateNewContainers(env.interval) # New containers info
	pdecision, orphaned = provisioner.provision()
	deployed, destroyed = env.addContainers(newcontainerinfos) # Deploy new containers and get container IDs
	start = time()
	selected = scheduler.selection() # Select container IDs for migration
	decision = scheduler.filter_placement(scheduler.placement(selected+deployed)) # Decide placement for selected container ids
	schedulingTime = time() - start
	migrations = env.simulationStep(decision) # Schedule containers
	workload.updateDeployedContainers(env.getCreationIDs(migrations, deployed)) # Update workload deployed using creation IDs
	print("Deployed containers' creation IDs:", env.getCreationIDs(migrations, deployed))
	print("Deployed:", len(env.getCreationIDs(migrations, deployed)), "of", len(newcontainerinfos), [i[0] for i in newcontainerinfos])
	print("Destroyed:", len(destroyed), "of", env.getNumActiveContainers())
	print("Containers in host:", env.getContainersInHosts())
	print("Num active containers:", env.getNumActiveContainers())
	print("Host allocation:", [(c.getHostID() if c else -1) for c in env.containerlist])
	print("Num Hosts:", len(env.hostlist))
	printDecisionAndMigrations(decision, migrations)

	stats.saveStats(deployed, migrations, destroyed, selected, decision, pdecision, schedulingTime)

def saveStats(stats, datacenter, workload, env, end=True):
	dirname = "logs/" + datacenter.__class__.__name__
	dirname += "_" + workload.__class__.__name__
	dirname += "_" + str(NUM_SIM_STEPS) 
	dirname += "_" + str(HOSTS)
	dirname += "_" + str(CONTAINERS)
	dirname += "_" + str(TOTAL_POWER)
	dirname += "_" + str(ROUTER_BW)
	dirname += "_" + str(INTERVAL_TIME)
	dirname += "_" + str(NEW_CONTAINERS)
	if not os.path.exists("logs"): os.mkdir("logs")
	if os.path.exists(dirname): shutil.rmtree(dirname, ignore_errors=True)
	os.mkdir(dirname)
	if not end: return
	stats.generateDatasets(dirname)
	stats.generateGraphs(dirname)
	# stats.generateCompleteDatasets(dirname)
	stats.env, stats.workload, stats.datacenter, stats.scheduler = None, None, None, None
	with open(dirname + '/' + dirname.split('/')[1] +'.pk', 'wb') as handle:
	    pickle.dump(stats, handle)

if __name__ == '__main__':
	env, mode = '', 0
	datacenter, workload, scheduler, provisioner, env, stats = initalizeEnvironment(env, logger)

	for step in range(NUM_SIM_STEPS):
		print(color.BOLD+"Simulation Interval:", step, color.ENDC)
		stepSimulation(workload, scheduler, provisioner, env, stats)
		saveStats(stats, datacenter, workload, env, False)

	saveStats(stats, datacenter, workload, env)
