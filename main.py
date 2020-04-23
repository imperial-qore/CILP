from environment.Fog.SimpleFog import *
from workload.StaticWorkload_StaticDistribution import *
from environment.Environment import *
from scheduler.Random import *

# Global constants
NUM_SIM_STEPS = 100
HOSTS = 10
CONTAINERS = 10
TOTAL_POWER = 1000
ROUTER_BW = 10000
INTERVAL_TIME = 300 # seconds
NEW_CONTAINERS = 2

def initalizeEnvironment(env, workload, datacenter):
	hostlist = datacenter.generateHosts()

if __name__ == '__main__':
	datacenter = SimpleFog(HOSTS)
	workload = StaticWorkload_StaticDistribution(NEW_CONTAINERS)
	scheduler = RandomScheduler()
	env = Environment(TOTAL_POWER, ROUTER_BW, scheduler, CONTAINERS, HOSTS, INTERVAL_TIME)

	for step in range(NUM_SIM_STEPS):
