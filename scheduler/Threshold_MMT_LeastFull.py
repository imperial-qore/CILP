from .Scheduler import *
import numpy as np
from copy import deepcopy

class TMMTLScheduler(Scheduler):
	def __init__(self):
		super().__init__()

	def selection(self):
		selectedHostIDs = self.ThresholdHostSelection()
		selectedVMIDs = self.MMTContainerSelection(selectedHostIDs)
		return selectedVMIDs

	def placement(self, containerIDs):
		return self.LeastFullPlacement(containerIDs)
