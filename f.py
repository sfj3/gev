import numpy as np
from numpy import inf
from scipy.stats import genextreme as gev
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.special import gamma
import torch
device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'

def ll(params):
	loss=0
	shape=params[0]
	scale = params[1]
	loc=params[2]#-0.0001
	if scale>0:
		b = np.array(gev.ppf(xc, c=shape,loc=loc,scale=scale))
		loss = np.sum([0.1 for i in b if i <= 0])
		b = torch.from_numpy(b)#.to(device='cuda')
		loss+=torch.log(torch.sum((x+(x+0.6)**10)*torch.abs(b - liss1)**2)+0.5*(torch.std(b)-torch.std(liss1))**2)
		return loss.item()# add one for log likelihood 
	else:
		return 10000
	
class MyTakeStep:
	def __init__(self, stepsize=1):
	   self.stepsize = stepsize
	   self.rng = np.random.default_rng()
	def __call__(self, x):
	   self.stepsize = self.stepsize*0.99#annealing 4.347851452142499
	   s = self.stepsize
	   if x[0]>3 or x[0]<3:
	   	x[0]=0.001
	   x[0] = -1*x[0]+torch.tensor(self.rng.normal(0.11,0.01*s)).float() #shape
	   x[2]= torch.tensor(self.rng.normal(x[2], 10*s)).float() #loc
	   x[1] = torch.abs(torch.tensor(self.rng.normal(x[1],3*s)).float()) #scale
	   self.x= torch.nan_to_num(torch.from_numpy(x),nan=self.rng.normal(1,1)).cpu().detach().numpy()
	   self.x = x
	   return x.tolist()

durations = [1]#days
file_paths = [
	"./Los Angeles_time_series.txt",#line by line precip block maxima in mm
]
for idx, file_path in enumerate(file_paths):
	b=0.9999#use any support
	a=0.1
	with open(file_path, 'r') as file:
	   data = [float(line.strip()) for line in file]
	   gg = np.array(data)  # Convert list to numpy array
	   xc = np.linspace(a,b, len(gg)) # we train on linspace you may be able to in logspace
	   x = torch.from_numpy(xc)#.to(device='cuda')
	   liss1=(torch.from_numpy(np.sort(gg))/(durations[0]))#.to(device='cuda')#, my cpu runs it faster than my gpu but this isincase
	shape=-0.01#initial estimate
	scale=1
	loc=5
	x1=[shape,scale,loc]
	minimizer_kwargs = {"method": "BFGS"}
	mytakestep = MyTakeStep()
	ret = basinhopping(ll,x1,minimizer_kwargs=minimizer_kwargs,niter=100,take_step=mytakestep)
	fig, ax = plt.subplots()
	x = np.linspace(a,b, len(gg))
	xx = np.log(np.divide(1,np.ones_like(x)-x))
	res1 = np.array(gev.ppf(x, ret.x[0], ret.x[2], ret.x[1])) #these are the estimated params
