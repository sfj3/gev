import numpy as np
from numpy import inf
from scipy.stats import genextreme as gev
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy.special import gamma
import torch
device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
import matplotlib.ticker as ticker
years = np.array([2, 50, 100, 250, 500, 1000])#years to show on plot axis
transformed_x_values = np.log(years)
# Define the years for which you want tick marks
def log_format(x, pos):
    """Custom formatter to return the exponent of the log scale."""
    return f'{np.exp(x):.0f}'
#need to do the rest of them cause I missed some
# if dict contains... then dont do the simulation
# see how to write to an existing dictionary
def ll(params):
	loss=0
	shape=params[0]
	scale = params[1]
	loc=params[2]#-0.0001
	if scale>0 and scale1>0:
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
	   #self.rng1 = np.random.default_rng()
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

durations = [1,2,3,4,7,10,20,30,45,60]

#gg = np.load("rcp4.npy")#rcp8 in the right tab
#print(gg.shape)
#gg = np.swapaxes(gg,0,3)
#gg = np.swapaxes(gg,1,2)
#gg = np.swapaxes(gg,1,0)
#print(gg.shape)
tab = {}
"""below you will enter any time series
the format of them will be raw data, line by line per block maxima ie
12
13
14
"""
file_paths = [
	"./Fresno_time_series.txt",
	"./Los Angeles_time_series.txt",
	"./Sacramento_time_series.txt",
	"./San Diego_time_series.txt",
	"./San Francisco_time_series.txt",
	"./San Jose_time_series.txt"
]
gev_params = {#any sets of params you want to compare to the learned data
    'Fresno': {
        'Stat': [0.0279537396334234, 1.83688399394871, 22.5678984603338],
        'Nonstat': [0.0317933140707047, 1.83991338695591, 22.9551301357531]
    },
    'Los Angeles': {
        'Stat': [-0.0118036096075670, 2.97414331328946, 40.9138086279802],
        'Nonstat': [-0.0177458225240232, 2.97994729403216, 40.8917524517012]
    },
    'Sacramento': {
        'Stat': [0.00788121850664761, 2.40347809268535, 33.4739259756107],
        'Nonstat': [0.0114845676916233, 2.41332692499420, 33.2778495697244]
    },
    'San Diego': {
        'Stat': [0.0330518047348983, 2.28414761697310, 25.3797609335986],
        'Nonstat': [0.0361723417609112, 2.28483993591864, 25.4768347770986]
    },
    'San Francisco': {
        'Stat': [0.0136980578925005, 2.35163854800503, 35.9392084612063],
        'Nonstat': [0.0119752942324680, 2.35066604999109, 35.9273593279362]
    },
    'San Jose': {
        'Stat': [-0.0318089686977196, 2.02160547404123, 23.8638059751154],
        'Nonstat': [-0.0237793681428630, 2.02274541230661, 24.0486886580316]
    }
}
# Colorblind-friendly palette
colors = {
    'raw_data': '#1f77b4',  # blue
    'model_estimate': '#ff7f0e',  # orange
    'empirical_stationary': '#2ca02c',  # green
    'empirical_nonstationary': '#d62728',  # red
}

# Create a 3x2 subplot structure
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.flatten()  # Flatten to 1D array for easy indexing

for idx, file_path in enumerate(file_paths):
	with open(file_path, 'r') as file:
	   data = [float(line.strip()) for line in file]
	   gg = np.array(data)  # Convert list to numpy array
	   xc = np.linspace(0.1,0.9999, len(gg))
	   x = torch.from_numpy(xc)#.to(device='cuda')
	   liss1=(torch.from_numpy(np.sort(gg))/(durations[0]))#.to(device='cuda')#, my cpu runs it faster than my gpu but this isincase
	#print(liss1)
	
	shape=-0.01
	scale=1
	loc=5
	shape1=0.000045
	scale1=3
	loc1=15
	x0=[shape,scale,loc,shape1,scale1,loc1]
	x1=[shape,scale,loc]
	minimizer_kwargs = {"method": "BFGS"}
	mytakestep = MyTakeStep()
	ret = basinhopping(ll,x1,minimizer_kwargs=minimizer_kwargs,niter=100,take_step=mytakestep)
	#m.sample()  # uniformly distributed in the range [0.0, 5.0)
	fig, ax = plt.subplots()
	
	x = np.linspace(0.1,0.9999, len(gg))
	xx = np.log(np.divide(1,np.ones_like(x)-x))
	#ax.hist(sample, density=True, histtype='stepfilled', alpha=0.2)
	#the confidence interval is binomial 90%
	res1 = np.array(gev.ppf(x, ret.x[0], ret.x[2], ret.x[1]))
	city_name = file_path.split('/')[-1].split('_')[0]

	# Plotting in the respective subplot
	ax = axs[idx]
	xx = np.log(np.divide(1, np.ones_like(x) - x))

	ax.plot(xx, liss1, label='Empirical Data', color=colors['raw_data'])
	ax.plot(xx, res1, label='Model Estimate', color=colors['model_estimate'])

	stat_params = gev_params[city_name]['Stat']
	nonstat_params = gev_params[city_name]['Nonstat']

	ax.plot(xx, gev.ppf(x, *stat_params), label='MCMC Stationary', linestyle='--', color=colors['empirical_stationary'])
	ax.plot(xx, gev.ppf(x, *nonstat_params), label='MCMC Nonstationary w/ Time', linestyle='-.', color=colors['empirical_nonstationary'])

	ax.set_xscale('log')
	ax.xaxis.set_major_formatter(ticker.FuncFormatter(log_format))

	# Set custom x-ticks starting from 2
	custom_ticks = np.array([25,50, 100, 500,1000])
	transformed_custom_ticks = np.log(custom_ticks)
	ax.set_xticks(transformed_custom_ticks)
	ax.set_xticklabels([f'{tick}' for tick in custom_ticks], rotation=90)

	# Conditionally set the x-axis label
	if idx >= 3:  # Only for the bottom row subplots
	    ax.set_xlabel('N-Year Event')
	ax.set_ylabel('1-Day Precipitation (cm)')
	ax.set_title(f'GEV Comparison for {city_name}')
	ax.legend()

	tab[file_path] = ret.x

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

# Save the results
np.save("zzz.npy", tab)
print("Model results saved.")
