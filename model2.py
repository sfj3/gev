import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme as gev
from scipy.optimize import basinhopping
import torch
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
from scipy.special import gamma
from scipy.stats import entropy
# Load CSV data
def interpolate_array(original_array, new_length):
    # Create an array of indices for the original array
    original_indices = np.logspace(0, len(original_array) - 1, num=len(original_array))

    # Create an array of indices for the new, interpolated array
    new_indices = np.logspace(0, len(original_array) - 1, num=new_length)

    # Perform the interpolation
    interpolated_array = np.interp(new_indices, original_indices, original_array)

    return interpolated_array

def lin_interp(original_array, new_length):
    # Create an array of indices for the original array
    original_indices = np.linspace(0, len(original_array) - 1, num=len(original_array))

    # Create an array of indices for the new, interpolated array
    new_indices = np.linspace(0, len(original_array) - 1, num=new_length)

    # Perform the interpolation
    interpolated_array = np.interp(new_indices, original_indices, original_array)

    return interpolated_array

csv_file = "rr.csv"  # Replace with your CSV file path
df = pd.read_csv(csv_file)
device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
import matplotlib.ticker as ticker
years = np.array([10,25 , 50, 100, 300, 1000])#years to show on plot axis
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
	   self.stepsize = self.stepsize*0.999#annealing 4.347851452142499
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
        'Stat': [0.00531941256118090, 22.5958008494514, 1.84561324526801],
        'Nonstat': [0.00364436144788064, 22.9304700947800, 1.84234938512680]
    },
    'Los Angeles': {
        'Stat': [-0.0118036096075670, 40.9138086279802, 2.97414331328946],
        'Nonstat': [-0.0177458225240232, 40.8917524517012, 2.97994729403216]
    },
    'Sacramento': {
        'Stat': [0.00788121850664761, 33.4739259756107, 2.40347809268535],
        'Nonstat': [0.0114845676916233, 33.2778495697244, 2.41332692499420]
    },
    'San Diego': {
        'Stat': [0.0330518047348983, 25.3797609335986, 2.28414761697310],
        'Nonstat': [0.0361723417609112, 25.4768347770986, 2.28483993591864]
    },
    'San Francisco': {
        'Stat': [0.0136980578925005, 35.9392084612063, 2.35163854800503],
        'Nonstat': [0.0119752942324680, 35.9273593279362, 2.35066604999109]
    },
    'San Jose': {
        'Stat': [-0.0318089686977196, 23.8638059751154, 2.02160547404123],
        'Nonstat': [-0.0237793681428630, 24.0486886580316, 2.02274541230661]
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
fig, axs = plt.subplots(3, 2, figsize=(15, 10))
axs = axs.flatten()  # Flatten to 1D array for easy indexing

for idx, file_path in enumerate(file_paths):
	with open(file_path, 'r') as file:
	   data = [float(line.strip()) for line in file]
	   gg = np.array(data)  # Convert list to numpy array
	   xc = np.linspace(0.1,0.999, len(gg))
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
	
	x = np.linspace(0.1,0.999, len(gg))
	xx=x





	# xx = np.log(np.divide(1,np.ones_like(x)-x))
	#ax.hist(sample, density=True, histtype='stepfilled', alpha=0.2)
	#the confidence interval is binomial 90%
	res1 = np.array(gev.ppf(x, ret.x[0], ret.x[2], ret.x[1]))
	city_name = file_path.split('/')[-1].split('_')[0]
	# Plotting in the respective subplot
	ax = axs[idx]
	xx = np.log(np.divide(1, np.ones_like(x) - x))
	ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
	ax.plot(xx, liss1, label='Empirical Data', color=colors['raw_data'],linestyle="dotted",linewidth=2)
	ax.plot(xx, res1, label='Model Estimate', color=colors['model_estimate'])

	"""stat_params = gev_params[city_name]['Stat']
	nonstat_params = gev_params[city_name]['Nonstat']

	ax.plot(xx, gev.ppf(x, *stat_params), label='MCMC Stationary', linestyle='--', color=colors['empirical_stationary'])
	ax.plot(xx, gev.ppf(x, *nonstat_params), label='MCMC Nonstationary w/ Time', linestyle='-.', color=colors['empirical_nonstationary'])
	"""
	start = np.min(xx)
	end = np.max(xx)
	num_points = 999

	# Calculate log values for 1 and 1000 years
	
	# Generate linspace between log_year_1 and log_year_1000
	xx1 = np.linspace(2, 999, 999)
	xx1 = np.log(xx1)
	city_name = file_path.split('/')[-1].split('_')[0]
	stat_csv_data = df[f'{city_name} stat']
	nonstat_csv_data = df[f'{city_name} nonstat']

	ax.plot(xx1, stat_csv_data[:len(xx1)], label='MCMC Stationary', linestyle='-.', color='purple')
	ax.plot(xx1, nonstat_csv_data[:len(xx1)], label='MCMC Nonstationary', linestyle='dashed', color='red')
	ax.set_xscale('log')
	ax.xaxis.set_major_formatter(ticker.FuncFormatter(log_format))

	# Set custom x-ticks starting from 2
	custom_ticks = np.array([10,25,50,100, 300,1000])
	x = np.linspace(0.5,0.999, 999)
	xx = np.log(np.divide(1,np.ones_like(x)-x))



	#print('function of dataset',f(liss1))
	# print('model data interpolated to proneva',liss1)
	# print('model',res1)
	# print('nonstat',nonstat_csv_data)
	
	# print('model_interp',model_interp(xx))
	# fig, ax = plt.subplots()
	# rmse_model_estimate = np.sqrt(mean_squared_error(liss1,res1)



	# Calculate RMSE between empirical data and stationary/nonstationary data
	# rmse_stationary = np.sqrt(mean_squared_error(interpolate_array(model_interp(liss1),999), stat_csv_data))
	# rmse_nonstationary = np.sqrt(mean_squared_error(interpolate_array(model_interp(liss1),999), nonstat_csv_data))
	# print(city_name)
	# print(f"RMSE (Model Estimate): {rmse_model_estimate}")
	# print(f"RMSE (Stationary): {rmse_stationary}")
	# print(f"RMSE (Nonstationary): {rmse_nonstationary}")
	# fig, axs = plt.subplots(2, 3, figsize=(15, 10))
	# axs = axs.flatten()
	# liss1_interpolated = interpolate_array(liss1, 999)
	# model_data_interpolated = np.array(gev.ppf(np.exp(xx1)/1000, ret.x[0], ret.x[2], ret.x[1]))
	# stat_data_interpolated = interpolate_array(stat_csv_data, 999)
	# nonstat_data_interpolated = interpolate_array(nonstat_csv_data, 999)
	# Calculate RMSE
	liss1 = lin_interp(liss1,999)
	res1 = lin_interp(res1,999)
	# rmse_model_estimate = np.sqrt(mean_squared_error(liss1,res1))
	
	# rmse_stationary = np.sqrt(mean_squared_error(liss1_interpolated, stat_data_interpolated))
	# rmse_nonstationary = np.sqrt(mean_squared_error(liss1_interpolated, nonstat_data_interpolated))


	# Set ticks and labels
	transformed_custom_ticks = np.log(custom_ticks)
	ax.set_xticks(transformed_custom_ticks, minor=False)
	ax.set_xticklabels([f'{tick}' for tick in custom_ticks], rotation=90)
	ax.xaxis.set_minor_formatter(ticker.NullFormatter())
	ax.xaxis.set_minor_locator(ticker.NullLocator())
	if idx >= 3:
	    ax.set_xlabel('N-Year Event')
	ax.set_ylabel('1-Day Precipitation (mm)')
	ax.set_title(f'{city_name}')
	ax.legend()

	# # Print RMSE results
	# print(city_name)
	new_support = np.linspace(0.1, 0.999, len(gg))
	stat_interp = np.interp(new_support, np.linspace(0.1, 0.999, len(stat_csv_data)), stat_csv_data)
	n = np.linspace(10, 1000,999)  # n values from 2 to 1001

	proneva_support = np.ones_like(n) - 1/n
	# print(proneva_support)
	q = np.quantile(liss1, proneva_support)  # Convert tensor to numpy array if needed
	res1_sup = np.array(gev.ppf(proneva_support, ret.x[0], ret.x[2], ret.x[1]))
	rmse_model_estimate = np.sqrt(mean_squared_error(q,res1_sup))
	
	rmse_stationary = np.sqrt(mean_squared_error(q,stat_csv_data))
	rmse_nonstationary = np.sqrt(mean_squared_error(q,nonstat_csv_data))
	# print('stat csv ',stat_csv_data)
	# print('stat new_support',quantiles_liss1)
	# print('actual',liss1)
	print(city_name)
	print(f"RMSE (Model Estimate): {rmse_model_estimate}")
	print(f"RMSE (Stationary): {rmse_stationary}")
	print(f"RMSE (Nonstationary): {rmse_nonstationary}")
	transformed_custom_ticks = np.log(custom_ticks)
	ax.set_xticks(transformed_custom_ticks, minor=False)  # Disable minor ticks
	ax.set_xticklabels([f'{tick}' for tick in custom_ticks], rotation=90)
	ax.xaxis.set_minor_formatter(ticker.NullFormatter())
	ax.xaxis.set_minor_locator(ticker.NullLocator())
	# Conditionally set the x-axis label
	if idx >= 3:  # Only for the bottom row subplots
	    ax.set_xlabel('N-Year Event')
	ax.set_ylabel('1-Day Precipitation (mm)')
	ax.set_title(f'{city_name}')
	# ax.legend()
	# plt.show()

	tab[file_path] = ret.x

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

# Save the results
np.save("zzz.npy", tab)
print("Model results saved.")