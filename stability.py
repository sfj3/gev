import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme as gev
from scipy.optimize import basinhopping
import torch as torch
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
		loss = np.sum([0.0001 for i in b if i <= 0])
		b = torch.from_numpy(b)#.to(device='cuda')
		loss+=torch.log(torch.sum((x+(x+0.1)**2)*torch.abs(b - liss1)**2)+0.5*(torch.std(b)-torch.std(liss1))**2)
		return loss.item()# add one for log likelihood 
	else:
		return 1000000
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

colors = {
    'raw_data': '#1f77b4',  # blue
    'model_estimate': '#ff7f0e',  # orange
    'empirical_stationary': '#2ca02c',  # green
    'empirical_nonstationary': '#d62728',  # red
}

# Create a 3x2 subplot structure



df = pd.read_csv('sf.csv')

# Assuming TMAX is the column of interest
temperatures = df['TMAX'].values

colors = {
    'raw_data': 'blue',
    'model_estimate': 'red'
}

# Process in rolling windows of 30 days, shifting 5 days forward for each iteration do 90 next +sf
window_size = 1000
shift = 500
info_path = []
maximas = [-9999,-9999,-9999,-9999,-9999]
minimas = [9999,9999,9999,9999,9999]

for start in range(0, len(temperatures) - window_size + 1, shift):
    end = start + window_size
    data = temperatures[start:end]
    gg = np.array(data)

    # Here, insert the training logic and plotting for each window
    # The following lines are placeholders and need to be adapted to your specific model training and plotting logic
    xc = np.linspace(0.01, 0.999, len(gg))#program this please
    x = torch.from_numpy(xc)
    liss1 = torch.from_numpy(np.sort(gg)) / 1  # Assuming 'durations[0]' is defined elsewhere

    shape, scale, loc, shape1, scale1, loc1 = -0.01, 10, 5, 0.000045, 10, 15
    x0 = [shape, scale, loc, shape1, scale1, loc1]
    x1 = [shape, scale, loc]
    minimizer_kwargs = {"method": "BFGS"}
    mytakestep = MyTakeStep()
    ret = basinhopping(ll, x1, minimizer_kwargs=minimizer_kwargs, niter=200, take_step=mytakestep)

    fig, ax = plt.subplots()
    x = np.linspace(0.01, 0.999, len(gg))#program this please later (when it gets integrated)
    res1 = np.array(gev.ppf(x, ret.x[0], ret.x[2], ret.x[1]))
    

    print("SHAPE",ret.x[0],"SCALE", ret.x[2],"LOC",ret.x[1])
    if ret.x[2]==5.0:#domain collapse, model instablity check
        continue
    maximas_iter = [gev.ppf(0.01, ret.x[0], ret.x[2], ret.x[1]),gev.ppf(0.5, ret.x[0], ret.x[2], ret.x[1]),gev.ppf(0.9, ret.x[0], ret.x[2], ret.x[1]),gev.ppf(0.99, ret.x[0], ret.x[2], ret.x[1]),gev.ppf(0.999, ret.x[0], ret.x[2], ret.x[1])]
    maximas_iter = [gev.ppf(0.01, ret.x[0], ret.x[2], ret.x[1]),gev.ppf(0.5, ret.x[0], ret.x[2], ret.x[1]),gev.ppf(0.9, ret.x[0], ret.x[2], ret.x[1]),gev.ppf(0.99, ret.x[0], ret.x[2], ret.x[1]),gev.ppf(0.999, ret.x[0], ret.x[2], ret.x[1])]
    
    minimas_iter = [gev.ppf(0.01, ret.x[0], ret.x[2], ret.x[1]),gev.ppf(0.5, ret.x[0], ret.x[2], ret.x[1]),gev.ppf(0.9, ret.x[0], ret.x[2], ret.x[1]),gev.ppf(0.99, ret.x[0], ret.x[2], ret.x[1]),gev.ppf(0.999, ret.x[0], ret.x[2], ret.x[1])]
    maximas = [max(pair) for pair in zip(maximas, maximas_iter)]
    minimas = [min(pair) for pair in zip(minimas, minimas_iter)]

    print('maximas',maximas)
    print('minimas',minimas)
    binomial_max = [gev.ppf(0.01+0.025/np.sqrt(1000)*np.sqrt(0.01*0.99), ret.x[0], ret.x[2], ret.x[1]),gev.ppf(0.5+0.025/np.sqrt(1000)*np.sqrt(0.5*0.5), ret.x[0], ret.x[2], ret.x[1]),gev.ppf(0.9+0.025/np.sqrt(1000)*np.sqrt(0.1*0.9), ret.x[0], ret.x[2], ret.x[1]),gev.ppf(0.99+0.025/np.sqrt(1000)*np.sqrt(0.01*0.99), ret.x[0], ret.x[2], ret.x[1]),gev.ppf(0.999+0.025/np.sqrt(1000)*np.sqrt(0.001*0.999), ret.x[0], ret.x[2], ret.x[1])]
    binomial_min = [gev.ppf(0.01-0.025/np.sqrt(1000)*np.sqrt(0.01*0.99), ret.x[0], ret.x[2], ret.x[1]),gev.ppf(0.5-0.025/np.sqrt(1000)*np.sqrt(0.5*0.5), ret.x[0], ret.x[2], ret.x[1]),gev.ppf(0.9-0.025/np.sqrt(1000)*np.sqrt(0.1*0.9), ret.x[0], ret.x[2], ret.x[1]),gev.ppf(0.99-0.025/np.sqrt(1000)*np.sqrt(0.01*0.99), ret.x[0], ret.x[2], ret.x[1]),gev.ppf(0.999-0.025/np.sqrt(1000)*np.sqrt(0.001*0.999), ret.x[0], ret.x[2], ret.x[1])]
    print('binomial max',binomial_max)
    print('binomial min',binomial_min)
    xx = np.log(np.divide(1, np.ones_like(x) - x))
    ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
    ax.plot(xx, liss1, label='Empirical Data', color=colors['raw_data'], linestyle="dotted", linewidth=2)
    ax.plot(xx, res1, label='Model Estimate', color=colors['model_estimate'])
    # plt.legend()
    # plt.show()
df_info_path = pd.DataFrame(info_path)

# Save the DataFrame to a CSV file
df_info_path.to_csv('./info_path.csv', index=False)

'''
gg = np.array(data)  # Convert list to numpy array
xc = np.linspace(0.1,0.999, len(gg))#this is highly dependant on the number of sample points if its being used for n-year return period. 
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
print(x, ret.x[0], ret.x[2], ret.x[1])
# Plotting in the respective subplot
xx = np.log(np.divide(1, np.ones_like(x) - x))
ax.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)
ax.plot(xx, liss1, label='Empirical Data', color=colors['raw_data'],linestyle="dotted",linewidth=2)
ax.plot(xx, res1, label='Model Estimate', color=colors['model_estimate'])
plt.show()
'''






#look at total info vs ln(loc)
#look into the size of the jumps 