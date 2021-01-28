import pandas as pd
import os
import numpy as np
from matplotlib import pyplot as plt
import theano.tensor as tt
import scipy.stats as stats
from scipy.stats import norm
import pymc3 as pm
import theano
from theano import tensor as T
from pymc3.math import switch
from datetime import datetime
import arviz as az
import ndjson



###      ###
### Data ###
###      ### 
 
#ombinus:
politiken = pd.read_csv(r'/home/au617011/Documents/CHCAA/hope/cds_intro_exam/data/tabular/da_politiken-print-091820_signal.csv')
berlingske = pd.read_csv(r'/home/au617011/Documents/CHCAA/hope/cds_intro_exam/data/tabular/da_berglinske-print-091820_signal.csv')
jyllandsposten = pd.read_csv(r'/home/au617011/Documents/CHCAA/hope/cds_intro_exam/data/tabular/da_jyllands-posten-print-091820_signal.csv')
kristeligtdagblad = pd.read_csv(r'/home/au617011/Documents/CHCAA/hope/cds_intro_exam/data/tabular/da_kristeligt-dagblad-print-091820_signal.csv')
#tabloid:
bt = pd.read_csv(r'/home/au617011/Documents/CHCAA/hope/cds_intro_exam/data/tabular/da_bt-print-091820_signal.csv')
ekstrabladet = pd.read_csv(r'/home/au617011/Documents/CHCAA/hope/cds_intro_exam/data/tabular/da_ekstrabladet-print-091820_signal.csv')


# Choose newspaper for analysis
df = politiken


###               ###
### Preprocessing ###
###               ### 

# Filtering away first and last 7 days
df = df[df.novelty != 0]
df = df[df.resonance != 0]


# Date variable for analysis
df = df.sort_values(by="date")
dates = [string[0:10] for string in df["date"]]  
dates = [datetime.strptime(datez, '%Y-%m-%d').date() for datez in dates]

#ndays
dates_int = [dates[i] - dates[0] for i in range(len(dates))]
dates_int = [x.days for x in dates_int] 

#dates
dates_str = [str(x) for x in dates]

# Novelty variables for analysis
novelty = df["novelty"]


# Inspecting data
plt.plot(dates_int, novelty, markersize=8, alpha=0.4)
plt.ylabel("novelty")
plt.xlabel("days") 
plt.title("")

###          ###
### Modeling ###
###          ###

# Creating model

novelty = np.array(novelty)
niter = 4000 #number of iterations for the MCMC algorithm
#t = np.arange(0,len(novelty)) #array of observation positions ('time')
t = np.array(dates_int) 
with pm.Model() as model: # context management

    #define uniform priors
    mu1 = pm.Uniform("mu1",0,0.5)
    mu2 = pm.Uniform("mu2",0,0.5)
    mu3 = pm.Uniform("mu3",0,0.5)
    sigma = pm.HalfCauchy("sigma",0.5)
    tau1 = pm.DiscreteUniform("tau1",t.min(),t.max())
    tau2 = pm.DiscreteUniform("tau2",tau1,t.max())

    #define stochastic variable mu
    _mu = T.switch(tau1>=t,mu1,mu2)
    mu = T.switch(tau2>=t,_mu,mu3)

    #define formula for log-likelihood
    logp = - T.log(sigma * T.sqrt(2.0 * np.pi)) \
           - T.sqr(novelty - mu) / (2.0 * sigma * sigma)
    def logp_func(novelty):
        return logp.sum()

    #find out log-likelihood of observed data
    L_obs = pm.DensityDist('L_obs', logp_func, observed=novelty)

    #start MCMC algorithm
    #iterate MCMC
    trace = pm.sample(niter, step=pm.NUTS(), random_seed=123, progressbar=True)

pm.traceplot(trace)

pm.summary(trace)


# Plotting

plt.figure(figsize=(10, 8))
plt.plot(dates_int, novelty, ".", alpha=0.6)
plt.ylabel("novelty")
plt.xlabel("date") 

102 < trace["tau1"]
plt.vlines((
    trace["tau1"].mean(), trace["tau2"].mean()), novelty.min(), novelty.max(), color=("C4","C1")
)
average_novelty = np.zeros_like(novelty, dtype="float")
for i, day in enumerate(dates_int):
    idx1 = day < trace["tau1"]
    idx2 = np.logical_and(trace["tau1"] <= day , day <= trace["tau2"])
    idx3 = trace["tau2"] < day 
    average_novelty_list = []
    average_novelty_list.append(trace["mu1"][idx1])
    average_novelty_list.append(trace["mu2"][idx2])
    average_novelty_list.append(trace["mu3"][idx3])
    average_novelty_list = [value for sublist in average_novelty_list for value in sublist]
    #print(f'day: {day},len: {len(average_novelty_list)}')

    average_novelty[i] = sum(average_novelty_list) / len(average_novelty_list)

sp_hpd = az.hdi(trace["tau1"])
plt.fill_betweenx(
    y=[novelty.min(), novelty.max()],
    x1=sp_hpd[0],
    x2=sp_hpd[1],
    alpha=0.5,
    color="C4",
)
sp_hpd = az.hdi(trace["tau2"])
plt.fill_betweenx(
    y=[novelty.min(), novelty.max()],
    x1=sp_hpd[0],
    x2=sp_hpd[1],
    alpha=0.5,
    color="C1",
)

# plotting with dates on x-axis
plt.plot(dates_int, average_novelty, "k--", lw=2)
locs, labels = plt.xticks()
index = [i for i in range(len(dates_int)) if dates_int[i] in locs]
strings = [dates_str[i] for i in index]

plt.xticks(ticks = np.arange(300, step = 50), labels = strings)
plt.title('')

# #old plot - number of days on x-axis
# plt.plot(dates_int, average_novelty, "k--", lw=2)
# plt.title('')

#pm.plot_posterior(trace, var_names = ["mu3"])

