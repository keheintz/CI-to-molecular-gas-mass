from __future__ import print_function, division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
import corner
from lmfit import Model, Parameters, Minimizer, report_fit
import matplotlib.pylab as pyl

hdul = fits.open('../CIemgaltab.fits')
data = hdul[1].data

log_alpha_ci = -1.13*data['met'] + 1.33
log_alpha_ci_err = 0.2
logmH2 = log_alpha_ci + np.log10(data['lci'])+10
alpha_co = log_alpha_ci + np.log10(data['lci']) - np.log10(data['lco'])
alpha_co_err = log_alpha_ci_err**2 + data['e_lci']**2 + data['e_lco']**2

params = Parameters()
params.add('slope', value=-1.0, min=-30.0, max=10.0)
params.add('intercept', value=0.6, min=-20.0, max=20.0)

x = data['met'][(np.isfinite(alpha_co)) & (data.mstq == 1)]
y = alpha_co[(np.isfinite(alpha_co)) & (data.mstq == 1)]
yerr = alpha_co_err[(np.isfinite(alpha_co)) & (data.mstq == 1)]
xerr = 0.1
c = data['zsp'][(np.isfinite(alpha_co)) & (data.mstq == 1)]

def residual(pars):
    p = pars.valuesdict()
    model = p['slope']*x + p['intercept']
    return (model - y) / np.sqrt(yerr**2 + xerr**2)

mini = Minimizer(residual, params, nan_policy='omit')
res = mini.emcee(burn=100, steps=1000, thin=1, nwalkers=100, is_weighted=True)

corner.corner(res.flatchain, labels=res.var_names, truths=list(res.params.valuesdict().values()));

plt.savefig('Corner.pdf')
plt.close()

report_fit(res.params)

X = np.linspace(-0.55,0.55,100)
model = res.params.valuesdict()['slope'] * X + res.params.valuesdict()['intercept']
scatter = np.outer(res.flatchain.slope,X) + np.tile(res.flatchain.intercept,(len(X),1)).T
up = np.mean(scatter,axis=0) + np.std(scatter,axis=0)
down = np.mean(scatter,axis=0) - np.std(scatter,axis=0)

scatter_kwargs = {'zorder':100}
error_kwargs = {'lw':.5, 'zorder':0}

plt.fill_between(X,up,down,color='gray',alpha=0.2)
plt.plot(X,model,'k-')
plt.errorbar(x, y, xerr=0.1,yerr=yerr, fmt='none',marker='none',mew=0,ecolor='k',**error_kwargs)
plt.scatter(x,y,c=c,marker='o',edgecolor='black',cmap = 'Spectral_r',vmin=-0.2, vmax=5.0,**scatter_kwargs)
plt.axis([-0.55,0.55,-0.7,2.3])
plt.xlabel("log (Z/Z$_{\odot}$)")
plt.ylabel(r"log $\alpha_{\rm CO}$  [M$_{\odot}$ (K km s$^{-1}$ pc$^2$)$^{-1}$]")
cbar = plt.colorbar(shrink=0.9)
cbar.set_label(r"$z_{\rm spec}$")
plt.savefig('AlphaCO_Metz.pdf')
