from __future__ import print_function, division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
import corner
from lmfit import Model, Parameters, Minimizer, report_fit

hdul = fits.open('../CIemgaltab.fits')
data = hdul[1].data

log_alpha_ci = -1.13*data['met'] + 1.33
log_alpha_ci_err = 0.2
logmH2 = log_alpha_ci + np.log10(data['lci'])+10
alpha_co = log_alpha_ci + np.log10(data['lci']) - np.log10(data['lco'])

yax = log_alpha_ci + np.log10(data['lci'])+10
y = yax[(np.isfinite(alpha_co)) & (data.mstq == 1)]
xax = np.log10(data['lco']*1e10 * 4.3)
x = xax[(np.isfinite(alpha_co)) & (data.mstq == 1)] 
dyerr = log_alpha_ci_err**2 + data['e_lci']**2
yerr = dyerr[(np.isfinite(alpha_co)) & (data.mstq == 1)]
dxerr = np.log10(data['lco']*4.3e10 + data['e_lco']*4.3e10) - np.log10(data['lco']*4.3e10)
xerr = dxerr[(np.isfinite(alpha_co)) & (data.mstq == 1)]
c = data['met'][(np.isfinite(alpha_co)) & (data.mstq == 1)]

X = np.linspace(-0.55,0.55,100)

scatter_kwargs = {'zorder':100}
error_kwargs = {'lw':.5, 'zorder':0}

plt.plot([8,13], [8,13], 'k--', linewidth=0.8)
plt.errorbar(x, y, xerr=xerr,yerr=yerr, fmt='none',marker='none',mew=0,ecolor='k',**error_kwargs)
plt.scatter(x,y,c=c,marker='o',edgecolor='black',cmap = 'Spectral_r',vmin=-0.4, vmax=0.4,**scatter_kwargs)
plt.axis([8.8,11.8,8.4,12.4])
plt.xlabel(r"log (M/M$_{\odot}$) w. $\alpha_{\rm CO}$ = 4.3 M$_{\odot}$ (K km s$^{-1}$ pc$^2$)$^{-1}$")
plt.ylabel(r"log (M/M$_{\odot}$) w. $\alpha_{\rm [CI]}$")
cbar = plt.colorbar(shrink=0.9)
cbar.set_label("log (Z/Z$_{\odot}$)")
plt.savefig('CompMmol.pdf')
