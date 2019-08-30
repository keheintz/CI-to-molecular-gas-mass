from __future__ import print_function, division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
import corner
from lmfit import Model, Parameters, Minimizer, report_fit

hdul = fits.open('../GRB_QDLA_sample.fits')
data = hdul[1].data

met = data['met']
e_met = data['e_met']
cim = data['cimh2']
e_cim = data['e_cimh2']

xg = met[(data.type == 1)]  
xgerr = e_met[(data.type == 1)]
yg = cim[(data.type == 1)]
ygerr = e_cim[(data.type == 1)]
xq = met[(data.type == 0) & (data.nhi > 21.7)]                          
yq = cim[(data.type == 0) & (data.nhi > 21.7)]
xqerr = e_met[(data.type == 0) & (data.nhi > 21.7)]
yqerr = e_cim[(data.type == 0) & (data.nhi > 21.7)]
xt = met[(data.type == 0) & (data.nhi < 21.7)]
yt = cim[(data.type == 0) & (data.nhi < 21.7)]
xterr = e_met[(data.type == 0) & (data.nhi < 21.7)]
yterr = e_cim[(data.type == 0) & (data.nhi < 21.7)]


params = Parameters()
params.add('slope', value=-1.0, min=-30.0, max=10.0)
params.add('intercept', value=1.3, min=-20.0, max=20.0)

def residual(pars):
    p = pars.valuesdict()
    model = p['slope']*met + p['intercept']
    return (model - cim) / np.sqrt(e_cim**2 + e_met**2)

mini = Minimizer(residual, params, nan_policy='omit')
res = mini.minimize(method='leastsq')

report_fit(res.params)

X = np.linspace(-2.0,1.0,100)
model = res.params.valuesdict()['slope'] * X + res.params.valuesdict()['intercept']

plt.plot(X,model,'k-')
plt.errorbar(xg, yg, xerr=xgerr,yerr=ygerr, marker='s',ms=8,mfc='indianred',mec='k',ecolor='k',fmt='o')
plt.errorbar(xq, yq, xerr=xqerr,yerr=yqerr, marker='s',ms=7,mfc='dodgerblue',mec='k',ecolor='k',fmt='o')
plt.errorbar(xt, yt, xerr=xterr,yerr=yterr, marker='o',ms=5, mfc='dodgerblue',mec='k',ecolor='k',fmt='o')

plt.savefig('CItoMmol.pdf')
