import numpy as np
import matplotlib.pyplot as plt
import os

from .processes import limb_side_finder
from .utils import load_fits, fits_get_sampling, iter_noise, gaussian_fit, gaus
from astropy.io import fits
import datetime

def dataset_colorbar(ax,im,location="top",label=None,xy=None,fontsize=9):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size="5%", pad=0.05)
    if location == 'top' or location == 'bottom':
        orientation = 'horizontal'
    else:
        orientation = 'vertical'
    if xy is None:
        cb = plt.colorbar(im, orientation=orientation, cax=cax, label=label)
    else:
        cb = plt.colorbar(im, orientation=orientation, cax=cax)
        cax.set_title(label,fontsize=fontsize,x=xy[0],y=xy[1])
        
    if location == 'top' or location == 'bottom':
        cax.xaxis.set_ticks_position(location)    
    else:
        cax.yaxis.set_ticks_position(location)
    
    cax.tick_params(labelsize=8)

    return cax

def bmag_cmap():
    """
    Create a colormap for the BMAG
    """
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib import colormaps
    cmap = colormaps['tab20c']
    addition = cmap(np.linspace(0,1,20))[-4:]
    cmap = colormaps['tab20b']
    tab24 = np.append(cmap(np.linspace(0,1,20)),addition,axis=0)
    del addition, cmap
    tab24 = LinearSegmentedColormap.from_list('tab24',tab24,N=24)

    return tab24


def show_image_array(arr, hdr, grayscales, panel_sz=3.3, row_labels=None, 
                     column_labels=None, titles=None,
                     fig_title=None, ax_order=None):
    """Show array images of shape (rows, columns, X, Y).

    Parameters
    ----------
    
    """

    import itertools

    # panel_sz = 3.3
    nl, ns, ny, nx = arr.shape
    _, _, _, sly, slx = limb_side_finder(arr[0,0],hdr,False)
    # fig_width, fig_height = plt.gcf().get_size_inches()
    # print(fig_width, fig_height)    

    fig, axs = plt.subplots(
        ns, nl,
        sharex=True, sharey=True,
        subplot_kw=dict(aspect=1),
        figsize=(nl * panel_sz, ns * panel_sz),
        layout='constrained',
        # gridspec_kw={'hspace': 0, 'wspace': 0},
        # **kwargs
    )

    # Sort plots
    if ax_order is not None:
        axs = axs.flatten()
        axs = [axs[i] for i in ax_order]

    axs = np.reshape(axs, (ns, nl))

    # plt.subplots_adjust(top=0.92)

    for i, j in itertools.product(range(nl), range(ns)):
        im = arr[i, j, :, :]

        mean = im[sly,slx].mean()

        ax = axs[j, i]
        
        # Print color scale range
        im = ax.imshow(im, cmap='gray', clim=grayscales[j],interpolation=None)
        if i == 0:
            ax.text(0.05, 0.94, f'{grayscales[j][0]:.1f} - {grayscales[j][1]:.1f}', transform=ax.transAxes, color='white')
        else:
            ax.text(0.05, 0.94, f'{grayscales[j][0]:.3f} - {grayscales[j][1]:.3f}', transform=ax.transAxes, color='white')
        # ax.text(0.05, 0.94, f'{grayscales[i][0]:.1f} - {grayscales[i][1]:.1f}', transform=ax.transAxes, color='white')
        # else:
        #     im = axs[i, j].imshow(im, cmap='gray', vmin=mean+grayscales[i][0], vmax=mean+grayscales[i][1],interpolation=None)
        #     ax.text(0.05, 0.94, f'{mean:.4f} $\pm$ {grayscales[i][1]:.3f}', transform=ax.transAxes, color='white')

    # Set row labels
    if row_labels is not None:
        for row_label, ax in zip(row_labels, axs[:, 0]):
            ax.set_ylabel(row_label)

    # Set column labels
    if column_labels is not None:
        for column_label, ax in zip(column_labels, axs[0, :]):
            ax.set_title(column_label)

    # Set panel titles
    if titles is not None:
        for title, ax in zip(titles, axs.flatten()):
            ax.set_title(title, fontsize=12)

    if fig_title is not None:
        fig.suptitle(fig_title, fontsize=14)

    return fig

def plot_l2_pdf(path,did,version=None,save_output=True,plot_noise=True,plot_stokes=True, **kwargs):
    """
    Generate standard plots for pipeline results

    kwargs = {
        'icnt_cmap':'gist_heat',
        'vlos_cmap':cmr.fusion.reversed(),
        'blos_cmap':hmimag,
        'binc_cmap':cmr.fusion,
        'bmag_cmap':bmag_cmap(), # 'gnuplot_r'
        'bazi_cmap':'hsv',
        'panel_sz': 4,
        'dpi': 300,
        'rows': 2,
        'columns': 3,
    }
    """

    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import glob
    from matplotlib.colors import LinearSegmentedColormap
    # import os
    # import re
    # from argparse import ArgumentParser
    # import numpy as np
    
    # import sys

    import matplotlib as mpl
    mpl.rc_file_defaults()
    mpl.rcParams['image.origin'] = 'lower'
    # plt.rcParams['figure.dpi'] = 300
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['image.cmap'] = 'gist_heat'
    plt.rcParams['image.interpolation'] = 'none'

    import cmasher as cmr

    
    pipe_dir = os.path.realpath(__file__)
    pipe_dir = pipe_dir.split('src/')[0]
    hmimag = LinearSegmentedColormap.from_list('hmimag', np.loadtxt(pipe_dir+'csv/hmimag.csv',delimiter=','), N=256)

    default_params = {
        'icnt_cmap':'gist_heat',
        'vlos_cmap':cmr.fusion.reversed(),
        'blos_cmap':hmimag,
        'binc_cmap':cmr.fusion,
        'bmag_cmap':'gnuplot_r', # bmag_cmap(), # 
        'bazi_cmap':'hsv',
        'icnt_clim':(.3,1.2),
        'vlos_clim':(-2,2),
        'blos_clim':(-1500,1500),
        'binc_clim':(0,180),
        'bmag_clim':(0,2000), # (0,3000) # 
        'bazi_clim':(0,180),
        'panel_sz': 4,
        'dpi': 300,
        'rows': 2,
        'columns': 3,
    }

    params = {**default_params, **kwargs}

    file_n = os.listdir(path)
    if type(did) != str:
        did = str(did).rjust(10,'0')
    if version is None:
        version = '*'

    # pdf_name = re.search('.*/(.*)$', args.path).groups()[0]
    
    # # -----------------------------------------------------------------------------
    # # Loading data
    # # -----------------------------------------------------------------------------

    dat = {}
    keys = ['icnt', 'vlos', 'blos', 'binc', 'bmag', 'bazi', 'chi2']
    missingKeys = []
    for key in keys:
        try:
            datfile = glob.glob(os.path.join(path, f'solo_L2_phi-hrt-{key}_*_{version}_{did}.fits*'))[0]
            dat[key], h = load_fits(datfile)
        except:
            print('Missing '+key)
            missingKeys += [key]
    
    for key in missingKeys:
        k = next((e for e in keys if e!=key), None)
        dat[key] = np.zeros(dat[k].shape)
    
    _, _, _, sly, slx = limb_side_finder(dat['icnt'], h, False)
    

    datfile = glob.glob(os.path.join(path, f'solo_L2_phi-hrt-stokes_*_{version}_{did}.fits*'))[0]
    stk, h = load_fits(datfile)
    if stk.shape[0] != 6 or stk.shape[1] != 4:
        stk = np.einsum('yxpl->lpyx',stk)
    
    wavelengths,_,_,cpos = fits_get_sampling(datfile)

    if version == '*':
        version = 'V'+h['VERSION']
    if save_output:
        save_file = os.path.join(path, f'plots_solo_L2_phi-hrt_{datfile.split('_')[-3]}_{version}_{did}.pdf')
        p = PdfPages(save_file)

    # # -----------------------------------------------------------------------------
    # # Plot inversion results
    # # -----------------------------------------------------------------------------

    # Plot parameters
    panel_sz = params['panel_sz']
    dpi = params['dpi']
    rows = params['rows']
    columns = params['columns']

    fig, axs = plt.subplots(
        rows, columns,
        sharey=True,
        subplot_kw={'aspect': 1},
        figsize=(columns * panel_sz + 2, rows * panel_sz), dpi=dpi,
        layout='constrained')

    # Continuum intensity
    ax = axs[0, 0]
    im = ax.imshow(dat['icnt'], cmap=params['icnt_cmap'], clim = params['icnt_clim'],interpolation='none')
    dataset_colorbar(ax,im,"right")
    ax.set_title('Continuum intensity')

    # vLOS
    ax = axs[0, 1]
    shape = dat['vlos'].shape
    avg = dat['vlos'][int(shape[0]//4):-int(shape[0]//4),int(shape[1]//4):-int(shape[1]//4)].mean()
    im = ax.imshow(dat['vlos'], cmap=params['vlos_cmap'], clim = (params['vlos_clim'][0]+avg, params['vlos_clim'][1]+avg),interpolation='none')
    dataset_colorbar(ax,im,"right", label='km/s')
    ax.set_title('LoS velocity')

    # BLOS
    ax = axs[0, 2]
    im = ax.imshow(dat['blos'], cmap=params['blos_cmap'], clim = params['blos_clim'],interpolation='none')
    dataset_colorbar(ax,im,"right", label='G')
    ax.set_title('LoS magnetic field')

    # B inclination
    ax = axs[1, 0]
    im = ax.imshow(dat['binc'], cmap=params['binc_cmap'], clim = params['binc_clim'],interpolation='none')
    dataset_colorbar(ax,im,"right", label='°')
    ax.set_title('Magn. field inclination')

    # B
    ax = axs[1, 1]
    im = ax.imshow(dat['bmag'], cmap=params['bmag_cmap'], clim = params['bmag_clim'],interpolation='none')
    dataset_colorbar(ax,im,"right", label='G')
    ax.set_title('Magn. field strength')

    # B azimuth
    ax = axs[1, 2]
    im = ax.imshow(dat['bazi'], cmap=params['bazi_cmap'], clim = params['bazi_clim'],interpolation='none')
    dataset_colorbar(ax,im,"right", label='°')
    ax.set_title('Magn. field azimuth')

    # Figure title
    timestp = h['FILENAME'].split('_')[-3]
    fig.suptitle(os.path.join(path, f'solo_L2_phi-hrt-*_{timestp}_{version}_{did}.fits'), fontsize=12)
    
    if save_output:
        fig.savefig(p, format='pdf')
        plt.close(fig)

    if plot_noise:
        panel_sz = 4
        dpi = 300
        rows = 2
        columns = 3

        fig, axs = plt.subplots(
            rows, columns,
            subplot_kw={'aspect': 1},
            figsize=(columns * panel_sz + 2, rows * panel_sz), dpi=dpi,
            layout='constrained')

        # Chisq
        ax = axs[0,0]
        im = ax.imshow(dat['chi2'], cmap='turbo', vmin=0, vmax=100,interpolation='none')
        dataset_colorbar(ax,im,"right")
        ax.set_title('$\chi^2$')


        # Blos Noise
        values = dat['blos'][sly,slx]

        ax = axs[0,1]
        hi = ax.hist(values.flatten(), bins=np.linspace(-2e2,2e2,200),)
        tmp = [0,0]
        tmp[0] = hi[0].astype('float64')
        tmp[1] = hi[1].astype('float64')

        #guassian fit + label
        pp = gaussian_fit(tmp, show = False)    
        xx=hi[1][:-1] + (hi[1][1]-hi[1][0])/2
        lbl = f'{pp[1]:.2e} $\pm$ {pp[2]:.2e} G'


        ax.plot(xx,gaus(xx,*pp),'r--', label=lbl)
        # try:
        #     p_iter, hi_iter = iter_noise(values,[1.,0.,10.],eps=1e-4); p_iter[0] = pp[0]
        #     ax.plot(xx,gaus(xx,*p_iter),'g-.', label= f"Iter Fit: {p_iter[1]:.2e} $\pm$ {p_iter[2]:.2e} G")
        #     # ax[1].scatter(0,0, color = 'white', s = 0, label = lbl) #also display the original fit in legend
        # except:
        #     print("Iterative Gauss Fit failed")
        ax.set_aspect('auto')
        ax.legend()
        ax.set_title(f"LoS magnetic field NSR")

        # Blos Transverse
        values = (dat['bmag']*np.sin(dat['binc']*np.pi/180))[sly,slx]

        ax = axs[0,2]
        hi = ax.hist(values.flatten(), bins=np.linspace(0,10e2,200))
        tmp = [0,0]
        tmp[0] = hi[0].astype('float64')
        tmp[1] = hi[1].astype('float64')

        #guassian fit + label
        pp = gaussian_fit(tmp, show = False)    
        xx=hi[1][:-1] + (hi[1][1]-hi[1][0])/2
        lbl = f'{pp[1]:.2e} $\pm$ {pp[2]:.2e} G'


        ax.plot(xx,gaus(xx,*pp),'r--', label=lbl)
        # try:
        #     p_iter, hi_iter = iter_noise(values,[1.,0.,1000.],eps=1e-4); p_iter[0] = pp[0]
        #     ax.plot(xx,gaus(xx,*p_iter),'g-.', label= f"Iter Fit: {p_iter[1]:.2e} $\pm$ {p_iter[2]:.2e} G")
        #     # ax[1].scatter(0,0, color = 'white', s = 0, label = lbl) #also display the original fit in legend
        # except:
        #     print("Iterative Gauss Fit failed")
        ax.set_aspect('auto')
        ax.legend()
        ax.set_title(f"Transverse magnetic field NSR")

        # Stokes Q Noise
        values = stk[cpos,1,sly,slx]

        ax = axs[1,0]
        hi = ax.hist(values.flatten(), bins=np.linspace(-1e-2,1e-2,200),)
        tmp = [0,0]
        tmp[0] = hi[0].astype('float64')
        tmp[1] = hi[1].astype('float64')

        #guassian fit + label
        pp = gaussian_fit(tmp, show = False)    
        xx=hi[1][:-1] + (hi[1][1]-hi[1][0])/2
        lbl = f'{pp[1]:.2e} $\pm$ {pp[2]:.2e}'


        ax.plot(xx,gaus(xx,*pp),'r--', label=lbl)
        # try:
        #     p_iter, hi_iter = iter_noise(values,[1.,0.,.1],eps=1e-6); p_iter[0] = pp[0]
        #     ax.plot(xx,gaus(xx,*p_iter),'g-.', label= f"Iter Fit: {p_iter[1]:.2e} $\pm$ {p_iter[2]:.2e}")
        #     # ax[1].scatter(0,0, color = 'white', s = 0, label = lbl) #also display the original fit in legend
        # except:
        #     print("Iterative Gauss Fit failed")
        ax.set_aspect('auto')
        ax.legend()
        ax.set_title(f"Stokes Q NSR")

        # Stokes U Noise
        values = stk[cpos,2,sly,slx]

        ax = axs[1,1]
        hi = ax.hist(values.flatten(), bins=np.linspace(-1e-2,1e-2,200),)
        tmp = [0,0]
        tmp[0] = hi[0].astype('float64')
        tmp[1] = hi[1].astype('float64')

        #guassian fit + label
        pp = gaussian_fit(tmp, show = False)    
        xx=hi[1][:-1] + (hi[1][1]-hi[1][0])/2
        lbl = f'{pp[1]:.2e} $\pm$ {pp[2]:.2e}'


        ax.plot(xx,gaus(xx,*pp),'r--', label=lbl)
        # try:
        #     p_iter, hi_iter = iter_noise(values,[1.,0.,.1],eps=1e-6); p_iter[0] = pp[0]
        #     ax.plot(xx,gaus(xx,*p_iter),'g-.', label= f"Iter Fit: {p_iter[1]:.2e} $\pm$ {p_iter[2]:.2e}")
        #     # ax[1].scatter(0,0, color = 'white', s = 0, label = lbl) #also display the original fit in legend
        # except:
        #     print("Iterative Gauss Fit failed")
        ax.set_aspect('auto')
        ax.legend()
        ax.set_title(f"Stokes U NSR")

        # Stokes V Noise
        values = stk[cpos,3,sly,slx]

        ax = axs[1,2]
        hi = ax.hist(values.flatten(), bins=np.linspace(-1e-2,1e-2,200),)
        tmp = [0,0]
        tmp[0] = hi[0].astype('float64')
        tmp[1] = hi[1].astype('float64')

        #guassian fit + label
        pp = gaussian_fit(tmp, show = False)    
        xx=hi[1][:-1] + (hi[1][1]-hi[1][0])/2
        lbl = f'{pp[1]:.2e} $\pm$ {pp[2]:.2e}'


        ax.plot(xx,gaus(xx,*pp),'r--', label=lbl)
        # try:
        #     p_iter, hi_iter = iter_noise(values,[1.,0.,.1],eps=1e-6); p_iter[0] = pp[0]
        #     ax.plot(xx,gaus(xx,*p_iter),'g-.', label= f"Iter Fit: {p_iter[1]:.2e} $\pm$ {p_iter[2]:.2e}")
        #     # ax[1].scatter(0,0, color = 'white', s = 0, label = lbl) #also display the original fit in legend
        # except:
        #     print("Iterative Gauss Fit failed")
        ax.set_aspect('auto')
        ax.legend()
        ax.set_title(f"Stokes V NSR")

        if save_output:
            fig.savefig(p, format='pdf')
            plt.close(fig)

    # # -----------------------------------------------------------------------------
    # # Plot Stokes images
    # # -----------------------------------------------------------------------------

    if plot_stokes:
        # dat = np.transpose(stk, (1, 0, 2, 3))  # re-arrange Stokes and wavelength axes from [l,p,y,x] to [p,l,y,x]

        grayscales = [(.3,1.2)] + [(-3e-3,3e-3)]*3 # [1] + [0.01] * 3  # I, Q, U, V
        row_labels = ['I', 'Q', 'U', 'V']
        column_labels = ['{:.3f} nm'.format(wave) for wave in wavelengths]
        title = os.path.basename(datfile) 

        fig = show_image_array(
            stk, h, grayscales, row_labels=row_labels,
            column_labels=column_labels, fig_title=title)

        if save_output:
            fig.savefig(p, format='pdf')
            plt.close(fig)
    if save_output:
        p.close()

def blos_noise(blos_file, iter=True, fs = None):
    """plot blos on left panel, and blos hist + Gaussian fit (w/ iterative fit option - only shown in legend)

    Parameters
    ----------
    blos_file : str
        path to blos file
    iter : bool, optional
        performs iterative Gaussian fit, by default True
    fs : array, optional
        field stop mask, by default None

    Returns
    -------
    p or p_iter: fit coefficients for Gaussian function
    """
    blos = fits.getdata(blos_file)
    hdr = fits.getheader(blos_file)
    #get the pixels that we want to consider (central 512x512 and limb handling)
    _, _, _, sly, slx = limb_side_finder(blos, hdr)
    values = blos[sly,slx]

    fig, ax = plt.subplots(1,2, figsize = (14,6))
    if fs is not None:
        idx = np.where(fs<1)
        blos[idx] = -300
    im1 = ax[0].imshow(blos, cmap = "gray", origin = "lower", vmin = -200, vmax = 200)
    fig.colorbar(im1, ax = ax[0], fraction=0.046, pad=0.04)
    hi = ax[1].hist(values.flatten(), bins=np.linspace(-2e2,2e2,200))
    tmp = [0,0]
    tmp[0] = hi[0].astype('float64')
    tmp[1] = hi[1].astype('float64')

    #guassian fit + label
    p = gaussian_fit(tmp, show = False)    
    xx=hi[1][:-1] + (hi[1][1]-hi[1][0])/2
    lbl = f'{p[1]:.2e} $\pm$ {p[2]:.2e} G'
    
    if iter:
        ax[1].plot(xx,gaus(xx,*p),'r--', label=lbl)
        try:
            p_iter, hi_iter = iter_noise(values,[1.,0.,10.],eps=1e-4); p_iter[0] = p[0]
            ax[1].plot(xx,gaus(xx,*p_iter),'g--', label= f"Iter Fit: {p_iter[1]:.2e} $\pm$ {p_iter[2]:.2e} G")
            # ax[1].scatter(0,0, color = 'white', s = 0, label = lbl) #also display the original fit in legend
        except:
            print("Iterative Gauss Fit failed")
            p_iter = p

    else:
        ax[1].plot(xx,gaus(xx,*p),'r--', label=lbl)

    ax[1].legend(fontsize=15)

    date = blos_file.split('blos_')[1][:15]
    dt_str = datetime.datetime.strptime(date, "%Y%m%dT%H%M%S")
    fig.suptitle(f"Blos {dt_str}")

    plt.tight_layout()
    plt.show()

    if iter:
        return p_iter
    else:
        return p

def stokes_noise(stokes_file, iter=True):
    """plot stokes V on left panel, and Stokes V hist + Gaussian fit (w/ iterative option)

    Parameters
    ----------
    stokes_file : str
        path to stokes file
    iter : bool, optional
        whether to use iterative Gaussian fit, by default True

    Returns
    -------
    p or p_iter: array
        Gaussian fit parameters
    """
    stokes = fits.getdata(stokes_file)
    if stokes.shape[0] == 6:
        stokes = np.einsum('lpyx->yxpl',stokes)
    hdr = fits.getheader(stokes_file)
    out = fits_get_sampling(stokes_file)
    cpos = out[3]
    #first get the pixels that we want (central 512x512 and limb handling)
    _, _, _, sly, slx = limb_side_finder(stokes[:,:,3,cpos], hdr)
    values = stokes[sly,slx,3,cpos]

    fig, ax = plt.subplots(1,2, figsize = (14,6))
    im1 = ax[0].imshow(stokes[:,:,3,cpos], cmap = "gist_heat", origin = "lower", vmin = -1e-2, vmax = 1e-2)
    fig.colorbar(im1, ax = ax[0], fraction=0.046, pad=0.04)
    hi = ax[1].hist(values.flatten(), bins=np.linspace(-1e-2,1e-2,200))
    #print(hi)
    tmp = [0,0]
    tmp[0] = hi[0].astype('float64')
    tmp[1] = hi[1].astype('float64')

    #guassian fit + label
    p = gaussian_fit(tmp, show = False)    
    xx=hi[1][:-1] + (hi[1][1]-hi[1][0])/2
    lbl = f'{p[1]:.2e} $\pm$ {p[2]:.2e}'
    
    if iter:
        ax[1].plot(xx,gaus(xx,*p),'r--', label=lbl)
        try:
            p_iter, hi_iter = iter_noise(values,[1.,0.,.1],eps=1e-6); p_iter[0] = p[0]
            ax[1].plot(xx,gaus(xx,*p_iter),'g--', label= f"Iter Fit: {p_iter[1]:.2e} $\pm$ {p_iter[2]:.2e}")
            # ax[1].scatter(0,0, color = 'white', s = 0, label = lbl) #also display the original fit in legend
        except:
            print("Iterative Gauss Fit failed")
            ax[1].plot(xx,gaus(xx,*p),'r--', label=lbl)
            p_iter = p

    else:
        ax[1].plot(xx,gaus(xx,*p),'r--', label=lbl)

    ax[1].legend(fontsize=15)

    date = stokes_file.split('stokes_')[1][:15]
    dt_str = datetime.datetime.strptime(date, "%Y%m%dT%H%M%S")
    fig.suptitle(f"Stokes {dt_str}")

    plt.tight_layout()
    plt.show()

    if iter:
        return p_iter
    else:
        return p
