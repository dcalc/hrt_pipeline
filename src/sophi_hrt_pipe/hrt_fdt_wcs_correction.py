import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from .utils import find_nearest, printc, bcolors, image_derivative, get_descriptor
from .coordinates import rotate_header, translate_header, center_coord, circular_mask, remap, fft_shift, image_register, Inv2, und
from .processes import limb_side_finder, elliptical_mask, double_gaussian_fit
# import argparse
from datetime import datetime as DT
from datetime import timedelta as TD
import glob, os
import sunpy.map
from sunpy.coordinates import Helioprojective
from sunpy.coordinates import propagate_with_solar_surface
import warnings, sunpy
from astropy import units as u
from packaging import version

if version.parse(sunpy.__version__) >= version.parse("5.0"):
    # SunPy ≥ 5.0  →  the new class exists
    from sunpy.coordinates import SphericalScreen
else:                                   # SunPy < 5.0  →  fall back
    SphericalScreen = Helioprojective.assume_spherical_screen

warnings.filterwarnings("ignore", category=sunpy.util.SunpyMetadataWarning)

VERSION = '1.0.1'

def closestFDT(filename):
    # hdr = fits.open(filename)
    # btype = hdr[0].header['BTYPE']
    LL_dates_corrupted = ['2024-03-17',
                          '2024-03-21',
                          '2024-03-24',
                          '2024-03-25']
    
    descriptor = get_descriptor(filename)

    date_obs = DT.strptime(filename.split('_')[-3], '%Y%m%dT%H%M%S')

    MAX_DAYS = 3
    delta_days = 1
    while delta_days <= MAX_DAYS:
        t0 = date_obs - TD(days=delta_days)
        dates = [(t0+TD(days=i)).strftime('%Y-%m-%d') for i in range(3)]

        fdt_files = []
        for d in dates:
            fdt_files += sorted(glob.glob(f'/data/solo/phi/data/fmdb/public/l2/{d}/*phi-fdt-{descriptor}*.fits.gz'))

        if len(fdt_files) == 0:
            print(f'No FDT L2 {descriptor} file, looking for NRT')
            for d in dates:
                fdt_files += sorted(glob.glob(f'/scratch/valori/nrt_fmdb/l2/{d}/*phi-fdt-{descriptor}*.fits.gz'))
        
        if len(fdt_files) == 0:
            print(f'No FDT NRT {descriptor} file, looking for LL')
            for d in dates:
                if d not in LL_dates_corrupted:
                    fdt_files += sorted(glob.glob(f'/data/solo/phi/data/fmdb/ll/{d}/*phi-fdt-{descriptor}*.fits.gz'))

        # if len(fdt_files) == 0:
        #     print(f'No FDT LL {descriptor} file, looking for L1')
        #     for d in dates:
        #         fdt_files += sorted(glob.glob(f'/data/solo/phi/data/fmdb/l1/{d}*phi-fdt-*lam*.fits.gz'))
        #     filename = filename.replace(descriptor, 'icnt')
        #     if os.path.isfile(filename):
        #         print('Using raw FDT data, so I will use HRT continuum intensity')
        #     else:
        #         raise FileNotFoundError('There is no HRT continuum intensity associated to this file!')

        if len(fdt_files) == 0:
            delta_days += 1
            if delta_days <= MAX_DAYS:
                print('Increasing delta_days to ',delta_days)
        else:
            break
    if len(fdt_files) == 0:
        raise FileNotFoundError('There is no FDT file associated to this HRT, sorry :-(')

    t_obs_fdt = [DT.fromisoformat(fits.getheader(f)['DATE-OBS']) for f in fdt_files]
    ind = find_nearest(t_obs_fdt, date_obs)
    fdt_filename = fdt_files[ind]

    return fdt_filename, descriptor

def limb_fixedR(img, hdr, field_stop, AR_mask):
    """Fits limb to the image using least squares method.

    Parameters
    ----------
    img : numpy.ndarray
        Image to fit limb to.
    hdr : astropy.io.fits.header.Header
        header of fits file
    field_stop : array
        field stop array
    AR_mask : array
        AR mask array
    
    Returns
    -------
    output: dictionary
        Output with most of the variables and outputs of the procedure.

    """

    def _residuals(p,x,y):
        """
        Finding the residuals of the fit

        Parameters
        ----------
        p : list
            [a,b,h,k,A] - ellipse axes (x,y), centers (x,y) and angle
        x : float
            test x coordinate
        y : float
            test y coordinate

        Returns
        -------
        residual = ((x-h)*np.cos(A)+(y-k)*np.sin(A))**2/a**2 + (-(x-h)*np.sin(A)+(y-k)*np.cos(A))**2/b**2 - 1
        """

        a,b,h,k,A = p
        residual = ((x-h)*np.cos(A)+(y-k)*np.sin(A))**2/a**2 + (-(x-h)*np.sin(A)+(y-k)*np.cos(A))**2/b**2 - 1

        return residual

    from scipy.optimize import least_squares
    from scipy.ndimage import binary_erosion, binary_dilation

    side, center, Rpix, sly, slx = limb_side_finder(img,hdr,verbose=False)

    s = 5
    temp = img[s:-s,s:-s][AR_mask[s:-s,s:-s]>0].flatten()
    hi = np.histogram(temp,bins=np.linspace(-0.07,temp.max(),100)); del temp
    gres, cov = double_gaussian_fit(hi,False,True)
    
    if (np.any((np.sqrt(np.diagonal(cov))/gres)[:3] > 100) or np.any(np.isnan(cov))) or gres[1] > gres[4]*0.7: # sometimes south pole limb is not found, so extra condition on fit
        printc('Despite the WCS, it looks like the Limb is not in the FoV',bcolors.WARNING)
        
        return {'hi':hi,'gres':gres,'cov':cov,'center':center,'Rpix':Rpix}

    xx=hi[1][:-1] + (hi[1][1]-hi[1][0])/2
    thr = xx[find_nearest(xx,min(gres[1],gres[4]))+np.argmin(hi[0][find_nearest(xx,min(gres[1],gres[4])):find_nearest(xx,max(gres[1],gres[4]))])]

    limb_mask = img[s:-s,s:-s]>thr;
    
    # dilation and erosion to remove any possible zeros coming from umbrae
    # border_value=1 in erosion to avoid black edges
    limb_mask = binary_erosion(binary_dilation(limb_mask,[[0,1,0],[1,1,1],[0,1,0]],iterations=20),[[0,1,0],[1,1,1],[0,1,0]],iterations=20,border_value=1)
    
    # erosion of field stop and AR_mask to avoid edges from there
    limb_edge = image_derivative(limb_mask)*binary_erosion(field_stop,[[0,1,0],[1,1,1],[0,1,0]],iterations=20)[s:-s,s:-s]\
                                           *binary_erosion(AR_mask,[[0,1,0],[1,1,1],[0,1,0]],iterations=20)[s:-s,s:-s]
    yi, xi = np.where(limb_edge>0.9)
    yi += s; xi += s;
    # max gradient along small vertical cuts
    if 'N' in side or 'S' in side:
        # print('N or S')
        xi,ar = np.unique(xi,return_index=True)
        yi = yi[ar]
        delta = 20
        img_der = image_derivative(img)
        cut = np.zeros((delta*2,img.shape[1]))
        new_yi = yi-delta
        count = 0
        for x0,y0 in zip(xi,yi):
            cut[:,x0] = img_der[y0-delta:y0+delta,x0]
            new_yi[count] += cut[:,x0].argmax()
            count += 1
        yi = new_yi

    # max gradient along small vertical cuts
    elif 'E' in side or 'W' in side:
        # print('E or W')
        yi,ar = np.unique(yi,return_index=True)
        xi = xi[ar]
        delta = 20
        img_der = image_derivative(img)
        cut = np.zeros((img.shape[0],delta*2))
        new_xi = xi-delta
        count = 0
        for x0,y0 in zip(xi,yi):
            cut[y0] = img_der[y0,x0-delta:x0+delta]
            new_xi[count] += cut[y0].argmax()
            count += 1
        xi = new_xi
    
    p = least_squares(_residuals,x0 = [Rpix,Rpix,center[0],center[1],0], args=(xi,yi),
                              bounds = ([Rpix-.1,Rpix-.1,center[0]-1000,center[1]-1000,-np.pi/2],[Rpix+.1,Rpix+.1,center[0]+1000,center[1]+1000,np.pi/2]))

    mask100 = elliptical_mask(img.shape,p.x)
    mask98 = elliptical_mask(img.shape,[p.x[0]*.98,p.x[1]*.98,p.x[2],p.x[3],p.x[4]])
    mask96 = elliptical_mask(img.shape,[p.x[0]*.96,p.x[1]*.96,p.x[2],p.x[3],p.x[4]])
    
    return {'mask100':mask100,'mask96':mask96,'hi':hi,'gres':gres,'thr':thr,'xx':xx,'limb_mask':limb_mask,'limb_edge':limb_edge,'yi':yi,'xi':xi,'p':p}
    
def correct_wcs_with_limb(img, hdr, field_stop, AR_mask, crota_manual_correction=0.15):
    good = False
    try:
        out = limb_fixedR(img, hdr, field_stop, AR_mask)
        p = out['p'].x.copy()
        h_hrt = rotate_header(hdr.copy(),-crota_manual_correction, center=[p[2],p[3]])
        center = center_coord(h_hrt)
        shift_center = [round(p[3] - center[1]), round(p[2] - center[0])] # (y,x)
        h_hrt = translate_header(h_hrt.copy(),np.asarray(shift_center), mode='crval')
        good = True
    except Exception as e:
        printc(f"There was an error in the WCS correction using the limb. This is the error: {e}", color=bcolors.FAIL)
        good = False
        h_hrt = hdr.copy()
    return h_hrt, good

def prepare_data(hrt_file, fdt_file, crota_manual_correction=0.15, undistortion=False, verbose=False):
    if isinstance(hrt_file, str):
        hdr_hrt = fits.open(hrt_file)
        hrt = hdr_hrt[0].data; h_hrt = hdr_hrt[0].header
    elif isinstance(hrt_file, tuple):
        hrt, h_hrt = hrt_file
    else:
        raise ValueError('hrt_file must be a string or a tuple')
    
    start_row = int(h_hrt['PXBEG2']-1)
    start_col = int(h_hrt['PXBEG1']-1)
    cpos = h_hrt['CONTPOS'] - 1

    h_hrt = rotate_header(h_hrt.copy(),-crota_manual_correction, center=center_coord(h_hrt))

    if hrt.ndim == 3:
        hrt = hrt[cpos*4]
    elif hrt.ndim == 4:
        if hrt.shape[0] == 6:
            hrt = hrt[cpos,0]            
        else:
            hrt = hrt[:,:,0,cpos]
    original_shape = hrt.shape

    if 'CAL_LIMB' in h_hrt:
        p = np.asarray(eval(h_hrt['CAL_LIMB']))
        n_erosion = 3
        p[0] -= n_erosion; p[1] -= n_erosion # reducing the axis of the ellipse
        ell = elliptical_mask(hrt.shape, p)
        printc(f'Masking out {n_erosion} pixels from the limb',bcolors.WARNING)
        hrt *= ell
        
        center = center_coord(h_hrt)
        shift_center = [round(p[3] - center[1],2), round(p[2] - center[0],2)] # (y,x)
        printc(f'Translating the header by {shift_center} pixels according to the center of the limb fit procedure',bcolors.WARNING)
        h_hrt = translate_header(h_hrt.copy(),np.asarray(shift_center), mode='crval')

    #TODO: FDT undistortion as input and crop HRT after undistortion (?)
    if undistortion:
        if hrt.shape[0] == 2048:
            und_hrt = und(hrt)
            h_hrt['CRPIX1'],h_hrt['CRPIX2'] = Inv2(1016,982,h_hrt['CRPIX1'],h_hrt['CRPIX2'],8e-9)
        else:
            hrt = np.pad(hrt,[(start_row,2048-(start_row+hrt.shape[0])),(start_col,2048-(start_col+hrt.shape[1]))])
            h_hrt['NAXIS1'] = 2048; h_hrt['NAXIS2'] = 2048
            h_hrt['PXBEG1'] = 1; h_hrt['PXBEG2'] = 1; h_hrt['PXEND1'] = 2048; h_hrt['PXEND2'] = 2048; 
            h_hrt['CRPIX1'] += start_col; h_hrt['CRPIX2'] += start_row
            hrt_map = sunpy.map.Map((und_hrt,h_hrt))
            und_hrt = und(hrt)
            h_hrt['CRPIX1'],h_hrt['CRPIX2'] = Inv2(1016,982,h_hrt['CRPIX1'],h_hrt['CRPIX2'],8e-9)
    else:
        und_hrt = hrt

    hrt_map = sunpy.map.Map((und_hrt,h_hrt))

    fdt_map = sunpy.map.Map(fdt_file)
    fdt_map_rot = fdt_map.rotate((fdt_map.fits_header['CROTA']-hrt_map.fits_header['CROTA'])*u.deg,method='opencv')
    if verbose:
        plot_fdt_hrt(fdt_map_rot, hrt_map)
    
    return hrt_map, fdt_map, fdt_map_rot

        
def correction(hrt_map, fdt_map, deriv=False,verbose=False,max_iterations = 10):
    from sunpy.coordinates import RotatedSunFrame
    from astropy.coordinates import SkyCoord

    ht = hrt_map.fits_header
    und_hrt = hrt_map.data.copy()
    t0 = ht['DATE-AVG']
    t_obs = DT.fromisoformat(ht['DATE-AVG'])

    h_hrt = ht.copy()
    ht['DATE-BEG'] = t0; ht['DATE-OBS'] = t0
    # ht['DATE-BEG'] = datetime.datetime.isoformat(datetime.datetime.fromisoformat(ht['DATE-BEG']) + dltt)
    # ht['DATE-OBS'] = datetime.datetime.isoformat(datetime.datetime.fromisoformat(ht['DATE-OBS']) + dltt)
    shift = [1,1]
    i = 0
    match = True

    fdt_map = sunpy.map.Map((np.nan_to_num(fdt_map.data,nan=0,posinf=0,neginf=0),fdt_map.fits_header))

    while np.any(np.abs(shift)>1e-1):
        hrt_map = sunpy.map.Map((und_hrt,ht))
        
        with propagate_with_solar_surface():
            with SphericalScreen(fdt_map.observer_coordinate, only_off_disk=True):
                
                bl = hrt_map.bottom_left_coord
                tr = hrt_map.top_right_coord

                # durations = ((fdt_map.date-hrt_map.date).value*u.day)
                # diffrot_tr = RotatedSunFrame(base=tr, duration=durations)
                # diffrot_tr = diffrot_tr.transform_to(fdt_map.coordinate_frame)
                # diffrot_bl = RotatedSunFrame(base=bl, duration=durations)
                # diffrot_bl = diffrot_bl.transform_to(fdt_map.coordinate_frame)
                
                # top_right = fdt_map.world_to_pixel(SkyCoord(diffrot_tr))
                # bottom_left = fdt_map.world_to_pixel(SkyCoord(diffrot_bl))
                top_right = fdt_map.world_to_pixel(tr)
                bottom_left = fdt_map.world_to_pixel(bl)

                tr_fdt_map = np.array([top_right.x.value,top_right.y.value])
                bl_fdt_map = np.array([bottom_left.x.value,bottom_left.y.value])

                fdt_submap = fdt_map.submap(bl_fdt_map*u.pix,top_right=tr_fdt_map*u.pix)
                _,c,R,_,_ = limb_side_finder(fdt_submap.data,fdt_submap.fits_header,verbose=False)
                mask = circular_mask(*fdt_submap.data.shape,c[:2],R)
                tt = fdt_submap.data; tt[mask==0] = 0
                fdt_submap = sunpy.map.Map((tt,fdt_submap.fits_header)); del tt

                hrt_remap = remap(fdt_submap, hrt_map, out_shape = fdt_submap.data.shape)

                if verbose:
                    plot_fdt_hrt(fdt_submap, hrt_remap)
            
        ref = np.nan_to_num(fdt_submap.data.copy(),True,0,0,0)
        temp = np.nan_to_num(hrt_remap.data.copy(),True,0,0,0)

        # masking to 99% of the sun to avoid limb fitting issues (mostly in LL)
        center_fdt = center_coord(fdt_submap.fits_header)
        Rfdt = fdt_submap.fits_header['RSUN_ARC'] / fdt_submap.fits_header['CDELT1']
        mask99 = circular_mask(*ref.shape,center_fdt,Rfdt*0.99)
        ref[mask99==0] = 0

        s = [1,1]
        shift = [0,0]
        it = 0

        deriv = False

        while np.any(np.abs(s)>1e-1) and it<10:
            if it == 0:
                _,s = image_register(ref,temp,False,deriv)
                # plt.figure(figsize=(18,6))
                # plt.subplot(131)
                # plt.imshow(cc,cmap='gray',origin='lower')
                # plt.subplot(132)
                # plt.imshow(ref,clim=(-50,50),cmap='bwr',origin='lower')
                # plt.subplot(133)
                # plt.imshow(temp,clim=(-50,50),cmap='bwr',origin='lower')
                # plt.show()

                if np.any(np.abs(s)==0):
                    _,s = image_register(ref,temp,True,deriv)
            else:
                _,s = image_register(ref,temp,True,deriv)
                # sr, sc, _ = SPG_shifts_FFT(np.asarray([ref,temp])); s = [sr[1],sc[1]]
            shift = [shift[0]+s[0],shift[1]+s[1]]
            # temp = fft_shift(hmi_map_wcs.data.copy(), shift); temp[np.isinf(temp)] = 0; temp[np.isnan(temp)] = 0
            temp = fft_shift(np.nan_to_num(hrt_remap.data.copy(),True,0,0,0), shift); temp[np.isinf(temp)] = 0; temp[np.isnan(temp)] = 0
            it += 1

        ht = translate_header(ht.copy(),-np.asarray(shift)*fdt_submap.fits_header['CDELT1']/hrt_map.fits_header['CDELT1'] * \
                                        fdt_map.fits_header['DSUN_OBS']/hrt_map.fits_header['DSUN_OBS'],
                                        mode='crval')
        print(it,'iterations shift FDT detector frame (x,y):',round(shift[1],2),round(shift[0],2))

        i+=1
        if i == max_iterations:
            print('Maximum iterations reached:',i)
            match = False
            break
    
    hrt_map = sunpy.map.Map((und_hrt,ht))
    # fdt_remap = remap(hrt_map, fdt_map, out_shape = hrt_map.data.shape)
    hrt_remap = remap(fdt_submap, hrt_map, out_shape = fdt_submap.data.shape)

    ht['DATE-BEG'] = h_hrt['DATE-BEG']
    ht['DATE-OBS'] = h_hrt['DATE-OBS']
    hrt_map = sunpy.map.Map((und_hrt,ht))     

    if match:# or np.any(np.abs(s)<0.5):
        newWCS = dict(DID=ht['PHIDATID'], 
                    CROTA=ht['CROTA'], PC1_1=ht['PC1_1'], PC1_2=ht['PC1_2'], PC2_1=ht['PC2_1'], PC2_2=ht['PC2_2'],
                    CRPIX1=ht['CRPIX1'], CRPIX2=ht['CRPIX2'], 
                    CRVAL1=ht['CRVAL1'], CRVAL2=ht['CRVAL2'])
    elif 'CAL_LIMB' in ht:
        newWCS = dict(DID=ht['PHIDATID'], 
                    CROTA=ht['CROTA'], PC1_1=ht['PC1_1'], PC1_2=ht['PC1_2'], PC2_1=ht['PC2_1'], PC2_2=ht['PC2_2'],
                    CRPIX1=ht['CRPIX1'], CRPIX2=ht['CRPIX2'], 
                    CRVAL1=ht['CRVAL1'], CRVAL2=ht['CRVAL2'])
    else:
        newWCS = dict(DID=None,
                    CROTA=None, PC1_1=None, PC1_2=None, PC2_1=None, PC2_2=None,
                    CRPIX1=None, CRPIX2=None, 
                    CRVAL1=None, CRVAL2=None)    
    
    return hrt_map, hrt_remap, newWCS, t0

def plot_fdt_hrt(fdt_map, hrt_map):
    fig = plt.figure(layout='tight',figsize=(7,7))
    ax = fig.add_subplot(projection=fdt_map)
    plot = dict(clim=(-50,50))
    with Helioprojective.assume_spherical_screen(fdt_map.observer_coordinate,True):
        fdt_map.plot(axes=ax, **plot, cmap='gray', zorder=0)
        hrt_map.plot(axes=ax, **plot, cmap='bwr', alpha=0.5,
                    autoalign=True, zorder=1)
        fdt_map.draw_limb(ax,color='y')

        x0 = fdt_map.world_to_pixel(hrt_map.pixel_to_world(x=1*u.pix,y=1*u.pix))
        x1 = fdt_map.world_to_pixel(hrt_map.pixel_to_world(x=1*u.pix,y=hrt_map.data.shape[0]*u.pix))
        x2 = fdt_map.world_to_pixel(hrt_map.pixel_to_world(x=hrt_map.data.shape[1]*u.pix,y=hrt_map.data.shape[0]*u.pix))
        x3 = fdt_map.world_to_pixel(hrt_map.pixel_to_world(x=hrt_map.data.shape[1]*u.pix,y=1*u.pix))

        ax.plot([x.x.value for x in [x0,x1,x2,x3,x0]],[x.y.value for x in [x0,x1,x2,x3,x0]],'r--',linewidth=1.5)
    plt.show()

    return fig, ax

def run_FDT_correction(data, header, verbose = False, **kwargs):

    parameters = dict(filename = None, print_values = False, crota_manual_correction = 0.15)
    parameters = {**parameters, **kwargs}

    if parameters['filename'] is not None:
        filename = parameters['filename']
        if not os.path.isfile(filename):
            print('filename is not a file, looking for files in directory')
            filename = sorted(glob.glob(filename))
            if len(filename) == 0:
                raise FileNotFoundError(f'File {filename} not found')
        else:
            filename = [filename]
    else:
        filename = header['FILENAME']
        descriptor = get_descriptor(filename)
        filename = filename.replace(descriptor, 'blos')
        filename = [filename]

    newWCS = dict(DID=[], 
                  CROTA=[], PC1_1=[], PC1_2=[], PC2_1=[], PC2_2=[],
                  CRPIX1=[], CRPIX2=[], 
                  CRVAL1=[], CRVAL2=[])
    
    if parameters['print_values']:
        ends = [',']*(len(newWCS.keys())-1)+['\n']
        cols = ''
        for e,k in zip(ends,newWCS.keys()):
            cols += k+e
    
    for f in filename:
        try:
            fdt_filename, descriptor = closestFDT(f)
        except FileNotFoundError as e:
            printc('No FDT file could be found for this HRT file, return None', color=bcolors.FAIL)
            for key in newWCS.keys():
                newWCS[key].append(None)
            continue

        print("")
        print(f'Processing the file: {f}')
        print(f'Closest FDT file: {fdt_filename}')
        print("")
        # print(f'Processing the file: {filename}')

        if parameters['filename'] is not None:
            hrt_map, fdt_map, fdt_map_rot = prepare_data(f, fdt_filename, parameters['crota_manual_correction'], undistortion=False, verbose=False)
        else:
            hrt_map, fdt_map, fdt_map_rot = prepare_data((data,header), fdt_filename, parameters['crota_manual_correction'], undistortion=False, verbose=False)

        try:
            hrt_map, hrt_remap, n, t0 = correction(hrt_map, fdt_map_rot, deriv=False,verbose=False)
        except Exception as e:
            printc(f"There was an error in the WCS correction, return None. This is the error: {e}", color=bcolors.FAIL)
            for key in newWCS.keys():
                newWCS[key].append(None)
            continue

        if parameters['print_values']:
            for e,v in zip(ends,n.values()):
                if isinstance(v,str):
                    cols += v+e
                elif v is None:
                    cols += 'None'+e
                else:
                    cols += '{:{width}.{prec}f}'.format(v,width=5,prec=3)+e
        
        for key, value in n.items():
            newWCS[key].append(value)

        if verbose:
            fig, ax = plot_fdt_hrt(fdt_map, hrt_remap)

    if parameters['print_values']:
        print('WCS correction results:')
        print(cols)

    return newWCS
    
