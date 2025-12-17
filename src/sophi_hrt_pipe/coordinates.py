from astropy.io import fits
import numpy as np
import os
import matplotlib.pyplot as plt
# import scipy.optimize as spo
# import scipy.signal as sps
# from datetime import datetime as dt
import datetime
# import time
# from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

# from scipy.ndimage import map_coordinates
from astropy import units as u
from astropy.wcs import WCS

from .utils import circular_mask, und, Inv2, fits_get_sampling, fft_shift, printc, bcolors, image_derivative

def mu_angle(hdr,coord=None):
    """get mu angle for a pixel

    Parameters
    ----------
    hdr : header or filename
        header of the fits file or filename path
    coord : array, optional
        pixel location (x,y) for which the mu angle is found (if None: center of the FoV), by default None.
        Shape has to be (2,Npix)

    Returns
    -------
    mu : float
        cosine of the heliocentric angle
    """
    if type(hdr) is str:
        hdr = fits.getheader(hdr)
    
    center=center_coord(hdr)
    try:
        Rpix=(hdr['RSUN_ARC']/hdr['CDELT1'])
    except:
        Rpix=(hdr['RSUN_OBS']/hdr['CDELT1'])
    if coord is None:
        coord = np.asarray([(hdr['PXEND1']-hdr['PXBEG1'])/2,
                            (hdr['PXEND2']-hdr['PXBEG2'])/2]) - center[:2]
    else:
        coord -= center[:2,np.newaxis]
    temp = Rpix**2 - (coord[0]**2 + coord[1]**2)
    if isinstance(temp, (list,np.ndarray)):
        temp[temp<0] = np.nan
    else:
        if temp < 0: temp = np.nan
    mu = np.sqrt(temp) / Rpix
    
    return mu

def center_coord(hdr):
    """calculate the center of the solar disk in the rotated reference system

    Parameters
    ----------
    hdr : header
        header of the fits file

    Returns
    -------
    center: [x,y,1] coordinates of the solar disk center (units: pixel)
    """
    pxsc = hdr['CDELT1']
    # sun_dist_m=(hdr['DSUN_AU']*u.AU).to(u.m).value #Earth
    # sun_dist_AU=hdr['DSUN_AU'] #Earth
    # rsun = hdr['RSUN_REF'] # m
    # pxbeg1 = hdr['PXBEG1']
    # pxend1 = hdr['PXEND1']
    # pxbeg2 = hdr['PXBEG2']
    # pxend2 = hdr['PXEND2']
    crval1 = hdr['CRVAL1']
    crval2 = hdr['CRVAL2']
    crpix1 = hdr['CRPIX1']
    crpix2 = hdr['CRPIX2']
    if 'PC1_1' in hdr:
        PC1_1 = hdr['PC1_1']
        PC1_2 = hdr['PC1_2']
        PC2_1 = hdr['PC2_1']
        PC2_2 = hdr['PC2_2']
    else:
        if 'CROTA2' in hdr:
            CROTA = hdr['CROTA2']
        else:
            CROTA = hdr['CROTA']
        PC1_1 = np.cos(CROTA*np.pi/180)
        PC1_2 = -np.sin(CROTA*np.pi/180)
        PC2_1 = np.sin(CROTA*np.pi/180) 
        PC2_2 = np.cos(CROTA*np.pi/180)
    
    HPC1 = 0
    HPC2 = 0
    
    x0 = crpix1 + 1/pxsc * (PC1_1*(HPC1-crval1) - PC1_2*(HPC2-crval2)) - 1
    y0 = crpix2 + 1/pxsc * (PC2_2*(HPC2-crval2) - PC2_1*(HPC1-crval1)) - 1
    
    return np.asarray([x0,y0,1])

def rotate_header(h,angle,center = [1024.5,1024.5]):
    """calculate new header when image is rotated by a fixed angle

    Parameters
    ----------
    h : astropy.io.fits.header.Header
        header of image to be rotated
    angle : float
        angle to rotate image by (in degrees)
    center : list or numpy.array
        center of rotation (x,y) in pixel, Default is [1024.5,1024.5]

    Returns
    -------
    h: astropy.io.fits.header.Header
        new header
    """
    if 'CROTA' in h:
        k = 'CROTA' # positive angle = clock-wise rotation of the reference system axes 
    else:
        k = 'CROTA2'
    h[k] -= angle
    h['PC1_1'] = np.cos(h[k]*np.pi/180)
    h['PC1_2'] = -np.sin(h[k]*np.pi/180)
    h['PC2_1'] = np.sin(h[k]*np.pi/180)
    h['PC2_2'] = np.cos(h[k]*np.pi/180)
    rad = angle * np.pi/180
    rot = np.asarray([[np.cos(rad),-np.sin(rad),0],[np.sin(rad),np.cos(rad),0],[0,0,1]])
    coords = np.asarray([h['CRPIX1'],h['CRPIX2'],1])
#     center = [1024.5,1024.5] # CRPIX from 1 to 2048, so 1024.5 is the center
    tr = np.asarray([[1,0,center[0]],[0,1,center[1]],[0,0,1]])
    invtr = np.asarray([[1,0,-center[0]],[0,1,-center[1]],[0,0,1]])

    M = tr @ rot @ invtr
    bl = M @ np.asarray([0,0,1])
    tl = M @ np.asarray([0,2048,1])
    br = M @ np.asarray([2048,0,1])
    tr = M @ np.asarray([2048,2048,1])

    O = -np.asarray([bl,tl,br,tr]).min(axis=0)[:-1]
    newO = np.asarray([[1,0,O[0]+1],[0,1,O[1]+1],[0,0,1]])
    newM = newO @ M
    new_coords = newM @ coords
    h['CRPIX1'] = round(new_coords[0],4)
    h['CRPIX2'] = round(new_coords[1],4)
    
    return h
    
def translate_header(h,tvec,mode='crpix'):
    """calculate new header when image is translated by a fixed vector

    Parameters
    ----------
    h : astropy.io.fits.header.Header
        header of image to be translated
    tvec : list
        vector to translate image by (in pixels) [y,x]
    mode : str
        if 'crpix' (Default) the shift will be applied to CRPIX*, if 'crval' the shift will be applied to CRVAL*

    Returns
    -------
    h: astropy.io.fits.header.Header
        new header
    """
    if mode == 'crval':
        tr = np.asarray([[1,0,-tvec[1]],[0,1,-tvec[0]],[0,0,1]])
        if 'CROTA' in h:
            angle = h['CROTA'] # positive angle = clock-wise rotation of the reference system axes 
        else:
            angle = h['CROTA2']
        rad = angle * np.pi/180
        vec = np.asarray([tvec[1],tvec[0],1])
        rot = np.asarray([[np.cos(rad),-np.sin(rad),0],[np.sin(rad),np.cos(rad),0],[0,0,1]])
        shift = rot @ vec
        shift[0] *= h['CDELT1']
        shift[1] *= h['CDELT2']
        h['CRVAL1'] = round(h['CRVAL1']-shift[0],4)
        h['CRVAL2'] = round(h['CRVAL2']-shift[1],4)
    elif mode == 'crpix':
        tr = np.asarray([[1,0,tvec[1]],[0,1,tvec[0]],[0,0,1]])
        coords = np.asarray([h['CRPIX1'],h['CRPIX2'],1])
        new_coords = tr @ coords
        h['CRPIX1'] = round(new_coords[0],4)
        h['CRPIX2'] = round(new_coords[1],4)
    else:
        print('mode not valid\nreturn old header')
    return h

def image_register(ref,im,subpixel=True,deriv=False,d=50,verbose=True):
    """
    credits: Marco Stangalini (2010, IDL version). Adapted for Python by Daniele Calchetti.
    return shift as **(y,x)**
    """
    try:
        import pyfftw.interfaces.numpy_fft as fft
    except:
        import numpy.fft as fft
        
    def _g2d(X, offset, amplitude, sigma_x, sigma_y, xo, yo, theta):
        import numpy as np
        (x, y) = X
        xo = float(xo)
        yo = float(yo)
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)
                                + c*((y-yo)**2)))
        return g.ravel()

    def _gauss2dfit(a,mask):
        import numpy as np
        from scipy.optimize import curve_fit
        sz = np.shape(a)
        X,Y = np.meshgrid(np.arange(sz[1])-sz[1]//2,np.arange(sz[0])-sz[0]//2)
        c = np.unravel_index(a.argmax(),sz)
        Xf = X[mask>0]; Yf = Y[mask>0]; af = a[mask>0]

        # y = a[c[0],:]
        # x = X[c[0],:]
        stdx = .5 #np.sqrt(abs(sum(y * (x - sum(x*y)/sum(y))**2) / sum(y)))
        # y = a[:,c[1]]
        # x = Y[:,c[1]]
        stdy = .5 #np.sqrt(abs(sum(y * (x - sum(x*y)/sum(y))**2) / sum(y)))
        initial_guess = [np.median(a), np.max(a), stdx, stdy, c[1] - sz[1]//2, c[0] - sz[0]//2, 0]
        bounds = ([-1,-1,0,0,initial_guess[4]-1,initial_guess[5]-1,-180],
                  [1,1,initial_guess[2]*4,initial_guess[2]*4,initial_guess[4]+1,initial_guess[5]+1,180])

        popt, pcov = curve_fit(_g2d, (Xf, Yf), af.ravel(), p0=initial_guess,bounds=bounds)
        return np.reshape(_g2d((X,Y), *popt), sz), popt
    
    def _one_power(array):
        return array/np.sqrt((np.abs(array)**2).mean())

    if deriv:
        ref = image_derivative(ref - np.mean(ref))
        im = image_derivative(im - np.mean(im))
        
    shifts=np.zeros(2)
    FT1=fft.fftn(ref - np.mean(ref))
    FT2=fft.fftn(im - np.mean(im))
    ss=np.shape(ref)
    r=np.real(fft.ifftn(_one_power(FT1) * _one_power(FT2.conj())))
    r = fft.fftshift(r)
    rmax=np.max(r)
    ppp = np.unravel_index(np.argmax(r),ss)
    shifts = [ppp[0]-ss[0]//2,ppp[1]-ss[1]//2]
    if subpixel:
        dd = [d,75,100,30]
        for d1 in dd:
            try:
                if d1>0:
                    mask = circular_mask(ss[0],ss[1],[ppp[1],ppp[0]],d1)
                else:
                    mask = np.ones(ss,dtype=bool); d = ss[0]//2
                g, A = _gauss2dfit(r,mask)
                break
            except RuntimeError as e:
                if verbose: print(f"Issue with gaussian fitting using mask with radius {d1}\nTrying new value...")
                if d1 == dd[-1]:
                    raise RuntimeError(e)
        shifts[0] = A[5]
        shifts[1] = A[4]
        del g
    del FT1, FT2
    return r, shifts

def remap(ref_map, temp_map, out_shape = None):
    """
    reproject hmi map onto hrt with hrt pixel size and observer coordinates
    from SO/PHI-HRT pipeline
    Parameters
    ----------
    ref_map : sunpy.map.GenericMap
        reference map
    temp_map : sunpy.map.GenericMap
        map to be remapped
    out_shape : tuple, None
        shape of output map, if None the out_shape is the shape of ref_map (Default: None)

    Returns
    -------
    temp_remap : sunpy.map.GenericMap
        remapped hmi map
    """
    import sunpy.map
    from reproject import reproject_adaptive
    from sunpy.coordinates import propagate_with_solar_surface
    from astropy.wcs import WCS

    if out_shape is None:
        out_shape = ref_map.data.shape

    # define new header for hmi map using hrt observer coordinates
    with propagate_with_solar_surface():
        out_header = sunpy.map.make_fitswcs_header(
            out_shape,
            ref_map.reference_coordinate.replicate(rsun=temp_map.reference_coordinate.rsun),
            scale=u.Quantity(ref_map.scale),
            instrument=temp_map.instrument,
            observatory=temp_map.observatory,
            wavelength=temp_map.wavelength
        )

        out_header['dsun_obs'] = ref_map.coordinate_frame.observer.radius.to(u.m).value
        out_header['hglt_obs'] = ref_map.coordinate_frame.observer.lat.value
        out_header['hgln_obs'] = ref_map.coordinate_frame.observer.lon.value
        out_header['detector'] = temp_map.detector
        out_header['crpix1'] = ref_map.fits_header['CRPIX1']
        out_header['crpix2'] = ref_map.fits_header['CRPIX2']
        out_header['crval1'] = ref_map.fits_header['CRVAL1']
        out_header['crval2'] = ref_map.fits_header['CRVAL2']

        if 'CROTA' in ref_map.fits_header:
            key = 'CROTA'
        else:
            key = 'CROTA2'
        out_header[key] = ref_map.fits_header[key]

        if 'PC1_1' in ref_map.fits_header:
            out_header['PC1_1'] = ref_map.fits_header['PC1_1']
            out_header['PC1_2'] = ref_map.fits_header['PC1_2']
            out_header['PC2_1'] = ref_map.fits_header['PC2_1']
            out_header['PC2_2'] = ref_map.fits_header['PC2_2']
        else:
            out_header['PC1_1'] = np.cos(ref_map.fits_header[key]*np.pi/180)
            out_header['PC1_2'] = -np.sin(ref_map.fits_header[key]*np.pi/180)
            out_header['PC2_1'] = np.sin(ref_map.fits_header[key]*np.pi/180)
            out_header['PC2_2'] = np.cos(ref_map.fits_header[key]*np.pi/180)

        out_wcs = WCS(out_header)

        # reprojection
        temp_origin = temp_map
        output, _ = reproject_adaptive(temp_origin, out_wcs, out_shape,kernel='Hann',boundary_mode='ignore')
        temp_remap = sunpy.map.Map((output, out_header))
    temp_remap.plot_settings = temp_origin.plot_settings

    return temp_remap

def subregion_selection(ht,start_row,start_col,original_shape,dsmax = 512,edge = 20):
    intcrpix1 = int(round(ht['CRPIX1']))
    intcrpix2 = int(round(ht['CRPIX2']))
    ds = min(dsmax,
             intcrpix2-start_row-edge,
             intcrpix1-start_col-edge,
             original_shape[0]+start_row-intcrpix2-edge,
             original_shape[1]+start_col-intcrpix1-edge)
    sly = slice(intcrpix2-ds,intcrpix2+ds)
    slx = slice(intcrpix1-ds,intcrpix1+ds)
    
    return sly, slx

def downloadClosestHMI(ht,t_obs,jsoc_email,verbose=False,path=False,cad='45',hmi_path=None):
    """
    Script to download the HMI m_45 or ic_45 cosest in time to the provided SO/PHI observation.
    TAI convention and light travel time are taken into consideration.
    
    Parameters
    ----------
    ht: astropy.io.fits.header.Header
        header of the SO/PHI observation
    t_obs: str, datetime.datetime
        observation time of the SO/PHI observation. A string can be provided, but it is expected to be isoformat
    jsoc_email: str
        email address to be used for JSOC connection
    verbose: bool
        if True, plot of the HMI map will be shown (DEFAULT: False)
    path: bool
        if True, the path of the cache directory and of the HMI dataset will return as output (DEFAULT: False)
    """
    
    import glob, drms
    import sunpy, sunpy.map
    from astropy.constants import c
    
    if type(t_obs) == str:
        t_obs = datetime.datetime.fromisoformat(t_obs)
    dtai = datetime.timedelta(seconds=37) # datetime.timedelta(seconds=94)
    
    if hmi_path is not None:
        hmi_f = sorted(glob.glob(hmi_path+'*.fits'))
        t_obs_hmi = [datetime.datetime.strptime(fits.getheader(f,1)['T_OBS'],'%Y.%m.%d_%H:%M:%S.%f_TAI') - dtai for f in hmi_f]
        hmi_f = [x for _, x in sorted(zip(t_obs_hmi, hmi_f))]
        t_obs_hmi.sort()

        dltt = datetime.timedelta(seconds=(((1*u.AU).to(u.m) - ht['DSUN_OBS']*u.m)/c).value)
        T_OBS = [np.abs((t - dltt - t_obs).total_seconds()) for t in t_obs_hmi]
        ind = np.argmin(T_OBS)
        hmi_file = hmi_f[ind]
        print('closest HMI file in time is:', hmi_file,'with time difference:',T_OBS[ind],'s')
        hmi_map = sunpy.map.Map(hmi_file,cache=False)
        cache_dir = sunpy.data.CACHE_DIR+'/'
        hmi_name = cache_dir + hmi_file.split("/")[-1]
    else:

        if type(cad) != str:
            cad = str(int(cad))
        if cad == '45':
            dcad = datetime.timedelta(seconds=35) # half HMI cadence (23) + margin
        elif cad == '720':
            dcad = datetime.timedelta(seconds=360+60) # half HMI cadence (23) + margin
        else:
            print('wrong HMI cadence, only 45 and 720 are accepted')
            return None
        
        dltt = datetime.timedelta(seconds=ht['EAR_TDEL']) # difference in light travel time S/C-Earth

        kwlist = ['T_REC','T_OBS','DATE-OBS','CADENCE','DSUN_OBS']
        
        client = drms.Client(email=jsoc_email) 

        lt = np.nan
        n = 0
        while np.isnan(lt):
            n += 2
            if ht['BTYPE'] == 'BLOS':
                keys = client.query('hmi.m_'+cad+'s['+(t_obs+dtai-dcad+dltt).strftime('%Y.%m.%d_%H:%M:%S')+'-'+
                                (t_obs+dtai+dcad+dltt).strftime('%Y.%m.%d_%H:%M:%S')+']',seg=None,key=kwlist,n=n)
            elif ht['BTYPE'] == 'VLOS':
                keys = client.query('hmi.v_'+cad+'s['+(t_obs+dtai-dcad+dltt).strftime('%Y.%m.%d_%H:%M:%S')+'-'+
                                (t_obs+dtai+dcad+dltt).strftime('%Y.%m.%d_%H:%M:%S')+']',seg=None,key=kwlist,n=n)
            else:
                keys = client.query('hmi.ic_'+cad+'s['+(t_obs+dtai-dcad+dltt).strftime('%Y.%m.%d_%H:%M:%S')+'-'+
                                (t_obs+dtai+dcad+dltt).strftime('%Y.%m.%d_%H:%M:%S')+']',seg=None,key=kwlist,n=n)
            keys = keys[keys['T_OBS'] != 'MISSING']
            if np.size(keys['T_OBS']) > 0:
                lt = (np.nanmean(keys['DSUN_OBS'])*u.m - ht['DSUN_OBS']*u.m)/c
            else:
                print('adding 60s margin')
                dcad += datetime.timedelta(seconds=60)
            
        dltt = datetime.timedelta(seconds=lt.value) # difference in light travel time S/C-SDO


        T_OBS = [(ind,np.abs((datetime.datetime.strptime(t,'%Y.%m.%d_%H:%M:%S_TAI') - dtai - dltt - t_obs).total_seconds())) for ind, t in zip(keys.index,keys['T_OBS'])]
        ind = T_OBS[np.argmin([t[1] for t in T_OBS])][0]

        if ht['BTYPE'] == 'BLOS':
            name_h = 'hmi.m_'+cad+'s['+keys['T_REC'][ind]+']{Magnetogram}'
        elif ht['BTYPE'] == 'VLOS':
            name_h = 'hmi.v_'+cad+'s['+keys['T_REC'][ind]+']{Dopplergram}'
        else:
            name_h = 'hmi.ic_'+cad+'s['+keys['T_REC'][ind]+']{Continuum}'

        if np.abs((datetime.datetime.strptime(keys['T_OBS'][ind],'%Y.%m.%d_%H:%M:%S_TAI') - dtai - dltt - t_obs).total_seconds()) > np.ceil(int(cad)/2):
            print('WARNING: Closer file exists but has not been found.')
            print(name_h)
            print('T_OBS:',datetime.datetime.strptime(keys['T_OBS'][ind],'%Y.%m.%d_%H:%M:%S_TAI') - dtai - dltt)
            print('DATE-AVG:',t_obs)
            print('')
        else:
            print('HMI T_OBS (corrected for TAI and Light travel time):',datetime.datetime.strptime(keys['T_OBS'][ind],'%Y.%m.%d_%H:%M:%S_TAI') - dtai - dltt)
            print('PHI DATE-AVG:',t_obs)
        s45 = client.export(name_h,protocol='fits')
        hmi_map = sunpy.map.Map(s45.urls.url[0],cache=False)
        cache_dir = sunpy.data.CACHE_DIR+'/'
        hmi_name = cache_dir + s45.urls.url[0].split("/")[-1]

    if verbose:
        hmi_map.peek()
    if path:
        return hmi_map, cache_dir, hmi_name
    else:
        return hmi_map


def WCS_correction(file_name,jsoc_email,dir_out='./',remapping = 'remap',undistortion = False, allDID=False,verbose=False, deriv = True, values_only = False, subregion = None, crota_manual_correction = 0.15, hmi_file = None):
    """This function saves new version of the fits file with updated WCS.
    It works by correlating HRT data on remapped HMI data. 
    This function exports the nearest HMI data from JSOC. [Not downloaded to out_dir]
    Not validated on limb data. 
    Not tested on data with different viewing angle.
    icnt, stokes or ilam files are expected as input.
    
    Version: 1.0.0 (July 2022)
             1.1.0 (August 2023)
                Updates on data handling and correlation. New remapping function based on coordinates reprojecting functions.

    Parameters
    ----------
    file_name: str
        path to the fits file
    jsoc_email: str
        email address to be used for JSOC connection
    dir_out: str
        path to the output directory, DEFAULT: './', if None no file will be saved
    remapping: str
        type of remapping procedure. 'remap' uses the reprojection algorithm by DeForest, 'ccd' uses a coordinate translation from HMI to HRT based on function in this file (not working yet). DEFAULT: 'remap'
    undistortion: bool
        if True, HRT will be undistorted (DEFAULT: False).
    allDID: bool
        if True, all the fits file with the same DID in the directory of the input file will be saved with the new WCS.
    verbose: bool
        if True, plot of the maps will be shown (DEFAULT: False)
    deriv: bool
        if True, correlation is computed using the derivative of the image (DEFAULT: True)
    values_only: bool
        if True, new fits will not be saved (DEFAULT: False).
    subregion: tuple, None
        if None, automatic subregion. Accepted values are only tuples of slices (sly,slx)
    crota_manual_correction: float
        manual change to HRT CROTA value (deg). The value is added to the original one (DEFAULT: 0.15)
    Returns
    -------
    ht: astropy.io.fits.header.Header
        new header for hrt
    """
    import sunpy
    import sunpy.map
    from sunpy.coordinates import propagate_with_solar_surface
    # from reproject import reproject_interp, reproject_adaptive
    # from sunpy.coordinates import get_body_heliographic_stonyhurst
    
    from sunpy.coordinates import frames
    import warnings, sunpy
    warnings.filterwarnings("ignore", category=sunpy.util.SunpyMetadataWarning)

    
    # print('This is a preliminary procedure')
    # print('It has been optimized on raw, continuum and blos data')
    # print('This script is based on sunpy routines and examples')
    
    hdr_phi = fits.open(file_name)
    phi = hdr_phi[0].data; h_phi = hdr_phi[0].header
    start_row = int(h_phi['PXBEG2']-1)
    start_col = int(h_phi['PXBEG1']-1)
    _,_,_,cpos = fits_get_sampling(file_name)
    
    h_phi = rotate_header(h_phi.copy(),-crota_manual_correction, center=center_coord(h_phi))

    if phi.ndim == 3:
        phi = phi[cpos*4]
    elif phi.ndim == 4:
        if phi.shape[0] == 6:
            phi = phi[cpos,0]            
        else:
            phi = phi[:,:,0,cpos]
    original_shape = phi.shape
    
    if phi.shape[0] == 2048:
        if undistortion:
            und_phi = und(phi)
            h_phi['CRPIX1'],h_phi['CRPIX2'] = Inv2(1016,982,h_phi['CRPIX1'],h_phi['CRPIX2'],8e-9)
        else:
            und_phi = phi
        phi_map = sunpy.map.Map((und_phi,h_phi))
    else:
        phi = np.pad(phi,[(start_row,2048-(start_row+phi.shape[0])),(start_col,2048-(start_col+phi.shape[1]))])
        h_phi['NAXIS1'] = 2048; h_phi['NAXIS2'] = 2048
        h_phi['PXBEG1'] = 1; h_phi['PXBEG2'] = 1; h_phi['PXEND1'] = 2048; h_phi['PXEND2'] = 2048; 
        h_phi['CRPIX1'] += start_col; h_phi['CRPIX2'] += start_row
        if undistortion:
            und_phi = und(phi)
            h_phi['CRPIX1'],h_phi['CRPIX2'] = Inv2(1016,982,h_phi['CRPIX1'],h_phi['CRPIX2'],8e-9)
        else:
            und_phi = phi
        phi_map = sunpy.map.Map((und_phi,h_phi))
    
    if verbose:
        phi_map.peek()
    
    ht = phi_map.fits_header
    t0 = hdr_phi[10].data['EXP_START_TIME']
    if t0.size > 24:
        t0 = t0[int(round(t0.size//24/2,0))::t0.size//24]
    #             t0 = np.asarray([DT.datetime.fromisoformat(t0[i]) for i in range(len(t0))])
    t0 = [t0[i] for i in range(len(t0))]
    if cpos == 5:
        t0 = t0[20]
    else:
        t0 = t0[0]
        
    t_obs = datetime.datetime.fromisoformat(t0)
    if ht['BTYPE'] == 'BLOS':
        t0 = ht['DATE-AVG']
        t_obs = datetime.datetime.fromisoformat(ht['DATE-AVG'])
    
    try:
        if hmi_file is None:
            hmi_map, cache_dir, hmi_name = downloadClosestHMI(ht,t_obs,jsoc_email,verbose,True)
        else:
            if os.path.isfile(hmi_file):
                hmi_map = sunpy.map.Map(hmi_file)
                hmi_name = hmi_file.split('/')[-1]
            elif os.path.isdir(hmi_file):
                hmi_map, cache_dir, hmi_name = downloadClosestHMI(ht,t_obs,jsoc_email,verbose,True, hmi_path=hmi_file)

    except Exception as e:
        print("Issue with downloading HMI. The code stops here. Restults obtained so far will be saved. This was the error:")
        print(e)
        return h_phi['CROTA'], h_phi['CRPIX1'] - start_col, h_phi['CRPIX2'] - start_row, h_phi['CRVAL1'], h_phi['CRVAL2'], t0, False, None, None
        
    if verbose:
        hmi_map.peek()
    
    sly = slice(128*4,128*12)
    slx = slice(128*4,128*12)

    ht = h_phi.copy()
    ht['DATE-BEG'] = ht['DATE-AVG']; ht['DATE-OBS'] = ht['DATE-AVG']
    # ht['DATE-BEG'] = datetime.datetime.isoformat(datetime.datetime.fromisoformat(ht['DATE-BEG']) + dltt)
    # ht['DATE-OBS'] = datetime.datetime.isoformat(datetime.datetime.fromisoformat(ht['DATE-OBS']) + dltt)
    shift = [1,1]
    i = 0
    angle = 1
    match = True
    
    # hgsPHI = ccd2HGS(phi_map.fits_header)
    # hgsHMI = ccd2HGS(hmi_map.fits_header)
    # hmi_remap = hmi2phi(hmi_map,phi_map)

    try:
        while np.any(np.abs(shift)>5e-2):
            
            if subregion is not None:
                sly,slx = subregion
            else:
                sly, slx = subregion_selection(ht,start_row,start_col,original_shape,dsmax = 512)
                print('Subregion size:',sly.stop-sly.start)

            if remapping == 'remap':
                phi_map = sunpy.map.Map((und_phi,ht))
                with propagate_with_solar_surface():
                    bl = phi_map.pixel_to_world(slx.start*u.pix, sly.start*u.pix)
                    tr = phi_map.pixel_to_world((slx.stop-1)*u.pix, (sly.stop-1)*u.pix)
                    phi_submap = phi_map.submap(np.asarray([slx.start, sly.start])*u.pix,
                                        top_right=np.asarray([slx.stop-1, sly.stop-1])*u.pix)

                    hmi_remap = remap(phi_map, hmi_map, out_shape = (2048,2048))

                    # necessary when the FoV is close to the HMI limb
                    temp0 = hmi_remap.data.copy(); temp0[np.isinf(temp0)] = 0; temp0[np.isnan(temp0)] = 0
                    hmi_remap = sunpy.map.Map((temp0, hmi_remap.fits_header))

                    top_right = hmi_remap.world_to_pixel(tr)
                    bottom_left = hmi_remap.world_to_pixel(bl)
                    tr_hmi_map = np.array([top_right.x.value,top_right.y.value])
                    bl_hmi_map = np.array([bottom_left.x.value,bottom_left.y.value])
                    slyhmi = slice(int(round(bl_hmi_map[1])),int(round(tr_hmi_map[1]))+1)
                    slxhmi = slice(int(round(bl_hmi_map[0])),int(round(tr_hmi_map[0]))+1)
                    hmi_map_wcs = hmi_remap.submap(bl_hmi_map*u.pix,top_right=tr_hmi_map*u.pix)

            
            elif remapping == 'ccd':
                phi_map = sunpy.map.Map((und_phi,ht))
                # hgsPHI = ccd2HGS(phi_map.fits_header)
                
                hmi_remap = sunpy.map.Map((hmi2phi(hmi_map,phi_map),ht))
                
                phi_submap = phi_map.submap(np.asarray([slx.start, sly.start])*u.pix,
                                    top_right=np.asarray([slx.stop-1, sly.stop-1])*u.pix)
                slyhmi = sly
                slxhmi = slx
                hmi_map_wcs = hmi_remap.submap(np.asarray([slx.start, sly.start])*u.pix,
                                    top_right=np.asarray([slx.stop-1, sly.stop-1])*u.pix)

            ref = phi_submap.data.copy()
            temp = hmi_map_wcs.data.copy(); temp[np.isinf(temp)] = 0; temp[np.isnan(temp)] = 0
            s = [1,1]
            shift = [0,0]
            it = 0

            while np.any(np.abs(s)>1e-2) and it<10:
                if it == 0:
                    _,s = image_register(ref,temp,False,deriv)
                    if np.any(np.abs(s)==0):
                        _,s = image_register(ref,temp,True,deriv)
                else:
                    _,s = image_register(ref,temp,True,deriv)
                    # sr, sc, _ = SPG_shifts_FFT(np.asarray([ref,temp])); s = [sr[1],sc[1]]
                shift = [shift[0]+s[0],shift[1]+s[1]]
                # temp = fft_shift(hmi_map_wcs.data.copy(), shift); temp[np.isinf(temp)] = 0; temp[np.isnan(temp)] = 0
                temp = fft_shift(hmi_remap.data.copy(), shift)[slyhmi,slxhmi]; temp[np.isinf(temp)] = 0; temp[np.isnan(temp)] = 0
                it += 1
                
            hmi_map_shift = sunpy.map.Map((temp,hmi_map_wcs.fits_header))

            ht = translate_header(ht.copy(),np.asarray(shift),mode='crval')
            print(it,'iterations shift (x,y):',round(shift[1],2),round(shift[0],2))

            i+=1
            if i == 10:
                print('Maximum iterations reached:',i)
                match = False
                break
    except Exception as e:
        printc("Issue with co-alignment. The code stops here. Restults obtained so far will be saved. This was the error:",bcolors.FAIL)
        printc(e,bcolors.FAIL)
        print(f'{it} iterations shift (x,y): {round(shift[1],2),round(shift[0],2)}',bcolors.FAIL)
        return ht['CROTA'], ht['CRPIX1'] - start_col, ht['CRPIX2'] - start_row, ht['CRVAL1'], ht['CRVAL2'], t0, False, phi_map, hmi_remap
    
    if remapping == 'remap':
        phi_map = sunpy.map.Map((und_phi,ht))

        # bl = phi_map.pixel_to_world(slx.start*u.pix, sly.start*u.pix)
        # tr = phi_map.pixel_to_world((slx.stop-1)*u.pix, (sly.stop-1)*u.pix)
        # phi_submap = phi_map.submap(np.asarray([slx.start, sly.start])*u.pix,
        #                     top_right=np.asarray([slx.stop-1, sly.stop-1])*u.pix)
        with propagate_with_solar_surface():
            hmi_remap = remap(phi_map, hmi_map, out_shape = (2048,2048))
        
        ht['DATE-BEG'] = h_phi['DATE-BEG']
        ht['DATE-OBS'] = h_phi['DATE-OBS']
        phi_map = sunpy.map.Map((und_phi,ht))

        # top_right = hmi_remap.world_to_pixel(tr)
        # bottom_left = hmi_remap.world_to_pixel(bl)
        # tr_hmi_map = np.array([top_right.x.value,top_right.y.value])
        # bl_hmi_map = np.array([bottom_left.x.value,bottom_left.y.value])
        # hmi_map_wcs = hmi_remap.submap(bl_hmi_map*u.pix,top_right=tr_hmi_map*u.pix)

    elif remapping == 'ccd':
        phi_map = sunpy.map.Map((und_phi,ht))
        hmi_remap = sunpy.map.Map((hmi2phi(hmi_map,phi_map),ht))
        ht['DATE-BEG'] = h_phi['DATE-BEG']
        ht['DATE-OBS'] = h_phi['DATE-OBS']
        phi_map = sunpy.map.Map((und_phi,ht))

        phi_submap = phi_map.submap(np.asarray([slx.start, sly.start])*u.pix,
                            top_right=np.asarray([slx.stop-1, sly.stop-1])*u.pix)
        hmi_map_wcs = hmi_remap.submap(np.asarray([slx.start, sly.start])*u.pix,
                            top_right=np.asarray([slx.stop-1, sly.stop-1])*u.pix)
    
    
    if verbose:
        if remapping == 'remap':
            bl = phi_map.pixel_to_world(slx.start*u.pix, sly.start*u.pix)
            tr = phi_map.pixel_to_world((slx.stop-1)*u.pix, (sly.stop-1)*u.pix)
            top_right = hmi_remap.world_to_pixel(tr)
            bottom_left = hmi_remap.world_to_pixel(bl)
            tr_hmi_map = np.array([top_right.x.value,top_right.y.value])
            bl_hmi_map = np.array([bottom_left.x.value,bottom_left.y.value])
            hmi_map_wcs = hmi_remap.submap(bl_hmi_map*u.pix,top_right=tr_hmi_map*u.pix)
        fig = plt.figure(figsize=(10,6))
        ax1 = fig.add_subplot(1, 2, 1, projection=hmi_map_wcs)
        hmi_map_wcs.plot(axes=ax1, title='SDO/HMI image as seen from PHI/HRT')
        ax2 = fig.add_subplot(1, 2, 2, projection=phi_submap)
        phi_submap.plot(axes=ax2)
        # Set the HPC grid color to black as the background is white
        ax1.coords[0].grid_lines_kwargs['edgecolor'] = 'k'
        ax1.coords[1].grid_lines_kwargs['edgecolor'] = 'k'
        ax2.coords[0].grid_lines_kwargs['edgecolor'] = 'k'
        ax2.coords[1].grid_lines_kwargs['edgecolor'] = 'k'
    
    if os.path.isfile(hmi_name) and hmi_file is None:
        os.remove(hmi_name)
        import sqlite3
        # creating file path
        dbfile = cache_dir+'cache.db'
        if os.path.isfile(dbfile):
            # Create a SQL connection to our SQLite database
            con = sqlite3.connect(dbfile)
            # creating cursor
            cur = con.cursor()
            removeItem = "DELETE FROM cache_storage WHERE file_path = \'"+hmi_name+"\'"
            cur.execute(removeItem)
            con.commit()
        print(hmi_name.split("/")[-1]+' deleted')
    
    if values_only:
        return ht['CROTA'], ht['CRPIX1'] - start_col, ht['CRPIX2'] - start_row, ht['CRVAL1'], ht['CRVAL2'], t0, match, phi_map, hmi_remap
    else:
        if dir_out is not None:
            if allDID:
                did = h_phi['PHIDATID']
                name = file_name.split('/')[-1]
                new_name = name.replace("_phi-",".WCS_phi-")
    
                directory = file_name[:-len(name)]
                file_n = os.listdir(directory)
                if type(did) != str:
                    did = str(did)
                did_n = [directory+i for i in file_n if did in i]
                l2_n = ['stokes','icnt','bmag','binc','bazi','vlos','blos']
                for n in l2_n:
                    f = [i for i in did_n if n in i][0]
                    name = f.split('/')[-1]
                    new_name = name.replace("_phi-",".WCS_phi-")
                    with fits.open(f) as h:
                        h[0].header['CROTA'] = ht['CROTA']
                        h[0].header['CRPIX1'] = ht['CRPIX1'] - start_col
                        h[0].header['CRPIX2'] = ht['CRPIX2'] - start_row
                        h[0].header['CRVAL1'] = ht['CRVAL1']
                        h[0].header['CRVAL2'] = ht['CRVAL2']
                        h[0].header['PC1_1'] = ht['PC1_1']
                        h[0].header['PC1_2'] = ht['PC1_2']
                        h[0].header['PC2_1'] = ht['PC2_1']
                        h[0].header['PC2_2'] = ht['PC2_2']
                        h[0].header['HISTORY'] = 'WCS corrected via HRT - HMI cross correlation'
                        h.writeto(dir_out+new_name, overwrite=True)        
            else:
                with fits.open(file_name) as h:
                    h[0].header['CROTA'] = ht['CROTA']
                    h[0].header['CRPIX1'] = ht['CRPIX1'] - start_col
                    h[0].header['CRPIX2'] = ht['CRPIX2'] - start_row
                    h[0].header['CRVAL1'] = ht['CRVAL1']
                    h[0].header['CRVAL2'] = ht['CRVAL2']
                    h[0].header['PC1_1'] = ht['PC1_1']
                    h[0].header['PC1_2'] = ht['PC1_2']
                    h[0].header['PC2_1'] = ht['PC2_1']
                    h[0].header['PC2_2'] = ht['PC2_2']
                    h[0].header['HISTORY'] = 'WCS corrected via HRT - HMI cross correlation '
                    h.writeto(dir_out+new_name, overwrite=True)
        return ht['CROTA'], ht['CRPIX1'] - start_col, ht['CRPIX2'] - start_row, ht['CRVAL1'], ht['CRVAL2'], t0, match, phi_map, hmi_remap

def ccd2HPC(file,coords=None):
    """
    from CCD frame to Helioprojective Cartesian
    
    Input
    file: file_name, sunpy map or header
    coords: (x,y) or np.asarray([[x0,y0],[x1,y1],...])
            if None: built the coordinates map
            
    Output
    HPCx, HPCy, HPCd
    """
    import sunpy.map
    if type(file) == str:
        hdr = fits.getheader(file)
    elif type(file) == sunpy.map.mapbase.GenericMap or type(file) == sunpy.map.sources.sdo.HMIMap:
        hdr = file.fits_header
    elif type(file) == fits.header.Header:
        hdr = file
    
    if coords is not None: # coords must be (N,3), N is the number of pixels, the second axis is (x,y,1)
        if type(coords) == list or type(coords) == tuple:
            coords = np.asarray([coords[0],coords[1],1])
        elif type(coords) == np.ndarray:
            if coords.ndim == 1:
                if coords.shape[0] == 2:
                    coords = np.append(coords,1)
            else:
                if coords.shape[1] == 2:
                    coords = np.append(coords,np.ones((coords.shape[0],1)),axis=1)
        if coords.ndim == 1:
            coords = coords[np.newaxis]
        
    pxsc = hdr['CDELT1']
    sun_dist_m=hdr['DSUN_OBS'] #Earth
    rsun = hdr['RSUN_REF'] # m
    if 'PXBEG1' in hdr:
        pxbeg1 = hdr['PXBEG1']
    else:
        pxbeg1 = 1
    if 'PXBEG2' in hdr:
        pxbeg2 = hdr['PXBEG2']
    else:
        pxbeg2 = 1
    if 'PXEND1' in hdr:
        pxend1 = hdr['PXEND1']
    else:
        pxend1 = hdr['NAXIS1']
    if 'PXEND2' in hdr:
        pxend2 = hdr['PXEND2']
    else:
        pxend2 = hdr['NAXIS2']
    crval1 = hdr['CRVAL1']
    crval2 = hdr['CRVAL2']
    crpix1 = hdr['CRPIX1']
    crpix2 = hdr['CRPIX2']
    if 'PC1_1' in hdr:
        PC1_1 = hdr['PC1_1']
        PC1_2 = hdr['PC1_2']
        PC2_1 = hdr['PC2_1']
        PC2_2 = hdr['PC2_2']
    else:
        crota = hdr['CROTA*'][0]
        PC1_1 = np.cos(crota*np.pi/180)
        PC1_2 = -np.sin(crota*np.pi/180)
        PC2_1 = np.sin(crota*np.pi/180)
        PC2_2 = np.cos(crota*np.pi/180)
    
    if coords is None:
        X,Y = np.meshgrid(np.arange(1,pxend1-pxbeg1+2),np.arange(1,pxend2-pxbeg2+2))
    else:
        X = coords[:,0]+1
        Y = coords[:,1]+1
    
    HPC1 = crval1 + pxsc*(PC1_1*(X-crpix1)+PC1_2*(Y-crpix2))
    HPC2 = crval2 + pxsc*(PC2_1*(X-crpix1)+PC2_2*(Y-crpix2))

    th = np.arctan(np.sqrt(np.cos(HPC2/3600*np.pi/180)**2*np.sin(HPC1/3600*np.pi/180)**2+np.sin(HPC2/3600*np.pi/180)**2/
                          (np.cos(HPC2/3600*np.pi/180)*np.cos(HPC1/3600*np.pi/180))))
    b = np.arcsin(sun_dist_m/rsun*np.sin(th)) - th
    d = (sun_dist_m-rsun*np.cos(b))/np.cos(th)
    
    return HPC1, HPC2, d

def ccd2HCC(file,coords = None):
    """
    coordinate center in the center of the Sun
    x is pointing westward, y toward the north pole and z toward the observer (max for all should be Rsun)
    
    Input
    file: file_name, sunpy map or header
    coords: (x,y) or np.asarray([[x0,y0],[x1,y1],...])
            if None: built the coordinates map
            
    Output
    HCCx, HCCy, HCCz
    """
    import sunpy.map
    if type(file) == str:
#         smap = sunpy.map.Map(file)
        hdr = fits.getheader(file)
    elif type(file) == sunpy.map.mapbase.GenericMap or type(file) == sunpy.map.sources.sdo.HMIMap:
#         smap = file
        hdr = file.fits_header
    elif type(file) == fits.header.Header:
        hdr = file
    
    sun_dist_m=hdr['DSUN_OBS']
    
    HPCx, HPCy, HPCd = ccd2HPC(file,coords)
    
    HCCx = HPCd * np.cos(HPCy/3600*np.pi/180) * np.sin(HPCx/3600*np.pi/180)
    HCCy = HPCd * np.sin(HPCy/3600*np.pi/180)
    HCCz = sun_dist_m - HPCd * np.cos(HPCy/3600*np.pi/180) * np.cos(HPCx/3600*np.pi/180)
    
    return HCCx,HCCy,HCCz

def ccd2HGS(file, coords = None):
    """
    From CCD frame to Heliographic Stonyhurst coordinates
    
    Input
    file: file_name, sunpy map or header
    coords: (x,y) or np.asarray([[x0,y0],[x1,y1],...])
            if None: built the coordinates map
            
    Output
    r, THETA, PHI
    """
    import sunpy.map
    if type(file) == str:
        hdr = fits.getheader(file)
    elif type(file) == sunpy.map.mapbase.GenericMap or type(file) == sunpy.map.sources.sdo.HMIMap:
        hdr = file.fits_header
    elif type(file) == fits.header.Header:
        hdr = file
    
    if 'HGLT_OBS' in hdr:
        B0 = hdr['HGLT_OBS']*np.pi/180
    else:
        B0 = hdr['CRLT_OBS']*np.pi/180
    if 'HGLN_OBS' in hdr:
        PHI0 = hdr['HGLN_OBS']*np.pi/180
    else:
        L0time = datetime.datetime.fromisoformat(hdr['DATE-OBS'])
        PHI0 = hdr['CRLN_OBS'] - sunpy.coordinates.sun.L0(L0time).value
    
    HCCx, HCCy, HCCz = ccd2HCC(file,coords)
        
    r = np.sqrt(HCCx**2 + HCCy**2 + HCCz**2)
    THETA = np.arcsin((HCCy*np.cos(B0) + HCCz*np.sin(B0))/r)*180/np.pi
    PHI = PHI0*180/np.pi + np.arctan(HCCx/(HCCz*np.cos(B0) - HCCy*np.sin(B0)))*180/np.pi
    
    # THETA == LAT; PHI == LON
    return r, THETA, PHI

def HPC2ccd(file, coords):
    """
    from Helioprojective Cartesian to CCD frame
    
    Input
    file: file_name, sunpy map or header
    coords: Helioprojective Cartesian coordinates (sse output of ccd2HPC)
            
    Output
    x,y (in pixels)
    
    """
    import sunpy.map
    if type(file) == str:
        hdr = fits.getheader(file)
    elif type(file) == sunpy.map.mapbase.GenericMap or type(file) == sunpy.map.sources.sdo.HMIMap:
        hdr = file.fits_header
    elif type(file) == fits.header.Header:
        hdr = file
            
    try:
        HPC1, HPC2 = coords
    except:
        HPC1, HPC2, d = coords
    assert (np.shape(HPC1) == np.shape(HPC2))

    if type(HPC1) == list or type(HPC2) == tuple:
        HPC1 = np.asarray(HPC1)
        HPC2 = np.asarray(HPC2)
        
    pxsc = hdr['CDELT1']
    crval1 = hdr['CRVAL1']
    crval2 = hdr['CRVAL2']
    crpix1 = hdr['CRPIX1']
    crpix2 = hdr['CRPIX2']
    if 'PC1_1' in hdr:
        PC1_1 = hdr['PC1_1']
        PC1_2 = hdr['PC1_2']
        PC2_1 = hdr['PC2_1']
        PC2_2 = hdr['PC2_2']
    else:
        crota = hdr['CROTA*'][0]
        PC1_1 = np.cos(crota*np.pi/180)
        PC1_2 = -np.sin(crota*np.pi/180)
        PC2_1 = np.sin(crota*np.pi/180)
        PC2_2 = np.cos(crota*np.pi/180)
    
    x = crpix1 + 1/pxsc * (PC1_1*(HPC1-crval1) + PC2_1*(HPC2-crval2)) - 1
    y = crpix2 + 1/pxsc * (PC1_2*(HPC1-crval1) + PC2_2*(HPC2-crval2)) - 1
        
    return x,y

def HCC2ccd(file,coords):
    """
    from Heliocentric Cartesian to CCD frame
    
    Input
    file: file_name, sunpy map or header
    coords: Heliocentric Cartesian coordinates (sse output of ccd2HCC)
            
    Output
    x,y (in pixels)
    
    """
    import sunpy.map
    if type(file) == str:
#         smap = sunpy.map.Map(file)
        hdr = fits.getheader(file)
    elif type(file) == sunpy.map.mapbase.GenericMap or type(file) == sunpy.map.sources.sdo.HMIMap:
#         smap = file
        hdr = file.fits_header
    elif type(file) == fits.header.Header:
        hdr = file
    
    HCCx,HCCy,HCCz = coords
    assert (np.shape(HCCx) == np.shape(HCCy) and np.shape(HCCx) == np.shape(HCCz))

    if type(HCCx) == list or type(HCCx) == tuple:
        HCCx = np.asarray(HCCx)
        HCCy = np.asarray(HCCy)
        HCCz = np.asarray(HCCz)

    sun_dist_m=hdr['DSUN_OBS']
    
    HPCd = np.sqrt(HCCx**2+HCCy**2+(sun_dist_m-HCCz)**2)
    HPCx = np.arctan(HCCx/(sun_dist_m-HCCz))*180/np.pi*3600
    HPCy = np.arcsin(HCCy/HPCd)*180/np.pi*3600
    
    
    x,y = HPC2ccd(hdr,(HPCx,HPCy,HPCd))
    
    return x,y

def HGS2ccd(file, coords):
    """
    from Heliographic Stonyhurst to CCD frame
    
    Input
    file: file_name, sunpy map or header
    coords: Heliographic Stonyhurst coordinates (sse output of ccd2HGS)
            
    Output
    x,y (in pixels)
    
    """
    
    import sunpy.map
    if type(file) == str:
        hdr = fits.getheader(file)
    elif type(file) == sunpy.map.mapbase.GenericMap or type(file) == sunpy.map.sources.sdo.HMIMap:
        hdr = file.fits_header
    elif type(file) == fits.header.Header:
        hdr = file
    
    r, THETA, PHI = coords
    assert (np.shape(r) == np.shape(THETA) and np.shape(r) == np.shape(PHI))

    if type(r) == list or type(r) == tuple:
        r = np.asarray(r)
        THETA = np.asarray(THETA)
        PHI = np.asarray(PHI)

    if 'HGLT_OBS' in hdr:
        B0 = hdr['HGLT_OBS']*np.pi/180
    else:
        B0 = hdr['CRLT_OBS']*np.pi/180
    if 'HGLN_OBS' in hdr:
        PHI0 = hdr['HGLN_OBS']*np.pi/180
    else:
        L0time = datetime.datetime.fromisoformat(hdr['DATE-OBS'])
        PHI0 = hdr['CRLN_OBS'] - sunpy.coordinates.sun.L0(L0time).value
    
    THETA = THETA * np.pi/180
    PHI = PHI * np.pi/180
    
    HCCx = r * np.cos(THETA) * np.sin(PHI-PHI0)
    HCCy = r * (np.sin(THETA)*np.cos(B0) - np.cos(THETA)*np.cos(PHI-PHI0)*np.sin(B0))
    HCCz = r * (np.sin(THETA)*np.sin(B0) + np.cos(THETA)*np.cos(PHI-PHI0)*np.cos(B0))
    
    
    x, y = HCC2ccd(hdr,(HCCx,HCCy,HCCz))
    
    return x,y

def hmi2phi(hmi_map, phi_map,  order=1):
    
    from scipy.ndimage import map_coordinates

    print('WARNING: deprecated function')
    hgsPHI = ccd2HGS(phi_map.fits_header)
    new_coord = HGS2ccd(hmi_map.fits_header,hgsPHI)
    hmi_remap = map_coordinates(hmi_map.data,[new_coord[1],new_coord[0]],order=order)
    
    return hmi_remap

def solarRotation(hdr):
    # vrot from hathaway et al., 2011, values in deg/day
    # proper vlos projection without thetarho ~ 0 approximation from Schuck et al., 2016
    X = ccd2HGS(hdr)
    HPCx, HPCy, HPCd = ccd2HPC(hdr)
    thetarho = np.arctan(np.sqrt(np.cos(HPCy*u.arcsec)**2*np.sin(HPCx*u.arcsec)**2+np.sin(HPCy*u.arcsec)**2) / 
                      (np.cos(HPCy*u.arcsec)*np.cos(HPCx*u.arcsec)))
    psi = np.arctan(-(np.cos(HPCy*u.arcsec)*np.sin(HPCx*u.arcsec)) / np.sin(HPCy*u.arcsec))
    psi[np.logical_and(HPCy>=0,HPCx<0)] += 0 * u.rad
    psi[np.logical_and(HPCy<0,HPCx<0)] += np.pi * u.rad
    psi[np.logical_and(HPCy<0,HPCx>=0)] += np.pi * u.rad
    psi[np.logical_and(HPCy>=0,HPCx>=0)] += 2*np.pi * u.rad
    
    a = (14.437 * u.deg/u.day).to(u.rad/u.s); 
    b = (-1.48 * u.deg/u.day).to(u.rad/u.s); 
    c = (-2.99 * u.deg/u.day).to(u.rad/u.s); 
    vrot = (a + b*np.sin(X[1]*u.deg)**2 + c*np.sin(X[1]*u.deg)**4)*np.cos(X[1]*u.deg)* hdr['RSUN_REF'] * u.m/u.rad
    B0 = hdr['HGLT_OBS']*u.deg
    THETA = (X[1])*u.deg # lat
    PHI = (X[2]-hdr['HGLN_OBS'])*u.deg # lon
    It = -np.cos(B0)*np.sin(PHI)*np.cos(thetarho) + \
         (np.cos(PHI)*np.sin(psi)-np.sin(B0)*np.sin(PHI)*np.cos(psi))*np.sin(thetarho)
    vlos = -(vrot) * It
    
    return vlos.value

def SCVelocityResidual(hdr,wlcore):
    # s/c velocity signal (considering line shift compensation) in m/s
#     X = ccd2HGS(hdr)
    HPCx, HPCy, HPCd = ccd2HPC(hdr)
    thetarho = np.arctan(np.sqrt(np.cos(HPCy*u.arcsec)**2*np.sin(HPCx*u.arcsec)**2+np.sin(HPCy*u.arcsec)**2) / 
                      (np.cos(HPCy*u.arcsec)*np.cos(HPCx*u.arcsec)))
    psi = np.arctan(-(np.cos(HPCy*u.arcsec)*np.sin(HPCx*u.arcsec)) / np.sin(HPCy*u.arcsec))
    psi[np.logical_and(HPCy>=0,HPCx<0)] += 0 * u.rad
    psi[np.logical_and(HPCy<0,HPCx<0)] += np.pi * u.rad
    psi[np.logical_and(HPCy<0,HPCx>=0)] += np.pi * u.rad
    psi[np.logical_and(HPCy>=0,HPCx>=0)] += 2*np.pi * u.rad

    vsc = hdr['OBS_VW']*np.sin(thetarho)*np.sin(psi) - hdr['OBS_VN']*np.sin(thetarho)*np.cos(psi) + hdr['OBS_VR']*np.cos(thetarho)
    c = 299792.458
    wlref = 6173.341
    vsc_compensation = (wlcore-wlref)/wlref*c*1e3
    
    return vsc.value - vsc_compensation

def meridionalFlow(hdr):
    # from hathaway et al., 2011, values in m/s
    # proper vlos projection without thetarho ~ 0 approximation from Schuck et al., 2016
    X = ccd2HGS(hdr)
    HPCx, HPCy, HPCd = ccd2HPC(hdr)
    thetarho = np.arctan(np.sqrt(np.cos(HPCy*u.arcsec)**2*np.sin(HPCx*u.arcsec)**2+np.sin(HPCy*u.arcsec)**2) / 
                      (np.cos(HPCy*u.arcsec)*np.cos(HPCx*u.arcsec)))
    psi = np.arctan(-(np.cos(HPCy*u.arcsec)*np.sin(HPCx*u.arcsec)) / np.sin(HPCy*u.arcsec))
    psi[np.logical_and(HPCy>=0,HPCx<0)] += 0 * u.rad
    psi[np.logical_and(HPCy<0,HPCx<0)] += np.pi * u.rad
    psi[np.logical_and(HPCy<0,HPCx>=0)] += np.pi * u.rad
    psi[np.logical_and(HPCy>=0,HPCx>=0)] += 2*np.pi * u.rad
    
    d = 29.7 * u.m/u.s; e = -17.7 * u.m/u.s; 
    vmer = (d*np.sin(X[1]*u.deg) + e*np.sin(X[1]*u.deg)**3)*np.cos(X[1]*u.deg)
    B0 = hdr['HGLT_OBS']*u.deg
    THETA = (X[1])*u.deg
    PHI = (X[2]-hdr['HGLN_OBS'])*u.deg
    It = (np.sin(B0)*np.cos(THETA) - np.cos(B0)*np.cos(PHI)*np.sin(THETA))*np.cos(thetarho) - \
        (np.sin(PHI)*np.sin(THETA)*np.sin(psi) + \
        (np.sin(B0)*np.cos(PHI)*np.sin(THETA) + np.cos(B0)*np.cos(THETA))*np.cos(psi))*np.sin(thetarho)
    
    vlos = (-vmer) * It
    
    return vlos.value

def SCGravitationalRedshift(hdr):
    # gravitational redshift (theoretical) from a distance dsun from the sun in m/s
    dsun = hdr['DSUN_OBS'] # m
    c = 299792.458e3 # m/s
    Rsun = hdr['RSUN_REF'] # m
    Msun = 1.9884099e30 # kg
    G = 6.6743e-11 # m3/kg/s2
    vg = G*Msun/c * (1/Rsun - 1/dsun)
    
    return vg

def convectiveBlueshift(hdr):
    # from Castellanos Duran et al. 2021, eq. (5). Valid for HMI.
    # coords = sunpy.map.all_coordinates_from_map(smap,).transform_to(frames.Heliocentric())
    shape = (hdr['NAXIS2'],hdr['NAXIS1'])
    center=center_coord(hdr)
    try:
        Rpix=(hdr['RSUN_ARC']/hdr['CDELT1']) # PHI
    except:
        Rpix=(hdr['RSUN_OBS']/hdr['CDELT1']) # HMI
    X,Y = np.meshgrid(np.arange(shape[1]) - center[0],np.arange(shape[0]) - center[1])
    mu = np.sqrt(Rpix**2 - (X**2 + Y**2)) / Rpix
    
    #HCCx, HCCy, HCCz = ccd2HCC(hdr)
    # rho = np.sqrt(HCCx**2+HCCy**2)
    # mu = np.sqrt(1- (rho / hdr['RSUN_REF'])).value
    
    v = 134 - 1179*mu - 2029*mu**2 + 9112*mu**3 - 10409*mu**4 + 4096*mu**5 # m/s
    return v

def muSO_map(h,shape):
    if type(h) is str:
        h = fits.getheader(h)
    center=center_coord(h)
    try:
        Rpix=(h['RSUN_ARC']/h['CDELT1']) # PHI
    except:
        Rpix=(h['RSUN_OBS']/h['CDELT1']) # HMI
    
    X,Y = np.meshgrid(np.arange(shape[1]) - center[0],np.arange(shape[0]) - center[1])
    mu = np.sqrt(Rpix**2 - (X**2 + Y**2)) / Rpix
    return mu

def CLV(mu):
    # input from muSO_map
    p = np.asarray([0.32519, 1.26432, -1.44591, 1.55723, -0.87415, 0.17333]) # from Pierce & Slughter 1977
    clv = np.zeros(mu.shape)
    for i in range(6):
        clv += p[i] * mu**i
        
    return clv
