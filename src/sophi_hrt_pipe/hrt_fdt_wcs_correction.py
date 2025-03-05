import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from .utils import find_nearest, printc, bcolors
from .coordinates import rotate_header, translate_header, center_coord, circular_mask, remap, fft_shift, image_register, Inv2, und
from .processes import limb_side_finder
# import argparse
from datetime import datetime as DT
from datetime import timedelta as TD
import glob, os
import sunpy.map
from sunpy.coordinates import Helioprojective
from sunpy.coordinates import propagate_with_solar_surface
import warnings, sunpy
from astropy import units as u

warnings.filterwarnings("ignore", category=sunpy.util.SunpyMetadataWarning)

VERSION = '1.0'

def get_descriptor(filename, telescope='hrt'):
    descriptor = filename.split('phi-{0}-'.format(telescope))[1].split('_')[0]
    return descriptor

def closestFDT(filename):
    # hdr = fits.open(filename)
    # btype = hdr[0].header['BTYPE']

    descriptor = get_descriptor(filename)

    date_obs = DT.strptime(filename.split('_')[-3], '%Y%m%dT%H%M%S')

    t0 = date_obs - TD(days=1)
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
            fdt_files += sorted(glob.glob(f'/data/solo/phi/data/fmdb/ll/{d}/*phi-fdt-{descriptor}*.fits.gz'))

    if len(fdt_files) == 0:
        print(f'No FDT LL {descriptor} file, looking for L1')
        for d in dates:
            fdt_files += sorted(glob.glob(f'/data/solo/phi/data/fmdb/l1/{d}*phi-fdt-*lam*.fits.gz'))
        filename = filename.replace(descriptor, 'icnt')
        if os.path.isfile(filename):
            print('Using raw FDT data, so I will use HRT continuum intensity')
        else:
            raise FileNotFoundError('There is no HRT continuum intensity associated to this file!')

    if len(fdt_files) == 0:
        raise FileNotFoundError('There is no FDT file associated to this HRT, sorry :-(')

    t_obs_fdt = [DT.fromisoformat(fits.getheader(f)['DATE-OBS']) for f in fdt_files]
    ind = find_nearest(t_obs_fdt, date_obs)
    fdt_filename = fdt_files[ind]

    return fdt_filename, descriptor

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
            with Helioprojective.assume_spherical_screen(fdt_map.observer_coordinate,True):
                
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

                hrt_remap = remap(fdt_submap, hrt_map, out_shape = fdt_submap.data.shape, verbose=False)

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
        print(it,'iterations shift (x,y):',round(shift[1],2),round(shift[0],2))

        i+=1
        if i == max_iterations:
            print('Maximum iterations reached:',i)
            match = False
            break
    
    hrt_map = sunpy.map.Map((und_hrt,ht))
    # fdt_remap = remap(hrt_map, fdt_map, out_shape = hrt_map.data.shape, verbose=verbose)
    hrt_remap = remap(fdt_submap, hrt_map, out_shape = fdt_submap.data.shape, verbose=False)

    ht['DATE-BEG'] = h_hrt['DATE-BEG']
    ht['DATE-OBS'] = h_hrt['DATE-OBS']
    hrt_map = sunpy.map.Map((und_hrt,ht))     

    newWCS = dict(DID=ht['PHIDATID'], 
                  CROTA=ht['CROTA'], PC1_1=ht['PC1_1'], PC1_2=ht['PC1_2'], PC2_1=ht['PC2_1'], PC2_2=ht['PC2_2'],
                  CRPIX1=ht['CRPIX1'], CRPIX2=ht['CRPIX2'], 
                  CRVAL1=ht['CRVAL1'], CRVAL2=ht['CRVAL2'])
    
    return hrt_map, hrt_remap, newWCS, t0, match

def plot_fdt_hrt(fdt_map, hrt_map):
    fig = plt.figure(layout='tight',figsize=(7,7))
    ax = fig.add_subplot(projection=fdt_map)
    plot = dict(clim=(-100,100))
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
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Filename (with path)')
#     parser.add_argument('filename', type=str, help='The name of the file to correct or directory with wildcard')
#     parser.add_argument('-v','--verbose', action='store_true',help='plot corrected images if true')

#     args = parser.parse_args()
    
#     verbose = args.verbose
#     filename = args.filename
    
    # filename can be a string with wildcard or directory path

    parameters = dict(filename = None, print_values = False)
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
            hrt_map, fdt_map, fdt_map_rot = prepare_data(f, fdt_filename, 0.15, undistortion=False, verbose=False)
        else:
            hrt_map, fdt_map, fdt_map_rot = prepare_data((data,header), fdt_filename, 0.15, undistortion=False, verbose=False)

        try:
            hrt_map, hrt_remap, n, t0, match = correction(hrt_map, fdt_map_rot, deriv=False,verbose=False)
        except:
            printc('There was an error in the WCS correction, return None', color=bcolors.FAIL)
            for key in newWCS.keys():
                newWCS[key].append(None)
            pass

        if parameters['print_values']:
            for e,v in zip(ends,n.values()):
                if isinstance(v,str):
                    cols += v+e
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
    
