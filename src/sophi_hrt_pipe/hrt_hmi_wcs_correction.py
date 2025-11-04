import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from .utils import find_nearest, printc, bcolors
from .coordinates import rotate_header, translate_header, center_coord, circular_mask, remap, fft_shift, image_register, Inv2, und, downloadClosestHMI
from .processes import limb_side_finder, elliptical_mask
# import argparse
from datetime import datetime as DT
from datetime import timedelta as TD
import glob, os
import sunpy.map
from sunpy.coordinates import Helioprojective
from sunpy.coordinates import propagate_with_solar_surface
import warnings, sunpy
from astropy import units as u
from sunpy.coordinates import SphericalScreen

warnings.filterwarnings("ignore", category=sunpy.util.SunpyMetadataWarning)

VERSION = '0.0.1'

def get_descriptor(filename, telescope='hrt'):
    descriptor = filename.split('phi-{0}-'.format(telescope))[1].split('_')[0]
    return descriptor

def prepare_data(hrt_file, hmi_map, crota_manual_correction=0.15, undistortion=False, verbose=False):
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

    hmi_map_rot = hmi_map.rotate((hmi_map.fits_header['CROTA2']-hrt_map.fits_header['CROTA'])*u.deg,method='opencv')
    if ('CROTA2' not in hmi_map_rot.fits_header) or ('CROTA' not in hmi_map_rot.fits_header):
        temph = hmi_map_rot.fits_header; temph['CROTA2'] = hrt_map.fits_header['CROTA']
        hmi_map_rot = sunpy.map.Map((np.nan_to_num(hmi_map_rot.data,True,0,0,0),temph)); del temph
    if verbose:
        plot_fdt_hrt(hmi_map_rot, hrt_map)
    
    return hrt_map, hmi_map_rot

        
def correction(hrt_map, hmi_map, deriv=False,verbose=False,max_iterations = 10):
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

    hmi_map = sunpy.map.Map((np.nan_to_num(hmi_map.data,nan=0,posinf=0,neginf=0),hmi_map.fits_header))

    while np.any(np.abs(shift)>1e-1):
        hrt_map = sunpy.map.Map((und_hrt,ht))
        
        with propagate_with_solar_surface():
            with SphericalScreen(hmi_map.observer_coordinate, only_off_disk=True):
                bl = hrt_map.bottom_left_coord
                tr = hrt_map.top_right_coord

                top_right = hmi_map.world_to_pixel(tr)
                bottom_left = hmi_map.world_to_pixel(bl)

                tr_hmi_map = np.array([top_right.x.value,top_right.y.value])
                bl_hmi_map = np.array([bottom_left.x.value,bottom_left.y.value])

                hmi_submap = hmi_map.submap(bl_hmi_map*u.pix,top_right=tr_hmi_map*u.pix)
                
                hpc_coords = sunpy.map.all_coordinates_from_map(hmi_submap)
                mask = ~sunpy.map.coordinate_is_on_solar_disk(hpc_coords)
                
                tt = hmi_submap.data; tt[mask==1] = 0
                hmi_submap = sunpy.map.Map((tt,hmi_submap.fits_header)); del tt

                hrt_remap = remap(hmi_submap, hrt_map, out_shape = hmi_submap.data.shape)

                if verbose:
                    plot_fdt_hrt(hmi_submap, hrt_remap)
            
        ref = np.nan_to_num(hmi_submap.data.copy(),True,0,0,0)
        temp = np.nan_to_num(hrt_remap.data.copy(),True,0,0,0)

        # masking to 99% of the sun to avoid limb fitting issues (mostly in LL)
        center_hmi = center_coord(hmi_submap.fits_header)
        Rhmi = hmi_submap.fits_header['RSUN_OBS'] / hmi_submap.fits_header['CDELT1']
        mask99 = circular_mask(*ref.shape,center_hmi,Rhmi*0.99)
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
            shift = [shift[0]+s[0],shift[1]+s[1]]
            temp = fft_shift(np.nan_to_num(hrt_remap.data.copy(),True,0,0,0), shift); temp[np.isinf(temp)] = 0; temp[np.isnan(temp)] = 0
            it += 1

        ht = translate_header(ht.copy(),-np.asarray(shift)*hmi_submap.fits_header['CDELT1']/hrt_map.fits_header['CDELT1'] * \
                                        hmi_map.fits_header['DSUN_OBS']/hrt_map.fits_header['DSUN_OBS'],
                                        mode='crval')
        print(it,'iterations shift (x,y):',round(shift[1],2),round(shift[0],2))

        i+=1
        if i == max_iterations:
            print('Maximum iterations reached:',i)
            match = False
            break
    
    hrt_map = sunpy.map.Map((und_hrt,ht))
    
    hrt_remap = remap(hmi_submap, hrt_map, out_shape = hmi_submap.data.shape)

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

def run_HMI_correction(data, header, verbose = False, **kwargs):

    parameters = dict(filename = None, print_values = False, hmi_path = None)
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
            if parameters['filename'] is not None:
                ht = fits.getheader(f)
                hmi_map, cache_dir, hmi_name = downloadClosestHMI(ht,ht['DATE-AVG'],"calchetti@mps.mpg.de",path=True,hmi_path=parameters['hmi_path'])
            else:
                hmi_map, cache_dir, hmi_name = downloadClosestHMI(header,header['DATE-AVG'],"calchetti@mps.mpg.de",path=True,hmi_path=parameters['hmi_path'])
        except FileNotFoundError as e:
            printc('No HMI file could be found for this HRT file, return None', color=bcolors.FAIL)
            for key in newWCS.keys():
                newWCS[key].append(None)
            continue

        print("")
        print(f'Processing the file: {f}')
        print(f'Closest HMI file: {hmi_name}')
        print("")
        # print(f'Processing the file: {filename}')

        if parameters['filename'] is not None:
            hrt_map, hmi_map_rot = prepare_data(f, hmi_map, 0.15, undistortion=False, verbose=False)
        else:
            hrt_map, hmi_map_rot = prepare_data((data,header), hmi_map, 0.15, undistortion=False, verbose=False)

        # try:
        hrt_map, hrt_remap, n, t0 = correction(hrt_map, hmi_map_rot, deriv=False,verbose=False)
        # except Exception as e:
        #     printc(f"There was an error in the WCS correction, return None. This is the error: {e}", color=bcolors.FAIL)
        #     for key in newWCS.keys():
        #         newWCS[key].append(None)
        #     continue
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
            fig, ax = plot_fdt_hrt(hmi_map, hrt_remap)

    if parameters['print_values']:
        print('WCS correction results:')
        print(cols)

    # empty the cache
    if os.path.isfile(hmi_name) and parameters['hmi_path'] is None:
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
    return newWCS
    
