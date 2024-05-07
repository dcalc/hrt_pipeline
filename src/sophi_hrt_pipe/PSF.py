"""
PSF deconvolution provided by Francisco Javier Bailén (IAA)
"""

import numpy as np
import datetime
from tqdm import tqdm
from photutils import CircularAperture
from scipy.fftpack import fftshift, ifftshift, fft2, ifft2
import sys

#PHI parameters
telescope='HRT' #'FDT' or 'HRT'
wvl=617.3341e-9 #Wavelength [m]
gamma1=1 #Gamma defined by Lofdahl and Scharmer 1994
# gamma2=0.1 #Gamma factor defined by Zhang 2017 to avoid divergence of "Q"
if telescope=='HRT':
    # N=1448 #128, 300 750 or 1448 (full_image) #Number of pixels
    fnum=29.4643 # f-number
    plate_scale=0.5 #Plate scale (arcsec/pixel)
elif telescope=='FDT':
    # N=2048 #600 (fine), 128 (coarse), 1536 (Stokes), 1279 (nominal PD), 200 (PD_new)
    fnum=33.1 #33.1
    plate_scale=3.75 #Plate scale (arcsec/pixel)
Delta_x=10e-6 #Size of the pixel [m]
# nuc=1/(wvl*fnum) #Critical frequency (m^(-1))
# inc_nu=1/(N*Delta_x)
# R=(1/2)*nuc/inc_nu #Pupil radius [pixel]
# nuc=2*R #critical frequency (pixels)
# nuc-=1#1 or 2 #To avoid boundary zeros when calculating the OTF (Julian usa 1)

# if gamma2!=0:
#     print('WARNING: gamma2 equal to %g'%gamma2)

def fourier2(f,s=None):
    """
    This function calculates the Direct Fast Fourier Transform of an
    array and shifts it to center its spectrum. Input must be a real numpy
    array
    Input:
        I: (real) 2D numpy array (image)
    Output:
        O: 2D numpy array with the Fourier transform of the input
    """
    #if flag==1:
    #    fft2=pyfftw.interfaces.numpy_fft.fft2
    #    pyfftw.interfaces.cache.enable() #Turn on cache for optimum performance

    #F=fftshift(fft2(f,s=s))
    F=fftshift(fft2(f))
    return F

def corr(f,g,norma=False):
    """
    This function returns the correlation of two vector or 2D matrices f and g.
    It is important to notice that the order MATTERS in correlations, in contrast
    to in convolution. The normalization factor is chosen from Bonet
    "crosscorr_c.pro".
    Parameters:
        f,g: Numpy vectors or 2D matrices
    """

    n=f.shape[1]
    F=fft2(f)
    G=fft2(g)
    power=n*n*np.conj(F)*G #Normalized correlation
    c=ifft2(power)
    norma_corr=np.abs(c[0,0])
    #c=np.real(c) #This is only true for the auto-correlation
    c=ifftshift(c)

    if norma==True:
        return norma_corr,c
    else:
        return c

def apod(nx,ny,perc):
   """
   Apodization window of size nx x ny. The input parameter
   perc accounts for the percentage of the window that is apodized
   """
   nx = int(nx)
   ny = int(ny)

   wx = np.ones(nx)
   wy = np.ones(ny)

   nxw = int(perc*nx/100.)
   nyw = int(perc*ny/100.)

   wi_x = 0.5*(1.-np.cos(np.pi*np.arange(0,nxw,1)/nxw))
   wi_y = 0.5*(1.-np.cos(np.pi*np.arange(0,nxw,1)/nxw))
   wx[0:nxw] = wi_x
   wx[nx-nxw:nx]= wi_x[::-1]
   wy[0:nyw] = wi_y
   wy[ny-nyw:ny]= wi_x[::-1]

   win = np.outer(wx,wy)
   return win

def FTpad(IM,Nout):
    """
    Carries out zero-padding to upsample an image IM in Fourier domain
    Input:
        IM: Numpy array in Fourier domain
        outsize: size of the new array

    """
    Nin=IM.shape[0]
    pd=int((Nout-Nin)/2)
    IM=np.fft.fftshift(IM)
    IMout=np.pad(IM,((pd,pd),(pd,pd)),'constant')
    IMout=np.fft.ifftshift(IMout)*Nout*Nout/(Nin*Nin)
    return IMout

def dftups(M,n_out,kappa,roff,coff):
    """
    Upsampled cross-correlation obtained by matrix multiplication
    Inputs:
        M: input image for calculation of the DFT
        n_out: number of pixels in the output upsampled DFT
        kappa: inverse of subpixel precision (kappa=20 -> 0.005 pixel precision)
        roff, coff: row and column offsets to shift the output array to a
            region of interest
    """
    nr,nc=M.shape
    kernc=np.exp((-1j*2*np.pi/(nc*kappa))*np.outer(\
    np.fft.ifftshift(np.arange(0,nc).T-np.floor(nc/2)),np.arange(0,n_out)-coff))

    kernr=np.exp((-1j*2*np.pi/(nr*kappa))*np.outer(\
    np.arange(0,n_out)-roff,np.fft.ifftshift(np.arange(0,nr).T-np.floor(nr/2))))
    return kernr @ M @ kernc

def dftreg(F,G,kappa):
    """
    Calculates the shift between a couple of images 'f' and 'g' with subpixel
    accuracy following the second method presented in
    Sicairos 2008, Efficient subpixel image registration algorithm.
    Input:
        F,G: ffts of images 'f' and 'g' without applying any fftshift
        kappa: inverse of subpixel precision (kappa=20 -> 0.005 pixel precision)
    Output:

    """
    nr,nc=np.shape(F)
    Nr=np.fft.ifftshift(np.arange(-np.fix(nr/2),np.ceil(nr/2)))
    Nc=np.fft.ifftshift(np.arange(-np.fix(nc/2),np.ceil(nc/2)))
    CC=np.fft.ifft2(FTpad(F*np.conj(G),2*nr))
    CCabs=np.abs(CC)
    ind = np.unravel_index(np.argmax(CCabs, axis=None), CCabs.shape)
    CCmax=CC[ind]*nr*nc
    Nr2=np.fft.ifftshift(np.arange(-np.fix(nr),np.ceil(nr)))
    Nc2=np.fft.ifftshift(np.arange(-np.fix(nc),np.ceil(nc)))
    row_shift=Nr2[ind[0]]/2
    col_shift=Nr2[ind[1]]/2

    #Initial shift estimate in upsampled grid
    row_shift=round(row_shift*kappa)/kappa
    col_shift=round(col_shift*kappa)/kappa
    dftshift=np.fix(np.ceil(kappa*1.5)/2)

    #DFT by matrix multiplication
    CC=np.conj(dftups(G*np.conj(F),np.ceil(kappa*1.5),kappa,\
    dftshift-row_shift*kappa,dftshift-col_shift*kappa))
    CCabs=np.abs(CC)
    ind = np.unravel_index(np.argmax(CCabs, axis=None), CCabs.shape)
    CCmax=CC[ind]
    rloc,cloc=ind-dftshift
    row_shift=row_shift+rloc/kappa
    col_shift=col_shift+cloc/kappa
    rg00=np.sum(np.abs(F)**2)
    rf00=np.sum(np.abs(G)**2)
    error=np.sqrt(1-np.abs(CCmax)**2/(rg00*rf00))
    Nc,Nr=np.meshgrid(Nc,Nr)
    Gshift=G*np.exp(1j*2*np.pi*(-row_shift*Nr/nr-col_shift*Nc/nc))
    return error,row_shift,col_shift,Gshift

def pmask(nuc,N):
    """
    Mask equal to 1 where cir_aperture is 0
    """
    pmask=np.where(cir_aperture(R=nuc,N=N,ct=0)==0,1,0)
    #pmask=1-cir_aperture(R=nuc,N=N)
    #pmask=np.where(pmask<0,0,pmask)
    return pmask

def noise_power(Of,nuc,N,filterfactor=1.5):
    """
    Average level of noise power of the image. Based on Bonet's program
    'noise_level.pro'. Noise is computed on 4 quadrants of the
    FFT of the image, beyond the critical frequency, to elliminate
    the dreadful cross (horizontal and vertical spurious signal in the FFT)
    that appears because of the finite Nyquist frecuency of the detector.
    """
    #Circular obscuration mask to calculate the noise beyond the critical freq.
    cir_obs=pmask(nuc=nuc,N=N)

    #Calculation of noise
    #power=np.sum(np.abs(Of)**2*cir_obs)/np.sum(cir_obs)
    #1st quadrant
    x2=int(np.floor(N/2-nuc*np.sqrt(2)/2))
    power=np.sum((np.abs(Of)**2*cir_obs)[0:x2,0:x2])/np.sum(cir_obs[0:x2,0:x2])
    #2nd quadrant
    x3=N-x2
    power+=np.sum((np.abs(Of)**2*cir_obs)[x3:N,0:x2])/np.sum(cir_obs[x3:N,0:x2])
    #3rd quadrant
    power+=np.sum((np.abs(Of)**2*cir_obs)[0:x2,x3:N])/np.sum(cir_obs[0:x2,x3:N])
     #4th quadrant
    power+=np.sum((np.abs(Of)**2*cir_obs)[x3:N,x3:N])/np.sum(cir_obs[x3:N,x3:N])

    #To obtain a more conservative filter in filter_sch
    power=filterfactor*power
    return power

def prepare_PD(ima,nuc,N,wind=True,kappa=100):
    """
    This function calculates gamma for each subpatch, apodizes the subpatch
    and calculates the Fourier transform of the focused and defocused images
    """
    #Initialization of arrays
    if ima.ndim==3:
        Nima=ima.shape[2]
    elif ima.ndim==2: #The array contains a single image
        ima=ima[..., np.newaxis] #To create a 3rd dummy dimension
        Nima=1    
    gamma=np.zeros(Nima)
    gamma[0]=1
    Ok=np.zeros((ima.shape[0],ima.shape[1],Nima),dtype='complex128')

    #Calculation of gamma before apodizing
    Of=fourier2(ima[:,:,0])
    for i in range(1,Nima):
        #Fourier transforms of the images and of the PSFs
        Ok[:,:,i]=fourier2(ima[:,:,i])
        gamma[i]=noise_power(Of,nuc=nuc,N=N)/noise_power(Ok[:,:,i],nuc=nuc,N=N)

    #Normalization to get mean 0 and apodization
    if wind==True:
        wind=apod(ima.shape[0],ima.shape[1],10) #Apodization of subframes
    else:
        wind=np.ones((ima.shape[0],ima.shape[1]))

    #Apodization and FFT of focused image
    susf=np.sum(wind*ima[:,:,0])/np.sum(wind)
    of=(ima[:,:,0]-susf)*wind
    #Of=mf.fourier2(of)
    Of=fft2(of)
    Of=Of/(N**2)
    Ok[:,:,0]=fftshift(Of)

    #Apodization and FFTs of each of the K images
    for i in range(1,Nima):
        susi=np.sum(wind*ima[:,:,i])/np.sum(wind)
        imak=(ima[:,:,i]-susi)*wind

        #Fourier transforms of the images and of the PSFs
        #Ok[:,:,i]=mf.fourier2(imak)
        Ok[:,:,i]=fft2(imak)
        Ok[:,:,i]=Ok[:,:,i]/(N**2)

        #Compute and correct the shifts between images
        error,row_shift,col_shift,Gshift=dftreg(Of,Ok[:,:,i],kappa)
        Ok[:,:,i]=Gshift #We shift Ok[:,:,i]
        #Shift to center the FTTs
        Ok[:,:,i]=fftshift(Ok[:,:,i])
    return Ok, gamma, wind, susf

def sampling2(N,R):
    """
    It returns RHO and THETA with the sampling of the pupil carried out
    in fatima.py. RHO is already normalized to the unity.
    """
    # DC change on 2024-05-02 as in phase() from PSF_interp.py
    
    # r=1
    # x=np.linspace(-r,r,2*int(R))
    # y=np.copy(x)
    # [X,Y]=np.meshgrid(x,y)
    # RHO = np.sqrt(X**2+Y**2)
    # THETA = np.arctan2(Y, X)
    # RHO0=np.zeros((N,N))
    # THETA0=np.copy(RHO0)
    # RHO0[N//2-int(R):N//2+int(R),N//2-int(R):N//2+int(R)]=RHO
    # THETA0[N//2-int(R):N//2+int(R),N//2-int(R):N//2+int(R)]=THETA

    x = (np.arange(0,N) - np.floor(N/2))/R
    y = (np.arange(0,N) - np.floor(N/2))/R

    [X,Y] = np.meshgrid(x,y) 
    RHO0 = np.sqrt(X**2+Y**2)
    THETA0 = np.arctan2(Y, X)

    THETA0[RHO0 > 1] = 0
    RHO0[RHO0 > 1] = 0
    
    return RHO0,THETA0

def cir_aperture(R,N,ct=None):

    #Offset to be subtracted to center the circle
    if ct==None:
        if N%2 != 0: #If N is odd (impar)
            #print('Number of pixels should be an even integer!\n')
            ct = 1/2
        else: #If N is even (par)
            # DC change on 2024-05-02 to set the center as in sampling2
            # ct = -1/2
            ct = 0

    N = int(N)
    R = int(R)
    A = CircularAperture((N/2-ct,N/2-ct),r=R) #Circular mask (1s in and 0s out)
    # DC change on 2024-05-02 to set the center as in sampling2
    # A = A.to_mask(method='exact') #Mask with exact value in edge pixels
    x = (np.arange(0,N) - np.floor(N/2))/R
    y = (np.arange(0,N) - np.floor(N/2))/R

    [X,Y] = np.meshgrid(x,y) 
    A = np.sqrt(X**2+Y**2) <= 1


    #A = CircularAperture((N/2-ct,N/2-ct),r=R-1)
    #A = A.to_mask(method='center')
    
    # DC change on 2024-05-02 to set the center as in sampling2
    # A = A.to_image(shape=(N,N)) #Conversion from mask to image
    return A

def aperture(N,R,cobs=0,spider=0):
    """
    This function calculates a simple aperture function that is 1 within
    a circle of radius R, takes and intermediate value between 0
    and 1 in the edge and 0 otherwise. The values in the edges are calculated
    according to the percentage of area corresponding to the intersection of the
    physical aperture and the edge pixels.
    http://photutils.readthedocs.io/en/stable/aperture.html
    Input:
        N: 1D size of the detector
        R: radius (in pixel units) of the pupil in the detector
        cobs: central obscuration (as percentage of radius)
        spider: width of spider arms (in pixels)
    Output:
        A: 2D array with 0s and 1s
    """
    A=cir_aperture(R=R,N=N)

    #If central obscuration:
    if (cobs != 0):
        if N%2 != 0:
            #print('Number of pixels should be an even integer!\n')
            ct = 0.
        else:
            ct = 1/2
        B=CircularAperture((N/2-ct,N/2-ct),r = R*cobs/100.)
        B=B.to_mask(method='exact') #Mask with exact value in edge pixels
        B=B.to_image(shape=(N,N)) #Conversion from mask to image
        A=A-B
        A = np.where(A<=0,0,A)

    #If spider:
    if (spider != 0):
        from scipy.misc import imresize
        C=np.array((N*10,N*10))
        M = N*10 + 10
        S = R*spider/10.
        C[int(M/2-S/2):int(M/2+S/2),:] = 1.0
        C[:,int(M/2-S/2):int(M/2+S/2)] = 1.0
        nC = imresize(C, (N, N))#, interp="bilinear")
        nC = nC / np.max(nC)
        nC=np.where(nC<=0,0,nC)
        A = A - nC
        A = np.where(A<=0,0,A)
    return A

def radialpol(m,n,rho):
    """
    This funcion calculates the radial polynomials of the Zernike polynomials
    Arguments:
        m,n: azimuthal and radial degrees of the Zernike polynomials
        rho: radial coordinate normalized to the unit (vector or array)
    """
    from scipy import special as sc
    l=(n-np.abs(m))/2
    t=(n+np.abs(m))/2
    if np.size(rho)==1:
        r=0
    else:
        r=np.zeros(rho.shape)
    for s in range(int(l)+1):
        r+=(((-1)**s*sc.factorial(n-s))/(sc.factorial(s)*\
        sc.factorial(t-s)*sc.factorial(l-s)))*rho**(n-2*s)
    return r

def kroneckerDelta(m,n):
    """
    This function calculates the kroneker delta for indices m and n
    """
    if m==n:
        delta=1
    else:
        delta=0
    return delta

def zernike(m,n,rho,theta):
    """
    This function calculates the Zernike polinomial of degree m,n
    Arguments:
        m,n: azimuthal and radial degrees of the Zernike polynomials
        rho: radial coordinate normalized to the unit (vector or array)
        theta: polar angle (vector or array)
    """
    N=np.sqrt((2*(n+1))/(1+kroneckerDelta(m,0)))
    if m>0:
        Z=N*radialpol(m,n,rho)*np.cos(m*theta)
    elif m<0:
        Z=N*radialpol(m,n,rho)*np.sin(-m*theta);
    else:
        Z=N*radialpol(m,n,rho)

    Z=np.roll(Z, 1, axis=1)
    Z=np.roll(Z, 1, axis=0)
    return Z

def zernikej_Noll(j,rho,theta):
    """
    Returns the zernike polinomial using the equivalence between single
    indices (j) and athimuthal (m) and radial (n) degrees.
    Ref: Thibos, L. N. et al (2002). "Standards for reporting the optical
    aberrations of eyes"
    """
    zernike_equiv=np.loadtxt('/data/slam/home/calchetti/hrt_pipeline/csv/j_to_Noll.txt',dtype='int')

    m=zernike_equiv[j-1,1]
    n=zernike_equiv[j-1,2]
    zj=zernike(m,n,rho,theta)
    return zj

def pupil(a,a_d,RHO,THETA,A):
    """
    This function calculates the generalized pupil function of a circular
    aperture for an incident wavefront with an aberration given by the
    Zernike coefficients following Noll's order.
    The aperture function is given by A ('aperture.py').
    Input:
        a: 1D array with Zernike coefficients (ordered following Noll's order
           The first element,a[0], is the Piston, a[1] and a[2] are Tilts
           in the X and Y direction ...), except if tiptilt is True
        a_d: Defocus Zernike coefficient for defocused image (float or int)
            or array with ALL the aberrations introduced by the defocused image
        RHO,THETA: meshgrid with polar coordinates (rho normalised to 1)
        A: aperture 2D array ('aperture')
    Output:
        p: 2D complex array describing the pupil function
    """
    try:
        Jmax=len(a) #Max. number of Zernikes to be employed
    except TypeError:
        print('Error in "pupil" function: "a" must be a Numpy array or a list')
        sys.exit()

    #Phase induced during PD
    if isinstance(a_d, (float,int)): #If float, then induce a defocus a_d
        phi=a_d*zernike(0,2,RHO,THETA)#diversity phase
    #If vector, a_d contain all the aberrations of the defocused image
    else:
        print('WARNING: a_d interpreted as a vector with the aberrations of the\
        PD plate in "pupil.py"')
        #If array, then get zernike coefficients from a_d
        Jmax_div=len(a_d)
        jj=0
        phi=0*RHO #To have an array filled with zeroes
        while jj<Jmax_div:
            jj+=1
            phi+=a_d[jj-1]*zernikej_Noll(jj,RHO,THETA)

    #Wavefront error produced by Zernike ('a') coefficients
    if a_d==0: #If focused image, then do not consider offset and tip/tilt
        a[:3]=0

    j=0
    while j<Jmax:
        j+=1
        phi+=a[j-1]*zernikej_Noll(j,RHO,THETA)
    phi=A*phi
    p=A*np.exp(1j*phi)
    return p

def select_tiptilt(a,i,K):
    """
    Returns Zernike vector (a) with selected tip tilt terms for each position
    (i) along the series of K images
    """
    tiltx=float(a[(2*i+1)])
    tilty=float(a[(2*i+2)])
    firsta=np.zeros((3,1))
    firsta[0]=0
    firsta[1]=tiltx
    firsta[2]=tilty
    a1=np.concatenate((firsta,a[(2*K+1):])) #0 is for the offset term
    return a1

def OTF(a,a_d,RHO,THETA,ap,norm=None,K=2,tiptilt=True):
    """
    This function calculates the OTFs of a circular aperture for  incident
    wavefronts with aberrations given by a set of Zernike coefficients.
    The OTF is calculated as the autocorrelation of the pupil.
    Input:
        a,a_d,RHO,THETA,A: defined in 'pupil.py'
            a_d can be an array with shape K, where K is the number
            of images with different defocuses. In such a case, a_d
            contains the defocus coefficient of each image and the program
            returns an multidimensional array of OTFs
        norm:{None,True}, optional. 'True' for normalization purpose. Default
         is None
        tiptilt: {True,False). If True, the first 2*(K-1) Zernike terms
        correspond to tip/tilt terms.
    Output:
        otf: 2D complex array representing the OTF if len(a_d)!=K or
            a 3D complex array whose 3rd dimenstion indicates the  OTF of
            of each defocused image.
        norm: normalization factor of the OTF

    """
    #If a_d is a number or a vector containing the aberrations of the PD plate
    if isinstance(a_d, (float,int)) or len(a_d)>K:
        #Pupil
        p=pupil(a,a_d,RHO,THETA,ap)
        #OTF
        norma_otf,otf=corr(p,p,norma=True)
        if norm==True:
            norma=norma_otf
            #norma=np.max(np.abs(otf)[:])
            otf=otf/norma #Normalization of the OTF
        else:
            norma=1
        otf=otf[...,np.newaxis]#To create a 3rd dummy axis    
    #If a_d is an array containing K diversities
    elif len(a_d)==K:
        otf=np.zeros((RHO.shape[0],RHO.shape[0],K),dtype='complex128')
        norma=np.zeros(K)

        for i in range(K):
            #Select tiptilt terms for each image along the series
            if tiptilt is True:
                #Offset and tiptilt terms corresponding to each position
                a1=select_tiptilt(a,i,K)
            else:
                a1=a
            #Pupil computation
            p=pupil(a1,a_d[i],RHO,THETA,ap)

            #OTF
            norma_otf,otf[:,:,i]=corr(p,p,norma=True)
            if norm==True:
                norma[i]=norma_otf
                #norma=np.max(np.abs(otf)[:])
                otf[:,:,i]=otf[:,:,i]/norma[i] #Normalization of the OTF
            else:
                norma[i]=1
    return otf,norma

def Qfactor(Hk,nuc,N,gamma=gamma1,reg=0.1):
    """
    Q factor defined in Lofdahl and Scharmer 1994 for construction of the merit
    function
    Input:
        Hk: 3D array with the OTFs of the system for each defocus
    Output:
        Q: 2D complex numpy array representing the Q factor
    """
    np.seterr(divide='ignore')
    #For the effect of gamma2 (reg) to increase with the frequency in the way
    #described in Martinez Pillet (2011), Sect. 9.2)
    if reg>0:
        nx=Hk.shape[0]
        ny=Hk.shape[1]
        x=np.linspace(-int(nx/2),int(nx/2),nx)
        y=np.linspace(-int(ny/2),int(ny/2),ny)
        [X,Y]=np.meshgrid(x,y)
        Q=1/np.sqrt(np.sum(gamma*np.abs(Hk)**2,axis=2)+reg*np.sqrt(X**2+Y**2)/nuc) #Linear
        #Q=1/np.sqrt(np.sum(gamma*np.abs(Hk)**2,axis=2)+reg*(X**2+Y**2)/nuc**2) #Quadratic
        #Q=1/np.sqrt(np.sum(gamma*np.abs(Hk)**2,axis=2)+reg*np.sqrt(np.sqrt(X**2+Y**2)/nuc)) #Sqrt
    else:
        Q=1/np.sqrt(np.sum(gamma*np.abs(Hk)**2,axis=2)+reg)
    
    Q=np.nan_to_num(Q, nan=0, posinf=0, neginf=0)
    Q=Q*cir_aperture(R=nuc,N=N,ct=0)
    return Q

def Qinv(Hk,nuc,N,gamma=gamma1):
    """
    Inverse of Q. Defined for test purposes
    Input:
        Hk: 3D array with the OTFs of the system for each defocus
    Output:
        Qinv:
    """
    Qinv=np.sqrt(np.sum(gamma*np.abs(Hk)**2,axis=2))
    Qinv=Qinv*cir_aperture(R=nuc,N=N,ct=0)
    return Qinv

def Ffactor(Q,Ok,Hk,gamma=gamma1):
    """
    F factor defined by Lofdahl and Scharmer 1994 (Eq. 5) in the general
    form of Paxman 1992 (Eq. 19). Gamma is added, too.
    Input:
        Q: Q returned in Qfactor
        Ok: 3D array with the FFTs of each of the PD images with different defoc.
        Hk: 3D array with the OTFs of the system for each defocus
    Output:
        Zk: 2D array with the Zk factor
    """
    return Q**2*np.sum(gamma*Ok*np.conj(Hk),axis=2)

def filter_sch(Q,Ok,Hk,nuc,N,low_f=0.2,gamma=gamma1):
    """
    Filter of the Fourier transforms of the focused and defocused
    object (Eqs. 18-19 of Lofdahl & Scharmer 1994). Based on
    filter_sch.pro (Bonet's IDL programs)
    Input:
        Q: Q returned in Qfactor
        Ok: 3D array with the FFTs of each of the PD images with different defoc.
        Hk: 3D array with the OTFs of the system for each defocus
    Output:
        filter:2D array (float64) with filter
    """
    from scipy.ndimage import median_filter, uniform_filter
    denom=np.abs(Ffactor(Q,Ok,Hk,gamma=gamma))**2\
    *Qinv(Hk,gamma=gamma,nuc=nuc,N=N)**2

    filter=noise_power(Ok[:,:,0],nuc=nuc,N=N)/uniform_filter(denom, size=3)
    filter=(filter+np.flip(filter))/2
    filter=1-filter
    filter=np.where(filter<low_f,0,filter)
    filter=np.where(filter>1,1,filter)
    #filter=erosion(filter, square(3)) #To remove isolated peaks (also decreases
                                       #the area of the central region)

    filter=median_filter(filter,size=9)
    filter=uniform_filter(filter,size=3)
    filter=filter*cir_aperture(R=nuc,N=N,ct=0)
    filter=np.nan_to_num(filter, nan=0, posinf=0, neginf=0)

    #Apply low_filter again to elliminate the remaining isolated peaks
    filter=np.where(filter<low_f,0,filter)
    return filter

def arraymerit(A,B):
    """
    This function returns calculates the individual error
    functions of Eq. 20 in Paxman and it stores them in
    an array of shape (N,N,nk), where nk=(K-1)*K/2.
    Input:
        A,B: 3D arrays of shape (N,N,K) containing Ok and Hk or derHk
    """
    K=A.shape[2] #Number of diversities (including focused image)
    nk=int((K-1)*K/2)
    arraymerit=np.zeros((A.shape[0],A.shape[1],nk),dtype='complex128') #Initialization of E
    i=-1
    for j in range(K-1):
        for k in range(j+1,K):
            i+=1
            arraymerit[:,:,i]=A[:,:,j]*B[:,:,k]-A[:,:,k]*B[:,:,j]
    return arraymerit

def meritE(Ok,Hk,Q):
    """
    Merit function 'E' for phase diversity optimization
    Ref: Lofdahl and Scharmer 1994, Eq.9
    Input:
        Q: 2D array returned by Qfactor
        Ok: 3D array with the FFTs of each of the PD images with different defoc.
        Hk: 3D array with the OTFs of the system for each defocus
    Ouput:
        E: 2D complex array with E factor for construction of the merit function
            or 3D complex array with the error metrics of each possible
            pair of the K images
        function
    """
    #Lofdahl error metric
    #E=Q*(Ok[:,:,1]*Hk[:,:,0]-Ok[:,:,0]*Hk[:,:,1])

    #General error metrix from Paxman merit function
    #E=Q*np.sum(Ok*np.conj(Hk),axis=2)

    #Alternative general error metric
    #E=Q*sumamerit(Ok,Hk)

    #Array with invidual error metrics
    E=arraymerit(Ok,Hk)

    if E.ndim==3:
        for i in range(E.shape[2]):
            E[:,:,i]=Q*E[:,:,i]
    elif E.ndim==2:
        E=Q*E
    return E

def merite(E):
    """
    Inverse transform of 'E' ('meritE' function).
    If E is a 3D array, it returns also a 3D array with the IFFT of each metric.
    Input:
        E: 2D complex array with E factor for construction of the
            classical (Löfdahl & Scharmer) merit function
            or 3D complex array with the error metrics for each possible
            combination of pairs of the K images
    Output:
        e: IFFT of E (2D array or 3D depending of the shape of E)
    """
    if E.ndim==3: #Array with individual merit functions of K diverities
        e=np.zeros(E.shape)
        for i in range(E.shape[2]):
            e0=ifftshift(E[:,:,i])
            e0=ifft2(e0)
            n=E.shape[0]
            e0=n*n*e0.real
            e[:,:,i]=e0-np.mean(e0)#We substract the mean value (Bonet e_func.pro (CORE2007))
    elif E.ndim==2:
        e=ifftshift(E)
        e=ifft2(e)
        n=E.shape[0]
        e=n*n*e.real
        e-=np.mean(e)#We substract the mean value (Bonet e_func.pro (CORE2007))
    return e

def meritl(e,cut=None):
    """
    Merit function "L" defined by Lofdahl and Scharmer 1994.
    If 'e' is a 3D array, the merit function is generalized to work for
    K images with different defocuses
    """
    cutx=cut
    cuty=cut
    L=0
    if e.ndim==3: #If dim=3: sum individual merit functions
        #print('Individual merit functions:')
        for i in range(e.shape[2]):
            L+=np.sum(np.abs(e[cutx:-cuty,cutx:-cuty,i])**2)/e.shape[0]**2
        #    print(i,np.sum(np.abs(e[cutx:-cuty,cutx:-cuty,i])**2)/e.shape[0]**2)
    elif e.ndim==2:
        if cut!=None:
            L=np.sum(np.abs(e[cutx:-cuty,cutx:-cuty])**2)/e.shape[0]**2
        else:
            L=np.sum(np.abs(e[:,:])**2)
    return L

def object_estimate(ima,a,a_d,reg=0.1,wind=True,cobs=0,cut=29,low_f=0.2,tiptilt=False,
                    noise='default',aberr_cor=False):
    """
    This function restores an image or an array of images employing a given
    set of Zernike coefficients.
    Inputs:
        ima: 2D array with the image to be restored or 3D array with PD
            images at different focus positions.
        a: array containing the set of Zernike coefficients in Noll's order    
        a_d: 0 if 'ima' is a 2D array or Numpy array with the list of
            focus positions if 'ima' is a 3D array
        reg: regularization parameter (also gamma2). Default: 0.1
        wind: True or False. True to apodize the image an False otherwise.
        cobs: central obscuration of the telescope (0 if off-axis).
        cut: number of pixels to be excluded near the edges of the image (to
            compute the merit function).
        low_f: low limit of the noise filter, as defined in 'filter_sch'         
        tiptilt: True or False. If True, the first Zernike coefficients refer to
            the Tip/Tilt term of each of the images. False by Default because
            the Tip/Tilt terms usually vary a lot along the image and images are
            aligned with subpixel accuracy by a cross-correlation technique in
            'prepare_PD'
        noise: 'default' to be computed as filt_scharmer. Otherwise, this variable
            should contain a 2x2 array with the filter.
    Output:
        object: restored image
        susf: offset defined in 'prepare_PD' 
        noise_filt: noise filter applied to deconvolve the images        
    """
    #Pupil sampling according to image size
    N=ima.shape[0]
    nuc=1/(wvl*fnum) #Critical frequency (m^(-1))
    inc_nu=1/(N*Delta_x)
    R=(1/2)*nuc/inc_nu #Pupil radius [pixel]
    nuc=2*R #critical frequency (pixels)
    nuc-=1
    ap=aperture(N,R,cobs=cobs)
    RHO,THETA=sampling2(N=N,R=R)

    #Fourier transform images
    Ok, gamma, wind, susf=prepare_PD(ima,nuc=nuc,N=N,wind=wind)

    if isinstance(a_d, np.ndarray): #If a_d is an array...
        if a_d[0]==a_d[1]:#In this case, the 2nd image is a dummy image
            if a_d[0]==0:
                gamma=[1,0] #To account only for the 1st image

    #OTFs
    Hk,normhk=OTF(a,a_d,RHO,THETA,ap,norm=True,K=Ok.shape[2],tiptilt=tiptilt)
    
    #Restoration
    Q=Qfactor(Hk,gamma=gamma,reg=reg,nuc=nuc,N=N)

    if type(noise) is str:
        if noise=='default':
            noise_filt=filter_sch(Q,Ok,Hk,gamma=gamma,nuc=nuc,N=N,low_f=low_f)
        else:
            print('WARNING. Only default is accepted as a string input. It will run anyway.')
            noise_filt=filter_sch(Q,Ok,Hk,gamma=gamma,nuc=nuc,N=N,low_f=low_f)
    else:
        noise_filt=noise

    #if gamma[1]==0 and N==1536: #For FDT images
    #    noise_filt=noise_filt*cir_aperture(R=nuc-200,N=N,ct=0)

    Nima=Ok.shape[2]
    for i in range(0,Nima):
        #Filtering in Fourier domain
        Ok[:,:,i]=noise_filt*Ok[:,:,i]

    #Compute and print merit function
    if Nima>1:
        E=meritE(Ok,Hk,Q)
        e=merite(E)
        L=meritl(e,cut=cut)#/(Nima*(Nima-1)/2)
        if a.any()!=0:
            print('Optimized merit function (K=%g):'%Nima,L)

    #Restoration
    O=Ffactor(Q,Ok,Hk,gamma=gamma)

    #Apply MTF of ideal telescope
    if aberr_cor:
        Hk_th,_ = OTF(np.zeros(a.shape),a_d,RHO,THETA,ap,norm=True,K=Ok.shape[2],tiptilt=tiptilt)
        O=Hk_th[...,0]*O

    Oshift=np.fft.fftshift(O)
    o=np.fft.ifft2(Oshift)
    #o=np.fft.fftshift(o)
    o_restored=N**2*o.real
    object=o_restored+susf
    return object,susf,noise_filt

def is_notebook() -> bool:
    """
    from https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    progressing bars are not shown in jupyter using this function
    """
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except (NameError,ImportError):
        return False      # Probably standard Python interpreter

def stokes_restoration(stokes_data,coefs,rest='lofdahl', gamma2=0.1,denoise=False,
                       wind_opt=True, low_f=0.1,num_iter=10,aberr_cor=False):
    """
    This function restores a cube of Stokes data from a given set of
    Zernike coefficients.
    Input:
        stokes_data: cube with Stokes vector (y,x,p,l)
        coefs: list of Zernike coefficients ordered following Noll's notation
        rest: type of deconvolution to be applied
            -> lofdahl: deconvolution of Löfdahl & Scharmer (1994) with a regularization
            parameter that depends on the frequency in the way described in Martínez-Pillet
            (2011), Sect. 9.2.
            -> lucy-richardson: Lucy-Richardson algorithm
            -> unsupervised_wiener: self-tuned Wiener filter defined in
            https://scikit-image.org/docs/stable/auto_examples/filters/plot_restoration.html
        gamma2: regularization parameter (Default: 0.1)
        denoise (True or False): if True, it applies a denoising algorithm to the original images.
        wind_opt (True or False): True to apodize the images.
        low_f: cut-off frequency of the noise filter in the function 'filter_sch'w
        num_iter: number of iterations if the Lucy-Richardson option is enable.

    """
    size = stokes_data[:,:,0,0].shape[0]
    # if size < 2048:
    #     edge_mask = np.zeros((size,size))
    #     edge_mask[3:-3,3:-3] = 1
    # else:
    #     edge_mask = np.ones((size,size))
    pad_width = int(size*10/(100-10*2))
    res_stokes = np.zeros((size+pad_width*2,size+pad_width*2,4,6))


    # if modulation is True:
    #     intensity_data=mod_hrt() @ stokes_data
    print('Restoration method:',rest)
    k=-1
    for i in tqdm(range(4),disable=is_notebook()):
        for j in tqdm(range(6),disable=is_notebook()):
            k+=1
            #Padding and restoration
            # if modulation is True:
            #     im0=intensity_data[:,:,i,j]
            # elif modulation is False:    
            im0 = stokes_data[:,:,i,j] #* edge_mask
            im0 = np.pad(im0, pad_width=((pad_width, pad_width), (pad_width, pad_width)),\
                          mode='symmetric')

                    
            #Denoising 
            # if denoise is True:
            #     sigma_est = restoration.estimate_sigma(im0, average_sigmas=True)
            #     im0=restoration.denoise_wavelet(im0, method='BayesShrink',
            #                                             mode='soft',sigma=sigma_est)

            #Restoration
            if rest=='lofdahl':
                if i==0 and j==0:#Optimum filter computed only for Stokes I at cont.
                    noise='default'#Löfdahl & Scharmer's (1994) optimum filter 
                else:
                    noise=noise_filt #To use always the same noise_filt (I at continuum)   
                
                #Restoration using mean power of noise at I_cont
                res_stokes[:,:,i,j],susf,noise_filt=object_estimate(im0,coefs,0,
                    reg=gamma2,wind=wind_opt,low_f=low_f,noise=noise,aberr_cor=aberr_cor)
                
            # not used in SO/PHI-HRT
            # elif rest=='unsupervised_wiener' or rest=='lucy-richardson':
            #     if i>0:# and modulation is False: 
            #         im0=im0+1 #Sum a term +1 to avoid change signs in Q, U, and V
            #     if k==0:#PSF computation and normalization at first iteration only
            #         #PSF is computed just once
            #         N=im0.shape[0]
            #         nuc,R=compute_nuc(N)
            #         nuc-=1
            #         ap=aperture(N,R)
            #         RHO,THETA=sampling2(N=N,R=R)
            #         PSF0=PSF(coefs,0,RHO,THETA,ap)
            #         maxPSF=np.max(PSF0)
            #         PSF0=PSF0/maxPSF
            #     if rest=='unsupervised_wiener':    
            #         res_stokes[:,:,i,j], _ = restoration.unsupervised_wiener(im0, PSF0)
            #         res_stokes[:,:,i,j]=np.sum(PSF0)*res_stokes[:,:,i,j]
            #     elif rest=='lucy-richardson':
            #         res_stokes[:,:,i,j] = restoration.richardson_lucy(im0, PSF0, clip=False,
            #                                                         num_iter=num_iter)
            #         if i>0:# and modulation is False: 
            #             im0=im0-1 #Substract now the +1 term that was summed to im0
            #             res_stokes[:,:,i,j]=res_stokes[:,:,i,j]-1

    # if modulation is True:
    #     #Demodulate data
    #     res_stokes=demod_hrt_fran() @ res_stokes

    #We extract only the subfield we are interested in
    res_stokes = res_stokes[pad_width:-pad_width,pad_width:-pad_width] #* edge_mask[:,:,np.newaxis,np.newaxis]
    return res_stokes

def edge_masking(stokes, mask):
    """
    This function masks the field stop and the off-limb area with the closest value to the mask
    # stokes (y,x,p,l)
    # mask (y,x)
    """
    from scipy.ndimage import binary_erosion, gaussian_filter
    stokes_edge = stokes.copy()

    struct = [[0,1,0],[1,1,1],[0,1,0]]
    mask_exp = binary_erosion(mask > 0, struct, iterations=7)
    mask_exp = ~mask_exp & binary_erosion(mask > 0, struct, iterations=6)

    ye, xe = np.where(mask_exp)
    yf, xf = np.where(mask==0)

    dd = np.sqrt((xf-xe[:,np.newaxis])**2 + (yf-ye[:,np.newaxis])**2); idx = np.argmin(dd,axis=0)
    stokes_edge[yf,xf] = stokes[ye[idx],xe[idx]]

    for l in range(6):
        for p in range(4):
            stokes_edge[mask==0,p,l] = gaussian_filter(stokes_edge[:,:,p,l],20,mode='reflect')[mask==0]
    
    return stokes_edge

def fran_restore(stokes_data, tobs, mask=None, rest='lofdahl', gamma2 = 0.1, low_f=0.1, denoise=False, num_iter=10, aberr_cor = False):
    #Input parameters
    # mask=None # mask of the field_stop and limb
    # rest='lofdahl' #'lofdahl','lucy-richardson'or 'unsupervised_wiener'. Type of restoration (Here only lofdahl is implemented)
    # gamma2=0.1 # regularization parameter
    # low_f=0.1 # Lower limit for the noise filter
    # denoise=False #True or False. Denoise of the original images.
    # num_iter=10 #If Lucy-Richardson is employed for the restoration
    # modulation=False #Modulate data to deconvolve intensity images instead of the Stokes parameters
    # fname='solo_L2_phi-hrt-stokes_20230407T032009_VnoPSF_0344070601'
    # ext='.fits' 
    # ffolder='./fits/Windows/stokes/HRT-TEST-DATA-DECONVOLUTION'#Folder containing the FITS file
    # aberr_corr=False # if True, convolution with Airy disk is applied
    
    #Restoration parameters
    wind_opt=True #True to apodize the image

    #Subregion to be shown for plots
    # x0=400
    # xf=800
    # y0=200
    # yf=600

    import csv
    file_name = '/data/slam/home/calchetti/hrt_pipeline/csv/2023_PD_result.csv'
    f = open(file_name,'r')
    reader = csv.reader(f,delimiter=',')
    Z = {}
    for row in reader:
        if row[0] == 'Date':
            dates = [datetime.datetime.strptime(r,'%d-%m-%y') for r in row[1:]]
        elif 'Z' in row[0]:
            Z[row[0]] = np.asarray([float(z) * 2*np.pi for z in row[1:]]) # *2*npi to convert to radians

    idx = np.argmin([abs(tobs - t) for t in dates])

    coefs = [Z[k][idx] for k in Z.keys()]

    #Zernike coefficients (in radians), starting from Z1 (offset)
    # coefs= [0.0,0.0,0.0,
    #     2.5384410288410995,
    #     0.10884970414084948,
    #     0.5778766523678903,
    #     0.17499516023395728,
    #     -0.22788919399655982,
    #     -0.10533576475415085,
    #     1.7010925728045585,
    #     1.3308455704245563,
    #     0.023446694681437074,
    #     -0.0667308907394535,
    #     -0.05345526313091036,
    #     0.03671330897504666,
    #     -0.05111485547951494,
    #     0.02619538247956514,
    #     0.12905269063257063,
    #     -0.015503522073734656,
    #     0.16477602560109075,
    #     -0.07481142465157851]
    coefs=np.array(coefs) #Convert into numpy array

    print('Zernike coefficients from PD dataset acquired on',dates[idx])
    print(np.round(coefs,5))
    if aberr_cor:
        print('Aberration correction is ON')

    #Read image and shift the axes
    # ima=pdf.read_image(ffolder+'/'+fname,ext=ext)
    # ima=np.moveaxis(ima,-2,0)
    # ima=np.moveaxis(ima,-1,1)
    # ima=np.moveaxis(ima,2,3)
    # stokes_data=ima

    if mask is not None:
        stokes_data_edge = edge_masking(stokes_data, mask)
    else:
        stokes_data_edge = stokes_data.copy()


    res_stokes=stokes_restoration(stokes_data_edge,coefs,rest=rest, gamma2=gamma2,
                                    denoise=denoise,
                                    wind_opt=wind_opt,low_f=low_f,
                                    num_iter=num_iter, aberr_cor=aberr_cor)
    
    if mask is not None:
        res_stokes[mask==0] = stokes_data[mask==0]
    
    return res_stokes, coefs
