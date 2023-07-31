# cimport the Cython declarations for numpy
# cython: language_level=3,
cimport numpy as np
import numpy as np

DTYPE_INT = np.intc
DTYPE_DOUBLE = np.float_
ctypedef np.npy_intp SIZE_t

# if you want to use the Numpy-C-API from Cython
# (not strictly necessary for this example, but good practice)
np.import_array()

# cdefine the signature of our c function
cdef extern from "./lib/milos.h":
    void call_milos(const int *options,
        const int *size,
        const double *waveaxis,
        double *weight,
        const double *initial_model,
        const double *inputdata,
        const double *cavity,
        double *outputdata)

# create the wrapper code, with numpy type annotations
def _py_milos(np.ndarray[int, ndim=1, mode="c"] options not None,
    np.ndarray[double, ndim=1, mode="c"] waveaxis not None,
    np.ndarray[double, ndim=1, mode="c"] weight not None,
    np.ndarray[double, ndim=1, mode="c"] initial_model not None,
    np.ndarray[double, ndim=1, mode="c"] inputdata not None,
    np.ndarray[double, ndim=1, mode="c"] cavity not None,
    np.ndarray[double, ndim=1, mode="c"] outputdata not None):

    # assert inputdata.dtype == DTYPE_DOUBLE and outputdata.dtype == DTYPE_DOUBLE and waveaxis.dtype == DTYPE_DOUBLE
    # assert options.dtype == DTYPE_INT
    # assert options[0] == len(waveaxis)
    size = np.array(inputdata.shape[0])

    call_milos(<int*> np.PyArray_DATA(options),
        <int*> np.PyArray_DATA(size),
        <double*> np.PyArray_DATA(waveaxis),
        <double*> np.PyArray_DATA(weight),
        <double*> np.PyArray_DATA(initial_model),
        <double*> np.PyArray_DATA(inputdata),
        <double*> np.PyArray_DATA(cavity),
        <double*> np.PyArray_DATA(outputdata))

    return

def pmilos(options,input_data,waveaxis,weight = None,initial_model = None,
    cavity: np.ndarray = np.empty([], dtype=float) ):


    if options.dtype != DTYPE_INT:
        options = options.astype(DTYPE_INT)
    if options.flags['C_CONTIGUOUS'] != True:
        print('non contiguous options')
        options = options.copy(order='c')

    # check if we are in synthesis or inversion mode:
    if options[3] in {1,2}: #Synthesis mode

        nmodels = len(input_data)//9
        print('models (pymilos.pyx): ',nmodels)

        try:
            assert input_data.ndim == 2
        except Exception:
            print('INPUT DATA DOES NOT HAVE PROPER DIMENSIONS. Check input data and options.')
            raise

        input_data = input_data.flatten(order='C')
        print('------ flattened: ',input_data.shape)

        if input_data.dtype != DTYPE_DOUBLE:
            input_data = input_data.astype(DTYPE_DOUBLE)
        if input_data.flags['C_CONTIGUOUS'] != True:
            print('non contiguous data')
            input_data = input_data.copy(order='C')

        if not cavity.shape:
            cavity = np.zeros((nmodels))
        if cavity.dtype != DTYPE_DOUBLE:
            cavity = cavity.astype(DTYPE_DOUBLE)
        if cavity.flags['C_CONTIGUOUS'] != True:
            print('non contiguous data')
            cavity = cavity.copy(order='C')

        if waveaxis.dtype != DTYPE_DOUBLE:
            waveaxis = waveaxis.astype(DTYPE_DOUBLE)
        if waveaxis.flags['C_CONTIGUOUS'] != True:
            print('non contiguous waveaxis')
            waveaxis = waveaxis.copy(order='C')

        if not type(weight) is np.ndarray:
            weight = np.array([1.,10.,10.,4.])

        if weight.dtype != DTYPE_DOUBLE:
            weight = weight.astype(DTYPE_DOUBLE)
        if weight.flags['C_CONTIGUOUS'] != True:
            print('non contiguous weight')
            weight = weight.copy(order='C')

        if not type(initial_model) is np.ndarray:
            M_B = 400
            M_GM = 30
            M_AZI = 120
            M_ETHA0 = 3
            M_LAMBDADOPP = 0.025
            M_AA = 1.0
            M_VLOS = 0.01
            M_S0 = 0.15
            M_S1 = 0.85
            initial_model = np.array([M_B,M_GM,M_AZI,M_ETHA0,M_LAMBDADOPP,M_AA,M_VLOS,M_S0,M_S1])

        if initial_model.dtype != DTYPE_DOUBLE:
            initial_model = initial_model.astype(DTYPE_DOUBLE)
        if initial_model.flags['C_CONTIGUOUS'] != True:
            print('non contiguous initial_model')
            initial_model = initial_model.copy(order='C')

        if options.shape[0] != 4 and options.shape[0] != 7:
            print("milos: Error en el numero de parametros: %d . Pruebe: milos NLAMBDA MAX_ITER CLASSICAL_ESTIMATES RFS [FWHM(in A) DELTA(in A) NPOINTS] perfil.txt\n")
            print("O bien: milos NLAMBDA MAX_ITER CLASSICAL_ESTIMATES RFS [DELTA(in A)] perfil.txt")
            print("Note : CLASSICAL_ESTIMATES=> 0: Disabled, 1: Enabled, 2: Only Classical Estimates.")
            print("RFS : 0: Disabled     1: Synthesis      2: Synthesis and Response Functions")
            print("Note when RFS>0: perfil.txt is considered as models.txt.")
            raise ValueError("Error in options")

        print("Options: ")
        print(options)

        assert options[0] == len(waveaxis)
        assert cavity.size == nmodels

        if options[3] == 1:   #No RFS
            output_data = np.zeros((len(waveaxis)*4*nmodels),dtype=DTYPE_DOUBLE)
        elif options[3] == 2: #RFS  #TODO This still does not work (memory issues)
            output_data = np.zeros((len(waveaxis)*4*nmodels),dtype=DTYPE_DOUBLE) #PREL
            # output_data = np.zeros((10*len(waveaxis)*4*nmodels),dtype=DTYPE_DOUBLE)

        _py_milos(options,waveaxis,weight,initial_model,input_data,cavity,output_data)

        if options[3] == 1:   #No RFS
            if nmodels == 1:
                return np.reshape(output_data,(4,len(waveaxis)))
            else:
                return np.reshape(output_data,(nmodels,4,len(waveaxis)))
        if options[3] == 2: #RFS #TODO This still does not work (memory issues)
            if nmodels == 1:
                return np.reshape(output_data,(4,len(waveaxis))) #PREL
                # return np.reshape(output_data,(10,4,len(waveaxis)))
            else:
                return np.reshape(output_data,(4,len(waveaxis))) #PREL
                # return np.reshape(output_data,(nmodels,10,4,len(waveaxis)))

    if options[3] == 0: #Inversion mode

        #prepare input
        nyx, npol, nwave = input_data.shape
        print('input_data shape: ',nyx, npol, nwave)

        input_data = input_data.flatten(order='C')
        print('------ flattened: ',input_data.shape)

        if input_data.dtype != DTYPE_DOUBLE:
            input_data = input_data.astype(DTYPE_DOUBLE)
        if input_data.flags['C_CONTIGUOUS'] != True:
            print('non contiguous data')
            input_data = input_data.copy(order='C')

        if not cavity.shape:
            cavity = np.zeros((nyx))
        if cavity.dtype != DTYPE_DOUBLE:
            cavity = cavity.astype(DTYPE_DOUBLE)
        if cavity.flags['C_CONTIGUOUS'] != True:
            print('non contiguous data')
            cavity = cavity.copy(order='C')

        if waveaxis.dtype != DTYPE_DOUBLE:
            waveaxis = waveaxis.astype(DTYPE_DOUBLE)
        if waveaxis.flags['C_CONTIGUOUS'] != True:
            print('non contiguous waveaxis')
            waveaxis = waveaxis.copy(order='c')

        if not type(weight) is np.ndarray:
            weight = np.array([1.,10.,10.,4.])

        if weight.dtype != DTYPE_DOUBLE:
            weight = weight.astype(DTYPE_DOUBLE)
        if weight.flags['C_CONTIGUOUS'] != True:
            print('non contiguous weight')
            weight = weight.copy(order='C')

        if not type(initial_model) is np.ndarray:
            M_B = 400
            M_GM = 30
            M_AZI = 120
            M_ETHA0 = 3
            M_LAMBDADOPP = 0.025
            M_AA = 1.0
            M_VLOS = 0.01
            M_S0 = 0.15
            M_S1 = 0.85
            initial_model = np.array([M_B,M_GM,M_AZI,M_ETHA0,M_LAMBDADOPP,M_AA,M_VLOS,M_S0,M_S1])

        if initial_model.dtype != DTYPE_DOUBLE:
            initial_model = initial_model.astype(DTYPE_DOUBLE)
        if initial_model.flags['C_CONTIGUOUS'] != True:
            print('non contiguous initial_model')
            initial_model = initial_model.copy(order='C')

        if options.shape[0] != 4 and options.shape[0] != 7:
            print("milos: Error en el numero de parametros: %d . Pruebe: milos NLAMBDA MAX_ITER CLASSICAL_ESTIMATES RFS [FWHM(in A) DELTA(in A) NPOINTS] perfil.txt\n")
            print("O bien: milos NLAMBDA MAX_ITER CLASSICAL_ESTIMATES RFS [DELTA(in A)] perfil.txt")
            print("Note : CLASSICAL_ESTIMATES=> 0: Disabled, 1: Enabled, 2: Only Classical Estimates.")
            print("RFS : 0: Disabled     1: Synthesis      2: Synthesis and Response Functions")
            print("Note when RFS>0: perfil.txt is considered as models.txt.")
            raise ValueError("Error in options")

        print("Options: ")
        print(options)
        print("Weights: ")
        print(weight)
        print("Initial Model: ")
        print(initial_model)

        #	if(CLASSICAL_ESTIMATES!=0 && CLASSICAL_ESTIMATES != 1 && CLASSICAL_ESTIMATES != 2){#
        #		printf("milos: Error in CLASSICAL_ESTIMATES parameter. [0,1,2] are valid values. Not accepted: %d\n",CLASSICAL_ESTIMATES);
        #		return -1;
        #	}

        #	if(RFS != 0 && RFS != 1 && RFS != 2){
        #		printf("milos: Error in RFS parameter. [0,1,2] are valid values. Not accepted: %d\n",RFS);
        #		return -1;
        #	}

        assert options[0] == len(waveaxis)
        assert cavity.size == nyx

        length = len(input_data)
        output_data = np.zeros((length//len(waveaxis)//4 * 12),dtype=DTYPE_DOUBLE)

        _py_milos(options,waveaxis,weight,initial_model,input_data,cavity,output_data)

        return np.reshape(output_data,(nyx,12))

    else:
        print('ERROR')
        return 0