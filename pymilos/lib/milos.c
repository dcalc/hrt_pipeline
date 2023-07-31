
//    _______             _______ _________ _        _______  _______
//   (  ____ \           (       )\__   __/( \      (  ___  )(  ____ \
//   | (    \/           | () () |   ) (   | (      | (   ) || (    \/
//   | |         _____   | || || |   | |   | |      | |   | || (_____
//   | |        (_____)  | |(_)| |   | |   | |      | |   | |(_____  )
//   | |                 | |   | |   | |   | |      | |   | |      ) |
//   | (____/\           | )   ( |___) (___| (____/\| (___) |/\____) |
//   (_______/           |/     \|\_______/(_______/(_______)\_______)
//
//
// CMILOS v0.90(2015)
// CMILOS v0.91 (July - 2021) - Pre-Adapted to python by D. Orozco Suarez
// CMILOS v0.92 (Oct - 2021) - Cleaner version D. Orozco Suarez
// CMILOS v0.93 (Oct - 2021) - added void
// CMILOS v0.94 (March - 2022) - modified CI for different continuum (DC)
// CMILOS v0.95 (May - 2023) - Added cavity in pmilos (DOS)
// CMILOS v0.96 (May - 2023) - Added synthesis in pmilos (DOS)
// CMILOS v0.97 (June - 2023) - PSF changes and much cleaning (DOS)
// RTE INVERSION C code for SOPHI (based on the IDL code MILOS by D. Orozco)
// juanp (IAA-CSIC)
//
// How to use:
//
//  >> milos NLAMBDA MAX_ITER CLASSICAL_ESTIMATES RFS [FWHM DELTA NPOINTS] profiles_file.txt > output.txt
//
//   NLAMBDA number of lambda of input profiles
//   MAX_ITER of inversion
//   CLASSICAL_ESTIMATES use classical estimates? 1 yes, 0 no, 2 only CE
//   RFS : 0 RTE, 1 Spectral Synthesis, 2 Spectral Synthesis + Response Funcions
//   [FWHM DELTA NPOINTS] use convolution with a gaussian? if the tree parameteres are defined yes, else no. Units in A. NPOINTS has to be odd.
//   profiles_file.txt name of input profiles file
//   output.txt name of output file
//
//

#include <time.h>
#include "defines.h"
#include "milos.h"  // for python compatibility

#include "nrutil.h"
#include "svdcmp.c"
#include "svdcordic.c"
//#include "tridiagonal.c"
#include "convolution.c"
#include <string.h>

float pythag(float a, float b);

void weights_init(int nlambda,double *sigma,PRECISION *weight,int nweight,PRECISION **wOut,PRECISION **sigOut,double noise);

int check(Init_Model *Model);
int lm_mils(Cuantic *cuantic,double * wlines,int nwlines,double *lambda,int nlambda,PRECISION *spectro,int nspectro,
		Init_Model *initModel, PRECISION *spectra,int err,double *chisqrf, int *iterOut,
		double slight, double toplim, int miter, PRECISION * weight,int nweight, int * fix,
		PRECISION *sigma, double filter, double ilambda, double noise, double *pol,
		double getshi,int triplete);

int mil_svd(PRECISION *h,PRECISION *beta,PRECISION *delta);

int multmatrixIDL(double *a,int naf,int nac, double *b,int nbf,int nbc,double **resultOut,int *fil,int *col);
int multmatrix_transposeD(double *a,int naf,int nac, double *b,int nbf,int nbc,double *result,int *fil,int *col);
int multmatrix3(PRECISION *a,int naf,int nac,double *b,int nbf,int nbc,double **result,int *fil,int *col);
double * leeVector(char *nombre,int tam);
double * transpose(double *mat,int fil,int col);

double total(double * A, int f,int c);
int multmatrix(PRECISION *a,int naf,int nac, PRECISION *b,int nbf,int nbc,PRECISION *result,int *fil,int *col);
int multmatrix2(double *a,int naf,int nac, PRECISION *b,int nbf,int nbc,double **result,int *fil,int *col);

int covarm(PRECISION *w,PRECISION *sig,int nsig,PRECISION *spectro,int nspectro,PRECISION *spectra,PRECISION  *d_spectra,
		PRECISION *beta,PRECISION *alpha);

int CalculaNfree(PRECISION *spectro,int nspectro);

double fchisqr(PRECISION * spectra,int nspectro,PRECISION *spectro,PRECISION *w,PRECISION *sig,double nfree);

void AplicaDelta(Init_Model *model,PRECISION * delta,int * fixed,Init_Model *modelout);
void FijaACeroDerivadasNoNecesarias(PRECISION * d_spectra,int *fixed,int nlambda);
void reformarVector(PRECISION **spectro,int neje);
void spectral_synthesis_convolution();
void response_functions_convolution();

void estimacionesClasicas(PRECISION lambda_0,double *lambda,int nlambda, PRECISION *spectro,Init_Model *initModel);

Cuantic* cuantic;   // Variable global, está hecho así, de momento,para parecerse al original
char * concatena(char *a, int n,char*b);

PRECISION ** PUNTEROS_CALCULOS_COMPARTIDOS;
int POSW_PUNTERO_CALCULOS_COMPARTIDOS;
int POSR_PUNTERO_CALCULOS_COMPARTIDOS;

PRECISION *gp1,*gp2,*dt,*dti,*gp3,*gp4,*gp5,*gp6,*etai_2;
PRECISION *gp4_gp2_rhoq,*gp5_gp2_rhou,*gp6_gp2_rhov;


PRECISION *dgp1,*dgp2,*dgp3,*dgp4,*dgp5,*dgp6,*d_dt;
PRECISION *d_ei,*d_eq,*d_eu,*d_ev,*d_rq,*d_ru,*d_rv;
PRECISION *dfi,*dshi;
PRECISION CC,CC_2,sin_gm,azi_2,sinis,cosis,cosis_2,cosi,sina,cosa,sinda,cosda,sindi,cosdi,sinis_cosa,sinis_sina;
PRECISION *fi_p,*fi_b,*fi_r,*shi_p,*shi_b,*shi_r;
PRECISION *etain,*etaqn,*etaun,*etavn,*rhoqn,*rhoun,*rhovn;
PRECISION *etai,*etaq,*etau,*etav,*rhoq,*rhou,*rhov;
PRECISION *parcial1,*parcial2,*parcial3;
PRECISION *nubB,*nupB,*nurB;
PRECISION **uuGlobalInicial;
PRECISION **HGlobalInicial;
PRECISION **FGlobalInicial;
PRECISION *perfil_instrumental;
PRECISION *spectral_psf;
int FGlobal,HGlobal,uuGlobal;

PRECISION *d_spectra,*spectra,*spectra_tmp,*output;

//Number of lambdas in the input profiles
int NLAMBDA = 0;
int PSF_ON_OFF = 0;
int CONT_POS = 0;
int CONT_POS_SHIFT = 1;
int PSF_SAMPLING_POINTS;

//Convolutions values
int n_samples	= 0;
int INPUT_PSF = 0;
int CLASSICAL_ESTIMATES = 0;
int RFS = 0;

void call_milos(const int *options,
	const int *size,
	const double *waveaxis,
	double *weight,
	const double *initial_model,
	const double *inputdata,
	const double *cavity,
	double *outputdata) {
    // size_t index;
    // for (index = 0; index < size; ++index)
    //     outputdata[index] = inputdata[index] * 2.0;

	double * wlines;
	int nwlines;
	double *lambda;
	double *init_lambda;
	int nlambda;
	PRECISION *spectro;
	// int ny,i,j;
	Init_Model initModel;
	int err;
	double chisqrf;
	int iter;
	double slight;
	double toplim;
	int miter;
	//PRECISION weight[4]={1.,10.,10.,4.}; 20 Oct 2022
	// PRECISION weight[4]={1.,12.,12.,10.};
	int nweight;

	int fix[]={1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,0.};  //Parametros invertidos
	//----------------------------------------------

	double sigma[NPARMS];
	double vsig;
	double filter;
	double ilambda;
	double noise;
	double *pol;
	double getshi;

	double dat[7]={CUANTIC_NWL,CUANTIC_SLOI,CUANTIC_LLOI,CUANTIC_JLOI,CUANTIC_SUPI,CUANTIC_LUPI,CUANTIC_JUPI};

	int Max_iter;
	// added 20 Oct 2022 (now initial model is an input from pymilos)
    PRECISION INITIAL_MODEL_B = initial_model[0];
    PRECISION INITIAL_MODEL_GM = initial_model[1];
    PRECISION INITIAL_MODEL_AZI = initial_model[2];
    PRECISION INITIAL_MODEL_ETHA0 = initial_model[3];
    PRECISION INITIAL_MODEL_LAMBDADOPP = initial_model[4];  //en A
    PRECISION INITIAL_MODEL_AA = initial_model[5];
    PRECISION INITIAL_MODEL_VLOS = initial_model[6]; // Km/s
    PRECISION INITIAL_MODEL_S0 = initial_model[7];
    PRECISION INITIAL_MODEL_S1 = initial_model[8];

	NLAMBDA = options[0];
	Max_iter = options[1];
	CLASSICAL_ESTIMATES = options[2];
	RFS = options[3];

	if (options[4] != 0){
		PSF_ON_OFF = 1;
		PRECISION FWHM = options[4]/1000.;
		PRECISION DELTA = options[5]/1000.;
		PSF_SAMPLING_POINTS = options[6];
		if(PSF_SAMPLING_POINTS % 2 == 0) PSF_SAMPLING_POINTS = PSF_SAMPLING_POINTS + 1; //we need to force to be odd (impar) number

		spectral_psf = calloc(PSF_SAMPLING_POINTS,sizeof(PRECISION)); // allocate memory (zeros)

		gaussian_psf(FWHM,PSF_SAMPLING_POINTS,DELTA,spectral_psf);

	}

	nlambda=NLAMBDA;

	cuantic=create_cuantic(dat);

	// Inicializar_Puntero_Calculos_Compartidos();

	toplim=1e-18;

	CC=PI/180.0;
	CC_2=CC*2;

	filter=0;
	getshi=0;
	nweight=4;

	nwlines=1;
	wlines=(double*) calloc(2,sizeof(double));
	wlines[0]=1;
	wlines[1]= CENTRAL_WL;

	vsig=NOISE_SIGMA; //original 0.001
	sigma[0]=vsig;
	sigma[1]=vsig;
	sigma[2]=vsig;
	sigma[3]=vsig;
	pol=NULL;

	noise=NOISE_SIGMA;
	ilambda=ILAMBDA;
	iter=0;
	miter=Max_iter;

	lambda=calloc(nlambda,sizeof(double));
	init_lambda=calloc(nlambda,sizeof(double));
	spectro=calloc(nlambda*4,sizeof(PRECISION));

	// double lin;
	// int rfscanf;
	// int totalIter=0;

	ReservarMemoriaSinteisisDerivadas(nlambda);

	//initializing weights
	PRECISION *w,*sig;
	weights_init(nlambda,sigma,weight,nweight,&w,&sig,noise);

	// int indaux;
	// indaux=0;

	int contador, neje, nsub, np, cnt, cnt_model, landa_loop;
	double iin,qin,uin,vin;
	contador = 0;
	np = 0;

	for(landa_loop=0;landa_loop<NLAMBDA;landa_loop++){
		init_lambda[landa_loop] = waveaxis[landa_loop];
		// printf("value is = %f\n",lambda[landa_loop]);
	}

    PRECISION d1, d2;
    d1 = (PRECISION)init_lambda[0] - (PRECISION)init_lambda[1];
    d2 = (PRECISION)init_lambda[nlambda-2] - (PRECISION)init_lambda[nlambda-1];
    if (fabs(d1)<fabs(d2)){
        CONT_POS = nlambda -1;
		CONT_POS_SHIFT = 0;
    }

	// If RFS = 0 meaning you are using inversion mode
	if(!RFS){
		cnt_model = 0;

		// find number of profiles to invert
		int n_profiles = (*size)/4/NLAMBDA;
		printf("profiles to invert = %d\n",n_profiles);

		// initialize counter for n_profiles
		register int profile_idx = 0;

		do{
			nsub=0;
			neje=0;
			np = profile_idx*NLAMBDA*4;
			while (nsub<NLAMBDA){
				spectro[nsub] = inputdata[nsub + np];
				spectro[nsub+NLAMBDA] = inputdata[nsub + NLAMBDA + np];
				spectro[nsub+NLAMBDA*2] = inputdata[nsub + NLAMBDA*2 + np];
				spectro[nsub+NLAMBDA*3] = inputdata[nsub + NLAMBDA*3 + np];
				// printf("Spectra is = %.10e\n",spectro[nsub]);
				// printf("Spectra is = %.10e\n",spectro[nsub+NLAMBDA]);
				// printf("Spectra is = %.10e\n",spectro[nsub+NLAMBDA*2]);
				// printf("Spectra is = %.10e\n",spectro[nsub+NLAMBDA*3]);
				nsub++;
				neje++;
			}

			//Initial Model
			initModel.eta0 = INITIAL_MODEL_ETHA0;
			initModel.B = INITIAL_MODEL_B; //200 700
			initModel.gm = INITIAL_MODEL_GM;
			initModel.az = INITIAL_MODEL_AZI;
			initModel.vlos = INITIAL_MODEL_VLOS; //km/s 0
			initModel.mac = 0.0;
			initModel.dopp = INITIAL_MODEL_LAMBDADOPP;
			initModel.aa = INITIAL_MODEL_AA;
			initModel.alfa = 1;							//0.38; //stray light factor
			initModel.S0 = INITIAL_MODEL_S0;
			initModel.S1 = INITIAL_MODEL_S1;

			if(cavity[profile_idx]!=0){
				for(landa_loop=0;landa_loop<NLAMBDA;landa_loop++){
					lambda[landa_loop] = init_lambda[landa_loop] - cavity[profile_idx];
				}
			}
			else{
				for(landa_loop=0;landa_loop<NLAMBDA;landa_loop++){
					lambda[landa_loop] = init_lambda[landa_loop];
				}
			}

			if(CLASSICAL_ESTIMATES){

				estimacionesClasicas(wlines[1],lambda,nlambda,spectro,&initModel);

				//Se comprueba si el resultado fue "nan" en las CE
				if(isnan(initModel.B))
					initModel.B = 1;
				if(isnan(initModel.vlos))
					initModel.vlos = 1e-3;
				if(isnan(initModel.gm))
					initModel.gm=1;
				if(isnan(initModel.az))
					initModel.az = 1;
			}

			//inversion
			if(CLASSICAL_ESTIMATES!=2 ){

				//Se introduce en S0 el valor de Blos si solo se calculan estimaciones clásicas
				//Aqui se anula esa asignación porque se va a realizar la inversion RTE completa
				initModel.S0 = INITIAL_MODEL_S0;

				// printf("\n\n Input\n");
				// printf("%f\n",initModel.S0);
				// printf("%f\n",initModel.S1);
				// printf("%f\n",initModel.B);
				// printf("%f\n",initModel.gm);
				// printf("%f\n",initModel.az);
				// printf("%f\n",initModel.vlos);

				lm_mils(cuantic,wlines,nwlines,lambda, nlambda,spectro,nlambda,&initModel,spectra,err,&chisqrf,&iter,slight,toplim,miter,
					weight,nweight,fix,sig,filter,ilambda,noise,pol,getshi,0);
			}

			// printf("\n\n Output\n");
			// printf("%f\n",initModel.S0);
			// printf("%f\n",initModel.S1);
			// printf("%f\n",initModel.B);
			// printf("%f\n",initModel.gm);
			// printf("%f\n",initModel.az);
			// printf("%f\n",initModel.vlos);
			// printf("chi2 is = %f\n",chisqrf);

			// [contador;iter;B;GM;AZI;etha0;lambdadopp;aa;vlos;S0;S1;final_chisqr];
			outputdata[cnt_model] = profile_idx;
			outputdata[cnt_model+1] = iter;
			outputdata[cnt_model+2] = initModel.B;
			outputdata[cnt_model+3] = initModel.gm;
			outputdata[cnt_model+4] = initModel.az;
			outputdata[cnt_model+5] = initModel.eta0;
			outputdata[cnt_model+6] = initModel.dopp;
			outputdata[cnt_model+7] = initModel.aa;
			outputdata[cnt_model+8] = initModel.vlos;
			outputdata[cnt_model+9] = initModel.S0;
			outputdata[cnt_model+10] = initModel.S1;
			outputdata[cnt_model+11] = chisqrf;
			cnt_model = cnt_model + 12;
			profile_idx++;

			// for(landa_loop=0;landa_loop<12;landa_loop++){
			// 	printf("value is = %f\n",outputdata[landa_loop]);
			// }

		} while(profile_idx < n_profiles);
	}
	else{   // If RFS != 0 meaning you are using synthesis and RFS mode

		// find number of profiles to synthesize
		int n_profiles = (*size)/9;
		printf("profiles to synthesize = %d\n",n_profiles);

		// initialize counter for n_profiles
		register int p_idx = 0;
		// initialize counter for wavelength
		register int w_idx = 0;
		// initialize counter for derivatives (0-9)
		register int d_idx = 0;

		// these two are, so far, constant always
		initModel.mac  = 0.0;
		initModel.alfa = 1;

		do{
			// jump from 9 to 9 model parameters
			np = p_idx*9;

			initModel.B    = inputdata[np];
			initModel.gm   = inputdata[np + 1];
			initModel.az   = inputdata[np + 2];
			initModel.eta0 = inputdata[np + 3];
			initModel.dopp = inputdata[np + 4];
			initModel.aa   = inputdata[np + 5];
			initModel.vlos = inputdata[np + 6];
			initModel.S0   = inputdata[np + 7];
			initModel.S1   = inputdata[np + 8];

			if(cavity[p_idx]!=0){
				for(landa_loop=0;landa_loop<NLAMBDA;landa_loop++){
					lambda[landa_loop] = init_lambda[landa_loop] - cavity[p_idx];
				}
			}
			else{
				for(landa_loop=0;landa_loop<NLAMBDA;landa_loop++){
					lambda[landa_loop] = init_lambda[landa_loop];
				}
			}

			// synthesize the profiles
			mil_sinrf(cuantic,&initModel,wlines,nwlines,lambda,nlambda,spectra,AH,slight,0,filter);

			// convolve the profiles
			spectral_synthesis_convolution();

			// this will be always an output
			for(w_idx=NLAMBDA;w_idx--;){
				outputdata[w_idx +             4*NLAMBDA*p_idx] = spectra[w_idx];
				outputdata[w_idx + NLAMBDA   + 4*NLAMBDA*p_idx] = spectra[w_idx + NLAMBDA];
				outputdata[w_idx + NLAMBDA*2 + 4*NLAMBDA*p_idx] = spectra[w_idx + NLAMBDA*2];
				outputdata[w_idx + NLAMBDA*3 + 4*NLAMBDA*p_idx] = spectra[w_idx + NLAMBDA*3];
			}

			// if the RFS are activated, calculate them
			if(RFS==2){
				// calculate the RFS
				me_der(cuantic,&initModel,wlines,nwlines,lambda,nlambda,d_spectra,AH,slight,0,filter);

				// convolve the RFS
				response_functions_convolution();

				// for(w_idx=NLAMBDA;w_idx--;){
			 	// 	for(d_idx = 0;d_idx<NTERMS;d_idx++){
				// 		outputdata[w_idx + NLAMBDA*d_idx +                    4*NLAMBDA*(p_idx+1)*n_profiles ] = d_spectra[w_idx + NLAMBDA*d_idx  ];
				// 		outputdata[w_idx + NLAMBDA*d_idx + NLAMBDA*NTERMS   + 4*NLAMBDA*(p_idx+1)*n_profiles ] = d_spectra[w_idx + NLAMBDA*d_idx + NLAMBDA * NTERMS    ];
				// 		outputdata[w_idx + NLAMBDA*d_idx + NLAMBDA*NTERMS*2 + 4*NLAMBDA*(p_idx+1)*n_profiles ] = d_spectra[w_idx + NLAMBDA*d_idx + NLAMBDA * NTERMS * 2];
				// 		outputdata[w_idx + NLAMBDA*d_idx + NLAMBDA*NTERMS*3 + 4*NLAMBDA*(p_idx+1)*n_profiles ] = d_spectra[w_idx + NLAMBDA*d_idx + NLAMBDA * NTERMS * 3];
				// 	}
				// }
			}

			p_idx++;  //next profile

		} while(p_idx < n_profiles);
	}

	free(spectro);
	free(lambda);
	free(init_lambda);
	free(cuantic);
	free(wlines);

	LiberarMemoriaSinteisisDerivadas();
	Liberar_Puntero_Calculos_Compartidos();

	if(PSF_ON_OFF) free(spectral_psf);

}

int main(int argc,char **argv){

	double * wlines;
	int nwlines;
	double *lambda;
	int nlambda;
	PRECISION *spectro;
	int ny,i,j;
	Init_Model initModel;
	int err;
	double chisqrf;
	int iter;
	double slight;
	double toplim;
	int miter;
	// PRECISION weight[4]={1.,10.,10.,4.};
	PRECISION weight[4]={1.,4.,5.4,4.1}; // DC update for HRT (with ISS off)
	int nweight;

	clock_t t_ini, t_fin;
	double secs, total_secs;

	// double *chisqrf_array;
	// chisqrf_array =(double*) calloc(883*894*4,sizeof(double));

	// CONFIGURACION DE PARAMETROS A INVERTIR
	//INIT_MODEL=[eta0,magnet,vlos,landadopp,aa,gamma,azi,B1,B2,macro,alfa]
	int fix[]={1.,1.,1.,1.,1.,1.,1.,1.,1.,0.,0.};  //Parametros invertidos
	//----------------------------------------------

	double sigma[NPARMS];
	double vsig;
	double filter;
	double ilambda;
	double noise;
	double *pol;
	double getshi;

	double dat[7]={CUANTIC_NWL,CUANTIC_SLOI,CUANTIC_LLOI,CUANTIC_JLOI,CUANTIC_SUPI,CUANTIC_LUPI,CUANTIC_JUPI};

	char *nombre,*input_iter;
	int Max_iter;

	// Since initial model was removed from defines.h, we need to add it here for CMILOS
	// 20 Oct 2022
    PRECISION INITIAL_MODEL_B = 400;
    PRECISION INITIAL_MODEL_GM = 30;
    PRECISION INITIAL_MODEL_AZI = 120;
    PRECISION INITIAL_MODEL_ETHA0 = 1; // DC change for HRT
    PRECISION INITIAL_MODEL_LAMBDADOPP = 0.05;  //en A // DC change for HRT
    PRECISION INITIAL_MODEL_AA = 1.5; // DC change for HRT
    PRECISION INITIAL_MODEL_VLOS = 0.01; // Km/s
    PRECISION INITIAL_MODEL_S0 = 0.22; // DC change for HRT
    PRECISION INITIAL_MODEL_S1 = 0.85;

	//milos NLAMBDA MAX_ITER CLASSICAL_ESTIMATES RFS [FWHM DELTA NPOINTS] profiles.txt

	if(argc!=6 && argc != 7 && argc !=9){
		printf("milos: Error in number of input parameters: %d .\n Try: milos NLAMBDA MAX_ITER CLASSICAL_ESTIMATES RFS [FWHM(in A) DELTA(in A) NPOINTS] perfil.txt\n",argc);
		printf("Or else: milos NLAMBDA MAX_ITER CLASSICAL_ESTIMATES RFS [DELTA(in A)] profiles.txt\n  --> for using internally stored PSF\n");
		printf("Note : CLASSICAL_ESTIMATES=> 0: Disabled, 1: Enabled, 2: Only Classical Estimates.\n");
		printf("RFS : 0: Disabled     1: Synthesis      2: Synthesis and Response Functions\n");
		printf("Note when RFS>0: profiles.txt is considered as models.txt. \n");

		return -1;
	}

	NLAMBDA = atoi(argv[1]);

	input_iter = argv[2];
	Max_iter = atoi(input_iter);
	CLASSICAL_ESTIMATES = atoi(argv[3]);
	RFS = atoi(argv[4]);

	if(CLASSICAL_ESTIMATES!=0 && CLASSICAL_ESTIMATES != 1 && CLASSICAL_ESTIMATES != 2){
		printf("milos: Error in CLASSICAL_ESTIMATES parameter. [0,1,2] are valid values. Not accepted: %d\n",CLASSICAL_ESTIMATES);
		return -1;
	}

	if(RFS != 0 && RFS != 1 && RFS != 2){
		printf("milos: Error in RFS parameter. [0,1,2] are valid values. Not accepted: %d\n",RFS);
		return -1;
	}

	if(argc ==6){ //if no filter declared
		nombre = argv[5]; //input file name
	}
	else{
		if (atoi(argv[5]) != 0){
			PSF_SAMPLING_POINTS = 0;
			PRECISION FWHM = 0;
			PRECISION DELTA = 0;
			PSF_ON_OFF = 1;
			if(argc ==7){
				INPUT_PSF = 1;
				DELTA = atof(argv[5]);
				nombre = argv[6];
				FWHM = 0.035;
			}
			else{
				INPUT_PSF = 0;
				// FWHM = atof(argv[5])/1000.;
				// DELTA = atof(argv[6])/1000.;
				sscanf(argv[5], "%lf", &FWHM);
				sscanf(argv[6], "%lf", &DELTA);
				FWHM = FWHM / 1000.0 ;
				DELTA = DELTA / 1000.0 ;
				PSF_SAMPLING_POINTS = atoi(argv[7]);
				nombre = argv[8];
			}
    	if(PSF_SAMPLING_POINTS % 2 == 0) PSF_SAMPLING_POINTS = PSF_SAMPLING_POINTS + 1; //we need to force to be odd (impar) number
		spectral_psf = calloc(PSF_SAMPLING_POINTS,sizeof(PRECISION));
		gaussian_psf(FWHM,PSF_SAMPLING_POINTS,DELTA,spectral_psf);
		}
		nombre = argv[8];
	}

	nlambda=NLAMBDA;

	cuantic=create_cuantic(dat);
	Inicializar_Puntero_Calculos_Compartidos();

	toplim=1e-18;

	CC=PI/180.0;
	CC_2=CC*2;

	filter=0;
	getshi=0;
	nweight=4;

	nwlines=1;
	wlines=(double*) calloc(2,sizeof(double));
	wlines[0]=1;
	wlines[1]= CENTRAL_WL;

	vsig=NOISE_SIGMA; //original 0.001
	sigma[0]=vsig;
	sigma[1]=vsig;
	sigma[2]=vsig;
	sigma[3]=vsig;
	pol=NULL;

	noise=NOISE_SIGMA;
	ilambda=ILAMBDA;
	iter=0;
	miter=Max_iter;


	lambda=calloc(nlambda,sizeof(double));
	spectro=calloc(nlambda*4,sizeof(PRECISION));

	FILE *fichero;

	fichero= fopen(nombre,"r");
	if(fichero==NULL){
		printf("Error de apertura, es posible que el fichero no exista.\n");
		printf("Milos: Error de lectura del fichero. ++++++++++++++++++\n");
		return 1;
	}

	char * buf;
	buf=calloc(strlen(nombre)+15+19,sizeof(char));
	buf = strcat(buf,nombre);
	buf = strcat(buf,"_CL_ESTIMATES");

	int neje;
	double lin;
	double iin,qin,uin,vin;
	int rfscanf;
	int contador;

	int totalIter=0;

	contador=0;

	ReservarMemoriaSinteisisDerivadas(nlambda);

	//initializing weights
	PRECISION *w,*sig;
	weights_init(nlambda,sigma,weight,nweight,&w,&sig,noise);

	int nsub,indaux;
	indaux=0;

    PRECISION d1, d2;

	if(!RFS){ // SI RFS ==0
		do{
			neje=0;
			nsub=0;
			 while (neje<NLAMBDA && (rfscanf=fscanf(fichero,"%lf %lf %lf %lf %lf",&lin,&iin,&qin,&uin,&vin))!= EOF){

				lambda[nsub]=lin;
				spectro[nsub]=iin;
				spectro[nsub+NLAMBDA]=qin;
				spectro[nsub+NLAMBDA*2]=uin;
				spectro[nsub+NLAMBDA*3]=vin;
				nsub++;
				neje++;
			}
		    d1 = (PRECISION)lambda[0] - (PRECISION)lambda[1];
		    d2 = (PRECISION)lambda[nlambda-2] - (PRECISION)lambda[nlambda-1];
		    if (fabs(d1)<fabs(d2)){
		        CONT_POS = nlambda -1;
				CONT_POS_SHIFT = 0;
		    }
			if(rfscanf!=EOF ){  //   && contador==8

				//Initial Model
				initModel.eta0 = INITIAL_MODEL_ETHA0;
				initModel.B = INITIAL_MODEL_B; //200 700
				initModel.gm = INITIAL_MODEL_GM;
				initModel.az = INITIAL_MODEL_AZI;
				initModel.vlos = INITIAL_MODEL_VLOS; //km/s 0
				initModel.mac = 0.0;
				initModel.dopp = INITIAL_MODEL_LAMBDADOPP;
				initModel.aa = INITIAL_MODEL_AA;
				initModel.alfa = 1;							//0.38; //stray light factor
				initModel.S0 = INITIAL_MODEL_S0;
				initModel.S1 = INITIAL_MODEL_S1;


				if(CLASSICAL_ESTIMATES && !RFS){

					t_ini = clock();
					estimacionesClasicas(wlines[1],lambda,nlambda,spectro,&initModel);
					t_fin = clock();

					//Se comprueba si el resultado fue "nan" en las CE
					if(isnan(initModel.B))
						initModel.B = 1;
					if(isnan(initModel.vlos))
						initModel.vlos = 1e-3;
					if(isnan(initModel.gm))
						initModel.gm=1;
					if(isnan(initModel.az))
						initModel.az = 1;
				}

				//inversion
				if(CLASSICAL_ESTIMATES!=2 ){

					//Se introduce en S0 el valor de Blos si solo se calculan estimaciones clásicas
					//Aqui se anula esa asignación porque se va a realizar la inversion RTE completa
					initModel.S0 = INITIAL_MODEL_S0;

					lm_mils(cuantic,wlines,nwlines,lambda, nlambda,spectro,nlambda,&initModel,spectra,err,&chisqrf,&iter,slight,toplim,miter,
						weight,nweight,fix,sig,filter,ilambda,noise,pol,getshi,0);
				}


				secs = (double)(t_fin - t_ini) / CLOCKS_PER_SEC;
				//printf("\n\n%.16g milisegundos\n", secs * 1000.0);

				total_secs += secs;
				totalIter += iter;

				// [contador;iter;B;GM;AZI;etha0;lambdadopp;aa;vlos;S0;S1;final_chisqr];
				printf("%d\n",contador);
				printf("%d\n",iter);
				printf("%f\n",initModel.B);
				printf("%f\n",initModel.gm);
				printf("%f\n",initModel.az);
				printf("%f \n",initModel.eta0);
				printf("%f\n",initModel.dopp);
				printf("%f\n",initModel.aa);
				printf("%f\n",initModel.vlos); //km/s
				//printf("alfa \t:%f\n",initModel.alfa); //stay light factor
				printf("%f\n",initModel.S0);
				printf("%f\n",initModel.S1);
				printf("%.10e\n",chisqrf);

				contador++;

			}

		}while(rfscanf!=EOF ); //&& contador<10000
	}
	else{   //when RFS is activated

		// lambda[0] =6.17320100e+003;
		// lambda[1] =6.1732710e+003;
		// lambda[2] =6.17334130e+003;
		// lambda[3] =6.17341110e+003;
		// lambda[4] =6.17348100e+003;
		// lambda[5] =6.17376100e+003;
		// get wavelength axis // DOS 28 March 2023
		neje=0;
		nsub=0;

		while (neje<NLAMBDA && (rfscanf=fscanf(fichero,"%lf",&lin))!= EOF){
			lambda[nsub]=lin;
			// printf("%le\n",lambda[nsub]);
			nsub++;
			neje++;
		}
	    d1 = (PRECISION)lambda[0] - (PRECISION)lambda[1];
	    d2 = (PRECISION)lambda[nlambda-2] - (PRECISION)lambda[nlambda-1];
	    if (fabs(d1)<fabs(d2)){
	        CONT_POS = nlambda -1;
			CONT_POS_SHIFT = 0;
	    }

		// int kkk;
		// printf("heck\n");
		// for(kkk=0;kkk<NLAMBDA;kkk++){
		// 	printf("%lf \n",lambda[kkk]);
		// }
		// printf("heck\n");
		do{
			int contador,iter;
			double chisqr;
			int NMODEL=10; //Numero de parametros del modelo


			//num,iter,B,GM,AZ,ETA0,dopp,aa,vlos,S0,S1,chisqr,
			if((rfscanf=fscanf(fichero,"%d",&contador))!= EOF){
				//rfscanf=fscanf(fichero,"%d",&iter);      // DOS 28 March 2023
				//rfscanf=fscanf(fichero,"%d",&iter);     // DOS 28 March 2023
				rfscanf=fscanf(fichero,"%lf",&initModel.B);
				rfscanf=fscanf(fichero,"%lf",&initModel.gm);
				rfscanf=fscanf(fichero,"%lf",&initModel.az);
				rfscanf=fscanf(fichero,"%lf",&initModel.eta0);
				rfscanf=fscanf(fichero,"%lf",&initModel.dopp);
				rfscanf=fscanf(fichero,"%lf",&initModel.aa);
				rfscanf=fscanf(fichero,"%lf",&initModel.vlos);
				rfscanf=fscanf(fichero,"%lf",&initModel.S0);
				rfscanf=fscanf(fichero,"%lf",&initModel.S1);
				//rfscanf=fscanf(fichero,"%le",&chisqr);

				// printf("%d\n",contador);
				// printf("%f\n",initModel.B);
				// printf("%f\n",initModel.gm);
				// printf("%f\n",initModel.az);
				// printf("%f \n",initModel.eta0);
				// printf("%f\n",initModel.dopp);
				// printf("%f\n",initModel.aa);
				// printf("%f\n",initModel.vlos); //km/s
				// //printf("alfa \t:%f\n",initModel.alfa); //stay light factor
				// printf("%f\n",initModel.S0);
				// printf("%f\n",initModel.S1);

				// int kkk;
				// for(kkk=0;kkk<nlambda;kkk++){
				// 	printf("l-> %lf \n",lambda[kkk]);
				// }

				mil_sinrf(cuantic,&initModel,wlines,nwlines,lambda,nlambda,spectra,AH,slight,0,filter);
				// int kkk;

				// for(kkk=0;kkk<nlambda;kkk++){
				// 	printf("Before -> %lf \n",spectra[kkk]);
				// }

				spectral_synthesis_convolution();

				// for(kkk=0;kkk<nlambda;kkk++){
				// 	printf("After -> %lf \n",spectra[kkk]);
				// }

				me_der(cuantic,&initModel,wlines,nwlines,lambda,nlambda,d_spectra,AH,slight,0,filter);

				response_functions_convolution();

				int kk;
				for(kk=0;kk<NLAMBDA;kk++){
					printf("%lf %le %le %le %le \n",lambda[kk],spectra[kk],spectra[kk + NLAMBDA],spectra[kk + NLAMBDA*2],spectra[kk + NLAMBDA*3]);
				}

				if(RFS==2){
					int number_parametros = 0;
					for(number_parametros=0;number_parametros<NTERMS;number_parametros++){
						for(kk=0;kk<NLAMBDA;kk++){
							printf("%lf %le %le %le %le \n",lambda[kk],
												d_spectra[kk + NLAMBDA * number_parametros],
												d_spectra[kk + NLAMBDA * number_parametros + NLAMBDA * NTERMS],
												d_spectra[kk + NLAMBDA * number_parametros + NLAMBDA * NTERMS*2],
												d_spectra[kk + NLAMBDA * number_parametros + NLAMBDA * NTERMS*3]);
						}
					}
				}
			}

		}while(rfscanf!=EOF );
	}

	fclose(fichero);

	//printf("\n\n TOTAL sec : %.16g segundos\n", total_secs);

	free(spectro);
	free(lambda);
	free(cuantic);
	free(wlines);

	LiberarMemoriaSinteisisDerivadas();
	Liberar_Puntero_Calculos_Compartidos();

	free(spectral_psf);

	return 0;
}


/*
 *
 * nwlineas :   numero de lineas espectrales
 * wlines :		lineas spectrales
 * lambda :		wavelength axis in angstrom
			longitud nlambda
 * spectra : IQUV por filas, longitud ny=nlambda
 */

int lm_mils(Cuantic *cuantic,double * wlines,int nwlines,double *lambda,int nlambda,PRECISION *spectro,int nspectro,
		Init_Model *initModel, PRECISION *spectra,int err,double *chisqrf, int *iterOut,
		double slight, double toplim, int miter, PRECISION * weight,int nweight, int * fix,
		PRECISION *sigma, double filter, double ilambda, double noise, double *pol,
		double getshi,int triplete)
{

	int * diag;
	int	iter;
	int i,j,In,*fixed,nfree;
	static PRECISION delta[NTERMS];
	double max[3],aux;
	int repite,pillado,nw,nsig;
	double *landa_store,flambda;
	static PRECISION beta[NTERMS],alpha[NTERMS*NTERMS];
	double chisqr,ochisqr;
	int nspectra,nd_spectra,clanda,ind;
	Init_Model model;

	iter=0;


	//nterms= 11; //numero de elementomodel->gms de initmodel
	nfree=CalculaNfree(spectro,nspectro);


	if(nfree==0){
		return -1; //'NOT ENOUGH POINTS'
	}

	flambda=ilambda;

	if(fix==NULL){
		fixed=calloc(NTERMS,sizeof(double));
		for(i=0;i<NTERMS;i++){
			fixed[i]=1;
		}
	}
	else{
		fixed=fix;
	}

	clanda=0;
	iter=0;
	repite=1;
	pillado=0;

	static PRECISION covar[NTERMS*NTERMS];
	static PRECISION betad[NTERMS];

	PRECISION chisqr_mem;
	int repite_chisqr=0;


	/**************************************************************************/
	mil_sinrf(cuantic,initModel,wlines,nwlines,lambda,nlambda,spectra,AH,slight,triplete,filter);

	//convolucionamos los perfiles IQUV (spectra)
	spectral_synthesis_convolution();

	me_der(cuantic,initModel,wlines,nwlines,lambda,nlambda,d_spectra,AH,slight,triplete,filter);

	response_functions_convolution();

	covarm(weight,sigma,nsig,spectro,nlambda,spectra,d_spectra,beta,alpha);

	for(i=0;i<NTERMS;i++)
		betad[i]=beta[i];

	for(i=0;i<NTERMS*NTERMS;i++)
		covar[i]=alpha[i];

	/**************************************************************************/

	ochisqr=fchisqr(spectra,nspectro,spectro,weight,sigma,nfree);


	model=*initModel;
	do{
		chisqr_mem=(PRECISION)ochisqr;

		for(i=0;i<NTERMS;i++){
			ind=i*(NTERMS+1);
			covar[ind]=alpha[ind]*(1.0+flambda);
		}


		mil_svd(covar,betad,delta);

		AplicaDelta(initModel,delta,fixed,&model);

		check(&model);

		mil_sinrf(cuantic,&model,wlines,nwlines,lambda,nlambda,spectra,AH,slight,triplete,filter);

		//convolucionamos los perfiles IQUV (spectra)
		spectral_synthesis_convolution();


		chisqr=fchisqr(spectra,nspectro,spectro,weight,sigma,nfree);

		/**************************************************************************/
		if(chisqr-ochisqr < 0){

			flambda=flambda/10.0;

			*initModel=model;


			// printf("iteration=%d , chisqr = %f CONVERGE	- lambda= %e \n",iter,chisqr,flambda);


			me_der(cuantic,initModel,wlines,nwlines,lambda,nlambda,d_spectra,AH,slight,triplete,filter);

			//convolucionamos las funciones respuesta ( d_spectra )
			response_functions_convolution();

			//FijaACeroDerivadasNoNecesarias(d_spectra,fixed,nlambda);

			covarm(weight,sigma,nsig,spectro,nlambda,spectra,d_spectra,beta,alpha);
			for(i=0;i<NTERMS;i++)
				betad[i]=beta[i];

			for(i=0;i<NTERMS*NTERMS;i++)
				covar[i]=alpha[i];

			ochisqr=chisqr;
		}
		else{
			flambda=flambda*10;//10;

			// printf("iteration=%d , chisqr = %f NOT CONVERGE	- lambda= %e \n",iter,ochisqr,flambda);

		}

		iter++;

/*
		printf("\n-----------------------\n");
		printf("%d\n",iter);
		printf("%f\n",initModel->B);
		printf("%f\n",initModel->gm);
		printf("%f\n",initModel->az);
		printf("%f \n",initModel->eta0);
		printf("%f\n",initModel->dopp);
		printf("%f\n",initModel->aa);
		printf("%f\n",initModel->vlos); //km/s
		//printf("alfa \t:%f\n",initModel.alfa); //stay light factor
		printf("%f\n",initModel->S0);
		printf("%f\n",initModel->S1);
		printf("%.10e\n",ochisqr);
*/

	}while(iter<=miter); // && !clanda);

	*iterOut=iter;

	*chisqrf=ochisqr;

	if(fix==NULL)
		free(fixed);


	return 1;
}

int CalculaNfree(PRECISION *spectro,int nspectro){
	int nfree,i,j;
	nfree=0;

	nfree = (nspectro*NPARMS) - NTERMS;

	return nfree;
}


/*
*
*
* Cálculo de las estimaciones clásicas.
*
*
* lambda_0 :  centro de la línea
* lambda :    vector de muestras
* nlambda :   numero de muesras
* spectro :   vector [I,Q,U,V]
* initModel:  Modelo de atmosfera a ser modificado
*
*
*
* @Author: Juan Pedro Cobos Carrascosa (IAA-CSIC)
*		   jpedro@iaa.es
* @Date:  Nov. 2011
*
*/
void estimacionesClasicas(PRECISION lambda_0,double *lambda,int nlambda, PRECISION *spectro,Init_Model *initModel){

	// Modified by Daniele Calchetti (DC) calchetti@mps.mpg.de in March 2022

	PRECISION x,y,aux,LM_lambda_plus,LM_lambda_minus,Blos,beta_B,Ic,Vlos;
	PRECISION *spectroI,*spectroQ,*spectroU,*spectroV;
	PRECISION L,m,gamma, gamma_rad,tan_gamma,maxV,minV,C,maxWh,minWh;
	register int i,j;
	int i0 = 0;
	if (CONT_POS_SHIFT == 0) i0 = 1 ;

	//Es necesario crear un lambda en FLOAT para probar como se hace en la FPGA
	PRECISION *lambda_aux;
	lambda_aux= (PRECISION*) calloc(nlambda,sizeof(PRECISION));

	spectroI=spectro;
	spectroQ=spectro+nlambda;
	spectroU=spectro+nlambda*2;
	spectroV=spectro+nlambda*3;

	// counts from i=0, nlambda-1 so assumes that the continuum is in the red
	// to correct this behaviour when the continuum is in the blue, we added cont_pos parameter and ii to the counter

    Ic= spectro[CONT_POS]; // Continuo ultimo valor de I

	for(i=0;i<nlambda-1;i++){
		lambda_aux[i] = (PRECISION)lambda[i+CONT_POS_SHIFT];// added by DC for continuum position
	}

	x=0;
	y=0;
	for(i=0;i<nlambda-1;i++){
		aux = ( Ic - (spectroI[i+CONT_POS_SHIFT]+ spectroV[i+CONT_POS_SHIFT])); // added by DC for continuum position
		x = x +  aux * (lambda_aux[i]-lambda_0);
		y = y + aux;
	}

	//Para evitar nan
	if(fabs(y)>1e-15)
		LM_lambda_plus	= x / y;
	else
		LM_lambda_plus = 0;
	// LM_lambda_plus	= x / y;

	x=0;
	y=0;
	for(i=0;i<nlambda-1;i++){
		aux = ( Ic - (spectroI[i+CONT_POS_SHIFT] - spectroV[i+CONT_POS_SHIFT]));// added by DC for continuum position
		x= x +  aux * (lambda_aux[i]-lambda_0);
		y = y + aux;
	}

	if(fabs(y)>1e-15)
		LM_lambda_minus	= x / y;
	else
		LM_lambda_minus = 0;
	// LM_lambda_minus	= x / y;

	C = (CTE4_6_13 * lambda_0 * lambda_0 * cuantic->GEFF);
	beta_B = 1 / C;

	Blos = beta_B * ((LM_lambda_plus - LM_lambda_minus)/2);
	Vlos = ( VLIGHT / (lambda_0)) * ((LM_lambda_plus + LM_lambda_minus)/2);

	//inclinacion
	x = 0;
	y = 0;
	for(i=0;i<nlambda-1;i++){
		L = fabs( sqrtf( spectroQ[i+CONT_POS_SHIFT]*spectroQ[i+CONT_POS_SHIFT] + spectroU[i+CONT_POS_SHIFT]*spectroU[i+CONT_POS_SHIFT] )); // added by DC for continuum position
		m = fabs( (4 * (lambda_aux[i]-lambda_0) * L ));// / (3*C*Blos) ); //2*3*C*Blos mod abril 2016 (en test!)

		x = x + fabs(spectroV[i+CONT_POS_SHIFT]) * m; // added by DC for continuum position
		y = y + fabs(spectroV[i+CONT_POS_SHIFT]) * fabs(spectroV[i+CONT_POS_SHIFT]); // added by DC for continuum position
	}

	y = y * fabs((3*C*Blos));

	tan_gamma = fabs(sqrtf(x/y));

	gamma_rad = atan(tan_gamma); //gamma en radianes

	gamma = gamma_rad * (180/ PI); //gamma en grados

	PRECISION gamma_out = gamma;

    if (Blos<0)
        gamma = (180)-gamma;

	//azimuth

	PRECISION tan2phi,phi;
	int muestra;

	if(nlambda==6)
		muestra = CLASSICAL_ESTIMATES_SAMPLE_REF - i0; // added by DC for continuum position
	else
		muestra = nlambda*0.75;


	tan2phi=spectroU[muestra]/spectroQ[muestra];

	phi= (atan(tan2phi)*180/PI) / 2;  //atan con paso a grados

	if(spectroU[muestra] > 0 && spectroQ[muestra] > 0 )
		phi=phi;
	else
	if (spectroU[muestra] < 0 && spectroQ[muestra] > 0 )
		phi=phi + 180;
	else
	if (spectroU[muestra] < 0 && spectroQ[muestra] < 0 )
		phi=phi + 90;
	else
	if (spectroU[muestra] > 0 && spectroQ[muestra]< 0 )
			phi=phi + 90;

	PRECISION B_aux;

	B_aux = fabs(Blos/cos(gamma_rad)) * 2; // 2 factor de corrección

	//Vlos = Vlos * 1.5;
	if(Vlos < (-20))
		Vlos= -20;
	if(Vlos >(20))
		Vlos=(20);

	initModel->B = (B_aux>4000?4000:B_aux);
	initModel->vlos=Vlos;//(Vlos*1.5);//1.5;
	initModel->gm=gamma;
	initModel->az=phi;
	initModel->S0= Blos;

	//Liberar memoria del vector de lambda auxiliar
	free(lambda_aux);

}

void FijaACeroDerivadasNoNecesarias(PRECISION * d_spectra,int *fixed,int nlambda){

	int In,j,i;
	for(In=0;In<NTERMS;In++)
		if(fixed[In]==0)
			for(j=0;j<4;j++)
				for(i=0;i<nlambda;i++)
					d_spectra[i+nlambda*In+j*nlambda*NTERMS]=0;
}

void AplicaDelta(Init_Model *model,PRECISION * delta,int * fixed,Init_Model *modelout){

	//INIT_MODEL=[eta0,magnet,vlos,landadopp,aa,gamma,azi,B1,B2,macro,alfa]

	if(fixed[0]){
		modelout->eta0=model->eta0-delta[0]; // 0
	}
	if(fixed[1]){
		if(delta[1]< -800) //300
			delta[1]=-800;
		else
			if(delta[1] >800)
				delta[1]=800;
		modelout->B=model->B-delta[1];//magnetic field
	}
	if(fixed[2]){

		 if(delta[2]>2)
			 delta[2] = 2;

		 if(delta[2]<-2)
			delta[2] = -2;

		modelout->vlos=model->vlos-delta[2];
	}

	if(fixed[3]){

		if(delta[3]>1e-2)
			delta[3] = 1e-2;
		else
			if(delta[3]<-1e-2)
				delta[3] = -1e-2;

		modelout->dopp=model->dopp-delta[3];
	}

	if(fixed[4])
		modelout->aa=model->aa-delta[4];

	if(fixed[5]){
		if(delta[5]< -15) //15
			delta[5]=-15;
		else
			if(delta[5] > 15)
				delta[5]=15;

		modelout->gm=model->gm-delta[5]; //5
	}
	if(fixed[6]){
		if(delta[6]< -15)
			delta[6]=-15;
		else
			if(delta[6] > 15)
				delta[6]= 15;

		modelout->az=model->az-delta[6];
	}
	if(fixed[7])
		modelout->S0=model->S0-delta[7];
	if(fixed[8])
		modelout->S1=model->S1-delta[8];
	if(fixed[9])
		modelout->mac=model->mac-delta[9]; //9
	if(fixed[10])
		modelout->alfa=model->alfa-delta[10];
}

/*
	Tamaño de H es 	 NTERMS x NTERMS
	Tamaño de beta es 1xNTERMS

	return en delta tam 1xNTERMS
*/

int mil_svd(PRECISION *h,PRECISION *beta,PRECISION *delta){

	double epsilon,top;
	static PRECISION v2[TAMANIO_SVD][TAMANIO_SVD],w2[TAMANIO_SVD],v[NTERMS*NTERMS],w[NTERMS];
	static PRECISION h1[NTERMS*NTERMS],h_svd[TAMANIO_SVD*TAMANIO_SVD];
	static PRECISION aux[NTERMS*NTERMS];
	int i,j;
//	static double aux2[NTERMS*NTERMS];
	static	PRECISION aux2[NTERMS];
	int aux_nf,aux_nc;
	PRECISION factor,maximo,minimo,wmax,wmin;
	int posi,posj;

	epsilon= 1e-12;
	top=1.0;

	factor=0;
	maximo=0;
	minimo=1000000000;



	if(USE_SVDCMP){ // NUMERICAL RECIPES CASE

		/**/
		for(j=0;j<NTERMS*NTERMS;j++){
			h1[j]=h[j];
		}
		svdcmp(h1,NTERMS,NTERMS,w,v);

 		static PRECISION vaux[NTERMS*NTERMS],waux[NTERMS];

		for(j=0;j<NTERMS*NTERMS;j++){
				vaux[j]=v[j];//*factor;
		}

		for(j=0;j<NTERMS;j++){
				waux[j]=w[j];//*factor;
		}

		multmatrix(beta,1,NTERMS,vaux,NTERMS,NTERMS,aux2,&aux_nf,&aux_nc);

		// for(i=0;i<NTERMS;i++){
		// 	aux2[i]= aux2[i]*((fabs(waux[i]) > epsilon) ? (1/waux[i]): 0.0);
		// }

		wmax = 0.0;
		for(i=1;i<=NTERMS;i++) if (w[i] > wmax) wmax=w[i];
		wmin = wmax*1e-24;

		for(i=0;i<NTERMS;i++){
			aux2[i]= aux2[i]*((fabs(waux[i]) > wmin) ? (1/waux[i]): 0.0);
		}

		multmatrix(vaux,NTERMS,NTERMS,aux2,NTERMS,1,delta,&aux_nf,&aux_nc);

// 		float wmax,wmin,**un,*wn,**vn,*x; //*b,,*x; int i,j;

// 		for(i=1;i<=NTERMS;i++)
// 			for(j=1;j<=NTERMS;j++)
// 				un[i][j]=h[i*(j+1)];
// 		svdcmp(un,NTERMS,NTERMS,wn,vn);
// 		wmax=0.0;
// 		for(j=1;j<=NTERMS;j++) if (wn[j] > wmax) wmax=w[j];
// 		for(j=1;j<=NTERMS;j++) if (wn[j] < wmin) wn[j]=0.0;
// 		wmin=wmax*1e-24;
// 		svbksb(un,wn,vn,NTERMS,NTERMS,beta,x);

// void svbksb(u,w,v,m,n,b,x)
// float **u,w[],**v,b[],x[];
// int m,n;
// {
// 	int jj,j,i;
// 	float s,*tmp,*vector();
// 	void free_vector();

// 	tmp=vector(1,n);
// 	for (j=1;j<=n;j++) {
// 		s=0.0;
// 		if (w[j]) {
// 			for (i=1;i<=m;i++) s += u[i][j]*b[i];
// 			s /= w[j];
// 		}
// 		tmp[j]=s;
// 	}
// 	for (j=1;j<=n;j++) {
// 		s=0.0;
// 		for (jj=1;jj<=n;jj++) s += v[j][jj]*tmp[jj];
// 		x[j]=s;
// 	}
// 	free_vector(tmp,1,n);
// }

		return 1;

	}
	else{ // CORDIC CASE

		/**/
		for(j=0;j<NTERMS*NTERMS;j++){
			h1[j]=h[j];
		}

		for(j=0;j<NTERMS*NTERMS;j++){
				if(fabs(h[j])>maximo){
					maximo=fabs(h[j]);
				}
			}

		factor=maximo;

		if(!NORMALIZATION_SVD)
			factor = 1;

		for(j=0;j<NTERMS*NTERMS;j++){
			h1[j]=h[j]/(factor );
		}


		for(i=0;i<TAMANIO_SVD-1;i++){
			for(j=0;j<TAMANIO_SVD;j++){
				if(j<NTERMS)
					h_svd[i*TAMANIO_SVD+j]=h1[i*NTERMS+j];
				else
					h_svd[i*TAMANIO_SVD+j]=0;
			}
		}

		for(j=0;j<TAMANIO_SVD;j++){
			h_svd[(TAMANIO_SVD-1)*TAMANIO_SVD+j]=0;
		}

		svdcordic(h_svd,TAMANIO_SVD,TAMANIO_SVD,w2,v2,NUM_ITER_SVD_CORDIC);

		for(i=0;i<TAMANIO_SVD-1;i++){
			for(j=0;j<TAMANIO_SVD-1;j++){
				v[i*NTERMS+j]=v2[i][j];
			}
		}

		for(j=0;j<TAMANIO_SVD-1;j++){
			w[j]=w2[j]*factor;
		}

		static PRECISION vaux[NTERMS*NTERMS],waux[NTERMS];

		for(j=0;j<NTERMS*NTERMS;j++){
				vaux[j]=v[j];//*factor;
		}

		for(j=0;j<NTERMS;j++){
				waux[j]=w[j];//*factor;
		}

		multmatrix(beta,1,NTERMS,vaux,NTERMS,NTERMS,aux2,&aux_nf,&aux_nc);

		for(i=0;i<NTERMS;i++){
			aux2[i]= aux2[i]*((fabs(waux[i]) > epsilon) ? (1/waux[i]): 0.0);
		}

		multmatrix(vaux,NTERMS,NTERMS,aux2,NTERMS,1,delta,&aux_nf,&aux_nc);

		return 1;

	}

}



void weights_init(int nlambda,double *sigma,PRECISION *weight,int nweight,PRECISION **wOut,PRECISION **sigOut,double noise)
{
	int i,j;
	PRECISION *w,*sig;


	sig=calloc(4,sizeof(PRECISION));
	if(sigma==NULL){
		for(i=0;i<4;i++)
			sig[i]=	noise* noise;
	}
	else{

		for(i=0;i<4;i++)
			sig[i]=	(*sigma);// * (*sigma);
	}

	*wOut=w;
	*sigOut=sig;

}


int check(Init_Model *model){

	double offset=0;
	double inter;

	//Inclination
	/*	if(model->gm < 0)
		model->gm = -(model->gm);
	if(model->gm > 180)
		model->gm =180-(((int)floor(model->gm) % 180)+(model->gm-floor(model->gm)));//180-((int)model->gm % 180);*/

	//Magnetic field
	if(model->B < 0){
		//model->B = 190;
		model->B = -(model->B);
		model->gm = 180.0 -(model->gm);
	}
	if(model->B > 5000)
		model->B= 5000;

	//Inclination
	if(model->gm < 0)
		model->gm = -(model->gm);
	if(model->gm > 180){
		model->gm = 360.0 - model->gm;
		// model->gm = 179; //360.0 - model->gm;
	}

	//azimuth
	if(model->az < 0)
		model->az= 180 + (model->az); //model->az= 180 + (model->az);
	if(model->az > 180){
		model->az =model->az -180.0;
		// model->az = 179.0;
	}

	//RANGOS
	//Eta0
	if(model->eta0 < 0.1)
		model->eta0=0.1;

	// if(model->eta0 >8)
			// model->eta0=8;
	if(model->eta0 >2500)  //idl 2500
			model->eta0=2500;

	//velocity
	if(model->vlos < (-20)) //20
		model->vlos= (-20);
	if(model->vlos >20)
		model->vlos=20;

	//doppler width ;Do NOT CHANGE THIS
	if(model->dopp < 0.0001)
		model->dopp = 0.0001;

	if(model->dopp > 1.6)  // idl 0.6
		model->dopp = 1.6;


	if(model->aa < 0.0001)  // idl 1e-4
		model->aa = 0.0001;
	if(model->aa > 100)            //// changed from 10.0 to 100.0  May 2022
		model->aa = 100;

	//S0
	if(model->S0 < 0.0001)
		model->S0 = 0.0001;
	if(model->S0 > 1.500)
		model->S0 = 1.500;

	//S1
	if(model->S1 < 0.0001)
		model->S1 = 0.0001;
	if(model->S1 > 2.000)
		model->S1 = 2.000;

	//macroturbulence
	if(model->mac < 0)
		model->mac = 0;
	if(model->mac > 4)
		model->mac = 4;

	//filling factor
/*	if(model->S1 < 0)
		model->S1 = 0;
	if(model->S1 > 1)
		model->S1 = 1;*/

	return 1;
}

void spectral_synthesis_convolution(){

	register int i, j;
	int nlambda = NLAMBDA;

	if(PSF_ON_OFF){   //convolution of synthetic Stokes IQUV profiles starts here

		PRECISION Ic;

		//convolucion de I
		Ic=spectra[CONT_POS];

		for(i=0;i<nlambda-1;i++)
			spectra_tmp[i]=Ic-spectra[i+CONT_POS_SHIFT];

		// direct_convolution(spectra,nlambda-1,spectral_psf,NMUESTRAS_G,1);  //no convolucionamos el ultimo valor Ic
		// printf("CONT_POS %d %d %d \n",CONT_POS, i0, ii);
		convolution(spectra_tmp, spectral_psf, NLAMBDA - 1, PSF_SAMPLING_POINTS, output);

		for(i=0;i<nlambda-1;i++)
			spectra[i+CONT_POS_SHIFT]=Ic-output[i];

		//convolucion QUV
		for(j=1;j<NPARMS;j++){
			for(i=0;i<nlambda-1;i++)
				spectra_tmp[i] = spectra[i+CONT_POS_SHIFT + NLAMBDA*j];

			// for(i=0;i<nlambda;i++)
			// 	output[i] = 0.;

			convolution(spectra_tmp, spectral_psf, NLAMBDA - 1, PSF_SAMPLING_POINTS, output);

			for(i=0;i<nlambda-1;i++)
				spectra[i+CONT_POS_SHIFT + NLAMBDA*j] = output[i];

		}

		//convolucion QUV
		// for(i=1;i<NPARMS;i++)
		// 	direct_convolution(spectra+nlambda*i,nlambda-1,spectral_psf,NMUESTRAS_G,1);  //no convolucionamos el ultimo valor

	}
}

void response_functions_convolution(){

	register int i,j,k;
	int nlambda = NLAMBDA;

	//convolucionamos las funciones respuesta ( d_spectra )
	if(PSF_ON_OFF){
		for(i=0;i<NTERMS;i++){     // loop in physical parameters
			if(i!=7){ // S0 not need convolution
				for(j=0;j<NPARMS;j++){  // Loop in stokes parameter
					if(i==8 & j==0) {// S1 Stokes I only need low cont

						PRECISION Ic;
						Ic = d_spectra[CONT_POS+i*nlambda+j*nlambda*NTERMS];
						// for(k=0;k<nlambda-1;k++)
						for(k=nlambda-1;k--;)
							spectra_tmp[k] = Ic-d_spectra[k+CONT_POS_SHIFT+i*nlambda+j*nlambda*NTERMS];

						convolution(spectra_tmp, spectral_psf, NLAMBDA - 1, PSF_SAMPLING_POINTS, output);

						for(k=0;k<nlambda-1;k++)
							d_spectra[k+CONT_POS_SHIFT+i*nlambda+j*nlambda*NTERMS] = Ic-output[k];

					}
					else {
						for(k=0;k<nlambda-1;k++)
							spectra_tmp[k] = d_spectra[k+CONT_POS_SHIFT+i*nlambda+j*nlambda*NTERMS];

						convolution(spectra_tmp, spectral_psf, NLAMBDA - 1, PSF_SAMPLING_POINTS, output);

						for(k=0;k<nlambda-1;k++)
							d_spectra[k+CONT_POS_SHIFT+i*nlambda+j*nlambda*NTERMS] = output[k];
					}
				}
			}
		}
	}
}