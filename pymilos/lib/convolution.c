
/*

	@autor: Juan Pedro Cobos
	@Date: 31 Marzo 2011
	@loc: IAA- CSIC

	Convolucion para el caso Sophi: convolucion central de x con h.

	direct_convolution(x,h,delta)
	x spectro
	h gaussiana -perfil instrumental- � ojo, solo con longitud impar!
	delta anchura de muestreo

	--nota: como h es simetrico no se invierte su orden

	//result=calloc(nresult,sizeof(PRECISION*));
	//free();

	_juanp
*/

void convolution(PRECISION * input, PRECISION * kernel, int input_size, int kernel_size, PRECISION * output) {
    register int i = 0;
    register int j = 0;
    register int k = 0;

    int kernel_offset = kernel_size / 2;

    for (i = 0; i < input_size; i++) {
		output[i] = 0.;
        for (j = 0; j < kernel_size; j++) {
            k = i - kernel_offset + j;
            if (k >= 0 && k < input_size) {
                output[i] += input[k] * kernel[j];
            }
        }
    }
}

void direct_convolution(PRECISION * x, int nx,PRECISION * h, int nh,PRECISION delta){

	PRECISION *x_aux;
	int nresult,nx_aux;
	int k,j;


	nx_aux = nx + nh - 1; //tama�o de toda la convolucion
	x_aux=calloc(nx_aux,sizeof(PRECISION));

	int mitad_nh = nh/2; //nh should be odd (impar)

	//rellenamos el vector auxiliar
	for(k=0;k<nx_aux;k++){
		x_aux[k]=0;
	}

	for(k=0;k<nx;k++){
		x_aux[k+mitad_nh]=x[k];
	}

	//vamos a tomar solo la convolucion central

	for(k=0;k<nx;k++){
		x[k]=0;
		for(j=0;j<nh;j++){
			x[k]+=h[j]*x_aux[j+k];
		}
		x[k]*=delta;
	}

	free(x_aux);

}

/*
	Genera una gaussiana de anchura FWHM (en A)
	centrada en 0 y muestreada por delta

	�ojo  nmuestras_G debe ser un numero impar	!
*/
PRECISION * vgauss(PRECISION fwhm,int nmuestras_G,PRECISION delta){

	PRECISION * res;
	PRECISION sigma,alfa;
	PRECISION aux,line;
	int i;

	line=0; //es cero porque la gaussina esta centrada en cero
	sigma=fwhm / (2*sqrt(2*log(2)));
	alfa= 1   /  (sigma*sqrt(2*PI));

	res=calloc(nmuestras_G,sizeof(PRECISION));


	//se generan las posiciones de muestreo
	PRECISION *pos_muestras_G;
	pos_muestras_G=calloc(nmuestras_G,sizeof(PRECISION));

	int mitad_nmuestras_G;
	mitad_nmuestras_G=nmuestras_G/2;

	for(i=0;i<nmuestras_G;i++){
		pos_muestras_G[i]=-(mitad_nmuestras_G*delta)+((i+1)*delta); // DOS added (i+1) 17-02-2023 porque el continuo de la gausiana esta en azul y deberia ser al rojo
		// printf("%f\n",pos_muestras_G[i]);

	}


	//se genera la gaussiana
	for(i=0;i<nmuestras_G;i++){
		aux=pos_muestras_G[i]-line;
		res[i]=alfa*exp(-( 	(aux*aux)/((sigma*sigma)*2) ) );
		res[i] = res[i]*delta;
	}

	//se normaliza su �rea

	float sum = 0;
	for(i=0;i<nmuestras_G;i++){
		sum+=res[i];
	}
	for(i=0;i<nmuestras_G;i++){
		res[i]=res[i]/sum;
	}


	free(pos_muestras_G);

	return res;

}

/*
	Gaussian PSF provided FWHM (in mA) and step (in mA)
	Input n_samples should be an odd (impar) number !!!
	If not, the continuum in the PSF will be in the blue side, e.g.,
	for n_samples = 6 (even :: par): -0.24, -0.16, -0.08, 0.0, 0.08, 0.16

	This program is an evolution of vgauss
*/
void gaussian_psf(PRECISION fwhm,int n_samples,PRECISION delta, PRECISION * res){

	PRECISION sigma,alfa;
	PRECISION aux;
	int i;

	sigma=fwhm / (2*sqrt(2*log(2)));
	alfa= 1   /  (sigma*sqrt(2*PI));

	//generate sampling around the gauss peak

	int mitad_nmuestras_G;
	mitad_nmuestras_G=n_samples/2 + 1;

	//psf generation
	for(i=0;i<n_samples;i++){
		aux=-(mitad_nmuestras_G*delta)+((i+1)*delta);
		res[i] = alfa*exp(-( 	(aux*aux)/((sigma*sigma)*2) ) );
		res[i] = res[i]*delta;
	}
	//normalization
	float sum = 0;
	for(i=0;i<n_samples;i++){
		sum+=res[i];
	}
	for(i=0;i<n_samples;i++){
		res[i]=res[i]/sum;
	}
}







