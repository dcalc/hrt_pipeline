#ifndef MILOS_H
#define MILOS_H

void call_milos(const int *options,
	const int *size,
	const double *waveaxis,
	double *weight,
	const double *initial_model,
	const double *inputdata,
	const double *cavity,
	double *outputdata);

#endif