/*
    empty stub for cl.exe and link.exe - shows only when cl.exe and link.exe are called w/o doing anything

    Copyright (C) 2024 Alois Schloegl <alois.schloegl@gmail.com>
    This file is part of the "BioSig for python" repository
    at http://biosig.sf.net/
 */

#include <stdio.h>
int main(int argc, const char *argv[]) {
	fprintf(stderr,"# stub for cl.exe, link.exe - this shows all input arguments, but does not execute anything\n ");
	for (int k=0; k<argc; k++) {
		fprintf(stderr, "%s ",argv[k]);
	}
	fprintf(stderr,"\r\n");
	return 0;
}

