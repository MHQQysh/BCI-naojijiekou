/*

    Copyright (C) 2011,2013,2015,2022,2024,2025 Alois Schloegl <alois.schloegl@ist.ac.at>
    This file is part of the "BioSig for C/C++" repository
    (biosig4c++) at http://biosig.sf.net/

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 3
    of the License, or (at your option) any later version.

 */

#include "mex.h"
#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <biosig-dev.h>
#include <biosig.h>


#ifdef NDEBUG
#define VERBOSE_LEVEL 0 	// turn off debugging information, but its only used without NDEBUG
#else
extern int VERBOSE_LEVEL; 	// used for debugging
#endif

#ifdef tmwtypes_h
  #if (MX_API_VER<=0x07020000)
    typedef int mwSize;
  #endif
#endif

static const double EPS  = 0x1p-52;  /* about 2.220e-16 */

double getDouble(const mxArray *pm, size_t idx) {
	size_t n = mxGetNumberOfElements(pm);
	if (n == 0)   return(NAN);
	if (n <= idx) idx = n-1;

	switch (mxGetClassID(pm)) {
	case mxCHAR_CLASS:
	case mxLOGICAL_CLASS:
	case mxINT8_CLASS:
		return(*((int8_t*)mxGetData(pm) + idx));
	case mxUINT8_CLASS:
		return(*((uint8_t*)mxGetData(pm) + idx));
	case mxDOUBLE_CLASS:
		return(*((double*)mxGetData(pm) + idx));
	case mxSINGLE_CLASS:
		return(*((float*)mxGetData(pm) + idx));
	case mxINT16_CLASS:
		return(*((int16_t*)mxGetData(pm) + idx));
	case mxUINT16_CLASS:
		return(*((uint16_t*)mxGetData(pm) + idx));
	case mxINT32_CLASS:
		return(*((int32_t*)mxGetData(pm) + idx));
	case mxUINT32_CLASS:
		return(*((uint32_t*)mxGetData(pm) + idx));
	case mxINT64_CLASS:
		return(*((int64_t*)mxGetData(pm) + idx));
	case mxUINT64_CLASS:
		return(*((uint64_t*)mxGetData(pm) + idx));
/*
	case mxFUNCTION_CLASS:
	case mxUNKNOWN_CLASS:
	case mxCELL_CLASS:
	case mxSTRUCT_CLASS:
*/
	default:
		return(NAN);
	}
	return(NAN);
}

void mexFunction(
    int           nlhs,           /* number of expected outputs */
    mxArray       *plhs[],        /* array of pointers to output arguments */
    int           nrhs,           /* number of inputs */
    const mxArray *prhs[]         /* array of pointers to input arguments */
)

{
	int k;
	const mxArray	*arg;
	HDRTYPE		*hdr;
	size_t 		count;
	time_t 		T0;
	char 		*FileName;
	char 		tmpstr[128];
	int		CHAN = 0;
	double		*ChanList=NULL;
	int		NS = -1;
	char		FlagOverflowDetection = -1, FlagUCAL = -1;
	void 		*data = NULL;
	mxArray *p = NULL, *p1 = NULL, *p2 = NULL;
	uint16_t gdftyp = 0;
	int		istatus;
	double		*status;

#ifdef CHOLMOD_H
	cholmod_sparse RR,*rr=NULL;
	double dummy;
#endif

// ToDO: output single data
//	mxClassId	FlagMXclass=mxDOUBLE_CLASS;

	if (nrhs<1) {
		mexPrintf("   Usage of mexSSAVE:\n");
		mexPrintf("\tstatus=mexSSAVE(HDR,data)\n");
		mexPrintf("   Input:\n\tHDR\tHeader structure \n");
		mexPrintf("\tdata\tdata matrix, one channel per column\n");
		mexPrintf("\tHDR\theader structure\n\n");
		mexPrintf("\tstatus 0: file saves successfully\n\n");
		mexPrintf("\tstatus <>0: file could not saved\n\n");
		return;
	}

#ifndef NDEBUG
	VERBOSE_LEVEL=0;	// set debug level to minimum
#endif

	if (nlhs > 0) {
		plhs[0] = mxCreateDoubleMatrix(1,1, mxREAL);
		status = mxGetPr(plhs[0]);
		*status = 0.0;
	}
/*
 	improve checks for input arguments
*/
	/* process input arguments */
	if (mxIsNumeric(prhs[1]) &&
	    mxIsStruct( prhs[0])) {
		data      = (void*) mxGetData(prhs[1]);
		// get number of channels
		size_t NS = mxGetN (prhs[1]);	// TODO: CHECK SIZE

		// get number of events
		size_t NEvt=0;
		if ( (p = mxGetField(prhs[0], 0, "EVENT") ) != NULL ) {
			if ( (p1 = mxGetField(p, 0, "POS") ) != NULL ) {
				NEvt = mxGetNumberOfElements(p1);
			}
			if ( (p1 = mxGetField(p, 0, "TYP") ) != NULL ) {
				size_t n = mxGetNumberOfElements(p1);
				if (n>NEvt) NEvt = n;
			}
		}

		// allocate memory for header structure
		hdr       = constructHDR (NS, NEvt);
		hdr->NRec = mxGetM (prhs[1]);	// TODO: CHECK SIZE
		hdr->SPR  = 1;
		data = (biosig_data_type*) mxGetData (prhs[1]);

		switch (mxGetClassID(prhs[1])) {
		case mxDOUBLE_CLASS: gdftyp=17; break;
		case mxSINGLE_CLASS: gdftyp=16; break;
		case mxCHAR_CLASS:   gdftyp=1;  break;
		case mxINT8_CLASS:   gdftyp=1;  break;
		case mxUINT8_CLASS:  gdftyp=2;  break;
		case mxINT16_CLASS:  gdftyp=3;  break;
		case mxUINT16_CLASS: gdftyp=4;  break;
		case mxINT32_CLASS:  gdftyp=5;  break;
		case mxUINT32_CLASS: gdftyp=6;  break;
		case mxINT64_CLASS:  gdftyp=7;  break;
		case mxUINT64_CLASS: gdftyp=8;  break;

		case mxLOGICAL_CLASS:
		case mxVOID_CLASS:
		case mxFUNCTION_CLASS:
		case mxUNKNOWN_CLASS:
		case mxCELL_CLASS:
		case mxSTRUCT_CLASS:
		default:
			*status = -1.0;
			mexErrMsgTxt("mexSSAVE(HDR,data) failed datatype of data is not supported\n");
			return;
		}
	}
	else {
		*status = -2.0;
		mexErrMsgTxt("mexSSAVE(HDR,data) failed because HDR and data, are not a struct and numeric, resp.\n");
		return;
	}


	for (k = 2; k < nrhs; k++) {
		arg = prhs[k];

		if (mxIsChar(arg)) {
#ifdef DEBUG
			mexPrintf("arg[%i]=%s \n",k,mxArrayToString(prhs[k]));
#endif
			if (!strcmp(mxArrayToString(prhs[k]), "OVERFLOWDETECTION:ON"))
				FlagOverflowDetection = 1;
			else if (!strcmp(mxArrayToString(prhs[k]), "OVERFLOWDETECTION:OFF"))
				FlagOverflowDetection = 0;
			else if (!strcmp(mxArrayToString(prhs[k]), "UCAL:ON"))
				FlagUCAL = 1;
			else if (!strcmp(mxArrayToString(prhs[k]), "UCAL:OFF"))
				FlagUCAL = 0;
		}
		else {
			*status = -3.0;
			mexPrintf("%s: argument #%i is invalid.",__FILE__, k+1);
			mexErrMsgTxt("mexSOPEN fails because of unknown parameter\n");
		}
	}

	/***** SET INPUT ARGUMENTS *****/
#ifdef CHOLMOD_H
	hdr->FLAG.ROW_BASED_CHANNELS = (rr!=NULL);
#else
	hdr->FLAG.ROW_BASED_CHANNELS = 0;
#endif

	/***** CHECK INPUT HDR STUCTURE CONVERT TO libbiosig hdr *****/
	if (VERBOSE_LEVEL>7) mexPrintf("110: input arguments checked\n");

	if ( (p = mxGetField(prhs[0], 0, "FLAG") ) != NULL ) {
		if (FlagUCAL==-1)
			if ( (p1 = mxGetField(p, 0, "UCAL") ) != NULL ) {
				FlagUCAL = (fabs(getDouble(p1, 0)) < EPS);
			}
		if (FlagOverflowDetection==-1)
			if ( (p1 = mxGetField(p, 0, "OVERFLOWDETECTION") ) != NULL ) {
				FlagOverflowDetection = (fabs(getDouble(p1, 0)) < EPS);
			}
	}
	if (FlagUCAL==-1) FlagUCAL=0;
	hdr->FLAG.UCAL = FlagUCAL;
	if (FlagOverflowDetection==-1) FlagOverflowDetection=1;
	hdr->FLAG.OVERFLOWDETECTION = FlagOverflowDetection;

	if ( (p = mxGetField(prhs[0], 0, "TYPE") ) != NULL ) {
		mxGetString(p,tmpstr,sizeof(tmpstr));
		hdr->TYPE 	= GetFileTypeFromString(tmpstr);
	}
	if ( (p = mxGetField(prhs[0], 0, "VERSION") ) != NULL ) {
		switch (mxGetClassID(p)) {
		case mxCHAR_CLASS:
			mxGetString(p, tmpstr, sizeof(tmpstr));
			hdr->VERSION  = atof(tmpstr);
			break;
		case mxDOUBLE_CLASS:
			hdr->VERSION  = getDouble(p, 0);
			break;
		}
	}
	if ( (p = mxGetField(prhs[0], 0, "T0") ) != NULL ) 		hdr->T0 = (gdf_time)ldexp(getDouble(p, 0),32);
	if ( (p = mxGetField(prhs[0], 0, "tzmin") ) != NULL )
		hdr->tzmin 	= (int16_t)getDouble(p, 0);
	else {
#if __FreeBSD__ || __APPLE__ || __NetBSD__
		time_t t = gdf_time2t_time(hdr->T0);
		struct tm *tt = localtime(&t);
		hdr->tzmin    = tt->tm_gmtoff/60;
#else
		hdr->tzmin    = -timezone/60;
#endif
	}
	if ( (p = mxGetField(prhs[0], 0, "FileName") ) != NULL ) 	FileName 	= mxArrayToString(p);
	if ( (p = mxGetField(prhs[0], 0, "SampleRate") ) != NULL ) 	hdr->SampleRate = getDouble(p, 0);

#ifdef DEBUG
	mexPrintf("%s (line %d): TYPE=<%s><%s> VERSION=%f\n",__FILE__,__LINE__,tmpstr,GetFileTypeString(hdr->TYPE),hdr->VERSION);
#endif

	if ( (p = mxGetField(prhs[0], 0, "NS") ) != NULL ) {
		hdr->NS         = getDouble(p, 0);
		mexPrintf("HDR.NS=%d/%g is defined - but will not be used, size(data,2) is used instead;\n", hdr->NS, getDouble(p, 0));
	}
	p1=NULL; p2=NULL;
	if ( (p1 = mxGetField(prhs[0], 0, "SPR") ) != NULL ) {
		mexPrintf("HDR.SPR=%d/%g is defined - but will not be used, size(data,1) is used instead;\n", hdr->SPR, getDouble(p1, 0));
	}
	if ( (p2 = mxGetField(prhs[0], 0, "NRec") ) != NULL ) {
		mexPrintf("HDR.NRec=%d/%g is defined - but will not be used, size(data,1) is used instead;\n", hdr->NRec, getDouble(p2, 0));
	}

	if      ( p1 && p2) {
		hdr->SPR  = (size_t)getDouble(p1, 0);
		hdr->NRec = (size_t)getDouble(p2, 0);
	}
	else if (!p1 && p2) {
		hdr->SPR  = hdr->NRec;
		hdr->NRec = (size_t)getDouble(p2, 0);
		hdr->SPR  /= hdr->NRec;
	}
	else if ( p1 && !p2) {
		hdr->SPR  = (size_t)getDouble(p1, 0);
		hdr->NRec /= hdr->SPR;
	}
	else if (!p1 && !p2) {
		; /* use default values SPR=1, NREC = size(data,1) */
	}

	if (hdr->TYPE == SCP_ECG) {
		/* TODO: convert HDR.data to hdr->AS.rawdata including
			- rescaling (HDR.FLAG.UCAL)
			- row/column orientation
			- check whether dimensions fit [SPR*NRec,NS]
			- Check OnOff channels
			- conversion to int16
		*/
		gdftyp = 3;
	        // hdr->FLAG.ROW_BASED_CHANNELS = 1;

		mexPrintf("%s: WARNING: writing SCPECG is work in progress, has known bugs, and is not ready for production use !!!!\n",__FILE__);

#ifndef NDEBUG
		VERBOSE_LEVEL=8;	// writing SCP is not well tested - increase verbosity
#endif
		for (k = 0; k < hdr->NS; k++) {
			hdr->CHANNEL[k].GDFTYP = gdftyp;  // double
		}
	}
	else {
		if (hdr->NRec * hdr->SPR != mxGetM (prhs[1]) )
			mexPrintf("mexSSAVE: warning HDR.NRec * HDR.SPR (%i*%i = %i) does not match number of rows (%i) in data.", hdr->NRec, hdr->SPR, hdr->NRec*hdr->SPR, mxGetM(prhs[1]) );
	}

	if ( (p = mxGetField(prhs[0], 0, "LeadIdCode") ) != NULL ) {
		for (k = 0; k < hdr->NS; k++)
			hdr->CHANNEL[k].LeadIdCode = (uint8_t)getDouble(p,k);
	}
	else if ( (p = mxGetField(prhs[0], 0, "Label") ) != NULL ) {
		if ( mxIsCell(p) ) {
			for (k = 0; k < hdr->NS; k++)
				mxGetString(mxGetCell(p,k), hdr->CHANNEL[k].Label, MAX_LENGTH_LABEL+1);
		}
	}
	if ( (p = mxGetField(prhs[0], 0, "Transducer") ) != NULL ) {
		if ( mxIsCell(p) ) {
			for (k = 0; k < hdr->NS; k++)
				mxGetString(mxGetCell(p,k), hdr->CHANNEL[k].Transducer, MAX_LENGTH_LABEL+1);
		}
	}
	if ( (p = mxGetField(prhs[0], 0, "LowPass") ) != NULL ) {
		for (k = 0; k < hdr->NS; k++)
			hdr->CHANNEL[k].LowPass = (float)getDouble(p,k);
	}
	if ( (p = mxGetField(prhs[0], 0, "HighPass") ) != NULL ) {
		for (k = 0; k < hdr->NS; k++)
			hdr->CHANNEL[k].HighPass = (float)getDouble(p,k);
	}
	if ( (p = mxGetField(prhs[0], 0, "Notch") ) != NULL ) {
		for (k = 0; k < hdr->NS; k++)
			hdr->CHANNEL[k].Notch = (float)getDouble(p,k);
	}
	if ( (p = mxGetField(prhs[0], 0, "PhysMax") ) != NULL ) {
		for (k = 0; k < hdr->NS; k++)
			hdr->CHANNEL[k].PhysMax = (double)getDouble(p,k);
	}
	if ( (p = mxGetField(prhs[0], 0, "PhysMin") ) != NULL ) {
		for (k = 0; k < hdr->NS; k++)
			hdr->CHANNEL[k].PhysMin = (double)getDouble(p,k);
	}
	if ( (p = mxGetField(prhs[0], 0, "DigMax") ) != NULL ) {
		for (k = 0; k < hdr->NS; k++)
			hdr->CHANNEL[k].DigMax = (double)getDouble(p,k);
	}
	if ( (p = mxGetField(prhs[0], 0, "DigMin") ) != NULL ) {
		for (k = 0; k < hdr->NS; k++)
			hdr->CHANNEL[k].DigMin = (double)getDouble(p,k);
	}

	if ( (p = mxGetField(prhs[0], 0, "PhysDimCode") ) != NULL ) {
		for (k = 0; k < hdr->NS; k++)
			hdr->CHANNEL[k].PhysDimCode = (uint16_t)getDouble(p,k);
	}
	else if ( (p = mxGetField(prhs[0], 0, "PhysDim") ) != NULL ) {
		if ( mxIsCell(p) ) {
			for (k = 0; k < hdr->NS; k++)
				mxGetString(mxGetCell(p,k), tmpstr, sizeof(tmpstr));
				hdr->CHANNEL[k].PhysDimCode = PhysDimCode(tmpstr);
		}
	}

	if ( (p = mxGetField(prhs[0], 0, "OnOff") ) != NULL ) {
		for (k = 0; k < hdr->NS; k++)
			hdr->CHANNEL[k].OnOff = (uint8_t)getDouble(p,k);
	}
//* 	TODO: GDFTYP is derived from 'data', ignore this 
	if ( (p = mxGetField(prhs[0], 0, "GDFTYP") ) != NULL ) {
		for (k = 0; k < hdr->NS; k++)
			hdr->CHANNEL[k].GDFTYP = (uint16_t)getDouble(p,k);
	}
	if ( (p = mxGetField(prhs[0], 0, "TOffset") ) != NULL ) {
		for (k = 0; k < hdr->NS; k++)
			hdr->CHANNEL[k].TOffset = (float)getDouble(p,k);
	}
	if ( (p = mxGetField(prhs[0], 0, "Impedance") ) != NULL ) {
		for (k = 0; k < hdr->NS; k++)
			hdr->CHANNEL[k].Impedance = (float)getDouble(p,k);
	}
	if ( (p = mxGetField(prhs[0], 0, "fZ") ) != NULL ) {
		for (k = 0; k < hdr->NS; k++)
			hdr->CHANNEL[k].fZ = (float)getDouble(p,k);
	}
	if ( (p = mxGetField(prhs[0], 0, "AS") ) != NULL ) {
		if ( (p1 = mxGetField(p, 0, "SPR") ) != NULL ) {
			// define channel-based samplingRate, HDR.SampleRate*HDR.AS.SPR(channel)/HDR.SPR;
			for (k = 0; k < hdr->NS; k++)
				hdr->CHANNEL[k].SPR = (uint32_t)getDouble(p1,k);
		}
	}

	if ( (p = mxGetField(prhs[0], 0, "Patient") ) != NULL ) {
		if ( (p1 = mxGetField(p, 0, "Id") ) != NULL )
			if (mxIsChar(p1)) mxGetString(p1, hdr->Patient.Id, MAX_LENGTH_PID+1);
		if ( (p1 = mxGetField(p, 0, "Name") ) != NULL )
			if (mxIsChar(p1)) mxGetString(p1, hdr->Patient.Name, MAX_LENGTH_PID+1);
		if ( (p1 = mxGetField(p, 0, "Sex") ) != NULL ) {
			if (mxIsChar(p1)) {
				char sex = toupper(*mxGetChars(p1));
				hdr->Patient.Sex = (sex=='M') + 2*(sex=='F');
			}
			else
				hdr->Patient.Sex = (int8_t)getDouble(p1,0);
		}

		if ( (p1 = mxGetField(p, 0, "Handedness") ) != NULL )
			hdr->Patient.Handedness = (int8_t)getDouble(p1,0);
		if ( (p1 = mxGetField(p, 0, "Smoking") ) != NULL )
			hdr->Patient.Smoking = (int8_t)getDouble(p1,0);
		if ( (p1 = mxGetField(p, 0, "AlcoholAbuse") ) != NULL )
			hdr->Patient.AlcoholAbuse = (int8_t)getDouble(p1,0);
		if ( (p1 = mxGetField(p, 0, "DrugAbuse") ) != NULL )
			hdr->Patient.DrugAbuse = (int8_t)getDouble(p1,0);
		if ( (p1 = mxGetField(p, 0, "Medication") ) != NULL )
			hdr->Patient.Medication = (int8_t)getDouble(p1,0);
		if ( (p1 = mxGetField(p, 0, "Impairment") ) != NULL ) {
			if ( (p2 = mxGetField(p1, 0, "Visual") ) != NULL )
				hdr->Patient.Impairment.Visual = (int8_t)getDouble(p2,0);
			if ( (p2 = mxGetField(p1, 0, "Heart") ) != NULL )
				hdr->Patient.Impairment.Heart = (int8_t)getDouble(p2,0);
		}

		if ( (p1 = mxGetField(p, 0, "Weight") ) != NULL )
			hdr->Patient.Weight = (uint8_t)getDouble(p1,0);
		if ( (p1 = mxGetField(p, 0, "Height") ) != NULL )
			hdr->Patient.Height = (uint8_t)getDouble(p1,0);
		if ( (p1 = mxGetField(p, 0, "Birthday") ) != NULL )
			hdr->Patient.Birthday = (gdf_time)getDouble(p1,0);
	}

	if ( (p = mxGetField(prhs[0], 0, "ID") ) != NULL ) {
		if ( (p1 = mxGetField(p, 0, "Recording") ) != NULL )
			if (mxIsChar(p1)) mxGetString(p1, hdr->ID.Recording, MAX_LENGTH_RID+1);
		if ( (p1 = mxGetField(p, 0, "Technician") ) != NULL )
			if (mxIsChar(p1)) {
				char* str = mxArrayToString(p1);
				hdr->ID.Technician = strdup(str);
			}
		if ( (p1 = mxGetField(p, 0, "Hospital") ) != NULL && mxIsChar(p1) ) {
                        size_t len = mxGetN(p1)*mxGetN(p1);
                        hdr->ID.Hospital = (char*)realloc(hdr->ID.Hospital, len+1);
                        mxGetString(p1, hdr->ID.Hospital, len);
                }
		if ( (p1 = mxGetField(p, 0, "Equipment") ) != NULL )
			memcpy(&hdr->ID.Equipment,mxGetData(p1), 8);
		if ( (p1 = mxGetField(p, 0, "Manufacturer") ) != NULL ) {
			uint8_t pos = 0;
			if ( ( (p2 = mxGetField(p1, 0, "Name") ) != NULL ) &&  mxIsChar(p2)) {
					//hdr->ID.Manufacturer.Name=mxGetChars(p2);
					mxGetString(p1, hdr->ID.Manufacturer._field,MAX_LENGTH_MANUF);
					pos = strlen(hdr->ID.Manufacturer._field)+1;
				}
			else {
				hdr->ID.Manufacturer._field[pos++] = 0;
			}

			if ( ( (p2 = mxGetField(p1, 0, "Model") ) != NULL ) && mxIsChar(p2)) {
					//hdr->ID.Manufacturer.Model=mxGetChars(p2);
					mxGetString(p1, hdr->ID.Manufacturer._field + pos, MAX_LENGTH_MANUF);
					pos += strlen(hdr->ID.Manufacturer._field + pos)+1;
				}
			else {
				hdr->ID.Manufacturer._field[pos++] = 0;
			}

			if ( ( (p2 = mxGetField(p1, 0, "Version") ) != NULL ) && mxIsChar(p2)) {
					//hdr->ID.Manufacturer.Version=mxGetChars(p2);
					mxGetString(p1, hdr->ID.Manufacturer._field + pos, MAX_LENGTH_MANUF);
					pos += strlen(hdr->ID.Manufacturer._field+pos)+1;
				}
			else {
				hdr->ID.Manufacturer._field[pos++] = 0;
			}

			if ( ( (p2 = mxGetField(p1, 0, "SerialNumber") ) != NULL ) && mxIsChar(p2)) {
					//hdr->ID.Manufacturer.SerialNumber=mxGetChars(p2);
					mxGetString(p1, hdr->ID.Manufacturer._field + pos, MAX_LENGTH_MANUF);
					pos += strlen(hdr->ID.Manufacturer._field+pos)+1;
				}
			else {
				hdr->ID.Manufacturer._field[pos++] = 0;
			}
		}
	}

	if ( (p = mxGetField(prhs[0], 0, "FLAG") ) != NULL ) {
		if ( (p1 = mxGetField(p, 0, "OVERFLOWDETECTION") ) != NULL )
			hdr->FLAG.OVERFLOWDETECTION = (char)getDouble(p1,0);
		if ( (p1 = mxGetField(p, 0, "UCAL") ) != NULL )
			hdr->FLAG.UCAL = (char)getDouble(p1,0);
		if ( (p1 = mxGetField(p, 0, "ANONYMOUS") ) != NULL )
			hdr->FLAG.ANONYMOUS = (char)getDouble(p1,0);
		if ( (p1 = mxGetField(p, 0, "ROW_BASED_CHANNELS") ) != NULL )
			hdr->FLAG.ROW_BASED_CHANNELS = (char)getDouble(p1,0);
	}

	if ( (p = mxGetField(prhs[0], 0, "EVENT") ) != NULL ) {
		if ( (p1 = mxGetField(p, 0, "SampleRate") ) != NULL ) {
			hdr->EVENT.SampleRate = (double)getDouble(p1,0);
		}
		if ( (p1 = mxGetField(p, 0, "POS") ) != NULL ) {
			size_t n = mxGetNumberOfElements(p1);
			for (k = 0; k < n; k++)
				hdr->EVENT.POS[k] = (uint32_t)getDouble(p1,k) - 1;   // convert from 1-based indexing to 0-based indexing
		}
		if ( (p1 = mxGetField(p, 0, "TYP") ) != NULL ) {
			size_t n = mxGetNumberOfElements(p1);
			for (k = 0; k < n; k++)
				hdr->EVENT.TYP[k] = (uint16_t)getDouble(p1,k);
		}
		if ( (p1 = mxGetField(p, 0, "DUR") ) != NULL ) {
			size_t n = mxGetNumberOfElements(p1);
			for (k = 0; k < n; k++)
				hdr->EVENT.DUR[k] = (uint32_t)getDouble(p1,k);
		}
		if ( (p1 = mxGetField(p, 0, "CHN") ) != NULL ) {
			size_t n = mxGetNumberOfElements(p1);
			for (k = 0; k < n; k++)
				hdr->EVENT.CHN[k] = (uint16_t)getDouble(p1,k);
		}

		if ( (p1 = mxGetField(p, 0, "CodeDesc") ) != NULL ) {
			size_t n = mxGetNumberOfElements(p1);
			hdr->EVENT.LenCodeDesc = n+1;
			// get total size for storing all CodeDesc strings
			size_t memsiz = 1;
			for (k = 0; k < n; k++) {
				mxArray *p2 = mxGetCell(p1,k);
				memsiz += mxGetN(p2)+1;
			}
			/* allocate memory for
				hdr->.AS.auxBUF contains the \0-terminated CodeDesc strings,
				hdr->EVENT.CodeDesc contains the pointer to the strings
			*/
			hdr->EVENT.CodeDesc = (const char**) realloc(hdr->EVENT.CodeDesc, hdr->EVENT.LenCodeDesc*sizeof(char*));
			hdr->AS.auxBUF = (uint8_t*)realloc(hdr->AS.auxBUF, memsiz*sizeof(char));

			// first element is always the empty string.
			hdr->AS.auxBUF[0] = 0;
			hdr->EVENT.CodeDesc[0] = (char*)hdr->AS.auxBUF;
			size_t pos = 1;
			for (k = 0; k < n; k++) {
				mxArray *p2 = mxGetCell(p1,k);
				size_t buflen = mxGetN(p2)+1; //*mxGetM(p2)
				mxGetString(p2, (char*)hdr->AS.auxBUF+pos, buflen);
				hdr->EVENT.CodeDesc[k+1] = (char*)hdr->AS.auxBUF+pos;
				pos += buflen;
			}
		}
	}

	hdr = sopen(FileName, "w", hdr);
	istatus = serror2(hdr);
	if (istatus) {
		mexErrMsgTxt("mexSSAVE: sopen failed \n");
		*status = istatus;
	}

	swrite((biosig_data_type*)data, hdr->NRec, hdr);
	istatus = serror2(hdr);
	if (istatus) {
		mexErrMsgTxt("mexSSAVE: swrite failed \n");
		*status = istatus;
	}

	destructHDR(hdr);
	istatus = serror2(hdr);
	if (istatus) {
		mexErrMsgTxt("mexSSAVE: sclose failed \n");
		*status = istatus;
	}
};

