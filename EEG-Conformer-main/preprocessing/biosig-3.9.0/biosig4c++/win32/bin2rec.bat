: Conversion of ASCII-Header and binary channels into REC(GDF,EDF) data format
:
:    Copyright (C) 2008,2020 Alois Schloegl <alois.schloegl@gmail.com>
:    This file is part of the BioSig at http://biosig.sourceforge.io/
echo off
echo REC2BIN is part of BioSig http://biosig.sf.net and licensed with GNU GPL v3.
echo BIN2REC converts BIN data into other biosignal data formats (default GDF)
echo usage: bin2rec source dest
echo usage: bin2rec -f=FMT source dest
echo		FMT can be BDF,BIN,CFWB,EDF,GDF,HL7aECG,SCP_ECG
save2gdf.exe -f=GDF %1 %2 %3 %4 %5 %6 %7
