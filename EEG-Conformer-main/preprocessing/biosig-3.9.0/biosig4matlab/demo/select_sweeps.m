function [data,HDR] = select_sweeps(outfile, infile1, swlist1)
% SELECT_SWEEPS is a program to combine and select sweeps from 1 or more recordings
%
% Usage:
%    [data,HDR] = select_sweeps(outfile, infile1, swlist1)
% 	select sweeps
%    [data,HDR] = select_sweeps(outfile, {infile1, swlist1, infile2, swlist2,..});
%       combine sweeps from multiple files
%
%    outfile: 	output filename
%    infileX:	input filename(s)
%    swlistX: 	list of sweep numbers (1-indexed)
%
% Copyright (C) 2024,2025 Alois Schlögl, ISTA
%
%    BioSig is free software: you can redistribute it and/or modify
%    it under the terms of the GNU General Public License as published by
%    the Free Software Foundation, either version 3 of the License, or
%    (at your option) any later version.
%
%    BioSig is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%    GNU General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with BioSig.  If not, see <http://www.gnu.org/licenses/>.


if exist('mexSLOAD','file')==3,
	;
elseif exist('OCTAVE_VERSION','builtin')
	pkg load biosig
else
	%% addpath to biosig
end

% valid input arguments
if (nargin==2) && iscell(infile1) && size(infile1,2)==2,
	numInFiles=size(infile1,1);
	inFileList=infile1(:,1);
	ListOfsweeplists=infile1(:,2);

elseif (nargin==3) && ischar(infile1) && isvector(swlist1)
	numInFiles=1;
	inFileList={infile1};
	ListOfsweeplists={swlist1};
else
	error('invalid input')
end


outdata=[];
numsweeps=0;
for nf=1:numInFiles,
	inFile=inFileList{nf};
	swlist=ListOfsweeplists{nf};

	if ~exist(inFile,'file')
		fprintf(2,'Warning: file %s not found (ignored)\n', inFile);
		continue
	end

	[data,HDR]=mexSLOAD(inFile);
	six = find(HDR.EVENT.TYP==hex2dec('7ffe'));
	[ss,ssix] = sort(HDR.EVENT.POS(six));
	selpos = sort([1;ss;size(data,1)+1]);

	if numsweeps==0,
		HDR0 = HDR;
		HDR0.EVENT=[];
		HDR0.EVENT.POS=[];
		HDR0.EVENT.TYP=[];
	else
		% sanity checks
		if (HDR0.NS~=HDR.NS)
			fprintf(2,'Warning: number of channels do not match (%d!=%d) - File %s is skipped \n', HDR0.NS, HDR.NS, inFile);
			continue;
		end
		if (HDR0.SampleRate~=HDR.SampleRate)
			fprintf(2,'Warning: samplerates differ (%g!=%g) - File %s is skipped \n', HDR0.SampleRate, HDR.SampleRate, inFile);
			continue;
		end
		if ~all(HDR0.PhysDimCode==HDR.PhysDimCode)
			HDR0.PhysDimCode,HDR.PhysDimCode,
			fprintf(2,'Warning: physical units no not match - File %s is skipped \n', inFile);
			continue;
		end
	end;
	if any(swlist > (length(selpos)-1))
		swlist
		fprintf(2,'Warning: swlist has entries larger the sweep number (%d) of file %s (ignored)\n', length(selpos)-1, inFile);
		continue;
	end
	for sw=1:length(swlist),
		numsweeps=numsweeps+1;
		if numsweeps>1,
			HDR0.EVENT.POS(end+1,1)=size(outdata,1)+1;
			HDR0.EVENT.TYP(end+1,1)=hex2dec('7ffe');
		end
		ix = swlist(sw);
		outdata = [outdata; data(selpos(ix):selpos(ix+1)-1,:)];
	end
end


%%% write output file %%%
HDR0.TYPE='GDF';
HDR0.VERSION=3.0;
HDR0.FileName = outfile;
HDR0.NRec=size(outdata,1);
HDR0.SPR=1;
HDR.AS.SPR(:)=HDR.SPR;

mexSSAVE(HDR0,outdata);

