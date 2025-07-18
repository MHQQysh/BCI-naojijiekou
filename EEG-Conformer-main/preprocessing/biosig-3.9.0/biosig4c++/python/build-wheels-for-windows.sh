####### Build biosig wheels for windows 64bit ##################################
###
###  Copyright 2024 Alois Schloegl <alois.schloegl@gmail.com>
###  This file is part of the "BioSig for C/C++" repository
###  (biosig4c++) at http://biosig.sf.net/
###
################################################################################

# WINE=~/mywine-ea/bin/wine
if [ -z "$WINE" ]; then
	export WINE="wine"
fi	
if [ -z "$WINEPREFIX" ]; then
	export WINEPREFIX=${HOME}/.wine
fi	

if [ -z "$MXEBASE" ]; then
	### needs to point to local clone of https://github.com/schloegl/mxe 
	export MXEBASE=${HOME}/src/mxe.schloegl
fi

## make sure mingw-gcc, libbiosig.dll, libbiosig.dll.a are available 
(cd ${MXEBASE} && MXE_TARGETS=x86_64-w64-mingw32.static make biosig)

# mask calls to cl.exe and link.exe
${MXEBASE}/usr/bin/x86_64-w64-mingw32.static-gcc hello.c -o  ${WINEPREFIX}/drive_c/windows/cl.exe 
${MXEBASE}/usr/bin/x86_64-w64-mingw32.static-gcc hello.c -o  ${WINEPREFIX}/drive_c/windows/link.exe 

# cd src/biosig-code/biosig4c++/python
for PYVER in ${WINEPREFIX}/drive_c/users/$(whoami)/AppData/Local/Programs/Python/Python* ; do 
	VER=${PYVER/*Python/}
	${WINE} ${PYVER}/python.exe -m pip install build matplotlib numpy setuptools wheel
	### build dummy wheels ###
	${WINE} ${PYVER}/python.exe -m build --wheel

	${MXEBASE}/usr/bin/x86_64-w64-mingw32.static-gcc -mdll biosigmodule.c \
		-o build/lib.win-amd64-cpython-${VER}/biosig.cp${VER}-win_amd64.pyd \
		-I${PYVER}/Lib/site-packages/numpy/_core/include \
		-I${PYVER}/include \
		-L${PYVER}/libs/ \
		-L${MXEBASE}/usr/x86_64-w64-mingw32.static/lib/ \
		-lpython3 -lbiosig 

	cp ${MXEBASE}/usr/x86_64-w64-mingw32.static/bin/libbiosig-3.dll build/lib.win-amd64-cpython-${VER}/

	### build real wheel in dist/Biosig*.whl ###
	${WINE} ${PYVER}/python.exe -m build --wheel
done

