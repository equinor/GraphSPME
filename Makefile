ROOTDIR = $(CURDIR)

.PHONY: Rpack Rbuild Rinstall Rcheck

# Script to make a clean installable R package.
Rpack:
	rm -rf GraphSPME GraphSPME*.tar.gz
	cp -r R-package GraphSPME
	cp -r include GraphSPME/inst
	cp ./LICENSE GraphSPME

R ?= R

Rbuild: Rpack
	$(R) CMD build GraphSPME
	rm -rf GraphSPME
	
Rinstall: Rpack
	$(R) CMD install GraphSPME*.tar.gz

Rcheck: Rbuild
	$(R) CMD check --as-cran GraphSPME*.tar.gz

