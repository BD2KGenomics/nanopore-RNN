

all : python-recs nanonet signalalign marginAlign nanotensor

python-recs:
	pip install -U setuptools

marginAlign:
	pip install marginAlign
	pip install -U numpy
	pip install -U h5py

signalalign:
	cd signalAlign && make && python setup.py install

nanotensor:
	python setup.py install

nanonet:
	cd nanonet && python setup.py install

clean:
	cd signalAlign && make clean

test:
	cd signalAlign/bin && make test
	cd nanotensor && pytest

.PHONY: nanonet nanotensor signalalign python-recs marginAlign
