
all : python-recs signalalign chiron nanotensor

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

chiron:
	cd chiron && python setup.py install

clean:
	cd signalAlign && make clean

test:
	cd signalAlign && make test
	cd nanotensor && pytest
	cd python_utils && pytest

.PHONY: nanonet nanotensor signalalign python-recs marginAlign pypore chiron
