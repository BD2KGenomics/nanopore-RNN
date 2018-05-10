
all : python-recs basetensor signalalign chiron nanonet nanotensor

python-recs:
	pip install -U setuptools

marginAlign:
	pip install marginAlign
	pip install -U numpy
	pip install -U h5py


nanotensor:
	python setup.py install

chiron:
	cd submodules && cd chiron && python setup.py install

nanonet:
	cd submodules && cd nanonet && head -500 setup.py
	cd submodules && cd nanonet && python setup.py install

signalalign:
	cd submodules && cd signalAlign && make && python setup.py install

basetensor:
	cd submodules && cd BaseTensor && python setup.py install

clean:
	cd submodules && cd signalAlign && make clean

test:
	cd submodules && cd signalAlign && make test
	cd nanotensor && pytest

.PHONY: nanonet nanotensor signalalign python-recs marginAlign pypore chiron
