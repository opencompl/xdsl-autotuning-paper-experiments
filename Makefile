setup: setup-uica .venv

uica-staticdeps:
	git submodule update --init

setup-uica: uica-staticdeps
	cd uica-staticdeps && ./setup.sh

.venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r uica_requirements.txt
