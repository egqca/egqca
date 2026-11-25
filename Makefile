# Build documentation locally
# in bash:
# make docs-install
# --> Installs your package together with the docs dependencies via pip install -e .[docs].

# make docs-serve
# --> Starts a local MkDocs development server. You can then open the local URL it prints, usually http://127.0.0.1:8000/, and preview the docs in the browser

# make docs-build
# --> Builds the static documentation site into the site/ folder, the same kind of output that GitHub Pages would publish.

# make docs-clean
# --> Deletes the generated site/ folder.

.PHONY: docs-install docs-serve docs-build docs-clean

docs-install:
	python -m pip install -e .[docs]

docs-serve:
	mkdocs serve

docs-build:
	mkdocs build --clean

docs-clean:
	rm -rf site
