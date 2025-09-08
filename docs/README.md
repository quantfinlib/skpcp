### Documentation

The documentation is supported by [Sphinx](https://www.sphinx-doc.org/en/master/). 

To build the HTML pages locally, first make sure you have installed the package with its documentation dependencies:

```bash
uv pip install -e .[docs]
```

then run the following:

```bash
sphinx-build docs docs/_build
```

 from the package root directory. The documentation can then be viewed by opening `./docs/_build/html/index.html``. 

Github Actions is used for continuous integration, and the tests will fail if the documentation does not build.