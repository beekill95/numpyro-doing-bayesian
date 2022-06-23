# numpyro-doing-bayesian
My implementation of John K. Kruschke's Doing Bayesian Data Analysis 2nd edition using Python and Numpyro.
This implementation is not comprehensive,
I'll just focus on the generalized linear model only,
which is from chapter 16 onward.

## Dependencies

This project uses a combination of [Conda](https://docs.conda.io/en/latest/)
and [Poetry](https://python-poetry.org/) for dependencies management.
To install the dependencies for this project, make sure that you have `conda` installed on your system.

First, create a virtual environment managed by `conda`:

```
conda env create -f environment.yml
```

The above command will create a virtual environment named `doing_bayes`
and install `poetry` package manager into that environment.

After that, activate the environment `conda activate doing_bayes`
and use `poetry` to install the remaining dependencies:

```
poetry install
```

## Jupyter Notebook

Activate the `doing_bayes` environment,
and then start the `jupyter-lab` server:

```
jupyter-lab --no-browser
```

Then, you can click on the link to open notebooks on your browsers.

Each chapter's notebook are a normal python script thanks to [Jupytext](https://jupytext.readthedocs.io/en/latest/).
To generate a notebook for a chapter from the python script, you can follow this [instruction](https://jupytext.readthedocs.io/en/latest/paired-notebooks.html#how-to-open-scripts-with-either-the-text-or-notebook-view-in-jupyter).