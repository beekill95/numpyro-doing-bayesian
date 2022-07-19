# numpyro-doing-bayesian

My implementation of John K. Kruschke's
[Doing Bayesian Data Analysis 2nd edition](https://sites.google.com/site/doingbayesiandataanalysis/what-s-new-in-2nd-ed)
using Python and Numpyro.
This implementation is not comprehensive,
I'll just focus on the generalized linear model only,
which is from chapter 16 onward.
Suggestions for improvement are welcome!

## Chapters

* [Chapter 9: Hierarchical Models](https://www.nguyenmbquan.space/numpyro-doing-bayesian/chapter_09)
* Chapter 10: Model Comparison and Hierarchical Modelling
* Chapter 12: Bayesian Approaches to Testing a Point ("Null") Hypothesis
* [Chapter 16: Metric-Predicted Variable on One or Two Groups](https://www.nguyenmbquan.space/numpyro-doing-bayesian/chapter_16)
* [Chapter 17: Metric Predicted Variable with one Metric Predictor](https://www.nguyenmbquan.space/numpyro-doing-bayesian/chapter_17)
* [Chapter 18: Metric Predicted Variable with Multiple Metric Predictors](https://www.nguyenmbquan.space/numpyro-doing-bayesian/chapter_18)
* [Chapter 19: Metric Predicted Variable with One Nominal Predictor](https://www.nguyenmbquan.space/numpyro-doing-bayesian/chapter_19)
* (WIP) [Chapter 20: Metric Predicted Variable with Multiple Nominal Predictors](https://www.nguyenmbquan.space/numpyro-doing-bayesian/chapter_20)
* [Chapter 21: Dichotomous Predicted Variable](https://www.nguyenmbquan.space/numpyro-doing-bayesian/chapter_21)
    * [Exercise 21.3: Heterogeneous Concentration Parameters](https://www.nguyenmbquan.space/numpyro-doing-bayesian/chapter_21_exercise_21_3)
* [Chapter 22: Nominal Predicted Variable](https://www.nguyenmbquan.space/numpyro-doing-bayesian/chapter_22)
* [Chapter 23: Ordinal Predicted Variable](https://www.nguyenmbquan.space/numpyro-doing-bayesian/chapter_23)
    * Exercise 23.2: Handle Outliers
* [Chapter 24: Count Predicted Variable](https://www.nguyenmbquan.space/numpyro-doing-bayesian/chapter_24)

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

## Credits

My implementation refers to [JWarmenhoven's implementation](https://github.com/JWarmenhoven/DBDA-python) a lot,
especially those figures with data and posterior predictive distributions.
