# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: doing_bayes
#     language: python
#     name: doing_bayes
# ---

# %cd ..
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# # Chapter 17: Metric Predicted Variable with one Metric Predictor

# ## Simple Linear Regression

# +
x = np.random.uniform(-10, 10, size=500)
y = np.random.normal(10 + 2 * x, 2)

fig, ax = plt.subplots()
ax.scatter(x, y, c='black', s=4)
ax.set_title('Normal PDF around linear function')

xline = np.linspace(-10, 10, 1000)
ax.plot(xline, 10 + 2 * xline, lw=4, c='#87ceeb')

# TODO
for xinterval in [-7.5, -2.5, 2.5, 7.5]:
    y_ = np.linspace(xinterval - 6, xinterval + 6, 1000)
