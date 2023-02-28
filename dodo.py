from doit.tools import config_changed
import glob
import numpyro_glm
import os


def task_to_notebook():
    py_files = glob.iglob('notebooks/chapter_*.py')
    return (dict(
        file_dep=[file],
        name=os.path.basename(file),
        actions=['echo TODO'],
        uptodate=[config_changed(numpyro_glm.__version__)],
        verbosity=2,
    ) for file in py_files)
