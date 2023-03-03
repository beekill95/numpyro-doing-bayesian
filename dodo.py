from doit.tools import config_changed
import glob
import numpyro_glm
import os

OUTPUT_DIR = 'docs'
PY_FILES = glob.glob('notebooks/chapter_*.py')


def notebook_output_path(py_file: str):
    filename = os.path.basename(py_file)
    filename, _ = os.path.splitext(filename)
    dir = os.path.dirname(py_file)
    return os.path.join(dir, f'{filename}.ipynb')


def html_output_path(output_dir: str, py_file: str):
    filename = os.path.basename(py_file)
    filename, _ = os.path.splitext(filename)
    return os.path.join(output_dir, f'{filename}.html')


def task_notebook():
    def task_spec(path: str):
        return dict(
            file_dep=[path],
            name=os.path.basename(path),
            actions=[f'jupytext --to notebook --execute "{path}"'],
            targets=[notebook_output_path(path)],
            uptodate=[config_changed(numpyro_glm.__version__)],
            verbosity=2,
        )

    return (task_spec(file) for file in PY_FILES)


def task_publish():
    def task_spec(path: str):
        notebook = notebook_output_path(path)
        return dict(
            name=os.path.basename(path),
            file_dep=[notebook, './html_template/index.html.j2'],
            actions=[
                f'jupyter nbconvert --to html --embed-images --no-prompt --template ./html_template --output-dir="{OUTPUT_DIR}" "{notebook}"'],
            targets=[html_output_path(OUTPUT_DIR, path)],
            verbosity=2,
        )

    return (task_spec(file) for file in PY_FILES)
