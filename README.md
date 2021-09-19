# Final Project Repository Template

This is the final project repository template for
[Machine Learning with Probabilistic Programming](http://www.proditus.com/mlpp2021).

## Duplicating your own copy of this repository

Please follow these
[instructions](https://help.github.com/articles/duplicating-a-repository/)
to make a copy of this repository and push it to your own GitHub account.

Make sure to create a **new repository** on your own GitHub account before
starting this process.

## Final Project Notebook
We have included a example of a Jupyter notebook under
`/notebook-example/example.ipynb`. This shows how to use markdown along with
LaTeX to create section headings and typeset math.

Your final project notebook should go under
`/final-project/final-notebook.ipynb`. This notebook will be your final report.
We must be able to run it in a reasonable amount of time. (If your project
involves a massive dataset, please contact me.)

Your final report should be 8 pages long. Since it is hard to translate between
a Jupyter notebook and page numbers, we've come up with the following metric:
> the Markdown export of your notebook should be approximately 1500 words.

To compute this, save your Jupyter notebook as a Markdown file by going to
```
File > Download as > Markdown (.md)
```
and then counting the words
```
wc -w final-notebook.md
```

Since this includes your code as well, we encourage you to develop separate
python scripts and include these in your final notebook. My recommendation is
that you only do basic data loading, manipulation, and plotting within Jupyter;
do all of the heavy lifting in separate Python files. (Note our strict
guidelines on coding style below.)

### Structure
Your notebook should follow the basic structure described in the project
proposal template. Make sure to clearly indicate section headings and to
present a clear narrative structure. Every subsection of your report should
correspond to a particular step of Box's loop. Feel free to include images; you
can embed them in markdown cells.

## Development
Use Python 3.9.X.

Configure a virtual environment.
Follow the documentation
[here](https://docs.python.org/3.9/tutorial/venv.html).

Once you activate the virtual environment, use `pip` to install a foundational set of
packages.
```{bash}
(venv)$ pip install -r requirements.txt
```

This should install CmdStanPy, along with Jupyter and other useful libraries.

Please install cmdstan into its default location `~/.cmdstan` by following the instructions here: [Installing CmdStan](https://cmdstanpy.readthedocs.io/en/v0.9.77/installation.html#installing-cmdstan)

### Additional dependencies
If you introduce any new dependencies to your final project, you **MUST**
update `requirements.txt` with pinned versioning.

### Git stuff
There is a comprehensive `.gitignore` file in this repository. This should prevent you from committing any unnecessary files. Please edit it as needed and do not commit any large files to the repository. (Especially huge datasets.)

### Code styling
Any additional code you write must pass `flake8` linting. See this
[blog post](https://medium.com/python-pandemonium/what-is-flake8-and-why-we-should-use-it-b89bd78073f2) for details.

The first thing we will do after cloning your repository is:
```{bash}
(venv)$ flake8
```

If your repository fails any checks, we will **deduct 20%** from your final project grade.
