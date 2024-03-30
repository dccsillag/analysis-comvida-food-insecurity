# Code for 'Predicting food insecurity: a machine-learning analysis on a population-based survey'

All the Python analysis code can be found in `model_analysis.py`.

# Dependencies

All the required Python dependencies are declared via [Poetry](https://python-poetry.org/). To setup an environment with all the required packages, run (in the repository's directory):

```shell
poetry install
```

You can then run the Python code for the analyses through `poetry run`:

```shell
poetry run python model_analysis.py
```

Alternatively, you can open a shell within the environment via `poetry shell`, and then simply run the Python script directly:

```shell
poetry shell
python model_analysis.py
```

For more information on Poetry, see its manual: https://python-poetry.org/docs/.
