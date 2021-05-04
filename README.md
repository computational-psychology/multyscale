# Multyscale spatial filtering models of brightness/lightness perception

## Installing

Multyscale is not yet available on package registries (e.g., PyPI). Instead:

1. clone from GitLab (TUB):

   ```git
   git clone git@git.tu-berlin.de:computational-psychology/multyscale.git
   ```

1. Multyscale can then be installed using pip. From top-level directory (which contains `setup.py`) run:

    ```pip
    pip install .
    ```

    to install to your local python library.

- For developers, use:

    ```pip
    pip install -e .
    ```

    for an editable install;
    package now does not need to be reinstalled to make changes usable.

## Using

```python
import multyscale
```

- `multyscale.filters` contains functions to generate filters
- `multyscale.filterbank` contains classes defining specific sets (banks) of filters
- `multsycale.models` implements some common models from the literature
