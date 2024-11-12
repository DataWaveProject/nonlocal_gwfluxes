## Getting Started

### Obtaining `nonlocal_gwfluxes`

Clone the latest version of `nonlocal_gwfluxes` using the following command

```bash
git clone git@github.com:DataWaveProject/nonlocal_gwfluxes.git
```

### Install Dependencies

This project uses `poetry` to manage dependencies. To install the dependencies we first need poetry.

```bash
cd nonlocal_gwfluxes
python -m venv .nlgw
```

This will create a Python virtual environment inside our repository. Before we use this environment we must first `activate` it.

```bash
source .nlgw/bin/activate
```

> [!NOTE]
> Your shell prompt should update. Depending on your setup, you may have something like this:
> ```
> (.nlgw) demo@mypc:~/nonlocal_gwfluxes$
> ```

Now we can install `poetry`

```bash
pip install poetry
```

The following command installs all of the necessary dependencies for `nonlocal_gwfluxes`.

```bash
poetry install --no-root
```

If plan to develop `nonlocal_gwfluxes`, please add the optional `develop` dependencies.

```bash
poetry install --no-root --with develop
```
