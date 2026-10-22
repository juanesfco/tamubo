# Installing AutoIA 1.0.2

For the verified wheel in the current directory, install into the existing
Anaconda environment by invoking that environment's Python interpreter:

```bash
/opt/anaconda3/envs/autoia/bin/python -m pip install --upgrade --force-reinstall --no-deps ./autoia-1.0.2-py3-none-any.whl
```

Verify the installed version with:

```bash
/opt/anaconda3/envs/autoia/bin/python -c "import autoia; print(autoia.__version__)"
```

The expected output is `1.0.2`.

GPyTorch and BoTorch are optional integrations and are not installed as AutoIA
dependencies.

## Optional process support for closures

AutoIA's core install does not require `cloudpickle`. For spawn-based
`parallel="process"` execution of local functions or closures, install:

```bash
python -m pip install "autoia[process]"
```

Module-level models that are serializable by Python's standard `pickle` do not
require this extra.
