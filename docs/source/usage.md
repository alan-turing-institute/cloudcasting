.. _usage:

User guide
==========

**Contents:**

- :ref:`install`
- :ref:`optional`
- :ref:`getting_started`

.. _install:

Installation
------------

To use cloudcasting, first install it using pip:

```bash
git clone https://github.com/alan-turing-institute/cloudcasting
cd cloudcasting
python -m pip install .
```

.. _optional:

Optional dependencies
---------------------

cloudcasting supports optional dependencies, which are not installed by default. These dependencies are required for certain functionality.

To run the metrics on GPU:

```bash
python -m pip install --upgrade "jax[cuda12]"
```

To make changes to the library, it is necessary to install the extra `dev` dependencies, and install pre-commit:

```bash
python -m pip install ".[dev]"
pre-commit install
```

To create the documentation, it is necessary to install the extra `doc` dependencies:

```bash
python -m pip install ".[doc]"
```

.. _getting_started:

Getting started
---------------

Use the cli to download data:

```bash
cloudcasting download "2020-06-01 00:00" "2020-06-30 23:55" "path/to/data/save/dir"
```

Once you have developed a model, you can also validate the model, calculating a set of metrics with a standard dataset. 
To make use of the cli tool, use the [model github repo template](https://github.com/alan-turing-institute/ocf-model-template) to structure it correctly for validation. 

```bash
cloudcasting validate "path/to/config/file.yml" "path/to/model/file.py"
```
