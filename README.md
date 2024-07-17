# cloudcasting

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

Tooling and infrastructure to enable cloud nowcasting.

## Installation

From source (development mode):
```bash
git clone https://github.com/climetrend/cloudcasting
cd cloudcasting
python -m pip install ".[dev]"
```

Install pre-commit before making development changes:

```bash
pre-commit install
```

For making changes, see the [guidance on development](https://github.com/alan-turing-institute/python-project-template?tab=readme-ov-file#setting-up-a-new-project) from the template that generated this project.

## Usage

Example:

```bash
cloudcasting download "2020-06-01 00:00" "2020-06-30 23:55" "path/to/my/dir/data.zarr"
```

```bash
> cloudcasting download --help

 Usage: cloudcasting download [OPTIONS] START_DATE END_DATE OUTPUT_DIRECTORY

 Download a selection of the available EUMETSAT data.
 Each calendar year of data within the supplied date range will be saved to a
 separate file in the output directory.
 Args:     start_date: First datetime (inclusive) to download.     end_date: Last
 datetime (inclusive) to download.     data_inner_steps: Data will be sliced into
 data_inner_steps*5minute chunks.     output_directory: Directory to which the
 satellite data should be saved.     lon_min: The west-most longitude (in
 degrees) of the bounding box to download.     lon_max: The east-most longitude
 (in degrees) of the bounding box to download.     lat_min: The south-most
 latitude (in degrees) of the bounding box to download.     lat_max: The
 north-most latitude (in degrees) of the bounding box to download.     get_hrv:
 Whether to download the HRV data, else non-HRV is downloaded.
 override_date_bounds: Whether to override the date range limits.
 Raises:     FileNotFoundError: If the output directory doesn't exist.
 ValueError: If there are issues with the date range or if output files already
 exist.

╭─ Arguments ────────────────────────────────────────────────────────────────────╮
│ *    start_date            TEXT  Start date in 'YYYY-MM-DD HH:MM' format       │
│                                  [default: None]                               │
│                                  [required]                                    │
│ *    end_date              TEXT  End date in 'YYYY-MM-DD HH:MM' format         │
│                                  [default: None]                               │
│                                  [required]                                    │
│ *    output_directory      TEXT  Directory to save the satellite data          │
│                                  [default: None]                               │
│                                  [required]                                    │
╰────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ─────────────────────────────────────────────────────────────────────╮
│ --data-inner-steps                             INTEGER  Data will be sliced   │
│                                                         into                  │
│                                                         data_inner_steps*5mi… │
│                                                         chunks                │
│                                                         [default: 3]          │
│ --get-hrv               --no-get-hrv                    Whether to download   │
│                                                         HRV data              │
│                                                         [default: no-get-hrv] │
│ --override-date-bou…    --no-override-date…             Whether to override   │
│                                                         date range limits     │
│                                                         [default:             │
│                                                         no-override-date-bou… │
│ --lon-min                                      FLOAT    Minimum longitude     │
│                                                         [default: -16]        │
│ --lon-max                                      FLOAT    Maximum longitude     │
│                                                         [default: 10]         │
│ --lat-min                                      FLOAT    Minimum latitude      │
│                                                         [default: 45]         │
│ --lat-max                                      FLOAT    Maximum latitude      │
│                                                         [default: 70]         │
│ --valid-set             --no-valid-set                  Whether to filter     │
│                                                         data from 2022 to     │
│                                                         download the          │
│                                                         validation set (every │
│                                                         2 weeks).             │
│                                                         [default:             │
│                                                         no-valid-set]         │
│ --help                                                  Show this message and │
│                                                         exit.                 │
╰───────────────────────────────────────────────────────────────────────────────╯

```


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [MIT license](LICENSE).


<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/climetrend/cloudcasting/workflows/CI/badge.svg
[actions-link]:             https://github.com/climetrend/cloudcasting/actions
[pypi-link]:                https://pypi.org/project/cloudcasting/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/cloudcasting
[pypi-version]:             https://img.shields.io/pypi/v/cloudcasting
<!-- prettier-ignore-end -->
