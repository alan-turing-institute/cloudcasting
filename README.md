# cloudcasting

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

Tooling and infrastructure to enable cloud nowcasting.

## Installation

From source (development mode):

On macOS you first need to install `ffmpeg` with the following command. On other platforms this is 
not necessary.

```bash
brew install ffmpeg
```

Clone and install the repo.

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

Full options:

```bash
> cloudcasting download --help

 Usage: cloudcasting download [OPTIONS] START_DATE
                              END_DATE OUTPUT_DIRECTORY

╭─ Arguments ──────────────────────────────────────────╮
│ *    start_date            TEXT  Start date in       │
│                                  'YYYY-MM-DD HH:MM'  │
│                                  format              │
│                                  [default: None]     │
│                                  [required]          │
│ *    end_date              TEXT  End date in         │
│                                  'YYYY-MM-DD HH:MM'  │
│                                  format              │
│                                  [default: None]     │
│                                  [required]          │
│ *    output_directory      TEXT  Directory to save   │
│                                  the satellite data  │
│                                  [default: None]     │
│                                  [required]          │
╰──────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────╮
│ --download-f…                   TEXT   Frequency to  │
│                                        download data │
│                                        in pandas     │
│                                        datetime      │
│                                        format        │
│                                        [default:     │
│                                        15min]        │
│ --get-hrv        --no-get-h…           Whether to    │
│                                        download HRV  │
│                                        data          │
│                                        [default:     │
│                                        no-get-hrv]   │
│ --override-d…    --no-overr…           Whether to    │
│                                        override date │
│                                        range limits  │
│                                        [default:     │
│                                        no-override-… │
│ --lon-min                       FLOAT  Minimum       │
│                                        longitude     │
│                                        [default:     │
│                                        -16]          │
│ --lon-max                       FLOAT  Maximum       │
│                                        longitude     │
│                                        [default: 10] │
│ --lat-min                       FLOAT  Minimum       │
│                                        latitude      │
│                                        [default: 45] │
│ --lat-max                       FLOAT  Maximum       │
│                                        latitude      │
│                                        [default: 70] │
│ --test-2022-…    --no-test-…           Whether to    │
│                                        filter data   │
│                                        from 2022 to  │
│                                        download the  │
│                                        test set      │
│                                        (every 2      │
│                                        weeks).       │
│                                        [default:     │
│                                        no-test-2022… │
│ --verify-202…    --no-verif…           Whether to    │
│                                        download the  │
│                                        verification  │
│                                        data from     │
│                                        2023. Only    │
│                                        used at the   │
│                                        end of the    │
│                                        project       │
│                                        [default:     │
│                                        no-verify-20… |
│ --help                                 Show this     │
│                                        message and   │
│                                        exit.         │
╰──────────────────────────────────────────────────────╯
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
