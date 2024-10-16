# cloudcasting

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

Tooling and infrastructure to enable cloud nowcasting.

## Linked model repos
- [Optical Flow (Farneback)](https://github.com/alan-turing-institute/ocf-optical-flow)
- [Optical Flow (TVL1)](https://github.com/alan-turing-institute/ocf-optical-flow-tvl1)
- [Diffusion model](https://github.com/alan-turing-institute/ocf-diffusion) 
- [ConvLSTM](https://github.com/alan-turing-institute/ocf-convLSTM)

The model template repo on which these are based is found [here](https://github.com/alan-turing-institute/ocf-model-template). These repos can be used to validate the models.

## Installation

### For users:

```zsh
git clone https://github.com/alan-turing-institute/cloudcasting
cd cloudcasting
python -m pip install .
```

To run metrics on GPU:

```zsh
python -m pip install --upgrade "jax[cuda12]"
```
### For making changes to the library:

On macOS you first need to install `ffmpeg` with the following command. On other platforms this is
not necessary.

```bash
brew install ffmpeg
```

Clone and install the repo.

```bash
git clone https://github.com/alan-turing-institute/cloudcasting
cd cloudcasting
python -m pip install ".[dev]"
```

Install pre-commit before making development changes:

```bash
pre-commit install
```

For making changes, see the [guidance on development](https://github.com/alan-turing-institute/python-project-template?tab=readme-ov-file#setting-up-a-new-project) from the template that generated this project.

## Usage

### Validating a model
```bash
cloudcasting validate "path/to/config/file.yml" "path/to/model/file.py"
```

### Downloading data
```bash
cloudcasting download "2020-06-01 00:00" "2020-06-30 23:55" "path/to/data/save/dir"
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
[actions-badge]:            https://github.com/alan-turing-institute/cloudcasting/actions/workflows/ci.yml/badge.svg?branch=main
[actions-link]:             https://github.com/alan-turing-institute/cloudcasting/actions
[pypi-link]:                https://pypi.org/project/cloudcasting/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/cloudcasting
[pypi-version]:             https://img.shields.io/pypi/v/cloudcasting
<!-- prettier-ignore-end -->
