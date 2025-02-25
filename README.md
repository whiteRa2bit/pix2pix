# PIX2PIX

Image-to-Image Translation with Conditional Adversarial Networks

One of the homeworks as part of Deep Learning Course at Higher School of Economics.
A detailed task description can be found [here](https://docs.google.com/document/d/1IW--VvxmLI6enh5yYPQganIQfQ1loXATbn8x0liLD9o).

**Experiments report** can be found at [docs folder](docs/report.pdf)

## Examples

![Anime dataset example](docs/anime_example.png)
![Pokemon dataset example](docs/pokemon_example.png)

## Summary

  - [Getting Started](#getting-started)
  - [Installing](#installing)
  - [Data](#data)
  - [Train](#train)
  - [Coding style tests](#coding-style-tests)
  - [Authors](#authors)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Getting Started

To install the package go to installing notes. All requirements can be found at setup.py

## Installing

To install the package run the following command

    pip install -e .

## Data

For experiments the following two datasets were used:
- [Pokemon images dataset](https://www.kaggle.com/kvpratama/pokemon-images-dataset)
- [Anime faces](https://www.kaggle.com/soumikrakshit/anime-faces)

## Train

To run training modify config file accordingly and run

    cd tools
    python run_train.py

## Coding style tests

YAPF (0.22.0) is used for coding style validation

    python3 -m yapf -rp . --in-place


## Authors

  - **Pavel Fakanov** - *Wrote the code, run the experiments* -
    [whiteRa2bit](https://github.com/whiteRa2bit)

## License

This project is licensed under the [Apache License](LICENSE)

## Acknowledgments
  - Inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
