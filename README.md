# Introduction
This repo contains complimentary code for the QUAK LIGO paper.

# Set up
To start, install [`Miniconda`](https://docs.conda.io/en/latest/miniconda.html)
(if you don't already have it) to be able to create a virtual environment.

Once conda is there, setup a virtual environment using the file from the repo
```
conda env create --file environment.yaml
conda activate quak
```
Finally, install additional package inside the create virtual environment
```
pip install snakemake
```

Good thing to do before running (only has to be done once) is to create
an `output/` directory that is a symbolic link to your work space (
wherever you have a lot of available space), to be able to store all the data.
```
ln -s {your-work-path-to-output} output/
```
Otherwise `Snakemake` will create `output/` directory in the same location this repo is.

Now every time you want to run the code, first activate the virtual environment
```
conda activate quak
```


# Snakemake
The code is organized using [`Snakemake`](https://snakemake.readthedocs.io/en/stable/).
The Snakemake workflow management system is a tool to create reproducible and scalable data analyses.

To run snakemake do
```
snakemake -c1 {rule_name}
```
where `-c1` specifies number of cores provided (one in this case).
It became required to specify it in the latest versions of snakemake,
so to make life easier you can add
`alias snakemake="snakemake -c1"` to your `bash/zsch/whatever` profile
and afterwards simply run `snakemake {rule_name}`.

If you want to run a rule, but Snakemake tells you `Nothing to be done`, use `-f`
to force it. Use `-F` to also force all the upstream rules to be re-run.

# Analysis Pipeline
Each step of the analysis is represented as a corresponding rule in the `Snakefile`.
There are several steps starting with downloading and generating data samples,
pre-processing them and training the QUAK space using these samples.
Followed by the Pearson correlation calculation and the evaluation step.
The details of all the available rules are given below.
## Configuration parameters
All the parameters of generation and training are specified in `config.py`, that's
the only place they are specified and the only place through which they can be changed.

## Running Omicron
The analysis pipeline starts with fetching 03 background data and running Omicron software on it to
select loud and quite segments of data taking.
```
snakemake -c1 run_omicron
```

## Downloading LIGO data
Next, the LIGO data is fetched and saved with
```
snakemake -c1 fetch_data
```

## Data Generation and pre-processing
To generate all the necessary samples for the analysis from downloaded data and
pre-process them for training, use `generate_dataset` and `pre_processing_step` rules.
Pre-processing includes per-batch standartization
These rules are written using `Wildcards`, such that one rule can be used for
each of four datasets, eg BBH, SG, glitches and background.

## Training QUAK
Now that the samples are prepared, we can move to training autoencoders to build the QUAK space:
```
snakemake -c1 train_all_quak
```
You do not need to run each rule by hand. In case for example you want to generate data and train the autoencoders
from scratch, just run the rule above and `Snakemake` will figure out dependencies and will run needed rules to create
missing data.

## Pearson Evaluation
Then we need to evaluate Pearson correlation between detector sites by running
```
snakemake -c1 calculate_pearson
```
the current implementation is quite slow, so prepare yourself to wait a lot...

## Evolutionary Search training
```
snakemake -c1 pre_processing_step
```
