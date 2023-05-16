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
an `output/` directory that is a symbolic link to your `eos` space (or
wherever you have a lot of available space), to be able to store all the data.
```
ln -s {your-eos-path-to-output} output/
```
Otherwise `Snakemake` will create `output/` directory in the same location this repo is.

Now everytime you want to run the code, first activate the virtual environment
```
conda activate anomaly-base
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
## Downloading LIGO data
We use real data from the 03A run of LIGO.
To get the initial data go to
[O3a Data Release](https://gwosc.org/data/)
and download from [the following dataset](https://gwosc.org/archive/O3a_16KHZ_R1/)

```
O3a Time Range: April 1, 2019 through October 1, 2019
Dataset: O3a_16KHZ_R1
```

The following interval

```
GPS Time Interval: [1239134846, 1239140924]
Detector: L1, H1
```

## Data Generation
To generate all the necessary samples for the analysis from downloaded data just run
```
snakemake -c1 generate_data
```
This rule with call all four rules explained below for each of the data types.
### Background
We can start with downloading open data from the LIGO webpage
To select background-like samples, we apply a cut on SNR to be less than 6.
This step can be repeated with
```
snakemake -c1 generate_background
```
### Glitches
To create glitch samples we again use real data from the 03A applying a cut on SNR to be larger than 10.
This step can be repeated with
```
snakemake -c1 generate_glitch
```
### BBH
We generate BBH samples as in details explained in our paper.
Once the signals are generated, we inject them in the real background.
This step can be repeated with
```
snakemake -c1 generate_bbh
```
### SG
We generate BBH samples as in details explained in our paper.
Once the signals are generated, we inject them in the real background.
This step can be repeated with
```
snakemake -c1 generate_sg
```

## Data Pre-processing
There are two steps to prepare the data for training the QUAK space.
First, we need to split the dataset in training and testing parts
```
snakemake -c1 train_test_split
```
Next, we want to standardize the data to facilitate autoencoders learning
```
snakemake -c1 pre_processing_step
```

## Training QUAK
Now that the samples are prepared, we can move to training autoencoders to build the QUAK space
```
snakemake -c1 train_quak
```
## Pearson Evaluation
```
snakemake -c1 pre_processing_step
```
## Evolutionary Search training
```
snakemake -c1 pre_processing_step
```
