# Overview

This script takes a .csv file containing a confusion matrix as input, and outputs two files containing extracted features in binary format and text format, respectively.

# Requirement

- `numpy`
- `pandas`

# Arguments
Positional argument:

`file`: The path to the .csv file containing the confusion matrix. The first column and the first row are the names of the phonemes.

Optional arguments:

`--out`: The path to the output .csv file. Default: same path as to the input file.

`--preprocessing`: How to preprocess the data before feature extraction. Options: 'p' - calculating the probabilities of errors from each input phoneme (the probability of errors in each COLUMN adds up to 1 ); 'skip' - no preprocessing. Default: 'p'.

`--ratio`: If 'p' is selected as the preprocessing method, this argument assigns the ratio between numbers of correct mappings (the numbers in diagonal cells) and (the numbers in off-diagonal cells) incorrect mappings for each INPUT phoneme. Requires a number. Default: 1.

`--subtract`: If 'p' is selected as the preprocessing method, this argument specifies whether to subtract the smallest value in the matrix minus 1 from the whole matrix. Options: True or False. Default: True.

# Examples
`python extract.py cm_production.csv --out results/production.csv`
`python extract.py cm_perception.csv --out results/perception.csv`
`python extract.py cm_perception.csv --preprocessing p --ratio 2 --subtract False`