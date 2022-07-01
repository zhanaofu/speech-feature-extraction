# Overview

'extract.py' takes a .csv file containing a confusion matrix as input, and outputs two files containing extracted features in binary format and text format, respectively.

'find_class.py' takes a .csv file containing a set natural classes and a .csv file containing a feature matrix, and find the minimal set of features needed to define the natural classes. Note that this definition is by the conjunction of features only.

# Requirement

- `numpy`
- `pandas`

# extract.py

## Arguments
Positional argument:

`file`: The path to the .csv file containing the confusion matrix. The first column and the first row are the names of the phonemes.

Optional arguments:

`--out`: The path to the output .csv file. Default: same path as to the input file.

`--preprocessing`: How to preprocess the data before feature extraction. Options: 'p' - calculating the probabilities of errors from each input phoneme (the probability of errors in each COLUMN adds up to 1 ); 'skip' - no preprocessing. Default: 'p'.

`--ratio`: If 'p' is selected as the preprocessing method, this argument assigns the ratio between numbers of correct mappings (the numbers in diagonal cells) and (the numbers in off-diagonal cells) incorrect mappings for each INPUT phoneme. Requires a number. Default: 1.


## Examples
`python extract.py cm_production.csv --out results/production.csv`

`python extract.py cm_perception.csv --out results/perception.csv`

`python extract.py cm_perception.csv --preprocessing p --ratio 2 --subtract False`

# find_class.py

## Arguments
Positional argument:

`feature_file`: The path to the .csv file containing the feature matrix. The the first row contains the names of the phonemes.

`pattern_file`: The path to the .csv file containing the p-base patterns with classes added as the last column.

Optional arguments:

`--out`: The path to the output .csv file. Default: same path as to the input file.

## Examples

`python find_class.py results/production.csv pbase_eng.csv`

`python find_class.py results/perception.csv pbase_eng.csv`

`python find_class.py feature_spe.csv pbase_eng.csv`