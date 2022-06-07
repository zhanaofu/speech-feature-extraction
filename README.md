# Requirement

- `numpy`
- `pandas`

# Arguments
Positional argument:

`file`: The path to the .csv file containing the confusion matrix. The first column and the first row are the names of the phonemes.

Optional arguments:

`-resample`: Whether or not to resample the confusion matrix before feature extraction. Options: 'True' or 'False', default: 'True'.

# Example
`python extract.py cm_perception.csv -resample False`
