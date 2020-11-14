*********************
Directory Structure
*********************
This file contains information on the infrastructure of the
repository. It is meant to be a complement to the project report in
case that some of the functionalities are not fully clear

data
#####
The data folder contains a reduced version of the data. In particular,
we used the data from the NYC open data on car crashes. Because the
file is too big there is a reduced version of it called ``reduced_data.csv``.
The file ``initial_data_claening.ipynb`` contains an explanation of
how this data is generated from the original dataset.

To generate all of the datasets used for the project. Run the python
script ``generate_datasets.py``


test_notebooks
##############
The test_notebooks folders contain jupyter notebooks that attempt to
implement some of the models that we use but using fake data. We do
this as a way of making sure that our inference methods are sound and
we are not getting something with absurd parameters. 
