# JobXMLC

The repository is divided into 3 main parts -
1. Make initial embeddings. This is where data preprocessing, embeddings creation etc. are handled.
The relevant files are in `make_initial_embeddings`
2. JobXMLC - This module is motivated by GalaXC, with a few changes for neighborhood selection and handling our task. Relevant files are in `jobxlmc`
3. Results - This folder has files to replicate results in the paper. Relevant files are in `results`


## Dataset
The dataset should be present in `dataset/`, e.g. the stackoverflow dataset is present as `dataset/stackoverflow` with the following files:
1. trn_X_Y.txt: dataset in csr format
2. tst_X_Y.txt: dataset in csr format
3. tst_X.txt: job description for each data point
4. tst_X.txt: job description for each data point
5. Y.txt: labels, one in each row