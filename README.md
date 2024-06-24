## Repo management

To activate this environment, use

     $ conda activate bounding_box_detection_ham10000_torch

 To deactivate an active environment, use

     $ conda deactivate

To recreate the environment, use

    $ conda env create -f environment.yml

To export environment, use

    $ conda env export --from-history > environment.yml

## Notebook descriptions

1. [Basic ConvNN](./1_basic.ipynb) - a basic neural network using SmoothL1Loss. 
One of the observations here is that the code spends a lot of time in the cpu because of data loading.
Perhaps the network itself is so small that the DataLoader can't keep up with the training.


