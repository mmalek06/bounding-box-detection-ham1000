## Repo management

To activate this environment, use

     $ conda activate bounding_box_detection_ham10000_torch

 To deactivate an active environment, use

     $ conda deactivate

To recreate the environment, use

    $ conda env create -f environment.yml

To export environment, use

    $ conda env export --from-history > environment.yml

## Main notebook descriptions

1. [Basic ConvNN](./1_basic.ipynb) - a basic neural network using SmoothL1Loss. 
One of the observations here is that the code spends a lot of time in the cpu because of data loading.
The reason was that num_workers param has been so with a too large value.
2. [Basic, bigger ConvNN](./2_bigger_basic.ipynb) - architecture is basically the same as for the no. 1 
notebook, and the differing factor is the number of kernels in each layer. That change seems to have helped
as now the loss is slightly smaller.
3. [Basic, bigger, using CIoU](./3_bigger_basic_ciou.ipynb) - again, the same architecture, but different
loss function. CIoU is a more advanced loss calculation method for bounding box regression.
4. [Comparison](./tbd) - comparison between bounding boxes detected by each neural network.

## Other notebook descriptions

Other notebooks contains various processes that move the data around, or add some information to the original data.
