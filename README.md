# uef-seqworkshop2018
UEF - Sequence Modeling Workshop 2018

How to clone all the code and data provided to your computer:

```bash
git clone --recursive git@github.com:trungnt13/uef-seqworkshop2018.git
```
For Windows users, using github desktop may significantly simplify the process:
[https://desktop.github.com/](https://desktop.github.com/)

Recommended reading: [Deep learning and language identification](http://epublications.uef.fi/pub/urn_nbn_fi_uef-20170270/urn_nbn_fi_uef-20170270.pdf)

## Setting up python environment

#### Installing miniconda
Following the instruction and install Miniconda from this link:
[https://conda.io/miniconda.html](https://conda.io/miniconda.html)

#### Create the environment
> conda env create -f=environment.yml

#### Using installed environment
For activating and using our environment:
> source activate uefseq18

Deactivating environment:
> source deactivate

Listing installed packages:
> conda list

#### Delete environment
> conda remove --name uefseq18 --all

#### More tutorials for Windows users
[https://conda.io/docs/user-guide/install/windows.html#install-win-silent](https://conda.io/docs/user-guide/install/windows.html#install-win-silent)

