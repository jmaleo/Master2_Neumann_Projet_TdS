# Neumann

This repository provides evaluation functions for the following repository : https://github.com/dgilton/iterative_reconstruction_networks

## Data 

You need to have the CelebA dataset.
You can have it by going into this link : https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ
and downloading the zip folder, named img_align_celeba.zip.

You'll need to put it into your Neumann directory as "img_align_celeba".

## How to use 

Firstly, you need to change the path of the directory in all the files.py from the /scripts/celeba directory and from the /evaluations/ directory.

It is included on the 5'th line "sys.path.append('your path to the directory Neumann')"

Now, you'll be able to exploite what you want ;).

## Result directory

Here, you may find the resulting training files. 
They are the result of the computation of the files in the /scripts/celeba directory.

If you want to train these neural networks, you may change the parameters in the top of these files.py, and you may change the name of the save_location since you want to obtain new trainings.
