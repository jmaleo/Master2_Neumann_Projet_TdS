# Neumann

This repository provides evaluation functions for the following repository : https://github.com/dgilton/iterative_reconstruction_networks

You can see this repository also in a drive directory, in order to use google colab at this link : https://drive.google.com/drive/folders/1cOS6yDUY6j2VPJeaoEGsPEgv6nlX8pmK?usp=sharing

## Data 

You need to have the CelebA dataset.
You can have it by going into this link : https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg?resourcekey=0-rJlzl934LzC-Xp28GeIBzQ
and downloading the zip folder, named img_align_celeba.zip.

Otherwise, there is the thyroid dataset into this link : https://drive.google.com/drive/folders/1hkPaI7jjao9vWCSr4lmsKr_z3iAvnA4E?usp=sharing
You have to put this directory into data with the name "Thyroid".

You'll need to put it into your Neumann directory as "data/img_align_celeba".

## How to use 

Firstly, you need to change the path of the directory in all the files.py from the /scripts/celeba directory and from the /evaluations/ directory.

It is included on the 5'th line "sys.path.append('your path to the directory Neumann')"

Now, you'll be able to exploite what you want ;).

## Result directory

Here, you may find the resulting training files. 
They are the result of the computation of the files in the /scripts/celeba directory.

If you want to train these neural networks, you may change the parameters in the top of these files.py, and you may change the name of the save_location since you want to obtain new trainings.

## evalutions directions
  
There is 4 directories, in which you can find results of evaluations.
For these evaluations, you can find here the explanations :

- dry files : files without any effect, only resised.
- blur files : blured files with the fonction given by the author of the initial repository.
- resulted files : files computed with the given neural network, trained onto Celeba dataset or Thyroid.
    - there is the parameter nb_iteration (1, 5, 10).
    - there is the parameter (blur, no_blur) to say if the image has been blured or not before the evaluation.
    - there is the parameter.
    - All these informations are given into the name of the images.

