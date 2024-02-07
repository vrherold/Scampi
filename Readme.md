# MRI-SCAMPI

### This is the implemenation of the Scampi *(Sparsity Constrained Application of deep Magnetic resonance Priors for Image reconstruction)* reconstruction algorithm for undersampled MRI data. 

##### It is optimized to run on a gpu if available.



## **Installation (linux/Mac):**

	$ git clone [URL of the GitHub repository]
	$ cd Scampi


We recommend to use virtualenv to create the virtual environment.
    
    $ mkdir myvenv
    $ cd myvenv
	$ python3.9 -m venv .
    $ source bin/activate
	$ pip install -r ../requirements.txt


## **Examples:**


The jupyternotebooks (Cartesian_Scampi, NonCartesianScampi) with the included MR-sample data demonstrates the reconstruction of undersampled multicoil data.

    $ cd ..
    $ jupyter-notebook Cartesian_Scampi.ipynb

If you have any questions/comments, feel free to contact [Volker Herold](mailto:vrherold@physik.uni-wuerzburg.de).

[Scampi: Research-Paper](https://arxiv.org/abs/2209.08143)

**Enjoy**