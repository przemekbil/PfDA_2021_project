# The final project for the 'Programming for the Data Analysis' 2021 course

This is the repository for the final project for Programming for the Data Analysis 2021 course.


## What to Install
1. Download and install the following software
    - Python v3.x (https://www.python.org/downloads/)
    - Pandas (https://pandas.pydata.org/docs/getting_started/install.html)
    - Numpy (https://numpy.org/)
    - SciPy (https://scipy.org/)
    - Jupyter Notebooks (https://jupyter.org/)
    - Matplotlib (https://matplotlib.org/)
    - Seaborn (https://seaborn.pydata.org/)
    - imbalanced-learn (https://imbalanced-learn.org/stable/install.html)
2. Alternatively, most of the above packages are contained in the Anaconda Python distribution (https://www.anaconda.com/products/individual)
    - imbalanced-learn is the only package not included in Anaconda (https://imbalanced-learn.org/stable/install.html)
3. If running Windows
    - Download and install commander (https://cmder.net/)

## How to run this project
1. Download the project to your computer by pressing the green 'Code' button and then downloading the 'ZIP' file or cloning this repository using git.
2. Unzip the file (if ZIP file was downloaded), open the Cmder (or other Console of your choice) and navigate to downloaded project folder
3. When in the folder, type 'Jupyter Lab' or 'Jupyter Notebook'
    - Jupyter Lab/Notebook will open in your system default browser
4. Using Jupyter Lab/Notebook navigation Menu, open:
    - 'Project.ipynb' notebook 
    - make sure MySPC.py file is in the same working folder, as it contains definitions of functions used in the 'Project.ipynb'


## What does the 'Project.ipynb' do?

The 'Project.ipynb' is set up to simulate the effect of a single metal cutting process step from the manufacturing process of a fictional part with 20 independent dimensions. These parts are set up to be processed in lots of 10, each lot consisting of parts with the same nominal dimensions. First, dimensions are simulated with only inherent process variation present and then special cause variations are added. All the dimensions are generated using numpy.random package.

After all dimensions in all the parts are generated, resulting dimensions are compared against the dimensional tolerance to determine whether parts are machined correctly. Parts that are non-conformant (manufactured outside the tolerance) are marked as scrap.
Various tools are tested to check what is the best predictor for the lot consisting of scrap parts: Statistical Process Control charts (developed by me) and Process Performance Index. 
At the end, Logistic Regression is used to categorise if lots have scrapped parts or not. Logistic regression is executed using sklearn package with a help of imbalanced-learn package.