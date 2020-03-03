## CPSC 8650 Data Mining: Kaggle Housing Data
This project was created for CPSC 8650, Data Mining at Clemson University. The goal of the project is to accurately predict the sale price of the houses in the test set, after training a knn model with the training set.

## Tech/framework used
- Python

## Installation
1. Install Python 3.8.2
2. Run `scripts\new-environment-setup`. This will create a virtual environment containing only the dependencies of this project.

## Using environments.txt
1. Any time a dependency is added (or removed), we need to update environments.txt. There is a script  `scripts\requirements-text-generate` (untested) that will update the file.
2. Whenever you pull, it might be necessary to run `scripts\update-from-environments-txt`. This will install any new dependencies.

# Git Aliases
I cannot even work without these. I am way too dependent
```
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.st status
```