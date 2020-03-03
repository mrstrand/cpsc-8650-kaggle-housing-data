## CPSC 8650 Data Mining: Kaggle Housing Data
This project was created for CPSC 8650, Data Mining at Clemson University. The goal of the project is to accurately predict the sale price of the houses in the test set, after training a knn model with the training set.  
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/

## Tech/framework used
- Python

## Timeline  TODO 
- [x] Jan 30th - Project Proposal Due  
- [ ] Feb 3rd - Finish reading machine learning tutorials  
- [ ] Feb 10th - Determine which models we will use  
- [ ] Feb 17th - Turn data into numeric values that we can actually use  
- [ ] Feb 24th - Visualizing the data, determining which attributes correlate with sales price. Choose at least 2 machine learning models to test  
- [ ] Feb 27th - Intermediate Project Report Due  
- [ ] March 2 - Complete predictions using one model and submit to Kaggle  
- [ ] March 9 - Complete predictions using another model and submit to Kaggle  
- [ ] March 16 - Initial Draft of report - Decide if we should use another model  
- [ ] March 23 - Revision 1 of report - All models that we plan to report on completed  
- [ ] March 30 - Solid, well written and thought out report completed  
- [ ] Apr 16th - Final Report Due  

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
