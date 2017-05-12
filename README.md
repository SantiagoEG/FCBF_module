# FCBF_module
Fast Correlation-Based Feature Selection

This module implements FCBF [1] and FCBF# [2] algorithms in order to perform Feature Selection in Machine Learning problems. Additionally, this module implements a novel version of FCBF algorithm (FCBFiP). The new version includes a design parameter that allow the user to control the ratio between the intercorrelation among features contained in the resulting subset and the execution time. Note that if the parameter npieces is small, the intercorrelation penalty increases and the execution time too. Meanwhile, a larger value for the parameter npieces reduces the intercorrelation penalty in the resulting subset and speeds up the algorithm execution. 

Also note that npieces must be a divisor of the size of the original dataset. In short, if we want to get a feature subset from dataset with 70 variables, we could set npieces = 2, 5, 7, 10, 14 or 35. If the original size of the dataset is prime, for example 71, the algorithm detects this case automatically and reduces the set to 70 variables by removing the worst feature. Thereby, we could set npieces = 2, 5, 7, 10, 14 or 35. 

References:

[1] L. Yu and H. Liu. Feature Selection for High‐Dimensional Data: A Fast Correlation‐Based Filter Solution. In Proceedings of The Twentieth International Conference on Machine Leaning (ICML‐03), 856‐863, Washington, D.C., August 21‐24, 2003.

[2] B. Senliol, G. Gulgezen, et al. Fast Correlation Based Filter (FCBF) with a Different Search Strategy. In Computer and Information Sciences (ISCIS ‘08) 23rd International Symposium on, pages 1‐4. Istanbul, October 27‐29, 2008.



# Example of use

To have look to one usage example see "test.py".





