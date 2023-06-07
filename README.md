# moo_nsgaII
Multi-objective Optimization using NSGA- II


select the best features using NSGA-II for two datasets from the provided list. Your objectives will be minimizing the number of features and classification error.
Control Parameter Settings
Np=100, Cr=0.9, pm=0.01 (uniform crossover)
D = total number of features
Size of test set: 30% of entire data
Range of variables: binary (0: not-selected, 1: selected)
Number of runs: 15
Termination condition: Max_NFC=100*Np
Classifier: k-NN (k = 5)

1) Classification error using all features.
2) Plot of initial Pareto front (initial population) vs final Pareto front for one of the runs.
3) Minimum classification error (MCE) on test and training sets using the selected features from the resultant solutions on the Pareto front.
2 | P a g e
4) Number of features associated with MCE, i.e., number of features selected by the solution with minimum classification error on test and training sets.
5) The solution with minimum classification error on test (i.e., indices of selected feature).
6) Average HV over 15 runs.
7) You need to submit a PDF file as a report and your codes. Make sure you include a “Conclusion & Discussion” section written by you.
