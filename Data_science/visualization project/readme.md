# PISA 2012 analysis
## by Gad Mohamed 


## Dataset

> This is the exploration notebook of the PISA2012 dataset. PISA is an education assessment dataset based on survey's on students, parents and teachers. The dataset is huge, but in this analysis, I do a lot of cleaning to focus only on the features relevant to the questions of interest.

> The original PISA2012 dataset contains 485490 records and 636 features. After removing irrelevant features and merging parents' qualifications, study time, and subjects' scores, the clean dataset has 129199 records and 9 features.


## Summary of Findings

> After cleaning the dataset and extracting only relevant features, I found:

1. Although students' scores were increasing linearly with parents' qualifications, scores start to decrease at very high qualifications. May be this is a signal that highly qualified parents are busy caring about their children.

2. A perfect linear relationship between mothers' and fathers' educational qualifications i.e. parents usually have similar educational background

3. Surprisingly, average study time of students didn't have much influence on their grades, which might suggest a problem in this dataset or the process of collecting its records.


## Key Insights for Presentation

> In the presentation, I go through the most informative plots that clearly conveys my findings. Particularly, I visualize the following relationships:

1. Parents' qualifications vs students' grades

2. Fathers' vs Mothers' educational qualifications

3. Students' grades relation with parents' qualifications and studying time ( using heatmap) 

