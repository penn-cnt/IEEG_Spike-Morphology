%{

Spike Morphology project

This is the codebase for the project using interictal spike EEG data to diagnose mesial temporal lobe epilepsy (mTLE) 
vs. other types. Here we describe steps to begin with a dataset containing electrode contact-level features and 
calculate mesial-to-lateral spread patterns. Additionally, code to perform univariate analysis across features 
and initializing machine learning models to predict mTLE is included.


To run the analysis, follow these steps:
1) Download the codebase from:
   https://github.com/penn-cnt/IEEG_Spike-Morphology/
2) Create an envrionment using requirements.txt or environment.yaml
3) For full pipeline - Download the datasets from:
   https://upenn.box.com/s/d45set9nrrzxf2zbx4z18hsus1ivat84

Explaining codebase:
- Univariate Analysis folder contains the pooled (2 cohorts) analysis. Code here replicates figures. (complete run in <1 min).
- Machine Learning folder contains python script for creating our logistic regression model and post-hoc analysis.
- Process Spike Data folder contains the full pipeline. Code here takes ~5 mins to run through the full dataset of spikes (can be downloaded using drop link above).
- Dataset folder contains intermediate datasets that can be used to replicate study.
- Results folder stores all outputs of the analysis.
- tools folder contains some of the basic functions that are used in the analysis.




Carlos Aguila 

October 2024 

aguilac@seas.upenn.edu

}%
