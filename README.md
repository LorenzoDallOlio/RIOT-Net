# RIOT-Net package



### Step 1: Preprocessing

Our pipeline requires 2 csv to start: 
 - dataset\_raw\.csv: containing the raw counts on which we will perform the analysis, in the shape of N_patients (on the rows) x N_genes (on the columns)
                      this file should contain non-negative values, 
                      patients names should end with the format "XTY" meaning that this is the Y-th time point for the patient X
                    
 - metadata\.csv: containing the metadata regarding the dataset rows (patients), 
                  this file can contain any kind of values, but there should be a column named "ID" with extact 1 to 1 matches to the dataset_raw.csv row names
                  target columns (e.g., Treatment, Response to a terapy, etc.) should contain 1 out of 2 possible string values (e.g., "Treated" or "not-Treated")
                    
The first step is set the working directory within the file and to restrict metadata only to the available patients using the "ID" coulmn, therefore its final shape is going to be (N_patients x N_covariates)

Then, genes are filtered according to the criteria: at least "low_expression_threshold" raw counts in at least "patients_threshold"% of the total patients.

At this point, the standard DESeq2 pipeline to extract Variance Stabilizing Transformed data is performed, and its result is transposed (so patients will become the columns) stored in the same repository with filename "dataset_vst.csv".
[comment]: <> (We reccommend to not use a design formula at this step (use design=~1), or better do not introduce in your design formula object any column you could be interested furthermore, such as Treatment related columns in case you want to investigate it during further analysis)





### Step2: RIOT-Net core




##### Preparing data for RIOT-Net


Load the data, properly load and order the metadata and optionally (but recommended) load a gene name reference file, this last could be used to convert gene names from specific codes to common names.



###### \"make\_discrete\_large\_colormap()\"

At this point we no longer need DESeq2 R package, and therefore we can move (and remain) into a Python3 enviroment. The first step is not mandatory, but is helpful in contexts where we could expect 20+ clusters from the analysis. In fact our first step is running the function \"make\_discrete\_large\_colormap\", thanks to which we can specify a number of distinct colors with which we would like to build a new discrete colormap.
For \<20 expected clusters scenarios, we suggest to use \"tab20\" colormap in all further cluster plots for the parameters named \"cmap\".


###### \"soft\_threshold\_selection()\"

Then the acutal RIOT-Net pipeline starts, the first part consists in creating the correlation matrix among variance-stabilized genes expression, possibly using Spearman's correlation to investigate more general trends with respect to using Pearson's.
Such a correlation matrix could be interpreted as an adjacency matrix of a gene-gene network, where each entry represents the strengh of a link between two genes expression. The key assumption of the original WGCNA approach is that this network should behave as a scale-free network, which is a network in which the degrees of the nodes follow a power-law distribution.
Therefore, crucial part of this step is also properly tuning the epsilon parameter for soft thresholding the matrix, the idea is to raise the correlation matrix to a certain power element-wise, if such power is \>1 we are consequently shrinking close to 0 most of the noisy correlations (which will be smaller in absolute value) more than most of the significant correlations, partially cleanining the adjacency matrix from the random noise.
To best tune this parameter we implemented the function \"soft\_threshold\_selection\", which will compute the correlation matrix, element-wise raise it to an integer between 1 and 20, and then for each different exponent try to fit its degree (sum over rows, or sum over columns) to a power-law distribution. The output is the best soft threshold candidate and, optionally, the soft thresholded correlation matrix and two plots: one showing the best fitting exponent degrees distribution fit, and one showing the R^2 score for each fit. 
To correctly run the analysis we need to input a value for the fit\_threshold parameter, which is the minimum value we accept as a good R^2 fit score. 
The last part is to decide the criteria to select the best soft threshold candidate, in order to not use too high values (that would end up cancelling also large part of the signal, a.k.a. significant correlations) some would suggest to use the smallest value whose fit reaches an R^2 score above the parameter fit\_threshold.


##### Clusters formation

A Dimensionality reduction (UMAP specificially) is applied to (1-soft_threshold_correlation_matrix) (in order to have a measure behaving like a metric, with 0 for identical elements on 1 for the furthest) using the "_precomputed_" option in metrics.

Afterwards, HDBSCAN (highly performant on lower dimensional spaces) is runned on the lower dimensional embedding outputted by UMAP, HDBSCAN should be tuned by varying the cluster min size and min samples parameters at least, until visual inspection of clusters in the lower dimensional space seems good enough.


# Step 3: find_drug_targeted_genes
At this point, RIOT-Net requires the third step, and R script to find the drug targeted genes (by scraping online freely accessible database DRUG-bank) and their interactors (by scraping online freely accesible database STRING).


# Step 4: tests_and_plots
RIOT Net then concludes with python script 4 by testing which clusters are significantly enriched in drug-targeted genes, and then performing plots and biological processes enrichment tests using the freely accessible APIs of PANTHER-db.