################ ##
## Lorenzo Dall'Olio
## 15 dec 2022
## perform Variance Stabilizing Transformation (VST) on the raw read counts dataset
################ ##


# NOTE: NEVER use WGCNA on Differentially Expressed Genes -> you're introducing
#       a huge BIAS in the correlations. Instead, run DGE analysis and see the
#       resulting outcome on the WGCNA clusters



# SETUP ------------------------------------------------------------------------

# Attach the DESeq2 library
library(DESeq2)
# We will need this so we can use the pipe: %>%
library(magrittr)
# We'll be making some plots
library(dplyr)


# parameters for preprocessing
working.directory <- "/home/lyro/Github/RIOT-Net/"  
perform.vst.normalization = TRUE  (recommended)

# gene filtering parameters 
### to pass the filtering step, a gene needs at least "low.expression.threshold" 
### raw counts in at least "patients.threshold"% of total patients
low.expression.threshold = 10  
patients.threshold = 90#% of patients that must have at least "low.expression.threshold" raw counts


# load the data
print("loading input dataset and relative needed metadata.")
df =  read.csv(paste0(working.directory,"dataset_raw.csv"), header=TRUE, row.names=1)
metadata = read.csv(paste0(working.directory,"metadata.csv"), header=TRUE, row.names='ID')

# gene filtering
print(paste0("Performing gene filtering on input dataset. using criteria: at least '",
             low.expression.threshold, " raw counts on at least ", patients.threshold,
             "% of patients to keep gene in consideration for the further analysis'"))
genes.percentile = apply(df, 1, function(x) quantile(x, na.rm=TRUE, probs=patients.threshold*0.01)
                         > low.expression.threshold)
df = as.data.frame(dplyr::filter(df, genes.percentile))

# now select only metadata rows that correspond to a present df column (a patient)
metadata <- metadata[colnames(df),]



if (perform.vst.normalization) {

  print("Performing VST normalization on input dataset.")
  
  # Variance Stabilizing Transformation ------------------------------------------
  
  dds <- DESeqDataSetFromMatrix(countData=df, colData=metadata, design=~1)
  
  # Normalize and transform the data in the `DESeqDataSet` object using the `vst()`
  # function from the `DESEq2` R package
  dds.norm <- vst(dds)
  
  # Retrieve the normalized data from the `DESeqDataSet`
  normalized.counts <- assay(dds.norm) %>%
    t() # Transpose this data to have patients as rows and genes as columns in the stored db
  
  
  ## Save counts matrix
  write.csv(normalized.counts, file=paste0(working.directory,"dataset_vst.csv"))
}
