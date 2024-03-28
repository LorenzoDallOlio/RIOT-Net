################ ##
## Said el Bouhaddani
## 15 dec 2022
## Retrieve drug targets and their interactors in string-db
################ ##

library(tidyverse)
library(magrittr)
library(rvest)


# if SPARQL is not available download SPARQL_1.16.tar.gz from https://cran.r-project.org/src/contrib/Archive/SPARQL/
# and run: install.packages('Path/to/SPARQL_1.16.tar.gz', repos = NULL, type="source")


## YOU CAN ALSO COPY PASTE THE GENES FOR THE DRUGS AND
##   DIRECTLY GO TO STRING-DB
# devtools::install_github("interstellar-Consultation-Services/dbdataset")
# install.packages("dbparser")
library("dbdataset")
library("dbparser")

# BiocManager::install("STRINGdb")
library("STRINGdb")

## https://www.disgenet.org/static/disgenet2r/disgenet2r.html
# devtools::install_bitbucket("ibi_group/disgenet2r")
library(disgenet2r)

## Initialize string-db with 700 threshold
if(!exists("string_db")){
  string_db <- STRINGdb$new(version="11.5", species=9606, 
                            score_threshold=700, input_directory="")
}

## Searches for BOTH gene name and enzyme. 
## Returns a list per drug with a vector of gene names
refDrugs <- c("Leflunomide", "Hydroxychloroquine")  # change it with the drugs you are interest into
working.dir <- "./" # change it with the directory you want to save the results



refGenes <- sapply(refDrugs, function(e) {
  cat("\n Looking up ",e,"... \n", sep="")
  DBDrug <- dbdataset::Drugs %>% filter(name == e)
  if(nrow(DBDrug) > 0){
    DBhtml <- read_html(paste0("https://www.drugbank.ca/drugs/", DBDrug$primary_key))
    DBtext <- DBhtml %>% html_nodes(css='div.col-sm-12.col-lg-7') %>% html_text
    DBgene.loc <- DBtext %>% lapply(function(ee) ee %>% str_locate(c("Uniprot ", "Gene Name")) %>% diag %>% `[`(c(2,1)))
    if(length(DBgene.loc)>0) {
      DBgene <- sapply(1:length(DBgene.loc), function(ii) DBtext[ii] %>% str_sub(DBgene.loc[[ii]][1]+1, DBgene.loc[[ii]][2]-1))
      cat("  \U2714 -- Gene(s) found: ***", DBgene, "***"); return(DBgene)}
    else {cat("  \U274C -- Drug found, but gene NOT FOUND\n"); return(NA)}
  } else {cat("  \U274C -- DRUG NOT FOUND\n"); return(NA)}
}) %>% na.omit %>% c

sapply(seq_along(refGenes), 
       function(ii) {
         if(anyNA(refGenes[[ii]])) 
           cat(paste("\U274C -- NA found at position",
                     which(is.na(refGenes[[ii]])), "in DRUG", ii,
                     names(refGenes)[ii],"\n"))
         }) %>% invisible

## If you want to get rid of NAs. 
## Alternative: replace NA  manually
# examples
# refGenes <- sapply(refGenes, function(e) {if(anyNA(e)) e[-which(is.na(e))] else e })
# refGenes[[2]][8] <- "P19652"
# refGenes[[2]] %<>% c("P02763")

## Extract drug targets/enzymes and map to string-ID
## Returns a vector of neighbors/interactors for all the drug targets/enzymes
## The vector includes the drug targets/enzymes as well
## Gene SYMBOLs are returned
## NOTE: first time looking up something in string takes some time
tophits_nball <- lapply(seq_along(refGenes), function(ii) {
  refdrug <- names(refGenes[ii])
  print(paste0(ii, "/",length(refGenes),"  ", refdrug))
  refgene <- na.omit(refGenes[[ii]])
  str_id <- string_db$mp(refgene)
  str_id.nb <- string_db$get_neighbors(str_id)
  print(paste0("Nr of interactors:", length(str_id.nb)))
  gene.nb <- string_db$get_proteins() %>% tibble %>% 
    filter(protein_external_id %in% c(str_id, str_id.nb)) %>% 
    pull(preferred_name) %>% unique
  return(gene.nb)
})
names(tophits_nball) <- names(refGenes)


for (i in names(tophits_nball)){
  write_csv(data.frame(genes=c(tophits_nball[i])), paste0(working.dir, "/interesting_genes_", i,".csv" ) )
