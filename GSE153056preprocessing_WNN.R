library(Seurat)
library(dplyr)
library(readr)
adt = read.csv(file = 'GSM4633606_CITE_ADT_counts.tsv.gz', sep = '\t', header = TRUE, row.names = 1)
rna = read.csv(file = 'GSM4633605_CITE_cDNA_counts.tsv.gz', sep = '\t', header = TRUE, row.names = 1)

head(rna)

thp1 = CreateSeuratObject(counts = rna)
Assays(thp1)
adt_assay <- CreateAssayObject(counts = adt)
thp1[["ADT"]] <- adt_assay


rownames(thp1[["ADT"]])


# RNA preprocessing
DefaultAssay(thp1) <- 'RNA' #ce has 3 assays 'RNA' and 'ADT' and 'HTO' and 'GDO'

thp1<- NormalizeData(thp1) %>% FindVariableFeatures() %>% ScaleData()

DefaultAssay(thp1) <- 'ADT'
thp1<- NormalizeData(thp1) %>% FindVariableFeatures() %>% ScaleData()

rna.scale <- GetAssayData(thp1[["RNA"]], slot="scale.data")[VariableFeatures(thp1[["RNA"]]), ]
protein.scale <- GetAssayData(thp1[["ADT"]], slot="scale.data")

dirr = ''
write.csv(rna.scale, gzfile(paste(dirr,"rna_scale.csv.gz", sep = '')))
write.csv(protein.scale, gzfile(paste(dirr,"protein_scale.csv.gz", sep = '')))

