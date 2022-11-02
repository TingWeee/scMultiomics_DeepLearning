library(Seurat)
library(SeuratData)
library(cowplot)
library(dplyr)
library(readr)
# CITE-seq data with 13 antibodies 
InstallData(ds = "cbmc") 
cbmc <- LoadData(ds = "cbmc")

# remove mouse cells 
cbmc<- cbmc[, cbmc$rna_annotations !="Mouse"]

# RNA preprocessing 
DefaultAssay(cbmc) <- 'RNA' # object has 2 assays 'RNA' and 'ADT'
cbmc <- NormalizeData(cbmc) %>% FindVariableFeatures() %>% ScaleData() %>% RunPCA()

DefaultAssay(cbmc) <- 'ADT'
# Use all ADT features for dim reduction
VariableFeatures(cbmc) <- rownames(cbmc[["ADT"]])
cbmc <- NormalizeData(cbmc, normalization.method = 'CLR', margin = 2) 
cbmc <- ScaleData(cbmc) # Scale and center features in data 
# we set a dimensional reduction name to avoid overwriting
cbmc <- RunPCA(cbmc, reduction.name = 'apca')
# gave warning message - You're computing too large a percentage of total singular values, use a standard svd instead.
# this is a warning, not an error (so we can running partial SVD but computing most of the singular values)
# we can also set approx=FALSE to run standard SVD instea

# export scaled data for RNA and protein 
rna.scale <- GetAssayData(cbmc[["RNA"]], slot="scale.data")[VariableFeatures(cbmc[["RNA"]]), ]
protein.scale <- GetAssayData(cbmc[["ADT"]], slot="scale.data")
metadata <- cbmc@meta.data # in this case data is already annotated

print(dim(rna.scale)) # 2000 8009
print(dim(protein.scale)) # 10 8009

write_csv(as.DataFrame(rna.scale), gzfile("/scMultiomics_DeepLearning/Sample Datasets/GSE100866/rna_scale.csv.gz"))
write_csv(protein.scale, "/scMultiomics_DeepLearning/Sample Datasets/GSE100866/protein_scale.csv.gz")
write_csv(metadata, gzfile("Sample Datasets/GSE100866/metadata.csv"))


#write.csv(rna.scale, gzfile("WNN_autoencoder/rna_scale.csv.gz"))
#write.csv(protein.scale, gzfile("WNN_autoencoder/protein_scale.csv.gz"))
#write.csv(metadata, gzfile("WNN_autoencoder/metadata.csv.gz"))


