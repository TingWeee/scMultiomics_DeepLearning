install.packages(c("plyr", "tidyverse", "data.table", "pROC", "MASS", "limma", "mclust", "corrplot", "ggraph", "circlize", "pals", "ggsignif", "ggridges", "viridis", "clustree", "cowplot"))
install.packages("BiocManager")
BiocManager::install()
BiocManager::install(c("SingleCellExperiment", "scater", "scran", "fgsea", "tmod", "ComplexHeatmap"))

# Seurat ver. 2.3.4
source("https://z.umn.edu/archived-seurat")

# download and save the rds objects in dir and set dir to path

metadata <- readRDS(paste0(dir,"H1_day0_scranNorm_adtbatchNorm_dist_clustered_TSNE_labels.rds"))
meta_data <- metadata@meta.data
# first we need to change the "_" in the gene name to "-" 
new_rows = rownames(meta_data)
changed_rows<- sub("_","-",new_rows)
rownames(meta_data) = changed_rows
filename="/scbio7/home/tingwei/WNN_autoencoder/Kotliarov_2020/2020_meta_data.csv"
file = read.csv(filename)
old = c(file[["label"]])
new = c(file[["Celltype"]])
y = c(meta_data[["K3"]])
y[y %in% old]<- new[match(y,old,nomatch =0)]
temp_meta <- meta_data
temp_meta[['celltype']]<-y
## META DATA IS FINALLY CREATED 

# RDS files containing lists of Seurat objects with normalized and batch-corrected data for individual data 
norm_data <-readRDS(paste0(dir,"H1_day0_scranNorm_adtbatchNorm.rds"))

# RNA: 32738 x 53511 sparse Matrix of class "dgCMatrix"
new <- norm_data@data
new_cols=colnames(new)
changed_cols <- sub("_","-",new_cols)
colnames(new) = changed_cols


# ADT: 53511 cells and 87 proteins
new_adt <-norm_data@assay$CITE@data
new_cols_adt <- colnames(new_adt)
changed_cols_adt <- sub("_","-",new_cols_adt)
colnames(new_adt) = changed_cols_adt

library(Seurat)
pbmc <- CreateSeuratObject(new, assay = "RNA",
                   min.cells = 0, min.features = 0, names.field = 1,
                   names.delim = "_", meta.data = NULL)
Assays(pbmc) # we can see that it contains 1 assay 
# Create new assay to store ADT information
adt_assay <- CreateAssayObject(counts = new_adt)
# add this assay to previously created Seurat object 
pbmc[["ADT"]] <- adt_assay
# Validate that it now contains multiple assays 
Assays(pbmc)
rownames(pbmc[["ADT"]])
saveRDS(pbmc, "2020_paper_object.rds")
library(SeuratData)
library(cowplot)
library(dplyr)
bm <- pbmc
DefaultAssay(bm) <- 'RNA'
bm <- NormalizeData(bm) %>% FindVariableFeatures() %>% ScaleData() %>% RunPCA()

DefaultAssay(bm) <- 'ADT'
# we will use all ADT features for dimensional reduction
# we set a dimensional reduction name to avoid overwriting the 
bm@assays$ADT@data@x[is.na(bm@assays$ADT@data@x)] <- 0
VariableFeatures(bm) <- rownames(bm[["ADT"]])
bm<- ScaleData(bm) %>% RunPCA(reduction.name = "apca")

bm <- FindMultiModalNeighbors(
  bm, reduction.list = list("pca", "apca"), 
  dims.list = list(1:30, 1:18), modality.weight.name = "RNA.weight"
)

bm <- RunUMAP(bm, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")
bm <- FindClusters(bm, graph.name = "wsnn", algorithm = 3, resolution = 2, verbose = FALSE)
p1 <- DimPlot(bm, reduction = 'wnn.umap', label = TRUE, repel = TRUE, label.size = 2.5) + NoLegend()
saveRDS(bm, "2020_processed_object.rds")
# export scaled data for RNA and protein 
rna.scale <- GetAssayData(bm[["RNA"]], slot="scale.data")[VariableFeatures(bm[["RNA"]]), ]
protein.scale <- GetAssayData(bm[["ADT"]], slot="scale.data")
rna<- t(rna.scale)
adt <- t(protein.scale)

high_responders <- temp_meta[temp_meta$adjmfc.time =="d0 high",]
low_responders <- temp_meta[temp_meta$adjmfc.time =="d0 low",]

high_cells <- c(rownames(high_responders))
low_cells <- c(rownames(low_responders))

high_adt <- adt[adt$X %in% high_cells,]
low_adt <- adt[adt$X %in% low_cells,]

high_rna <- rna[rna$X %in% high_cells,]
low_rna <- rna[rna$X %in% low_cells,]

write.csv(high_rna,gzfile("/scbio7/home/tingwei/WNN_autoencoder/Kotliarov_2020/high_responders/2020_rna_scale.csv.gz"))
write.csv(high_adt,gzfile("/scbio7/home/tingwei/WNN_autoencoder/Kotliarov_2020/high_responders/2020_protein_scale.csv.gz"))
write.csv(high_responders,"/scbio7/home/tingwei/WNN_autoencoder/Kotliarov_2020/high_responders/2020_metadata.csv")

write.csv(low_rna,gzfile("/scbio7/home/tingwei/WNN_autoencoder/Kotliarov_2020/low_responders/2020_rna_scale.csv.gz"))
write.csv(low_adt,gzfile("/scbio7/home/tingwei/WNN_autoencoder/Kotliarov_2020/low_responders/2020_protein_scale.csv.gz"))
write.csv(low_responders,"/scbio7/home/tingwei/WNN_autoencoder/Kotliarov_2020/low_responders/2020_metadata.csv")



