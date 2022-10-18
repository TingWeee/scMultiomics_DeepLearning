# WNN analysis of CITE-seq 
library(Seurat)
library(SeuratData)
library(cowplot)
library(dplyr)
# devtools::install_github("satijalab/seurat-data", ref = 'develop')

# 30,672 scRNA-seq profiles + 25 antibodies from bone marrow
# Data contains 2 assays -> RNA + Antibody-derived tags 
InstallData("bmcite")
bm <- LoadData(ds = "bmcite")
bm_1 <-bm
# Pre-processing using standard normalization 
# but we can also change to SCTransform or other alternative methods 

# RNA preprocessing 
DefaultAssay(bm) <- 'RNA' #bm has 2 assays 'RNA' and 'ADT'
# bm <- NormalizeData(bm) %>% FindVariableFeatures() %>% ScaleData() %>% RunPCA()
bm <- NormalizeData(bm) 
bm <- FindVariableFeatures(bm)
bm <- ScaleData(bm) # Scale and center features in data 
bm <- RunPCA(bm)

# ADT preprocessing 
DefaultAssay(bm) <- 'ADT'
# we will use all ADT features for dimensional reduction 
VariableFeatures(bm) <- rownames(bm[["ADT"]])
# CD11a, CD11c, CD123, CD127-IL7Ra, CD16, CD161, CD197-CCR7, CD25, CD278-ICOS, CD3, CD45RA, CD45RO, CD56, CD57, CD79b, CD8a, HLA.DR
bm <- NormalizeData(bm, normalization.method = 'CLR', margin = 2) 
bm <- ScaleData(bm) # Scale and center features in data 
# we set a dimensional reduction name to avoid overwriting
bm <- RunPCA(bm, reduction.name = 'apca')
# gave warning message - You're computing too large a percentage of total singular values, use a standard svd instead.
# this is a warning, not an error (so we can running partial SVD but computing most of the singular values)
# we can also set approx=FALSE to run standard SVD instead

# For each cell, calculate closest neighbors in dataset base on a weighted combi of RNA and protein similarities 
# Identify multimodal neighbors. These will be stored in the neighbors slot, 
# and can be accessed using bm[['weighted.nn']]
# The WNN graph can be accessed at bm[["wknn"]], 
# and the SNN graph used for clustering at bm[["wsnn"]]
# Cell-specific modality weights can be accessed at bm$RNA.weight
bm <- FindMultiModalNeighbors(
  bm, reduction.list = list("pca", "apca"), 
  dims.list = list(1:30, 1:18), modality.weight.name = "RNA.weight"
)

# UMAP based on weighted combination of RNA and protein data 
# Graph base clustering 
bm <- RunUMAP(bm, nn.name = "weighted.nn", reduction.name = "wnn.umap", reduction.key = "wnnUMAP_")
bm <- FindClusters(bm, graph.name = "wsnn", algorithm = 3, resolution = 2, verbose = FALSE)

p1 <- DimPlot(bm, reduction = 'wnn.umap', label = TRUE, repel = TRUE, label.size = 2.5) + NoLegend()
p2 <- DimPlot(bm, reduction = 'wnn.umap', group.by = 'celltype.l2', label = TRUE, repel = TRUE, label.size = 2.5) + NoLegend()
ggsave("WNN_UMAP_bonemarrow.pdf", p1+p2)

# we can also run UMAP individually 
bm <- RunUMAP(bm, reduction = 'pca', dims = 1:30, assay = 'RNA', 
              reduction.name = 'rna.umap', reduction.key = 'rnaUMAP_')
bm <- RunUMAP(bm, reduction = 'apca', dims = 1:18, assay = 'ADT', 
              reduction.name = 'adt.umap', reduction.key = 'adtUMAP_')
p3 <- DimPlot(bm, reduction = 'rna.umap', group.by = 'celltype.l2', label = TRUE, 
              repel = TRUE, label.size = 2.5) + NoLegend()
p4 <- DimPlot(bm, reduction = 'adt.umap', group.by = 'celltype.l2', label = TRUE, 
              repel = TRUE, label.size = 2.5) + NoLegend()
ggsave("indiv_UMAP_BM.pdf", p3+p4)

# export scaled data for RNA and protein 
rna.scale <- GetAssayData(bm[["RNA"]], slot="scale.data")[VariableFeatures(bm[["RNA"]]), ]
protein.scale <- GetAssayData(bm[["ADT"]], slot="scale.data")

metadata <- bm@meta.data # in this case data is already annotated

print(dim(rna.scale)) # 2000 30672
print(dim(protein.scale)) # 25 30672

write.csv(rna.scale, "WNN_autoencoder/rna_scale.csv")
write.csv(protein.scale, "WNN_autoencoder/protein_scale.csv")
write.csv(metadata, "WNN_autoencoder/metadata.csv")

#write.csv(rna.scale, gzfile("WNN_autoencoder/rna_scale.csv.gz"))
#write.csv(protein.scale, gzfile("WNN_autoencoder/protein_scale.csv.gz"))
#write.csv(metadata, gzfile("WNN_autoencoder/metadata.csv.gz"))


