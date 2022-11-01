library(Seurat)
library(readr)
library(dplyr)
memory.limit(size=56000)
library(SeuratDisk)
reference_file = 'D:/NUS-SPS github/scMultiomics_DeepLearning/data/PBMC_Reference/pbmc_multimodal.h5seurat'
reference <- LoadH5Seurat(reference_file)
setwd('D:/NUS-SPS github/scMultiomics_DeepLearning/data/GSE166489')


#############
# GSE166489 #
#############



#############
# GSM5073055 #
#############

data_dir = 'MIS-C Severe/GSM5073055'
list.files(data_dir) # Should show barcodes.tsv.gz, features.tsv.gz, and matrix.mtx.gz
data <- Read10X(data.dir = data_dir)
seurat_object = CreateSeuratObject(counts = data$`Gene Expression`)
seurat_object[['ADT']] = CreateAssayObject(counts = data$`Antibody Capture`)
DefaultAssay(seurat_object) <- 'RNA'
seurat_object<- NormalizeData(seurat_object) %>% FindVariableFeatures() %>% ScaleData()

DefaultAssay(seurat_object) <- 'ADT'
seurat_object<- NormalizeData(seurat_object) %>% FindVariableFeatures() %>% ScaleData()

rna.scale <- GetAssayData(seurat_object[["RNA"]], slot="scale.data")[VariableFeatures(seurat_object[["RNA"]]), ]
protein.scale <- GetAssayData(seurat_object[["ADT"]], slot="scale.data")

write.csv(rna.scale, gzfile(paste(data_dir,"rna_scale.csv.gz", sep = '/')))
write.csv(protein.scale, gzfile(paste(data_dir,"protein_scale.csv.gz", sep = '/')))



#####################
# REFERENCE MAPPING #
#####################
#remotes::install_github("mojaveazure/seurat-disk")

seurat_object <- SCTransform(seurat_object, verbose = FALSE)
  

anchors <- FindTransferAnchors(
  reference = reference,
  query = seurat_object,
  normalization.method = "SCT",
  reference.reduction = "spca"
)


seurat_object <- MapQuery(
  anchorset = anchors,
  query = seurat_object,
  reference = reference,
  refdata = list(
    celltype.l1 = "celltype.l1",
    celltype.l2 = "celltype.l2",
    predicted_ADT = "ADT"
  ),
  reference.reduction = "spca", 
  reduction.model = "wnn.umap"
)


write.csv(seurat_object@meta.data, gzfile(paste(data_dir,"metadata.csv.gz", sep = '/')))




#############
# GSM5073056 #
#############

data_dir = 'MIS-C Severe/GSM5073056'
list.files(data_dir) # Should show barcodes.tsv.gz, features.tsv.gz, and matrix.mtx.gz
data <- Read10X(data.dir = data_dir)
seurat_object = CreateSeuratObject(counts = data$`Gene Expression`)
seurat_object[['ADT']] = CreateAssayObject(counts = data$`Antibody Capture`)
DefaultAssay(seurat_object) <- 'RNA'
seurat_object<- NormalizeData(seurat_object) %>% FindVariableFeatures() %>% ScaleData()

DefaultAssay(seurat_object) <- 'ADT'
seurat_object<- NormalizeData(seurat_object) %>% FindVariableFeatures() %>% ScaleData()

rna.scale <- GetAssayData(seurat_object[["RNA"]], slot="scale.data")[VariableFeatures(seurat_object[["RNA"]]), ]
protein.scale <- GetAssayData(seurat_object[["ADT"]], slot="scale.data")

write.csv(rna.scale, gzfile(paste(data_dir,"rna_scale.csv.gz", sep = '/')))
write.csv(protein.scale, gzfile(paste(data_dir,"protein_scale.csv.gz", sep = '/')))



#####################
# REFERENCE MAPPING #
#####################
#remotes::install_github("mojaveazure/seurat-disk")

seurat_object <- SCTransform(seurat_object, verbose = FALSE)


anchors <- FindTransferAnchors(
  reference = reference,
  query = seurat_object,
  normalization.method = "SCT",
  reference.reduction = "spca"
)


seurat_object <- MapQuery(
  anchorset = anchors,
  query = seurat_object,
  reference = reference,
  refdata = list(
    celltype.l1 = "celltype.l1",
    celltype.l2 = "celltype.l2",
    predicted_ADT = "ADT"
  ),
  reference.reduction = "spca", 
  reduction.model = "wnn.umap"
)


write.csv(seurat_object@meta.data, gzfile(paste(data_dir,"metadata.csv.gz", sep = '/')))



#############
# GSM5073070 #
#############

data_dir = 'Normal/GSM5073070'
list.files(data_dir) # Should show barcodes.tsv.gz, features.tsv.gz, and matrix.mtx.gz
data <- Read10X(data.dir = data_dir)
seurat_object = CreateSeuratObject(counts = data$`Gene Expression`)
seurat_object[['ADT']] = CreateAssayObject(counts = data$`Antibody Capture`)
DefaultAssay(seurat_object) <- 'RNA'
seurat_object<- NormalizeData(seurat_object) %>% FindVariableFeatures() %>% ScaleData()

DefaultAssay(seurat_object) <- 'ADT'
seurat_object<- NormalizeData(seurat_object) %>% FindVariableFeatures() %>% ScaleData()

rna.scale <- GetAssayData(seurat_object[["RNA"]], slot="scale.data")[VariableFeatures(seurat_object[["RNA"]]), ]
protein.scale <- GetAssayData(seurat_object[["ADT"]], slot="scale.data")

write.csv(rna.scale, gzfile(paste(data_dir,"rna_scale.csv.gz", sep = '/')))
write.csv(protein.scale, gzfile(paste(data_dir,"protein_scale.csv.gz", sep = '/')))



#####################
# REFERENCE MAPPING #
#####################
#remotes::install_github("mojaveazure/seurat-disk")

seurat_object <- SCTransform(seurat_object, verbose = FALSE)


anchors <- FindTransferAnchors(
  reference = reference,
  query = seurat_object,
  normalization.method = "SCT",
  reference.reduction = "spca"
)


seurat_object <- MapQuery(
  anchorset = anchors,
  query = seurat_object,
  reference = reference,
  refdata = list(
    celltype.l1 = "celltype.l1",
    celltype.l2 = "celltype.l2",
    predicted_ADT = "ADT"
  ),
  reference.reduction = "spca", 
  reduction.model = "wnn.umap"
)


write.csv(seurat_object@meta.data, gzfile(paste(data_dir,"metadata.csv.gz", sep = '/')))

#############
# GSM5073071 #
#############

data_dir = 'Normal/GSM5073071'
list.files(data_dir) # Should show barcodes.tsv.gz, features.tsv.gz, and matrix.mtx.gz
data <- Read10X(data.dir = data_dir)
seurat_object = CreateSeuratObject(counts = data$`Gene Expression`)
seurat_object[['ADT']] = CreateAssayObject(counts = data$`Antibody Capture`)
DefaultAssay(seurat_object) <- 'RNA'
seurat_object<- NormalizeData(seurat_object) %>% FindVariableFeatures() %>% ScaleData()

DefaultAssay(seurat_object) <- 'ADT'
seurat_object<- NormalizeData(seurat_object) %>% FindVariableFeatures() %>% ScaleData()

rna.scale <- GetAssayData(seurat_object[["RNA"]], slot="scale.data")[VariableFeatures(seurat_object[["RNA"]]), ]
protein.scale <- GetAssayData(seurat_object[["ADT"]], slot="scale.data")

write.csv(rna.scale, gzfile(paste(data_dir,"rna_scale.csv.gz", sep = '/')))
write.csv(protein.scale, gzfile(paste(data_dir,"protein_scale.csv.gz", sep = '/')))



#####################
# REFERENCE MAPPING #
#####################
#remotes::install_github("mojaveazure/seurat-disk")

seurat_object <- SCTransform(seurat_object, verbose = FALSE)


anchors <- FindTransferAnchors(
  reference = reference,
  query = seurat_object,
  normalization.method = "SCT",
  reference.reduction = "spca"
)


seurat_object <- MapQuery(
  anchorset = anchors,
  query = seurat_object,
  reference = reference,
  refdata = list(
    celltype.l1 = "celltype.l1",
    celltype.l2 = "celltype.l2",
    predicted_ADT = "ADT"
  ),
  reference.reduction = "spca", 
  reduction.model = "wnn.umap"
)


write.csv(seurat_object@meta.data, gzfile(paste(data_dir,"metadata.csv.gz", sep = '/')))

#############
# GSM5073072 #
#############

data_dir = 'Normal/GSM5073072'
list.files(data_dir) # Should show barcodes.tsv.gz, features.tsv.gz, and matrix.mtx.gz
data <- Read10X(data.dir = data_dir)
seurat_object = CreateSeuratObject(counts = data$`Gene Expression`)
seurat_object[['ADT']] = CreateAssayObject(counts = data$`Antibody Capture`)
DefaultAssay(seurat_object) <- 'RNA'
seurat_object<- NormalizeData(seurat_object) %>% FindVariableFeatures() %>% ScaleData()

DefaultAssay(seurat_object) <- 'ADT'
seurat_object<- NormalizeData(seurat_object) %>% FindVariableFeatures() %>% ScaleData()

rna.scale <- GetAssayData(seurat_object[["RNA"]], slot="scale.data")[VariableFeatures(seurat_object[["RNA"]]), ]
protein.scale <- GetAssayData(seurat_object[["ADT"]], slot="scale.data")

write.csv(rna.scale, gzfile(paste(data_dir,"rna_scale.csv.gz", sep = '/')))
write.csv(protein.scale, gzfile(paste(data_dir,"protein_scale.csv.gz", sep = '/')))



#####################
# REFERENCE MAPPING #
#####################
#remotes::install_github("mojaveazure/seurat-disk")

seurat_object <- SCTransform(seurat_object, verbose = FALSE)


anchors <- FindTransferAnchors(
  reference = reference,
  query = seurat_object,
  normalization.method = "SCT",
  reference.reduction = "spca"
)


seurat_object <- MapQuery(
  anchorset = anchors,
  query = seurat_object,
  reference = reference,
  refdata = list(
    celltype.l1 = "celltype.l1",
    celltype.l2 = "celltype.l2",
    predicted_ADT = "ADT"
  ),
  reference.reduction = "spca", 
  reduction.model = "wnn.umap"
)


write.csv(seurat_object@meta.data, gzfile(paste(data_dir,"metadata.csv.gz", sep = '/')))

