setwd('D:\\Uni notes\\Y4S1\\ZB4171\\Prog\\cite_seq\\data\\new')
library(Seurat)
library(dplyr)

rna = ReadMtx(mtx = 'GSM5008737_RNA_3P-matrix.mtx.gz', 
		features = 'GSM5008737_RNA_3P-features.tsv.gz',
		cells = 'GSM5008737_RNA_3P-barcodes.tsv.gz')

rna <- CreateSeuratObject(counts = rna)
gse164378 = rna


adt = ReadMtx(mtx = 'GSM5008738_ADT_3P-matrix.mtx.gz', 
		features = 'GSM5008738_ADT_3P-features.tsv.gz',
		cells = 'GSM5008738_ADT_3P-barcodes.tsv.gz')
adt_assay <- CreateAssayObject(counts = adt)

hto = ReadMtx(mtx = 'GSM5008739_HTO_3P-matrix.mtx.gz', 
		features = 'GSM5008739_HTO_3P-features.tsv.gz',
		cells = 'GSM5008739_HTO_3P-barcodes.tsv.gz')
hto_assay = CreateAssayObject(counts = hto)


gse164378[['ADT']] = adt_assay
gse164378[['HTO']] = hto_assay
Assays(gse164378)

DefaultAssay(gse164378) = 'RNA'
gse164378 = NormalizeData(gse164378) %>% FindVariableFeatures() %>% ScaleData() %>% RunPCA()

DefaultAssay(gse164378) <- 'ADT'
VariableFeatures(gse164378) <- rownames(gse164378[["ADT"]])
gse164378 = NormalizeData(gse164378) %>% FindVariableFeatures() %>% ScaleData() %>% RunPCA(reduction.name = 'apca')

DefaultAssay(gse164378) <- 'HTO'
VariableFeatures(gse164378) <- rownames(gse164378[["HTO"]])
gse164378 = NormalizeData(gse164378) %>% FindVariableFeatures() %>% ScaleData() %>% RunPCA(reduction.name = 'hpca')

gse164378<- FindMultiModalNeighbors(
  object= gse164378, reduction.list = list("pca", "apca", "hpca"), 
  dims.list = list(1:30, 1:30, 1:20))


rna.scale <- GetAssayData(gse164378[["RNA"]], slot="scale.data")[VariableFeatures(gse164378[["RNA"]]), ]
protein.scale <- GetAssayData(gse164378[["ADT"]], slot="scale.data")
hto.scale <- GetAssayData(gse164378[["HTO"]], slot="scale.data")

dirr = 'D:\\Uni notes\\Y4S1\\ZB4171\\Prog\\cite_seq\\data\\new\\'

write.csv(rna.scale, gzfile(paste(dirr,"rna_scale.csv.gz", sep = '')))
write.csv(protein.scale, gzfile(paste(dirr,"protein_scale.csv.gz", sep = '')))
write.csv(hto.scale , gzfile(paste(dirr,"hto_scale.csv.gz", sep = '')))