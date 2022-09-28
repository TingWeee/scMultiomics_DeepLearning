
InstallData("thp1.eccite")

data("thp1.eccite")

citation('thp1.eccite.SeuratData')

ce =  LoadData(ds = "thp1.eccite")
ce

# RNA preprocessing
DefaultAssay(ce) <- 'RNA' #ce has 3 assays 'RNA' and 'ADT' and 'HTO' and 'GDO'

library(dplyr)
ce<- NormalizeData(ce) %>% FindVariableFeatures() %>% ScaleData() %>% RunPCA()


# ADT preprocessing 
DefaultAssay(ce) <- 'ADT'
# we will use all ADT features for dimensional reduction 
VariableFeatures(ce) <- rownames(ce[["ADT"]])
ce <- NormalizeData(ce, normalization.method = 'CLR', margin = 2) 
ce<- ScaleData(ce) # Scale and center features in data 
# we set a dimensional reduction name to avoid overwriting
ce <- RunPCA(ce, reduction.name = 'apca')

# HTO preprocessing 
DefaultAssay(ce) <- 'HTO'
# we will use all ADT features for dimensional reduction 
VariableFeatures(ce) <- rownames(ce[["HTO"]])
ce <- NormalizeData(ce, normalization.method = 'CLR', margin = 2) 
ce<- ScaleData(ce) # Scale and center features in data 
# we set a dimensional reduction name to avoid overwriting
ce <- RunPCA(ce, reduction.name = 'hpca')

# GDO preprocessing 
DefaultAssay(ce) <- 'GDO'
# we will use all ADT features for dimensional reduction 
VariableFeatures(ce) <- rownames(ce[["GDO"]])
ce <- NormalizeData(ce, normalization.method = 'CLR', margin = 2) 
ce<- ScaleData(ce) # Scale and center features in data 
# we set a dimensional reduction name to avoid overwriting
ce <- RunPCA(ce, reduction.name = 'gpca')

ce <- FindMultiModalNeighbors(
  object= ce , reduction.list = list("pca", "apca", "hpca", "gpca"), 
  dims.list = list(1:30, 1:3, 1:11, 1:30))


rna.scale <- GetAssayData(ce[["RNA"]], slot="scale.data")[VariableFeatures(ce[["RNA"]]), ]
protein.scale <- GetAssayData(ce[["ADT"]], slot="scale.data")
hto.scale <- GetAssayData(ce[["HTO"]], slot="scale.data")
gdo.scale <- GetAssayData(ce[["GDO"]], slot="scale.data")

metadata <- ce@meta.data # in this case data is already annotated

dirr = 'D:/NUS-SPS github/scMultiomics_DeepLearning/data/thp1/'
write.csv(rna.scale, gzfile(paste(dirr,"rna_scale.csv.gz", sep = '')))
write.csv(protein.scale, gzfile(paste(dirr,"protein_scale.csv.gz", sep = '')))
write.csv(metadata, gzfile(paste(dirr,"metadata.csv.gz", sep = '')))

write.csv(hto.scale, gzfile(paste(dirr,"hto_scale.csv.gz", sep = '')))
write.csv(gdo.scale, gzfile(paste(dirr,"gdo_scale.csv.gz", sep = '')))

