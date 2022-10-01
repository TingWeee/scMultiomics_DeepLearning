# scMultiomics_DeepLearning
### Integrative analysis using Autoencoders
The emergence of single-cell multimodal omics enabled multiple molecular programs to be simultaneously measured in individual cells at unprecedented resolution. However, analysis of sc-multimodal omics data is challenging due to lack of methods (?) that can accurately integrate across multiple data modalities. Here, we present Deep-omics, an approach for integrative analysis using Autoencoders. 

## Sample Datasets
### GSE128639 
Human bone marrow mononuclear cells - CITE-seq (Stuart et al., 2019)
### GSE100866
CBMC (cord blood mononuclear cells) CITE-seq (Linderman et al., 2022)
### GSE153056
Human ECCITE-seq (Papalexi et al., 2021) Raw and processed sequencing data is available through the Gene Expression Omnibus (GEO accession number: GSE153056). Processed data is also available through [SeuratData](https://github.com/satijalab/seurat-data). To facilitate access with a single command: InstallData(ds = “thp1.eccite”)
### GSE164378
Human PBMC - CITE-seq, ECITE-seq (Hao et al., 2021)
Dataset contains two batches and cells in both batches were annotated to 31 cell types. Batch 1 contains 67k cells (11k RNA, 228 ADT) and batch 2 contains 94k cells (12k RNA, 228 ADT) 
### GSE166489
PBMC CITE-seq (Ramaswamy et al.,2021) with 189 surface antibody phenotypes. Of the 38 samples under GSE166489, 5 included CITE-seq data (2 MIC-C patients and 3 healthy donors)  
### E-MTAB-10026 
PBMC CITE-seq dataset (Stephenson et al., 2021 ) from healthy individuals and COVID-19 patients. 
### Human PBMC-CITE-seq (Kotliarov et., 2020) 
CITE-seq profiling of 82 surface proteins and transcriptomes of 53,201 single cells from healthy high and low influenza-vaccination responders. Dataset can be downloaded from [here](https://nih.figshare.com/collections/Data_and_software_code_repository_for_Broad_immune_activation_underlies_shared_set_point_signatures_for_vaccine_responsiveness_in_healthy_individuals_and_disease_activity_in_patients_with_lupus_Kotliarov_Y_Sparks_R_et_al_Nat_Med_DOI_https_d/4753772)
