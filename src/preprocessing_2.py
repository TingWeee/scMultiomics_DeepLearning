# Input would be the path FOLDER of the stuff
import glob
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import os
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import tensorflow as tf
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from sklearn.linear_model import LogisticRegression
import sklearn.neighbors
import seaborn as sns
import umap
from sklearn.decomposition import PCA
import dill
# from sknetwork.clustering import Louvain

def get_path(data_directory):
        print(f'Reading Data in {data_directory}/\n')
        path_dicts = {}
        for path in glob.glob(f'{data_directory}/*'):
                if 'meta' in path:
                        path_dicts['meta']=path
                if 'protein' in path:
                        path_dicts['protein']= path
                if 'rna' in path:
                        path_dicts['rna']= path
        print("Found",(", ".join(path_dicts.keys())), "data!") 
        return path_dicts



def load_data(data_directory, transpose = True, set_index = False):
        path_dicts = get_path(data_directory)
        
        pro = pd.read_csv(path_dicts['protein'], index_col = 0)
        rna = pd.read_csv(path_dicts['rna'], index_col = 0)

        if transpose: pro, rna = pro.T, rna.T

        if 'meta' in path_dicts:
                meta_data = pd.read_csv(path_dicts['meta'], index_col = 0)
                meta_data['cell_barcode'] = meta_data.index # to preserve cell barcode in encoding later 
        else:
                #meta_data = pd.DataFrame(rna.index, columns =['cell_barcode'])
                # Changed this meta_data generation code to give me an additional metadata to playt with.
                meta_data = pd.DataFrame({use_template_metadata[1]:['placeholder' for x in range(len(rna.index))]}).set_index(rna.index)
        if set_index:
                rna.set_index(set_index, inplace = True)
                pro.set_index(set_index, inplace = True)
                meta_data.set_index(set_index, inplace = True)

        # check if barcodes of rna dataset match protein dataset
        assert all(rna.index == pro.index), 'Make sure rna index matches with protein index'

        cite_data = pd.concat([rna, pro], axis = 1)
        
        return meta_data, pro, rna, cite_data


def split_cite(data, pro):
        n_pro = pro.shape[1] # number of proteins
        return data.iloc[:,:-n_pro], data.iloc[:,-n_pro:]  #gene_data, pro_data

def splitDataset(cite_data, random_state =0, train_size = 0.4):
        train_cite, test_cite = train_test_split(cite_data, random_state = random_state, train_size = train_size)
        gene_train_cite, pro_train_cite = split_cite(train_cite, pro)
        gene_test_cite, pro_test_cite = split_cite(test_cite, pro)
        return train_cite, test_cite, gene_train_cite, pro_train_cite, gene_test_cite, pro_test_cite


def get_predictionUMAP(original_dataset, bottleneck):
        # original_dataset = [rna,pro] or rna
        whole_predicted = bottleneck.predict(original_dataset)
        reducer = umap.UMAP()
        whole_encoded = reducer.fit_transform(whole_predicted)
        return whole_encoded


def get_plot(whole_encoded, metadata):
        plot_df = metadata.copy()
        plot_df["UMAP1"] = whole_encoded[:, 0]
        plot_df["UMAP2"] = whole_encoded[:, 1]
        return plot_df

def get_score(whole_encoded,metadata, referenceCol, log_max_iter = 400):
        logisticRegressor = LogisticRegression(max_iter = log_max_iter)
        logisticRegressor.fit(whole_encoded, metadata[referenceCol])
        score = logisticRegressor.score(whole_encoded, metadata[referenceCol])
        return score

class IndivData:
    def __init__(self, bottleneck, umap_encoding, umap_df, score):
        self.umap_encoding = umap_encoding
        self.score = score
        self.umap_df = umap_df
        self.bottleneck = bottleneck

def makeObj(original_dataset, bottleneck, metadata, referenceCol, log_max_iter = 400):
        if bottleneck == None:
                reducer = umap.UMAP()
                encoded = reducer.fit_transform(original_dataset)
        else:
            encoded = get_predictionUMAP(original_dataset, bottleneck)
            
        umap_df = get_plot(encoded, metadata)
        score = get_score(encoded, metadata, referenceCol, log_max_iter)
        return IndivData(bottleneck, encoded, umap_df, score)

def PCAobj(data, metadata, referenceCol, log_max_iter = 400):
        pca = PCA(n_components=10, svd_solver = 'auto')
        principal_components=pca.fit_transform(data)
        reducer = umap.UMAP()
        encoded = reducer.fit_transform(principal_components)
        score = get_score(encoded, metadata, referenceCol, log_max_iter)
        umap_df = get_plot(encoded, metadata)
        
        return IndivData(pca, encoded, umap_df, score)
        

# plot(obj) -> gets all dictionary
class Rdata():
        pass

def generate_palette(metadata, colName, colormap="bright"):
    groups = metadata[colName].unique()
    colors = sns.color_palette(colormap, len(groups))
    palette = {group: color for group, color in zip(groups, colors)}
    return palette

def plotReloadedObj(filePath, refCol, palette=None):
        reloadedObj = dill.load(open(filePath, "rb"))
        metadata=reloadedObj.gene_protein.umap_df
        if palette == None: palette = generate_palette(metadata, refCol)
        ##
        plot_names = (list(vars(reloadedObj).keys()))

        fig, ax = plt.subplots(2,2, figsize = (14,14))
        ind = 0

        # See the same change in `plotobjs()`
        if numPlots == 1:
                ax = [ax, None]
                
        for x in plot_names:
                n = getattr(reloadedObj,x)
                sns.scatterplot(data = n.umap_df,x="UMAP1", y = "UMAP2",
                                hue = refCol,palette=palette,s = 4,ax=ax[ind]).set(title = f'{x} score:{n.score}')
                ax[ind].get_legend().remove()
                ind+=1
        print(x, n.score)
        plt.legend(loc='center right',bbox_to_anchor=legendAnchor)
        
        # plotObjs(reloadedObj, metadata,refCol, palette)
        

def plotObjs(Rdata, metadata, refCol, figWidth = 10, figHeight = 7, legendAnchor=(1.25, 0.5), palette=None):

    if palette == None: palette = generate_palette(metadata, refCol)
    
    plot_names = (list(vars(Rdata).keys()))
    numPlots = len(plot_names)
    
    fig, ax = plt.subplots(numPlots,1, figsize = (figWidth,figHeight*numPlots))
    ind = 0

    # This is to handle the case that the numPlots is 1
    # To prevent the error of "axes" is not subscriptable :D
    if numPlots == 1:
        ax = [ax, None]

    for x in plot_names:
        n = getattr(Rdata,x)
        sns.scatterplot(data = n.umap_df,x="UMAP1", y = "UMAP2", 
                        hue = refCol,palette=palette,s = 4,ax=ax[ind]).set(title = f'{x} score:{n.score}')
        ax[ind].get_legend().remove()
        ind+=1
        print(x, n.score)
        
    plt.legend(loc='center right',bbox_to_anchor=legendAnchor)

        
