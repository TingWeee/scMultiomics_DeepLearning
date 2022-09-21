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

# Constants
data_directory = 'data/first'
cell_type_col = 'celltype.l2'


# This grabs the paths only
def get_paths_dict(data_directory):
	data_src = data_directory
	print(f'Reading Data in {data_directory}/\n')
	path_dicts = {}
	for i in glob.glob(f'{data_src}/*'):
		if 'meta' in i:
			meta_data_path = i
			print('Metadata found!')
			path_dicts['meta']= meta_data_path
		if 'protein' in i:
			protein_data_path = i
			print('Protein Data Found!')
			path_dicts['protein']= protein_data_path
		if 'rna' in i:
			rna_data_path = i
			print('RNA-seq data found!')
			path_dicts['rna']= rna_data_path
	return path_dicts


# This reads in the data using the dictionary provided in get_paths_dict
# it returns meta_data, protein, rna and the combined protein rna data
def read_data(data_directory, transpose = True):
	path_dicts = get_paths_dict(data_directory)
	print()
	print('Reading in the data!')

	# use Pandas to read the data
	# The index column refers to the barcode of each individual cell that
	# They performed scRNA-seq on
	# The transpose function used here is to so that 1 row = 1 cell
	# rather than 1 column = 1 cell. I rather this shape than the other shape
	# in pandas anw.
	for key in path_dicts.keys():
		if key == 'meta':
			meta_data = pd.read_csv(path_dicts[key], index_col = 0)
			print('Loaded metadata')
		if key == 'protein':
			pro = pd.read_csv(path_dicts[key], index_col = 0)
			print('Loaded protein data')
		if key == 'rna':
			rna = pd.read_csv(path_dicts[key], index_col = 0)
			print('Loaded rna data')
	
	if transpose:
		pro = pro.T
		rna = rna.T
	# This is just to check if all the barcodes in the rna dataset
	# is found in the protein barcode.
	assert all(rna.index == pro.index), 'Make sure rna index matches with protein index'
	
	# if they are, we can combine them column-wise.
	# so that we have single-celled data for both RNA counts + protein counts
	cite_seq_data = pd.concat([rna, pro], axis = 1)
	return meta_data, pro, rna, cite_seq_data

# This puts labels to our targets.
def compile_data(data_directory, cell_type_col):
	# Convert cell annotation to integers
	meta_data, pro, rna, cite_seq_data = read_data(data_directory)

	meta_data['celltype'] = meta_data[cell_type_col].str.split().str[0]

	labels_encoder = preprocessing.LabelEncoder()
	labels_encoder.fit(meta_data['celltype'])
	labels = labels_encoder.transform(meta_data['celltype'])
	data_with_targets = cite_seq_data.copy()
	data_with_targets['cell_int'] = labels
	return meta_data, pro, rna, cite_seq_data, labels_encoder, labels, data_with_targets


def generate_training(data_with_targets, pro, gene_only = False, random_state = 0, train_size = 0.6):
	# Shuffle using train test split
	# Unbind labels so that we remove this information from the machine learning algo.
	lab = data_with_targets.iloc[:,-1]
	data = data_with_targets.iloc[:,:-1]
	if gene_only:
		data = data_with_targets.iloc[:, :-(pro.shape[1]+1)]
	train_data, test_data, train_labels, test_labels = train_test_split(data, lab, random_state = random_state, train_size = train_size)
	return train_data, test_data, train_labels, test_labels


def split_training_with_labels(train_data, test_data,pro):
	gene_train_data = train_data.iloc[:,:-pro.shape[1]]
	pro_train_data = train_data.iloc[:,-pro.shape[1]:]

	gene_test_data = test_data.iloc[:,:-pro.shape[1]]
	pro_test_data = test_data.iloc[:,-pro.shape[1]:]

	return gene_train_data,pro_train_data,gene_test_data,pro_test_data

# This function builds our autoencoder, I returned both encoder and autoencoder so that the model
# can train by minimizing loss by using the decoder, but at the same time we can extract
# the 'bottleneck' layer using the encoder portion
# The input is the input shape of teh training_data
# encoding_dim is the number of nodes we want our bottleneck to have
def build_autoencoder(input_shape, encoding_dim, N_hidden = 2, division_rate = 4, actvn = 'sigmoid'):
	'''
	If the number of genes we have are 2,000. The number of nodes are
	
	Input    1st     2nd      bottleneck      2nd     1st     output
	2000 --> 500 --> 125 --> encoding_dim --> 125 --> 500 --> 2000
	'''
	# Activation function for all our layers, we might want to explore this next time
	# for now im just putting sigmoid activation
	# Might be cool to look at tanh?
	activation = actvn
	# Init the tensor using the input shape
	inputs = layers.Input(shape = (input_shape[1],))
	# essentially, the number of genes I'm passing in, so that we can compress properly
	feat_dim = input_shape[1]
	# The first fully connected layer to connect out inputs to is making the feature smaller by floor dividing by 4
	# Followed by a BatchNormalization layer that allows every layer of the network to do learning more independently
	encoding = layers.Dense(feat_dim//division_rate, activation = activation)(inputs)
	encoding = layers.BatchNormalization()(encoding)
	div_rate = division_rate
	for i in range(N_hidden-1):
		# The next layer is making the feature smaller by floor dividing by 4
		div_rate = div_rate*division_rate
		encoding = layers.Dense(feat_dim//div_rate, activation = activation)(encoding)
		encoding = layers.BatchNormalization()(encoding)
	# Bottleneck layer
	encoding = layers.Dense(encoding_dim, activation = activation)(encoding)
	encoding = layers.BatchNormalization()(encoding)
	# Start of the Decoding layer
	decoding = layers.Dense(feat_dim//div_rate, activation = activation)(encoding)
	decoding = layers.BatchNormalization()(decoding)
	for i in range(N_hidden-1):
		div_rate = div_rate//division_rate
		decoding = layers.Dense(feat_dim//div_rate, activation = activation)(decoding)
		decoding = layers.BatchNormalization()(decoding)
	# Remember to map back to our initial dimensions of feat_dim
	decoding = layers.Dense(feat_dim, activation = activation)(decoding)
	autoencoder = Model(inputs, decoding)
	encoder = Model(inputs, encoding)
	return autoencoder, encoder

# This builds the entire thing
def gene_only_encoder(train_data, encoding_dim, saved_model_dir_name, name, N_hidden = 2, division_rate = 4, actvn = 'sigmoid', epochs = 15, override = False):
	if not os.path.exists(f'saved_models/{saved_model_dir_name}/{name}_NHL{N_hidden}_DIV{division_rate}_EPOCHS{epochs}_EncodingDim{encoding_dim}_auto') or override:
		autoencoder, encoder = build_autoencoder((train_data.shape), encoding_dim = encoding_dim, 
												 N_hidden = N_hidden, division_rate = division_rate, 
												 actvn = actvn)
		# Compile using the common optimizer of adam, the loss has to be mean_squared_error
		autoencoder.compile(optimizer='adam', loss='mean_squared_error')
		# Fitting the model, epochs =  50 
		history = autoencoder.fit(train_data, train_data, epochs = epochs)
		# Save the model
		autoencoder.save(f'saved_models/{saved_model_dir_name}/{name}_NHL{N_hidden}_DIV{division_rate}_EPOCHS{epochs}_EncodingDim{encoding_dim}_auto')
		encoder.save(f'saved_models/{saved_model_dir_name}/{name}_NHL{N_hidden}_DIV{division_rate}_EPOCHS{epochs}_EncodingDim{encoding_dim}_encoder')
		return history, autoencoder, encoder
	else:
		print('MODEL ALREADY EXISTS, TO RETRAIN, SET PARAM "override = True"')
		autoencoder = tf.keras.models.load_model(f'saved_models/{saved_model_dir_name}/{name}_NHL{N_hidden}_DIV{division_rate}_EPOCHS{epochs}_EncodingDim{encoding_dim}_auto')
		encoder = tf.keras.models.load_model(f'saved_models/{saved_model_dir_name}/{name}_NHL{N_hidden}_DIV{division_rate}_EPOCHS{epochs}_EncodingDim{encoding_dim}_encoder')

		return '', autoencoder, encoder
# Like previously, I'm asking for the shapes of my genes and protein data

def build_all_encoder(protein_shape, gene_shape, embedding_dim, actvn = 'sigmoid',
					  N_hidden_gene = 2, N_hidden_protein = 1, division_rate = 4):
	activation = actvn
	# Make the layers for my gene encoder
	# This is to get the gene inputs
	gene_inputs = layers.Input(shape = (gene_shape[1],))
	gene_feat_dim = gene_shape[1]
	# Layer architecture
	'''
	If 2,000 genes the following is seen
	
	input    2nd     3rd
	2000 --> 500 --> 125
	'''
	gene_encoding = layers.Dense(gene_feat_dim//division_rate, activation = activation)(gene_inputs)
	gene_encoding = layers.BatchNormalization()(gene_encoding)
	div_rate_g = division_rate
	for i in range(N_hidden_gene-1):
		div_rate_g = div_rate_g * division_rate 
		gene_encoding = layers.Dense(gene_feat_dim//div_rate_g, activation = activation)(gene_encoding)
		gene_encoding = layers.BatchNormalization()(gene_encoding)
	# Make the layers for my protein encoder
	# This is to get the protein inputs
	pro_inputs = layers.Input(shape = (protein_shape[1],))
	pro_feat_dim = protein_shape[1]
	
	# Layer architecture
	'''
	if 20 proteins, the following is seen:
	
	input  1st
	20 --> 5
	'''
	protein_encoding = layers.Dense(pro_feat_dim//division_rate, activation = activation)(pro_inputs)
	protein_encoding = layers.BatchNormalization()(protein_encoding)
	div_rate_p = division_rate
	for i in range(N_hidden_protein-1):
		div_rate_p = div_rate_p * division_rate
		protein_encoding = layers.Dense(pro_feat_dim//div_rate_p, activation = activation)(protein_encoding)
		protein_encoding = layers.BatchNormalization()(protein_encoding)
	# Merge both gene and protein encoders so that I can pass them into a smaller layer
	merged = layers.Concatenate()([gene_encoding, protein_encoding])
	# This is the "bottleneck layer". It will attempt to extract important features from protein and gene
	merged = layers.Dense(embedding_dim, activation = activation)(merged)
	# Make the gene decoder separately, follow the reverse of gene encoder architecture
	gene_decoder = layers.Dense(gene_feat_dim//div_rate_g, activation = activation)(merged)
	gene_decoder = layers.BatchNormalization()(gene_decoder)
	for i in range(N_hidden_gene-1):
		div_rate_g = div_rate_g//division_rate
		gene_decoder = layers.Dense(gene_feat_dim//div_rate_g, activation = activation)(gene_decoder)
		gene_decoder = layers.BatchNormalization()(gene_decoder)
	gene_decoder = layers.Dense(gene_feat_dim, activation = activation)(gene_decoder)
	# Make the protien decoder separately, follow the reverse of protein encoder architecture
	protein_decoder = layers.Dense(pro_feat_dim//div_rate_p, activation=activation)(merged)
	protein_decoder = layers.BatchNormalization()(protein_decoder)
	for i in range(N_hidden_protein-1):
		div_rate_p = div_rate_p//division_rate
		protein_decoder = layers.Dense(pro_feat_dim//division_rate, activation = activation)(protein_decoder)
	protein_decoder = layers.Dense(pro_feat_dim, activation = activation)(protein_decoder)
	# "Compile" the model. Specify that it has gene and protein as input. Output as the gene and protein decoders
	# This is so that the model can compare the losses properly
	autodecoder = Model([gene_inputs, pro_inputs], outputs = [gene_decoder,protein_decoder])
	# Extract out the embedded model too :)
	merged = Model([gene_inputs, pro_inputs], merged)
	return merged, autodecoder
'''
def build_all_encoder(protein_shape, gene_shape, embedding_dim, actvn = 'sigmoid',
					  N_hidden_gene = 2, N_hidden_protein = 1, division_rate = 4):
'''
def gene_protein_encoder(pro_train_data, gene_train_data, encoding_dim, saved_model_dir_name, name, N_hidden_gene = 2, N_hidden_protein = 1, division_rate = 4, actvn = 'sigmoid', epochs = 15, override = False):
	if not os.path.exists(f'saved_models/{saved_model_dir_name}/{name}_NHG{N_hidden_gene}_NHP{N_hidden_protein}_DIV{division_rate}_EPOCHS{epochs}_EncodingDim{encoding_dim}_auto') or override:
		merged, autodecoder = build_all_encoder((pro_train_data.shape), (gene_train_data.shape), embedding_dim = encoding_dim,
												 N_hidden_gene = N_hidden_gene, N_hidden_protein = N_hidden_protein,
												 division_rate = division_rate, actvn = actvn)
		autodecoder.compile(optimizer = 'adam', loss = 'mean_squared_error')
		history = autodecoder.fit([gene_train_data,pro_train_data], [gene_train_data,pro_train_data], epochs = epochs)

		# Save the model
		autodecoder.save(f'saved_models/{saved_model_dir_name}/{name}_NHG{N_hidden_gene}_NHP{N_hidden_protein}_DIV{division_rate}_EPOCHS{epochs}_EncodingDim{encoding_dim}_auto')
		merged.save(f'saved_models/{saved_model_dir_name}/{name}_NHG{N_hidden_gene}_NHP{N_hidden_protein}_DIV{division_rate}_EPOCHS{epochs}_EncodingDim{encoding_dim}_merged')
		return history, autodecoder, merged

	else:
		print('MODEL ALREADY EXISTS, TO RETRAIN, SET PARAM "override = True"')
		autodecoder = tf.keras.models.load_model(f'saved_models/{saved_model_dir_name}/{name}_NHG{N_hidden_gene}_NHP{N_hidden_protein}_DIV{division_rate}_EPOCHS{epochs}_EncodingDim{encoding_dim}_auto')
		merged = tf.keras.models.load_model(f'saved_models/{saved_model_dir_name}/{name}_NHG{N_hidden_gene}_NHP{N_hidden_protein}_DIV{division_rate}_EPOCHS{epochs}_EncodingDim{encoding_dim}_merged')

		return '', autodecoder, merged

def load_gene_only_coders(encoding_dim, saved_model_dir_name, name, N_hidden = 2, division_rate = 4, actvn = 'sigmoid', epochs = 15):
	autoencoder = tf.keras.models.load_model(f'saved_models/{saved_model_dir_name}/{name}_NHL{N_hidden}_DIV{division_rate}_EPOCHS{epochs}_EncodingDim{encoding_dim}_auto')
	encoder = tf.keras.models.load_model(f'saved_models/{saved_model_dir_name}/{name}_NHL{N_hidden}_DIV{division_rate}_EPOCHS{epochs}_EncodingDim{encoding_dim}_encoder')
	return autoencoder, encoder

# This generates a color map for each unique label
# You do need to specify a cmap tho
def generate_colormap(colors, labels_encoder, labels):
	color_map = {}
	for key, values in enumerate(labels_encoder.inverse_transform(np.unique(labels))):
		color_map[values] = colors[key]
	return color_map

def vis_data2d(left_data_TSNE, right_data_TSNE, train_labels, labels_encoder, color_map, N_predict, left_label = 'Encoded TSNE', right_label = 'uncoded TSNE', spacer = 'geneOnly'):
	'''	
	left/right_data_TSNE comes from generating the following code, run it yourself using TSNE
	# Perform the TSNE on the bottleneck layer of the encoded data and the non encoded data
	N_predict = 2000
	# Make the encoder do its job. We can now store this as an output to a var
	training_predicted = encoder.predict(train_data[:N_predict])
	# Perform TSNE on 2 components so we can visualise it.
	left_data_TSNE = TSNE(n_components = 2, init = 'pca', learning_rate = 'auto',random_state = 0).fit_transform(training_predicted)

	# right_data_TSNE here is a sample, of not pushing it through the autoencoder
	right_data_TSNE = TSNE(n_components = 2, init = 'pca', learning_rate = 'auto',random_state = 0).fit_transform(train_data[:N_predict])'''

	# Change the encoded data to a pandas df because I like it
	left_df = pd.DataFrame(left_data_TSNE)
	# Tag the target
	left_df['Target'] = labels_encoder.inverse_transform(train_labels[:N_predict])
	# Rename columns for easier plotting later
	left_df = left_df.rename(columns = {0:'x', 1:'y'})

	right_df = pd.DataFrame(right_data_TSNE)
	right_df['Target'] = labels_encoder.inverse_transform(train_labels[:N_predict])
	right_df = right_df.rename(columns = {0:'x', 1:'y'})

	# Plot by Targets so that we can colour them
	fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,5))
	grouped = left_df.groupby('Target')
	for key,group in grouped:
		group.plot(ax=ax[0], kind = 'scatter', x='x', y='y', label=key, color=color_map[key], s = 2)

	grouped = right_df.groupby('Target')
	for key, group in grouped:
		group.plot(ax=ax[1], kind='scatter', x='x', y='y', label=key, color=color_map[key], s = 2)

	for axes in ax:
		axes.set_xlim(left_df.min()['x']*1.1, left_df.max()['x']*1.1)
		axes.set_ylim(left_df.min()['y']*1.1, left_df.max()['y']*1.1)

	ax[1].set_title(right_label)
	ax[0].set_title(left_label)
	ax[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")
	ax[0].legend().set_visible(False)
	plt.tight_layout()
	plt.savefig(f'anim/{spacer}/2d_plot.png', dpi = 150)
	plt.show()


'''meta_data, pro, rna, cite_seq_data, labels_encoder, labels, data_with_targets = compile_data(data_directory, cell_type_col)
train_data, test_data, train_labels, test_labels = generate_training(data_with_targets, pro)
#print(rna)
#history, autoencoder, encoder = gene_only_encoder(train_data, 64, 'ori data', 'gene_only')
autoencoder, encoder = load_gene_only_coders(64, 'ori data', 'gene_only')

# Perform the TSNE on the bottleneck layer of the encoded data and the non encoded data
N_predict = 2000

# Make the encoder do its job. We can now store this as an output to a var
training_predicted = encoder.predict(train_data[:N_predict])
# Perform TSNE on 2 components so we can visualise it.
train_encoded = TSNE(n_components = 2, init = 'pca', learning_rate = 'auto',random_state = 0).fit_transform(training_predicted)
train_unencoded = TSNE(n_components = 2, init = 'pca', learning_rate = 'auto',random_state = 0).fit_transform(train_data[:N_predict])



color_map = generate_colormap(['#7312FE','#FF9900','#36FF00','#9AFBA4','#010101','#0500FF','#00D0FF','#FF0603'], labels_encoder, labels)
'''
