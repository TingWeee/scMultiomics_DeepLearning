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
from sknetwork.clustering import Louvain

# Constants
data_directory = 'Sample data/first'
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

		if 'gdo' in i:
			gdo_data_path = i
			print('GDO data found!')
			path_dicts['gdo']= gdo_data_path

	return path_dicts


# This reads in the data using the dictionary provided in get_paths_dict
# it returns meta_data, protein, rna and the combined protein rna data
def read_data(data_directory, transpose = True, set_index = False, use_template_metadata = (False, 'celltype.l2')):
	path_dicts = get_paths_dict(data_directory)
	print()
	print('Reading in the data!')

	# use Pandas to read the data
	# The index column refers to the barcode of each individual cell that
	# They performed scRNA-seq on
	# The transpose function used here is to so that 1 row = 1 cell
	# rather than 1 column = 1 cell. I rather this shape than the other shape
	# in pandas anw.

	loaded_meta = False

	for key in path_dicts.keys():
		if key == 'meta':
			meta_data = pd.read_csv(path_dicts[key], index_col = 0)
			print('Loaded metadata')
			loaded_meta = True
		if key == 'protein':
			pro = pd.read_csv(path_dicts[key], index_col = 0)
			print('Loaded protein data')
		if key == 'rna':
			rna = pd.read_csv(path_dicts[key], index_col = 0)
			print('Loaded rna data')
	
	if transpose:
		pro = pro.T
		rna = rna.T
	if use_template_metadata[0] or not loaded_meta:
		print('Meta_data not found in the path, supplying a template metadata')
		# Code to come up with a placeholder metadata
		meta_data = pd.DataFrame({use_template_metadata[1]:['placeholder' for x in range(len(pro.index))]}).set_index(pro.index)


	if set_index != False:
		rna.set_index(set_index, inplace = True)
		pro.set_index(set_index, inplace = True)
		meta_data.set_index(set_index, inplace = True)
	# This is just to check if all the barcodes in the rna dataset
	# is found in the protein barcode.
	assert all(rna.index == pro.index), 'Make sure rna index matches with protein index'

	# if they are, we can combine them column-wise.
	# so that we have single-celled data for both RNA counts + protein counts
	cite_seq_data = pd.concat([rna, pro], axis = 1)
	return meta_data, pro, rna, cite_seq_data

# This puts labels to our targets.
def compile_data(data_directory, cell_type_col, transpose = True, set_index = False, use_template_metadata = False):
	# Convert cell annotation to integers
	meta_data, pro, rna, cite_seq_data = read_data(data_directory, transpose = transpose, set_index = set_index, use_template_metadata = (use_template_metadata, cell_type_col))

	meta_data['celltype'] = meta_data[cell_type_col].str.split().str[0]

	labels_encoder = preprocessing.LabelEncoder()
	labels_encoder.fit(meta_data['celltype'])
	labels = labels_encoder.transform(meta_data['celltype'])
	data_with_targets = cite_seq_data.copy()
	data_with_targets['cell_int'] = labels
	return meta_data, pro, rna, cite_seq_data, labels_encoder, labels, data_with_targets


def generate_training(data_with_targets, pro, gene_only = False, random_state = 0, train_size = 0.4):
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
	inputs = layers.Input(shape = (input_shape[1],), name = 'gene_input_layer')
	# essentially, the number of genes I'm passing in, so that we can compress properly
	feat_dim = input_shape[1]
	# The first fully connected layer to connect out inputs to is making the feature smaller by floor dividing by 4
	# Followed by a BatchNormalization layer that allows every layer of the network to do learning more independently
	encoding = layers.Dense(feat_dim//division_rate, activation = activation, name = 'gene_encoder_1')(inputs)
	encoding = layers.BatchNormalization(name = 'BatchNormGeneEncode1')(encoding)
	div_rate = division_rate
	i = 0
	for i in range(N_hidden-1):
		# The next layer is making the feature smaller by floor dividing by 4
		div_rate = div_rate*division_rate
		encoding = layers.Dense(feat_dim//div_rate, activation = activation, name = f'gene_encoder_{i+2}')(encoding)
		encoding = layers.BatchNormalization(name = f'BatchNormGeneEncode{i+2}')(encoding)
	# Bottleneck layer
	encoding = layers.Dense(encoding_dim, activation = activation, name = f'EmbeddingDimGene')(encoding)
	encoding = layers.BatchNormalization(name = f'EmbeddingDimGene{i+3}')(encoding)
	# Start of the Decoding layer
	decoding = layers.Dense(feat_dim//div_rate, activation = activation, name = f'gene_decoder_1')(encoding)
	decoding = layers.BatchNormalization(name = f'BatchNormGeneDecode1')(decoding)
	for i in range(N_hidden-1):
		div_rate = div_rate//division_rate
		decoding = layers.Dense(feat_dim//div_rate, activation = activation, name = f'gene_decoder_{i+2}')(decoding)
		decoding = layers.BatchNormalization(name = f'BatchNormGene{i+2}')(decoding)
	# Remember to map back to our initial dimensions of feat_dim
	decoding = layers.Dense(feat_dim, activation = activation, name = f'gene_decoder_{i+3}')(decoding)
	autoencoder = Model(inputs, decoding)
	encoder = Model(inputs, encoding)
	return autoencoder, encoder

# This builds the entire thing
def gene_only_encoder(train_data, test_data, encoding_dim, saved_model_dir_name, name, N_hidden = 2, division_rate = 4, actvn = 'sigmoid', epochs = 15, override = False):
	if not os.path.exists(f'saved_models/{saved_model_dir_name}/{name}_NHL{N_hidden}_DIV{division_rate}_EPOCHS{epochs}_EncodingDim{encoding_dim}_auto') or override:
		autoencoder, encoder = build_autoencoder((train_data.shape), encoding_dim = encoding_dim, 
												 N_hidden = N_hidden, division_rate = division_rate, 
												 actvn = actvn)
		# Compile using the common optimizer of adam, the loss has to be mean_squared_error
		autoencoder.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')
		encoder.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'mean_squared_error')
		# Fitting the model, epochs =  50 
		history = autoencoder.fit(train_data, train_data, epochs = epochs, validation_data=(test_data, test_data))
		# Save the model
		autoencoder.save(f'saved_models/{saved_model_dir_name}/{name}_NHL{N_hidden}_DIV{division_rate}_EPOCHS{epochs}_EncodingDim{encoding_dim}_auto')
		encoder.save(f'saved_models/{saved_model_dir_name}/{name}_NHL{N_hidden}_DIV{division_rate}_EPOCHS{epochs}_EncodingDim{encoding_dim}_encoder')
		tf.keras.utils.plot_model(autoencoder,to_file=f'saved_models/{saved_model_dir_name}/autodecoder_geneonly.png',show_shapes=True,show_layer_names=True,dpi=150,show_layer_activations=True)
		tf.keras.utils.plot_model(encoder,to_file=f'saved_models/{saved_model_dir_name}/Encoder_geneonly.png',show_shapes=True,show_layer_names=True,dpi=150,show_layer_activations=True)
		
		return history, autoencoder, encoder
	else:
		print('MODEL ALREADY EXISTS, TO RETRAIN, SET PARAM "override = True"')
		autoencoder = tf.keras.models.load_model(f'saved_models/{saved_model_dir_name}/{name}_NHL{N_hidden}_DIV{division_rate}_EPOCHS{epochs}_EncodingDim{encoding_dim}_auto')
		encoder = tf.keras.models.load_model(f'saved_models/{saved_model_dir_name}/{name}_NHL{N_hidden}_DIV{division_rate}_EPOCHS{epochs}_EncodingDim{encoding_dim}_encoder')
		tf.keras.utils.plot_model(autoencoder,to_file=f'saved_models/{saved_model_dir_name}/autodecoder_geneonly.png',show_shapes=True,show_layer_names=True,dpi=150,show_layer_activations=True)
		tf.keras.utils.plot_model(encoder,to_file=f'saved_models/{saved_model_dir_name}/Encoder_geneonly.png',show_shapes=True,show_layer_names=True,dpi=150,show_layer_activations=True)
		
		return '', autoencoder, encoder
# Like previously, I'm asking for the shapes of my genes and protein data

def build_all_encoder(protein_shape, gene_shape, embedding_dim, actvn = 'sigmoid',
					  N_hidden_gene = 2, N_hidden_protein = 1, division_rate = 4):
	activation = actvn
	# Make the layers for my gene encoder
	# This is to get the gene inputs
	gene_inputs = layers.Input(shape = (gene_shape[1],), name = 'Gene_Input_Layer')
	gene_feat_dim = gene_shape[1]
	# Layer architecture
	'''
	If 2,000 genes the following is seen
	
	input    2nd     3rd
	2000 --> 500 --> 125
	'''
	gene_encoding = layers.Dense(gene_feat_dim//division_rate, activation = activation, name = 'gene_encoder_1')(gene_inputs)
	gene_encoding = layers.BatchNormalization(name = 'BatchNormGeneEncode1')(gene_encoding)
	div_rate_g = division_rate
	i = 0
	for i in range(N_hidden_gene-1):
		div_rate_g = div_rate_g * division_rate 
		gene_encoding = layers.Dense(gene_feat_dim//div_rate_g, activation = activation, name = f'gene_encoder_{i+2}')(gene_encoding)
		gene_encoding = layers.BatchNormalization(name=f'BatchNormGeneEncode{i+2}')(gene_encoding)
	# Make the layers for my protein encoder
	# This is to get the protein inputs
	pro_inputs = layers.Input(shape = (protein_shape[1],), name = 'Protein_Input_Layer')
	pro_feat_dim = protein_shape[1]
	
	# Layer architecture
	'''
	if 20 proteins, the following is seen:
	
	input  1st
	20 --> 5
	'''
	protein_encoding = layers.Dense(pro_feat_dim//division_rate, activation = activation, name = 'protein_encoder_1')(pro_inputs)
	protein_encoding = layers.BatchNormalization(name = 'BatchNormProteinEncode1')(protein_encoding)
	div_rate_p = division_rate
	for i in range(N_hidden_protein-1):
		div_rate_p = div_rate_p * division_rate
		protein_encoding = layers.Dense(pro_feat_dim//div_rate_p, activation = activation, name = f'protein_encoder_{i+2}')(protein_encoding)
		protein_encoding = layers.BatchNormalization(name = f'BatchNormProteinEncode{i+2}')(protein_encoding)
	# Merge both gene and protein encoders so that I can pass them into a smaller layer
	merged = layers.Concatenate(name = 'ConcatenateLayer')([gene_encoding, protein_encoding])
	# This is the "bottleneck layer". It will attempt to extract important features from protein and gene
	merged = layers.Dense(embedding_dim, activation = activation, name = 'EmbeddingDimDense')(merged)
	# Make the gene decoder separately, follow the reverse of gene encoder architecture
	gene_decoder = layers.Dense(gene_feat_dim//div_rate_g, activation = activation, name = 'gene_decoder_1')(merged)
	gene_decoder = layers.BatchNormalization(name='BatchNormGeneDecode1')(gene_decoder)
	for i in range(N_hidden_gene-1):
		div_rate_g = div_rate_g//division_rate
		gene_decoder = layers.Dense(gene_feat_dim//div_rate_g, activation = activation, name = f'gene_decoder_{i+2}')(gene_decoder)
		gene_decoder = layers.BatchNormalization(name = f'BatchNormGeneDecode{i+2}')(gene_decoder)
	gene_decoder = layers.Dense(gene_feat_dim, activation = activation, name = 'gene_decoder_last')(gene_decoder)
	# Make the protien decoder separately, follow the reverse of protein encoder architecture
	protein_decoder = layers.Dense(pro_feat_dim//div_rate_p, activation=activation, name = 'protein_decoder_1')(merged)
	protein_decoder = layers.BatchNormalization(name = 'BatchNormProteinDecode1')(protein_decoder)
	for i in range(N_hidden_protein-1):
		div_rate_p = div_rate_p//division_rate
		protein_decoder = layers.Dense(pro_feat_dim//division_rate, activation = activation, name = f'protein_decoder_{i+2}')(protein_decoder)
		protein_decoder = layers.BatchNormalization(name = f'BatchNormProteinDecode{i+2}')(protein_decoder)
	protein_decoder = layers.Dense(pro_feat_dim, activation = activation, name = f'protein_decoder_last')(protein_decoder)
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
def gene_protein_encoder(pro_train_data, gene_train_data, pro_test_data, gene_test_data,encoding_dim, saved_model_dir_name, name, N_hidden_gene = 2, N_hidden_protein = 1, division_rate = 4, actvn = 'sigmoid', epochs = 15, override = False):
	if not os.path.exists(f'saved_models/{saved_model_dir_name}/{name}_NHG{N_hidden_gene}_NHP{N_hidden_protein}_DIV{division_rate}_EPOCHS{epochs}_EncodingDim{encoding_dim}_auto') or override:
		merged, autodecoder = build_all_encoder((pro_train_data.shape), (gene_train_data.shape), embedding_dim = encoding_dim,
												 N_hidden_gene = N_hidden_gene, N_hidden_protein = N_hidden_protein,
												 division_rate = division_rate, actvn = actvn)
		autodecoder.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'mean_squared_error')
		merged.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'mean_squared_error')
		history = autodecoder.fit([gene_train_data,pro_train_data], [gene_train_data,pro_train_data], epochs = epochs, validation_data=([gene_test_data,pro_test_data], [gene_test_data,pro_test_data]))

		# Save the model
		autodecoder.save(f'saved_models/{saved_model_dir_name}/{name}_NHG{N_hidden_gene}_NHP{N_hidden_protein}_DIV{division_rate}_EPOCHS{epochs}_EncodingDim{encoding_dim}_auto')
		merged.save(f'saved_models/{saved_model_dir_name}/{name}_NHG{N_hidden_gene}_NHP{N_hidden_protein}_DIV{division_rate}_EPOCHS{epochs}_EncodingDim{encoding_dim}_merged')
		tf.keras.utils.plot_model(autodecoder,to_file=f'saved_models/{saved_model_dir_name}/autodecoder_gp.png',show_shapes=True,show_layer_names=True,dpi=150,show_layer_activations=True)
		tf.keras.utils.plot_model(merged,to_file=f'saved_models/{saved_model_dir_name}/Encoder_gp.png',show_shapes=True,show_layer_names=True,dpi=150,show_layer_activations=True)
		return history, autodecoder, merged

	else:
		print('MODEL ALREADY EXISTS, TO RETRAIN, SET PARAM "override = True"')
		autodecoder = tf.keras.models.load_model(f'saved_models/{saved_model_dir_name}/{name}_NHG{N_hidden_gene}_NHP{N_hidden_protein}_DIV{division_rate}_EPOCHS{epochs}_EncodingDim{encoding_dim}_auto')
		merged = tf.keras.models.load_model(f'saved_models/{saved_model_dir_name}/{name}_NHG{N_hidden_gene}_NHP{N_hidden_protein}_DIV{division_rate}_EPOCHS{epochs}_EncodingDim{encoding_dim}_merged')
		tf.keras.utils.plot_model(autodecoder,to_file=f'saved_models/{saved_model_dir_name}/autodecoder_gp.png',show_shapes=True,show_layer_names=True,dpi=150,show_layer_activations=True)
		tf.keras.utils.plot_model(merged,to_file=f'saved_models/{saved_model_dir_name}/Encoder_gp.png',show_shapes=True,show_layer_names=True,dpi=150,show_layer_activations=True)
		return '', autodecoder, merged

###############
# START BLOCK #
###############
# This begins the start of implementing custom n-autoencoders

# Generalise the Encoder Layer Building
# Make it return stuff I need later for concat, decoding and model building
def build_custom_autoencoders(concatenated_shapes, saved_model_dir_name, train_data_lst,epochs = 15, override=  False,
								n_hidden_layers = (2,1), division_rate = 4, actvn = 'sigmoid', embedding_dim = 64):
	# Check if this is legal
	if len(concatenated_shapes) != len(n_hidden_layers):
		print(f'Length of concatenated_shapes {len(concatenated_shapes)} do not match the length of n_hidden_layers {len(n_hidden_layers)}')
		print('You need to supply n_hidden_layers such that each element corresponds to each data')
		return None

	# concatenated_shapes would be a list of shapes regarding how many n-omics.
	# eg concatenated_shapes = [gene_train.shape, pro_train.shape]
	activation = actvn

	# Store everything sequentially
	input_layers_list = []
	n_feat_dim_list = []
	global_node_layers = []
	encoding_model_list = []
	autoencoder_model_list = []

	# This is for encoding #
	for index, data_shape in enumerate(concatenated_shapes):
		inputs = layers.Input(shape = (data_shape[1],))
		n_feat_dim = data_shape[1]

		input_layers_list.append(inputs)
		n_feat_dim_list.append(n_feat_dim)

		node_layers = [n_feat_dim//division_rate]
		# Initialize the first dense layer
		encoding = layers.Dense(n_feat_dim//division_rate, activation = activation)(inputs)
		encoding = layers.BatchNormalization()(encoding)

		div_rate_g = division_rate

		for i in range(n_hidden_layers[index]-1):
			div_rate_g = div_rate_g*division_rate
			encoding = layers.Dense(n_feat_dim//div_rate_g, activation = activation)(encoding)
			encoding = layers.BatchNormalization()(encoding)
			node_layers.append(n_feat_dim//div_rate_g)

		global_node_layers.append(node_layers)
		encoding_model_list.append(encoding)

	# This is for concatenation and merging #
	merged = layers.Concatenate()(encoding_model_list)
	merged = layers.Dense(embedding_dim, activation = activation)(merged)

	# This is for decoder #
	for index in range(len(concatenated_shapes)):
		node_layers = global_node_layers[index]
		if node_layers[0] > node_layers[-1]:
			node_layers.reverse()


		decoder = layers.Dense(node_layers[0], activation = activation)(merged)
		decoder = layers.BatchNormalization()(decoder)
		for nodes in node_layers[1:]:
			decoder = layers.Dense(nodes, activation = activation)(decoder)
			decoder = layers.BatchNormalization()(decoder)
		decoder = layers.Dense(n_feat_dim_list[index], activation = activation)(decoder)

		autoencoder_model_list.append(decoder)


	if not os.path.exists(f'saved_models/{saved_model_dir_name}/custom_N-{len(input_layers_list)}-EPOCHS{epochs}_auto') or override:
		autodecoder_model = Model(input_layers_list, outputs = autoencoder_model_list)
		merged_m = Model(input_layers_list, outputs = merged)
		autodecoder_model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'mean_squared_error')
		merged_m.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'mean_squared_error')

		history = autodecoder_model.fit(train_data_lst, train_data_lst, epochs = epochs)
		# Save the model
		autodecoder_model.save(f'saved_models/{saved_model_dir_name}/custom_N-{len(input_layers_list)}-EPOCHS{epochs}_auto')
		merged_m.save(f'saved_models/{saved_model_dir_name}/custom_N-{len(input_layers_list)}-EPOCHS{epochs}_merged')
		tf.keras.utils.plot_model(autodecoder_model,to_file=f'saved_models/{saved_model_dir_name}/autodecoder_custom.png',show_shapes=True,show_layer_names=True,dpi=150,show_layer_activations=True)
		tf.keras.utils.plot_model(merged_m,to_file=f'saved_models/{saved_model_dir_name}/Encoder_custom.png',show_shapes=True,show_layer_names=True,dpi=150,show_layer_activations=True)
		return history, autodecoder_model, merged_m

	else:
		print('MODEL ALREADY EXISTS, TO RETRAIN, SET PARAM "override = True"')
		autodecoder = tf.keras.models.load_model(f'saved_models/{saved_model_dir_name}/custom_N-{len(input_layers_list)}-EPOCHS{epochs}_auto')
		merged_m = tf.keras.models.load_model(f'saved_models/{saved_model_dir_name}/custom_N-{len(input_layers_list)}-EPOCHS{epochs}_merged')

		tf.keras.utils.plot_model(autodecoder,to_file=f'saved_models/{saved_model_dir_name}/autodecoder_custom.png',show_shapes=True,show_layer_names=True,dpi=150,show_layer_activations=True)
		tf.keras.utils.plot_model(merged_m,to_file=f'saved_models/{saved_model_dir_name}/Encoder_custom.png',show_shapes=True,show_layer_names=True,dpi=150,show_layer_activations=True)
		return '', autodecoder, merged_m


# A fancier function of build_custom_autoencoders which saves and loads models
# Because I don't want to train the model everytime
def save_load_build_custom_autoencoders(inputs_list, decoder_list, merged, train_data_lst,saved_model_dir_name, epochs = 15, override=  False):
	if not os.path.exists(f'saved_models/{saved_model_dir_name}/custom_N-{len(inputs_list)}-EPOCHS{epochs}_auto') or override:
		autodecoder = Model(inputs_list, outputs = decoder_list)
		merged_m = Model(inputs_list, merged)
		autodecoder.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'mean_squared_error')
		merged_m.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'mean_squared_error')
		history = autodecoder.fit(train_data_lst, train_data_lst, epochs = epochs)

		# Save the model
		autodecoder.save(f'saved_models/{saved_model_dir_name}/custom_N-{len(inputs_list)}-EPOCHS{epochs}_auto')
		merged_m.save(f'saved_models/{saved_model_dir_name}/custom_N-{len(inputs_list)}-EPOCHS{epochs}_merged')
		return history, autodecoder, merged_m

	else:
		print('MODEL ALREADY EXISTS, TO RETRAIN, SET PARAM "override = True"')
		autodecoder = tf.keras.models.load_model(f'saved_models/{saved_model_dir_name}/custom_N-{len(inputs_list)}-EPOCHS{epochs}_auto')
		merged_m = tf.keras.models.load_model(f'saved_models/{saved_model_dir_name}/custom_N-{len(inputs_list)}-EPOCHS{epochs}_merged')

		return '', autodecoder, merged_m

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

def rgb_to_hex(rgb):
	return '#%02x%02x%02x' % rgb

	
def generate_color(labels_encoder, labels):
	color = np.linspace(0,360, len(labels_encoder.inverse_transform(np.unique(labels))))
	color = np.concatenate(((color/360).reshape(-1,1), np.array([[0.8,0.9]]*len(labels_encoder.inverse_transform(np.unique(labels))))), axis = 1)
	color = np.around(hsv_to_rgb(color)*255,0).astype(int).tolist()

	for i in range(len(color)):
		temp = rgb_to_hex(tuple(color[i]))
		
		color[i] = temp

	return color


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


	above_mean_left = (left_df.mean(numeric_only=True) + (3*left_df.std(numeric_only=True)))
	below_mean_left = (left_df.mean(numeric_only=True) - (3*left_df.std(numeric_only=True)))
	above_mean_right = (right_df.mean(numeric_only=True) + (3*right_df.std(numeric_only=True)))
	below_mean_right = (right_df.mean(numeric_only=True) - (3*right_df.std(numeric_only=True)))

	ax[0].set_xlim(below_mean_left['x'], above_mean_left['x'])
	ax[1].set_xlim(below_mean_right['x'], above_mean_right['x'])

	ax[0].set_ylim(below_mean_left['y'], above_mean_left['y'])
	ax[1].set_ylim(below_mean_right['y'], above_mean_right['y'])

	ax[1].set_title(right_label)
	ax[0].set_title(left_label)
	ax[1].legend(bbox_to_anchor=(1.04, 1), loc="upper left")
	ax[0].legend().set_visible(False)
	plt.tight_layout()


	if not os.path.exists(f'anim/'):
		os.makedirs('anim/')
	if not os.path.exists(f'anim/{spacer}/'):
		os.makedirs(f'anim/{spacer}/')

	plt.savefig(f'anim/{spacer}/2d_plot.png', dpi = 150)
	plt.show()


def hyper_tune_models(pro_train_data,gene_train_data,pro_test_data, gene_test_data,NHG, NHP, div_rates, encoding_dims,saved_model_dir_name, name,epochs = 7):
	val_loss_list = []
	for N_HIDDEN_GENE in NHG:
		for N_HIDDEN_PROTEIN in NHP:
			for div_rate in div_rates:
				for ENCODING_DIM in encoding_dims:
					history, autodecoder, merged = gene_protein_encoder(pro_train_data,gene_train_data,pro_test_data, gene_test_data, ENCODING_DIM, f'{saved_model_dir_name}_hypertune', name,
																		N_hidden_gene = N_HIDDEN_GENE, N_hidden_protein = N_HIDDEN_PROTEIN, division_rate = div_rate, epochs=epochs)
					#print(f'N_HIDDEN_GENE: {N_HIDDEN_GENE}, N_HIDDEN_PROTEIN: {N_HIDDEN_PROTEIN}, div_rate: {div_rate}, ENCODING_DIM: {ENCODING_DIM}')
					x= np.array(history.history["val_loss"]) < np.array(history.history['loss'])
					y = np.array(history.history["val_loss"])
					
					dct = {'epoch':0, 'val_loss':0, 'NHG':0, 'NHP':0, 'divrate':0, 'encoding_dim':0}
					
					dct['epoch'] = np.argmin(y[x])
					dct['val_loss'] = np.min(y[x])
					dct['NHG'] = N_HIDDEN_GENE
					dct['NHP'] = N_HIDDEN_PROTEIN
					dct['divrate'] = div_rate
					dct['encoding_dim'] = ENCODING_DIM

					val_loss_list.append(dct)


	min_loss = 10000
	indices = 0
	for index, model in enumerate(val_loss_list):
		if min_loss > model['val_loss']:
			min_loss = model['val_loss']
			indices = index

	print(f"""
		Ideal Conditions are:
		Number of Gene Layers: {val_loss_list[indices]['NHG']}
		Number of Protein Layers: {val_loss_list[indices]['NHP']}
		division rate: {val_loss_list[indices]['divrate']}
		Encoding Dimensions: {val_loss_list[indices]['encoding_dim']}""")
		
	return val_loss_list[indices]['epoch'], val_loss_list[indices]['NHG'], val_loss_list[indices]['NHP'], val_loss_list[indices]['divrate'], val_loss_list[indices]['encoding_dim']

def comparison_cluster(test_unencoded, test_encoded, test_labels, N_predict = 2000):
	logisticRegressor_uncoded = LogisticRegression(max_iter = 400)
	logisticRegressor_uncoded.fit(test_unencoded, test_labels[:N_predict])
	score_uncoded = logisticRegressor_uncoded.score(test_unencoded, test_labels[:N_predict])

	logisticRegressor_encoded = LogisticRegression(max_iter = 400)
	logisticRegressor_encoded.fit(test_encoded, test_labels[:N_predict])
	score_coded = logisticRegressor_encoded.score(test_encoded, test_labels[:N_predict])

	print(f'Clustering Score of "first-arg data": {score_uncoded}\nClustering Score of "second-arg data": {score_coded}')
	return score_uncoded, score_coded

def find_clusters(data, n_neighbours = 20):
	knn_graph = sklearn.neighbors.kneighbors_graph(data, n_neighbours)
	louvain = Louvain()
	labels = louvain.fit_transform(knn_graph)

	return labels

def plot_custom_labels(data, labels, spacer, other_spacer = '',x_axis = 0, y_axis = 1, s = 3):
	df = pd.DataFrame(data)
	df['labels'] = labels
	df = df.astype({'labels':str})

	df = df.rename({0:x_axis, 1:y_axis}, axis = 1)

	sns.scatterplot(x = x_axis, y = y_axis, data = df, hue = 'labels', s = s)


	x_mean = df.loc[:,x_axis].mean()
	x_std = df.loc[:, x_axis].std()
	y_mean = df.loc[:,y_axis].mean()
	y_std = df.loc[:,y_axis].std()

	plt.legend(bbox_to_anchor=(1.04,1), loc = 'upper left')
	plt.xlim(x_mean - (3*x_std), x_mean + (3*x_std))
	plt.ylim(y_mean - (3*y_std), y_mean + (3*y_std))
	plt.tight_layout()
	plt.savefig(f'anim/{spacer}/custom_labels_plot_{other_spacer}.png', dpi = 150)
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
