��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
.
Rsqrt
x"T
y"T"
Ttype:

2
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68�
�
gene_encoder_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_namegene_encoder_1/kernel
�
)gene_encoder_1/kernel/Read/ReadVariableOpReadVariableOpgene_encoder_1/kernel* 
_output_shapes
:
��*
dtype0

gene_encoder_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_namegene_encoder_1/bias
x
'gene_encoder_1/bias/Read/ReadVariableOpReadVariableOpgene_encoder_1/bias*
_output_shapes	
:�*
dtype0
�
BatchNormGeneEncode1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameBatchNormGeneEncode1/gamma
�
.BatchNormGeneEncode1/gamma/Read/ReadVariableOpReadVariableOpBatchNormGeneEncode1/gamma*
_output_shapes	
:�*
dtype0
�
BatchNormGeneEncode1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameBatchNormGeneEncode1/beta
�
-BatchNormGeneEncode1/beta/Read/ReadVariableOpReadVariableOpBatchNormGeneEncode1/beta*
_output_shapes	
:�*
dtype0
�
 BatchNormGeneEncode1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" BatchNormGeneEncode1/moving_mean
�
4BatchNormGeneEncode1/moving_mean/Read/ReadVariableOpReadVariableOp BatchNormGeneEncode1/moving_mean*
_output_shapes	
:�*
dtype0
�
$BatchNormGeneEncode1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*5
shared_name&$BatchNormGeneEncode1/moving_variance
�
8BatchNormGeneEncode1/moving_variance/Read/ReadVariableOpReadVariableOp$BatchNormGeneEncode1/moving_variance*
_output_shapes	
:�*
dtype0
�
gene_encoder_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�}*&
shared_namegene_encoder_2/kernel
�
)gene_encoder_2/kernel/Read/ReadVariableOpReadVariableOpgene_encoder_2/kernel*
_output_shapes
:	�}*
dtype0
~
gene_encoder_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*$
shared_namegene_encoder_2/bias
w
'gene_encoder_2/bias/Read/ReadVariableOpReadVariableOpgene_encoder_2/bias*
_output_shapes
:}*
dtype0
�
protein_encoder_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�9*)
shared_nameprotein_encoder_1/kernel
�
,protein_encoder_1/kernel/Read/ReadVariableOpReadVariableOpprotein_encoder_1/kernel*
_output_shapes
:	�9*
dtype0
�
protein_encoder_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*'
shared_nameprotein_encoder_1/bias
}
*protein_encoder_1/bias/Read/ReadVariableOpReadVariableOpprotein_encoder_1/bias*
_output_shapes
:9*
dtype0
�
BatchNormGeneEncode2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*+
shared_nameBatchNormGeneEncode2/gamma
�
.BatchNormGeneEncode2/gamma/Read/ReadVariableOpReadVariableOpBatchNormGeneEncode2/gamma*
_output_shapes
:}*
dtype0
�
BatchNormGeneEncode2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:}**
shared_nameBatchNormGeneEncode2/beta
�
-BatchNormGeneEncode2/beta/Read/ReadVariableOpReadVariableOpBatchNormGeneEncode2/beta*
_output_shapes
:}*
dtype0
�
 BatchNormGeneEncode2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*1
shared_name" BatchNormGeneEncode2/moving_mean
�
4BatchNormGeneEncode2/moving_mean/Read/ReadVariableOpReadVariableOp BatchNormGeneEncode2/moving_mean*
_output_shapes
:}*
dtype0
�
$BatchNormGeneEncode2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*5
shared_name&$BatchNormGeneEncode2/moving_variance
�
8BatchNormGeneEncode2/moving_variance/Read/ReadVariableOpReadVariableOp$BatchNormGeneEncode2/moving_variance*
_output_shapes
:}*
dtype0
�
BatchNormProteinEncode1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*.
shared_nameBatchNormProteinEncode1/gamma
�
1BatchNormProteinEncode1/gamma/Read/ReadVariableOpReadVariableOpBatchNormProteinEncode1/gamma*
_output_shapes
:9*
dtype0
�
BatchNormProteinEncode1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*-
shared_nameBatchNormProteinEncode1/beta
�
0BatchNormProteinEncode1/beta/Read/ReadVariableOpReadVariableOpBatchNormProteinEncode1/beta*
_output_shapes
:9*
dtype0
�
#BatchNormProteinEncode1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*4
shared_name%#BatchNormProteinEncode1/moving_mean
�
7BatchNormProteinEncode1/moving_mean/Read/ReadVariableOpReadVariableOp#BatchNormProteinEncode1/moving_mean*
_output_shapes
:9*
dtype0
�
'BatchNormProteinEncode1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*8
shared_name)'BatchNormProteinEncode1/moving_variance
�
;BatchNormProteinEncode1/moving_variance/Read/ReadVariableOpReadVariableOp'BatchNormProteinEncode1/moving_variance*
_output_shapes
:9*
dtype0
�
EmbeddingDimDense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*)
shared_nameEmbeddingDimDense/kernel
�
,EmbeddingDimDense/kernel/Read/ReadVariableOpReadVariableOpEmbeddingDimDense/kernel*
_output_shapes
:	�@*
dtype0
�
EmbeddingDimDense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameEmbeddingDimDense/bias
}
*EmbeddingDimDense/bias/Read/ReadVariableOpReadVariableOpEmbeddingDimDense/bias*
_output_shapes
:@*
dtype0
�
gene_decoder_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@}*&
shared_namegene_decoder_1/kernel

)gene_decoder_1/kernel/Read/ReadVariableOpReadVariableOpgene_decoder_1/kernel*
_output_shapes

:@}*
dtype0
~
gene_decoder_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*$
shared_namegene_decoder_1/bias
w
'gene_decoder_1/bias/Read/ReadVariableOpReadVariableOpgene_decoder_1/bias*
_output_shapes
:}*
dtype0
�
BatchNormGeneDecode1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*+
shared_nameBatchNormGeneDecode1/gamma
�
.BatchNormGeneDecode1/gamma/Read/ReadVariableOpReadVariableOpBatchNormGeneDecode1/gamma*
_output_shapes
:}*
dtype0
�
BatchNormGeneDecode1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:}**
shared_nameBatchNormGeneDecode1/beta
�
-BatchNormGeneDecode1/beta/Read/ReadVariableOpReadVariableOpBatchNormGeneDecode1/beta*
_output_shapes
:}*
dtype0
�
 BatchNormGeneDecode1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*1
shared_name" BatchNormGeneDecode1/moving_mean
�
4BatchNormGeneDecode1/moving_mean/Read/ReadVariableOpReadVariableOp BatchNormGeneDecode1/moving_mean*
_output_shapes
:}*
dtype0
�
$BatchNormGeneDecode1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*5
shared_name&$BatchNormGeneDecode1/moving_variance
�
8BatchNormGeneDecode1/moving_variance/Read/ReadVariableOpReadVariableOp$BatchNormGeneDecode1/moving_variance*
_output_shapes
:}*
dtype0
�
gene_decoder_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	}�*&
shared_namegene_decoder_2/kernel
�
)gene_decoder_2/kernel/Read/ReadVariableOpReadVariableOpgene_decoder_2/kernel*
_output_shapes
:	}�*
dtype0

gene_decoder_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_namegene_decoder_2/bias
x
'gene_decoder_2/bias/Read/ReadVariableOpReadVariableOpgene_decoder_2/bias*
_output_shapes	
:�*
dtype0
�
protein_decoder_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@9*)
shared_nameprotein_decoder_1/kernel
�
,protein_decoder_1/kernel/Read/ReadVariableOpReadVariableOpprotein_decoder_1/kernel*
_output_shapes

:@9*
dtype0
�
protein_decoder_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*'
shared_nameprotein_decoder_1/bias
}
*protein_decoder_1/bias/Read/ReadVariableOpReadVariableOpprotein_decoder_1/bias*
_output_shapes
:9*
dtype0
�
BatchNormGeneDecode2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameBatchNormGeneDecode2/gamma
�
.BatchNormGeneDecode2/gamma/Read/ReadVariableOpReadVariableOpBatchNormGeneDecode2/gamma*
_output_shapes	
:�*
dtype0
�
BatchNormGeneDecode2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameBatchNormGeneDecode2/beta
�
-BatchNormGeneDecode2/beta/Read/ReadVariableOpReadVariableOpBatchNormGeneDecode2/beta*
_output_shapes	
:�*
dtype0
�
 BatchNormGeneDecode2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" BatchNormGeneDecode2/moving_mean
�
4BatchNormGeneDecode2/moving_mean/Read/ReadVariableOpReadVariableOp BatchNormGeneDecode2/moving_mean*
_output_shapes	
:�*
dtype0
�
$BatchNormGeneDecode2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*5
shared_name&$BatchNormGeneDecode2/moving_variance
�
8BatchNormGeneDecode2/moving_variance/Read/ReadVariableOpReadVariableOp$BatchNormGeneDecode2/moving_variance*
_output_shapes	
:�*
dtype0
�
BatchNormProteinDecode1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*.
shared_nameBatchNormProteinDecode1/gamma
�
1BatchNormProteinDecode1/gamma/Read/ReadVariableOpReadVariableOpBatchNormProteinDecode1/gamma*
_output_shapes
:9*
dtype0
�
BatchNormProteinDecode1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*-
shared_nameBatchNormProteinDecode1/beta
�
0BatchNormProteinDecode1/beta/Read/ReadVariableOpReadVariableOpBatchNormProteinDecode1/beta*
_output_shapes
:9*
dtype0
�
#BatchNormProteinDecode1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*4
shared_name%#BatchNormProteinDecode1/moving_mean
�
7BatchNormProteinDecode1/moving_mean/Read/ReadVariableOpReadVariableOp#BatchNormProteinDecode1/moving_mean*
_output_shapes
:9*
dtype0
�
'BatchNormProteinDecode1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*8
shared_name)'BatchNormProteinDecode1/moving_variance
�
;BatchNormProteinDecode1/moving_variance/Read/ReadVariableOpReadVariableOp'BatchNormProteinDecode1/moving_variance*
_output_shapes
:9*
dtype0
�
gene_decoder_last/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*)
shared_namegene_decoder_last/kernel
�
,gene_decoder_last/kernel/Read/ReadVariableOpReadVariableOpgene_decoder_last/kernel* 
_output_shapes
:
��*
dtype0
�
gene_decoder_last/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*'
shared_namegene_decoder_last/bias
~
*gene_decoder_last/bias/Read/ReadVariableOpReadVariableOpgene_decoder_last/bias*
_output_shapes	
:�*
dtype0
�
protein_decoder_last/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	9�*,
shared_nameprotein_decoder_last/kernel
�
/protein_decoder_last/kernel/Read/ReadVariableOpReadVariableOpprotein_decoder_last/kernel*
_output_shapes
:	9�*
dtype0
�
protein_decoder_last/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameprotein_decoder_last/bias
�
-protein_decoder_last/bias/Read/ReadVariableOpReadVariableOpprotein_decoder_last/bias*
_output_shapes	
:�*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
�
Adam/gene_encoder_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*-
shared_nameAdam/gene_encoder_1/kernel/m
�
0Adam/gene_encoder_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gene_encoder_1/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/gene_encoder_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameAdam/gene_encoder_1/bias/m
�
.Adam/gene_encoder_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/gene_encoder_1/bias/m*
_output_shapes	
:�*
dtype0
�
!Adam/BatchNormGeneEncode1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/BatchNormGeneEncode1/gamma/m
�
5Adam/BatchNormGeneEncode1/gamma/m/Read/ReadVariableOpReadVariableOp!Adam/BatchNormGeneEncode1/gamma/m*
_output_shapes	
:�*
dtype0
�
 Adam/BatchNormGeneEncode1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/BatchNormGeneEncode1/beta/m
�
4Adam/BatchNormGeneEncode1/beta/m/Read/ReadVariableOpReadVariableOp Adam/BatchNormGeneEncode1/beta/m*
_output_shapes	
:�*
dtype0
�
Adam/gene_encoder_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�}*-
shared_nameAdam/gene_encoder_2/kernel/m
�
0Adam/gene_encoder_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gene_encoder_2/kernel/m*
_output_shapes
:	�}*
dtype0
�
Adam/gene_encoder_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*+
shared_nameAdam/gene_encoder_2/bias/m
�
.Adam/gene_encoder_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/gene_encoder_2/bias/m*
_output_shapes
:}*
dtype0
�
Adam/protein_encoder_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�9*0
shared_name!Adam/protein_encoder_1/kernel/m
�
3Adam/protein_encoder_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/protein_encoder_1/kernel/m*
_output_shapes
:	�9*
dtype0
�
Adam/protein_encoder_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*.
shared_nameAdam/protein_encoder_1/bias/m
�
1Adam/protein_encoder_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/protein_encoder_1/bias/m*
_output_shapes
:9*
dtype0
�
!Adam/BatchNormGeneEncode2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*2
shared_name#!Adam/BatchNormGeneEncode2/gamma/m
�
5Adam/BatchNormGeneEncode2/gamma/m/Read/ReadVariableOpReadVariableOp!Adam/BatchNormGeneEncode2/gamma/m*
_output_shapes
:}*
dtype0
�
 Adam/BatchNormGeneEncode2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*1
shared_name" Adam/BatchNormGeneEncode2/beta/m
�
4Adam/BatchNormGeneEncode2/beta/m/Read/ReadVariableOpReadVariableOp Adam/BatchNormGeneEncode2/beta/m*
_output_shapes
:}*
dtype0
�
$Adam/BatchNormProteinEncode1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*5
shared_name&$Adam/BatchNormProteinEncode1/gamma/m
�
8Adam/BatchNormProteinEncode1/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/BatchNormProteinEncode1/gamma/m*
_output_shapes
:9*
dtype0
�
#Adam/BatchNormProteinEncode1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*4
shared_name%#Adam/BatchNormProteinEncode1/beta/m
�
7Adam/BatchNormProteinEncode1/beta/m/Read/ReadVariableOpReadVariableOp#Adam/BatchNormProteinEncode1/beta/m*
_output_shapes
:9*
dtype0
�
Adam/EmbeddingDimDense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*0
shared_name!Adam/EmbeddingDimDense/kernel/m
�
3Adam/EmbeddingDimDense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/EmbeddingDimDense/kernel/m*
_output_shapes
:	�@*
dtype0
�
Adam/EmbeddingDimDense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/EmbeddingDimDense/bias/m
�
1Adam/EmbeddingDimDense/bias/m/Read/ReadVariableOpReadVariableOpAdam/EmbeddingDimDense/bias/m*
_output_shapes
:@*
dtype0
�
Adam/gene_decoder_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@}*-
shared_nameAdam/gene_decoder_1/kernel/m
�
0Adam/gene_decoder_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_1/kernel/m*
_output_shapes

:@}*
dtype0
�
Adam/gene_decoder_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*+
shared_nameAdam/gene_decoder_1/bias/m
�
.Adam/gene_decoder_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_1/bias/m*
_output_shapes
:}*
dtype0
�
!Adam/BatchNormGeneDecode1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*2
shared_name#!Adam/BatchNormGeneDecode1/gamma/m
�
5Adam/BatchNormGeneDecode1/gamma/m/Read/ReadVariableOpReadVariableOp!Adam/BatchNormGeneDecode1/gamma/m*
_output_shapes
:}*
dtype0
�
 Adam/BatchNormGeneDecode1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*1
shared_name" Adam/BatchNormGeneDecode1/beta/m
�
4Adam/BatchNormGeneDecode1/beta/m/Read/ReadVariableOpReadVariableOp Adam/BatchNormGeneDecode1/beta/m*
_output_shapes
:}*
dtype0
�
Adam/gene_decoder_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	}�*-
shared_nameAdam/gene_decoder_2/kernel/m
�
0Adam/gene_decoder_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_2/kernel/m*
_output_shapes
:	}�*
dtype0
�
Adam/gene_decoder_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameAdam/gene_decoder_2/bias/m
�
.Adam/gene_decoder_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_2/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/protein_decoder_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@9*0
shared_name!Adam/protein_decoder_1/kernel/m
�
3Adam/protein_decoder_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/protein_decoder_1/kernel/m*
_output_shapes

:@9*
dtype0
�
Adam/protein_decoder_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*.
shared_nameAdam/protein_decoder_1/bias/m
�
1Adam/protein_decoder_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/protein_decoder_1/bias/m*
_output_shapes
:9*
dtype0
�
!Adam/BatchNormGeneDecode2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/BatchNormGeneDecode2/gamma/m
�
5Adam/BatchNormGeneDecode2/gamma/m/Read/ReadVariableOpReadVariableOp!Adam/BatchNormGeneDecode2/gamma/m*
_output_shapes	
:�*
dtype0
�
 Adam/BatchNormGeneDecode2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/BatchNormGeneDecode2/beta/m
�
4Adam/BatchNormGeneDecode2/beta/m/Read/ReadVariableOpReadVariableOp Adam/BatchNormGeneDecode2/beta/m*
_output_shapes	
:�*
dtype0
�
$Adam/BatchNormProteinDecode1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*5
shared_name&$Adam/BatchNormProteinDecode1/gamma/m
�
8Adam/BatchNormProteinDecode1/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/BatchNormProteinDecode1/gamma/m*
_output_shapes
:9*
dtype0
�
#Adam/BatchNormProteinDecode1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*4
shared_name%#Adam/BatchNormProteinDecode1/beta/m
�
7Adam/BatchNormProteinDecode1/beta/m/Read/ReadVariableOpReadVariableOp#Adam/BatchNormProteinDecode1/beta/m*
_output_shapes
:9*
dtype0
�
Adam/gene_decoder_last/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*0
shared_name!Adam/gene_decoder_last/kernel/m
�
3Adam/gene_decoder_last/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_last/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/gene_decoder_last/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_nameAdam/gene_decoder_last/bias/m
�
1Adam/gene_decoder_last/bias/m/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_last/bias/m*
_output_shapes	
:�*
dtype0
�
"Adam/protein_decoder_last/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	9�*3
shared_name$"Adam/protein_decoder_last/kernel/m
�
6Adam/protein_decoder_last/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/protein_decoder_last/kernel/m*
_output_shapes
:	9�*
dtype0
�
 Adam/protein_decoder_last/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/protein_decoder_last/bias/m
�
4Adam/protein_decoder_last/bias/m/Read/ReadVariableOpReadVariableOp Adam/protein_decoder_last/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/gene_encoder_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*-
shared_nameAdam/gene_encoder_1/kernel/v
�
0Adam/gene_encoder_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gene_encoder_1/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/gene_encoder_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameAdam/gene_encoder_1/bias/v
�
.Adam/gene_encoder_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/gene_encoder_1/bias/v*
_output_shapes	
:�*
dtype0
�
!Adam/BatchNormGeneEncode1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/BatchNormGeneEncode1/gamma/v
�
5Adam/BatchNormGeneEncode1/gamma/v/Read/ReadVariableOpReadVariableOp!Adam/BatchNormGeneEncode1/gamma/v*
_output_shapes	
:�*
dtype0
�
 Adam/BatchNormGeneEncode1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/BatchNormGeneEncode1/beta/v
�
4Adam/BatchNormGeneEncode1/beta/v/Read/ReadVariableOpReadVariableOp Adam/BatchNormGeneEncode1/beta/v*
_output_shapes	
:�*
dtype0
�
Adam/gene_encoder_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�}*-
shared_nameAdam/gene_encoder_2/kernel/v
�
0Adam/gene_encoder_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gene_encoder_2/kernel/v*
_output_shapes
:	�}*
dtype0
�
Adam/gene_encoder_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*+
shared_nameAdam/gene_encoder_2/bias/v
�
.Adam/gene_encoder_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/gene_encoder_2/bias/v*
_output_shapes
:}*
dtype0
�
Adam/protein_encoder_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�9*0
shared_name!Adam/protein_encoder_1/kernel/v
�
3Adam/protein_encoder_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/protein_encoder_1/kernel/v*
_output_shapes
:	�9*
dtype0
�
Adam/protein_encoder_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*.
shared_nameAdam/protein_encoder_1/bias/v
�
1Adam/protein_encoder_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/protein_encoder_1/bias/v*
_output_shapes
:9*
dtype0
�
!Adam/BatchNormGeneEncode2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*2
shared_name#!Adam/BatchNormGeneEncode2/gamma/v
�
5Adam/BatchNormGeneEncode2/gamma/v/Read/ReadVariableOpReadVariableOp!Adam/BatchNormGeneEncode2/gamma/v*
_output_shapes
:}*
dtype0
�
 Adam/BatchNormGeneEncode2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*1
shared_name" Adam/BatchNormGeneEncode2/beta/v
�
4Adam/BatchNormGeneEncode2/beta/v/Read/ReadVariableOpReadVariableOp Adam/BatchNormGeneEncode2/beta/v*
_output_shapes
:}*
dtype0
�
$Adam/BatchNormProteinEncode1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*5
shared_name&$Adam/BatchNormProteinEncode1/gamma/v
�
8Adam/BatchNormProteinEncode1/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/BatchNormProteinEncode1/gamma/v*
_output_shapes
:9*
dtype0
�
#Adam/BatchNormProteinEncode1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*4
shared_name%#Adam/BatchNormProteinEncode1/beta/v
�
7Adam/BatchNormProteinEncode1/beta/v/Read/ReadVariableOpReadVariableOp#Adam/BatchNormProteinEncode1/beta/v*
_output_shapes
:9*
dtype0
�
Adam/EmbeddingDimDense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�@*0
shared_name!Adam/EmbeddingDimDense/kernel/v
�
3Adam/EmbeddingDimDense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/EmbeddingDimDense/kernel/v*
_output_shapes
:	�@*
dtype0
�
Adam/EmbeddingDimDense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/EmbeddingDimDense/bias/v
�
1Adam/EmbeddingDimDense/bias/v/Read/ReadVariableOpReadVariableOpAdam/EmbeddingDimDense/bias/v*
_output_shapes
:@*
dtype0
�
Adam/gene_decoder_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@}*-
shared_nameAdam/gene_decoder_1/kernel/v
�
0Adam/gene_decoder_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_1/kernel/v*
_output_shapes

:@}*
dtype0
�
Adam/gene_decoder_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*+
shared_nameAdam/gene_decoder_1/bias/v
�
.Adam/gene_decoder_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_1/bias/v*
_output_shapes
:}*
dtype0
�
!Adam/BatchNormGeneDecode1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*2
shared_name#!Adam/BatchNormGeneDecode1/gamma/v
�
5Adam/BatchNormGeneDecode1/gamma/v/Read/ReadVariableOpReadVariableOp!Adam/BatchNormGeneDecode1/gamma/v*
_output_shapes
:}*
dtype0
�
 Adam/BatchNormGeneDecode1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*1
shared_name" Adam/BatchNormGeneDecode1/beta/v
�
4Adam/BatchNormGeneDecode1/beta/v/Read/ReadVariableOpReadVariableOp Adam/BatchNormGeneDecode1/beta/v*
_output_shapes
:}*
dtype0
�
Adam/gene_decoder_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	}�*-
shared_nameAdam/gene_decoder_2/kernel/v
�
0Adam/gene_decoder_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_2/kernel/v*
_output_shapes
:	}�*
dtype0
�
Adam/gene_decoder_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameAdam/gene_decoder_2/bias/v
�
.Adam/gene_decoder_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_2/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/protein_decoder_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@9*0
shared_name!Adam/protein_decoder_1/kernel/v
�
3Adam/protein_decoder_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/protein_decoder_1/kernel/v*
_output_shapes

:@9*
dtype0
�
Adam/protein_decoder_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*.
shared_nameAdam/protein_decoder_1/bias/v
�
1Adam/protein_decoder_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/protein_decoder_1/bias/v*
_output_shapes
:9*
dtype0
�
!Adam/BatchNormGeneDecode2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/BatchNormGeneDecode2/gamma/v
�
5Adam/BatchNormGeneDecode2/gamma/v/Read/ReadVariableOpReadVariableOp!Adam/BatchNormGeneDecode2/gamma/v*
_output_shapes	
:�*
dtype0
�
 Adam/BatchNormGeneDecode2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/BatchNormGeneDecode2/beta/v
�
4Adam/BatchNormGeneDecode2/beta/v/Read/ReadVariableOpReadVariableOp Adam/BatchNormGeneDecode2/beta/v*
_output_shapes	
:�*
dtype0
�
$Adam/BatchNormProteinDecode1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*5
shared_name&$Adam/BatchNormProteinDecode1/gamma/v
�
8Adam/BatchNormProteinDecode1/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/BatchNormProteinDecode1/gamma/v*
_output_shapes
:9*
dtype0
�
#Adam/BatchNormProteinDecode1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:9*4
shared_name%#Adam/BatchNormProteinDecode1/beta/v
�
7Adam/BatchNormProteinDecode1/beta/v/Read/ReadVariableOpReadVariableOp#Adam/BatchNormProteinDecode1/beta/v*
_output_shapes
:9*
dtype0
�
Adam/gene_decoder_last/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*0
shared_name!Adam/gene_decoder_last/kernel/v
�
3Adam/gene_decoder_last/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_last/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/gene_decoder_last/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*.
shared_nameAdam/gene_decoder_last/bias/v
�
1Adam/gene_decoder_last/bias/v/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_last/bias/v*
_output_shapes	
:�*
dtype0
�
"Adam/protein_decoder_last/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	9�*3
shared_name$"Adam/protein_decoder_last/kernel/v
�
6Adam/protein_decoder_last/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/protein_decoder_last/kernel/v*
_output_shapes
:	9�*
dtype0
�
 Adam/protein_decoder_last/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/protein_decoder_last/bias/v
�
4Adam/protein_decoder_last/bias/v/Read/ReadVariableOpReadVariableOp Adam/protein_decoder_last/bias/v*
_output_shapes	
:�*
dtype0

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer_with_weights-10
layer-13
layer_with_weights-11
layer-14
layer_with_weights-12
layer-15
layer_with_weights-13
layer-16
layer_with_weights-14
layer-17
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses*
�
$axis
	%gamma
&beta
'moving_mean
(moving_variance
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
* 
�

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses*
�

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses*
�
?axis
	@gamma
Abeta
Bmoving_mean
Cmoving_variance
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses*
�
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses*
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses* 
�

[kernel
\bias
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses*
�

ckernel
dbias
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses*
�
kaxis
	lgamma
mbeta
nmoving_mean
omoving_variance
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses*
�

vkernel
wbias
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses*
�

~kernel
bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
	�iter
�beta_1
�beta_2

�decay
�learning_ratem�m�%m�&m�/m�0m�7m�8m�@m�Am�Km�Lm�[m�\m�cm�dm�lm�mm�vm�wm�~m�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�v�v�%v�&v�/v�0v�7v�8v�@v�Av�Kv�Lv�[v�\v�cv�dv�lv�mv�vv�wv�~v�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�*
�
0
1
%2
&3
'4
(5
/6
07
78
89
@10
A11
B12
C13
K14
L15
M16
N17
[18
\19
c20
d21
l22
m23
n24
o25
v26
w27
~28
29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41*
�
0
1
%2
&3
/4
05
76
87
@8
A9
K10
L11
[12
\13
c14
d15
l16
m17
v18
w19
~20
21
�22
�23
�24
�25
�26
�27
�28
�29*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

�serving_default* 
e_
VARIABLE_VALUEgene_encoder_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEgene_encoder_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
* 
* 
* 
ic
VARIABLE_VALUEBatchNormGeneEncode1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEBatchNormGeneEncode1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE BatchNormGeneEncode1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE$BatchNormGeneEncode1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
%0
&1
'2
(3*

%0
&1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*
* 
* 
e_
VARIABLE_VALUEgene_encoder_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEgene_encoder_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

/0
01*

/0
01*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*
* 
* 
hb
VARIABLE_VALUEprotein_encoder_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEprotein_encoder_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

70
81*

70
81*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
* 
* 
* 
ic
VARIABLE_VALUEBatchNormGeneEncode2/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEBatchNormGeneEncode2/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE BatchNormGeneEncode2/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE$BatchNormGeneEncode2/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
@0
A1
B2
C3*

@0
A1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEBatchNormProteinEncode1/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEBatchNormProteinEncode1/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#BatchNormProteinEncode1/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE'BatchNormProteinEncode1/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
K0
L1
M2
N3*

K0
L1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses* 
* 
* 
hb
VARIABLE_VALUEEmbeddingDimDense/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEEmbeddingDimDense/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

[0
\1*

[0
\1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses*
* 
* 
e_
VARIABLE_VALUEgene_decoder_1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEgene_decoder_1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

c0
d1*

c0
d1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*
* 
* 
* 
ic
VARIABLE_VALUEBatchNormGeneDecode1/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEBatchNormGeneDecode1/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE BatchNormGeneDecode1/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE$BatchNormGeneDecode1/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
l0
m1
n2
o3*

l0
m1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses*
* 
* 
e_
VARIABLE_VALUEgene_decoder_2/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEgene_decoder_2/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

v0
w1*

v0
w1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses*
* 
* 
ic
VARIABLE_VALUEprotein_decoder_1/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEprotein_decoder_1/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*

~0
1*

~0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
jd
VARIABLE_VALUEBatchNormGeneDecode2/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEBatchNormGeneDecode2/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE BatchNormGeneDecode2/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE$BatchNormGeneDecode2/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
mg
VARIABLE_VALUEBatchNormProteinDecode1/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEBatchNormProteinDecode1/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#BatchNormProteinDecode1/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE'BatchNormProteinDecode1/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
$
�0
�1
�2
�3*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
ic
VARIABLE_VALUEgene_decoder_last/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEgene_decoder_last/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
lf
VARIABLE_VALUEprotein_decoder_last/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEprotein_decoder_last/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
^
'0
(1
B2
C3
M4
N5
n6
o7
�8
�9
�10
�11*
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17*

�0
�1
�2*
* 
* 
* 
* 
* 
* 
* 
* 

'0
(1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

B0
C1*
* 
* 
* 
* 

M0
N1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

n0
o1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 

�0
�1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<

�total

�count
�	variables
�	keras_api*
<

�total

�count
�	variables
�	keras_api*
<

�total

�count
�	variables
�	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
��
VARIABLE_VALUEAdam/gene_encoder_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/gene_encoder_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/BatchNormGeneEncode1/gamma/mQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/BatchNormGeneEncode1/beta/mPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_encoder_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/gene_encoder_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_encoder_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_encoder_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/BatchNormGeneEncode2/gamma/mQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/BatchNormGeneEncode2/beta/mPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE$Adam/BatchNormProteinEncode1/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/BatchNormProteinEncode1/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/EmbeddingDimDense/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/EmbeddingDimDense/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_decoder_1/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/gene_decoder_1/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/BatchNormGeneDecode1/gamma/mQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/BatchNormGeneDecode1/beta/mPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_decoder_2/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/gene_decoder_2/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_decoder_1/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_decoder_1/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/BatchNormGeneDecode2/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/BatchNormGeneDecode2/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE$Adam/BatchNormProteinDecode1/gamma/mRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/BatchNormProteinDecode1/beta/mQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_decoder_last/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_decoder_last/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/protein_decoder_last/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/protein_decoder_last/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_encoder_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/gene_encoder_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/BatchNormGeneEncode1/gamma/vQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/BatchNormGeneEncode1/beta/vPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_encoder_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/gene_encoder_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_encoder_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_encoder_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/BatchNormGeneEncode2/gamma/vQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/BatchNormGeneEncode2/beta/vPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE$Adam/BatchNormProteinEncode1/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/BatchNormProteinEncode1/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/EmbeddingDimDense/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/EmbeddingDimDense/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_decoder_1/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/gene_decoder_1/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/BatchNormGeneDecode1/gamma/vQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/BatchNormGeneDecode1/beta/vPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_decoder_2/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/gene_decoder_2/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_decoder_1/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_decoder_1/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/BatchNormGeneDecode2/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/BatchNormGeneDecode2/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE$Adam/BatchNormProteinDecode1/gamma/vRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/BatchNormProteinDecode1/beta/vQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_decoder_last/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_decoder_last/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/protein_decoder_last/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/protein_decoder_last/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
 serving_default_Gene_Input_LayerPlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
#serving_default_Protein_Input_LayerPlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCall serving_default_Gene_Input_Layer#serving_default_Protein_Input_Layergene_encoder_1/kernelgene_encoder_1/bias$BatchNormGeneEncode1/moving_varianceBatchNormGeneEncode1/gamma BatchNormGeneEncode1/moving_meanBatchNormGeneEncode1/betaprotein_encoder_1/kernelprotein_encoder_1/biasgene_encoder_2/kernelgene_encoder_2/bias$BatchNormGeneEncode2/moving_varianceBatchNormGeneEncode2/gamma BatchNormGeneEncode2/moving_meanBatchNormGeneEncode2/beta'BatchNormProteinEncode1/moving_varianceBatchNormProteinEncode1/gamma#BatchNormProteinEncode1/moving_meanBatchNormProteinEncode1/betaEmbeddingDimDense/kernelEmbeddingDimDense/biasgene_decoder_1/kernelgene_decoder_1/bias$BatchNormGeneDecode1/moving_varianceBatchNormGeneDecode1/gamma BatchNormGeneDecode1/moving_meanBatchNormGeneDecode1/betaprotein_decoder_1/kernelprotein_decoder_1/biasgene_decoder_2/kernelgene_decoder_2/bias'BatchNormProteinDecode1/moving_varianceBatchNormProteinDecode1/gamma#BatchNormProteinDecode1/moving_meanBatchNormProteinDecode1/beta$BatchNormGeneDecode2/moving_varianceBatchNormGeneDecode2/gamma BatchNormGeneDecode2/moving_meanBatchNormGeneDecode2/betaprotein_decoder_last/kernelprotein_decoder_last/biasgene_decoder_last/kernelgene_decoder_last/bias*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:����������:����������*L
_read_only_resource_inputs.
,*	
 !"#$%&'()*+*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_188218
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�.
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)gene_encoder_1/kernel/Read/ReadVariableOp'gene_encoder_1/bias/Read/ReadVariableOp.BatchNormGeneEncode1/gamma/Read/ReadVariableOp-BatchNormGeneEncode1/beta/Read/ReadVariableOp4BatchNormGeneEncode1/moving_mean/Read/ReadVariableOp8BatchNormGeneEncode1/moving_variance/Read/ReadVariableOp)gene_encoder_2/kernel/Read/ReadVariableOp'gene_encoder_2/bias/Read/ReadVariableOp,protein_encoder_1/kernel/Read/ReadVariableOp*protein_encoder_1/bias/Read/ReadVariableOp.BatchNormGeneEncode2/gamma/Read/ReadVariableOp-BatchNormGeneEncode2/beta/Read/ReadVariableOp4BatchNormGeneEncode2/moving_mean/Read/ReadVariableOp8BatchNormGeneEncode2/moving_variance/Read/ReadVariableOp1BatchNormProteinEncode1/gamma/Read/ReadVariableOp0BatchNormProteinEncode1/beta/Read/ReadVariableOp7BatchNormProteinEncode1/moving_mean/Read/ReadVariableOp;BatchNormProteinEncode1/moving_variance/Read/ReadVariableOp,EmbeddingDimDense/kernel/Read/ReadVariableOp*EmbeddingDimDense/bias/Read/ReadVariableOp)gene_decoder_1/kernel/Read/ReadVariableOp'gene_decoder_1/bias/Read/ReadVariableOp.BatchNormGeneDecode1/gamma/Read/ReadVariableOp-BatchNormGeneDecode1/beta/Read/ReadVariableOp4BatchNormGeneDecode1/moving_mean/Read/ReadVariableOp8BatchNormGeneDecode1/moving_variance/Read/ReadVariableOp)gene_decoder_2/kernel/Read/ReadVariableOp'gene_decoder_2/bias/Read/ReadVariableOp,protein_decoder_1/kernel/Read/ReadVariableOp*protein_decoder_1/bias/Read/ReadVariableOp.BatchNormGeneDecode2/gamma/Read/ReadVariableOp-BatchNormGeneDecode2/beta/Read/ReadVariableOp4BatchNormGeneDecode2/moving_mean/Read/ReadVariableOp8BatchNormGeneDecode2/moving_variance/Read/ReadVariableOp1BatchNormProteinDecode1/gamma/Read/ReadVariableOp0BatchNormProteinDecode1/beta/Read/ReadVariableOp7BatchNormProteinDecode1/moving_mean/Read/ReadVariableOp;BatchNormProteinDecode1/moving_variance/Read/ReadVariableOp,gene_decoder_last/kernel/Read/ReadVariableOp*gene_decoder_last/bias/Read/ReadVariableOp/protein_decoder_last/kernel/Read/ReadVariableOp-protein_decoder_last/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp0Adam/gene_encoder_1/kernel/m/Read/ReadVariableOp.Adam/gene_encoder_1/bias/m/Read/ReadVariableOp5Adam/BatchNormGeneEncode1/gamma/m/Read/ReadVariableOp4Adam/BatchNormGeneEncode1/beta/m/Read/ReadVariableOp0Adam/gene_encoder_2/kernel/m/Read/ReadVariableOp.Adam/gene_encoder_2/bias/m/Read/ReadVariableOp3Adam/protein_encoder_1/kernel/m/Read/ReadVariableOp1Adam/protein_encoder_1/bias/m/Read/ReadVariableOp5Adam/BatchNormGeneEncode2/gamma/m/Read/ReadVariableOp4Adam/BatchNormGeneEncode2/beta/m/Read/ReadVariableOp8Adam/BatchNormProteinEncode1/gamma/m/Read/ReadVariableOp7Adam/BatchNormProteinEncode1/beta/m/Read/ReadVariableOp3Adam/EmbeddingDimDense/kernel/m/Read/ReadVariableOp1Adam/EmbeddingDimDense/bias/m/Read/ReadVariableOp0Adam/gene_decoder_1/kernel/m/Read/ReadVariableOp.Adam/gene_decoder_1/bias/m/Read/ReadVariableOp5Adam/BatchNormGeneDecode1/gamma/m/Read/ReadVariableOp4Adam/BatchNormGeneDecode1/beta/m/Read/ReadVariableOp0Adam/gene_decoder_2/kernel/m/Read/ReadVariableOp.Adam/gene_decoder_2/bias/m/Read/ReadVariableOp3Adam/protein_decoder_1/kernel/m/Read/ReadVariableOp1Adam/protein_decoder_1/bias/m/Read/ReadVariableOp5Adam/BatchNormGeneDecode2/gamma/m/Read/ReadVariableOp4Adam/BatchNormGeneDecode2/beta/m/Read/ReadVariableOp8Adam/BatchNormProteinDecode1/gamma/m/Read/ReadVariableOp7Adam/BatchNormProteinDecode1/beta/m/Read/ReadVariableOp3Adam/gene_decoder_last/kernel/m/Read/ReadVariableOp1Adam/gene_decoder_last/bias/m/Read/ReadVariableOp6Adam/protein_decoder_last/kernel/m/Read/ReadVariableOp4Adam/protein_decoder_last/bias/m/Read/ReadVariableOp0Adam/gene_encoder_1/kernel/v/Read/ReadVariableOp.Adam/gene_encoder_1/bias/v/Read/ReadVariableOp5Adam/BatchNormGeneEncode1/gamma/v/Read/ReadVariableOp4Adam/BatchNormGeneEncode1/beta/v/Read/ReadVariableOp0Adam/gene_encoder_2/kernel/v/Read/ReadVariableOp.Adam/gene_encoder_2/bias/v/Read/ReadVariableOp3Adam/protein_encoder_1/kernel/v/Read/ReadVariableOp1Adam/protein_encoder_1/bias/v/Read/ReadVariableOp5Adam/BatchNormGeneEncode2/gamma/v/Read/ReadVariableOp4Adam/BatchNormGeneEncode2/beta/v/Read/ReadVariableOp8Adam/BatchNormProteinEncode1/gamma/v/Read/ReadVariableOp7Adam/BatchNormProteinEncode1/beta/v/Read/ReadVariableOp3Adam/EmbeddingDimDense/kernel/v/Read/ReadVariableOp1Adam/EmbeddingDimDense/bias/v/Read/ReadVariableOp0Adam/gene_decoder_1/kernel/v/Read/ReadVariableOp.Adam/gene_decoder_1/bias/v/Read/ReadVariableOp5Adam/BatchNormGeneDecode1/gamma/v/Read/ReadVariableOp4Adam/BatchNormGeneDecode1/beta/v/Read/ReadVariableOp0Adam/gene_decoder_2/kernel/v/Read/ReadVariableOp.Adam/gene_decoder_2/bias/v/Read/ReadVariableOp3Adam/protein_decoder_1/kernel/v/Read/ReadVariableOp1Adam/protein_decoder_1/bias/v/Read/ReadVariableOp5Adam/BatchNormGeneDecode2/gamma/v/Read/ReadVariableOp4Adam/BatchNormGeneDecode2/beta/v/Read/ReadVariableOp8Adam/BatchNormProteinDecode1/gamma/v/Read/ReadVariableOp7Adam/BatchNormProteinDecode1/beta/v/Read/ReadVariableOp3Adam/gene_decoder_last/kernel/v/Read/ReadVariableOp1Adam/gene_decoder_last/bias/v/Read/ReadVariableOp6Adam/protein_decoder_last/kernel/v/Read/ReadVariableOp4Adam/protein_decoder_last/bias/v/Read/ReadVariableOpConst*~
Tinw
u2s	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *(
f#R!
__inference__traced_save_189255
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegene_encoder_1/kernelgene_encoder_1/biasBatchNormGeneEncode1/gammaBatchNormGeneEncode1/beta BatchNormGeneEncode1/moving_mean$BatchNormGeneEncode1/moving_variancegene_encoder_2/kernelgene_encoder_2/biasprotein_encoder_1/kernelprotein_encoder_1/biasBatchNormGeneEncode2/gammaBatchNormGeneEncode2/beta BatchNormGeneEncode2/moving_mean$BatchNormGeneEncode2/moving_varianceBatchNormProteinEncode1/gammaBatchNormProteinEncode1/beta#BatchNormProteinEncode1/moving_mean'BatchNormProteinEncode1/moving_varianceEmbeddingDimDense/kernelEmbeddingDimDense/biasgene_decoder_1/kernelgene_decoder_1/biasBatchNormGeneDecode1/gammaBatchNormGeneDecode1/beta BatchNormGeneDecode1/moving_mean$BatchNormGeneDecode1/moving_variancegene_decoder_2/kernelgene_decoder_2/biasprotein_decoder_1/kernelprotein_decoder_1/biasBatchNormGeneDecode2/gammaBatchNormGeneDecode2/beta BatchNormGeneDecode2/moving_mean$BatchNormGeneDecode2/moving_varianceBatchNormProteinDecode1/gammaBatchNormProteinDecode1/beta#BatchNormProteinDecode1/moving_mean'BatchNormProteinDecode1/moving_variancegene_decoder_last/kernelgene_decoder_last/biasprotein_decoder_last/kernelprotein_decoder_last/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/gene_encoder_1/kernel/mAdam/gene_encoder_1/bias/m!Adam/BatchNormGeneEncode1/gamma/m Adam/BatchNormGeneEncode1/beta/mAdam/gene_encoder_2/kernel/mAdam/gene_encoder_2/bias/mAdam/protein_encoder_1/kernel/mAdam/protein_encoder_1/bias/m!Adam/BatchNormGeneEncode2/gamma/m Adam/BatchNormGeneEncode2/beta/m$Adam/BatchNormProteinEncode1/gamma/m#Adam/BatchNormProteinEncode1/beta/mAdam/EmbeddingDimDense/kernel/mAdam/EmbeddingDimDense/bias/mAdam/gene_decoder_1/kernel/mAdam/gene_decoder_1/bias/m!Adam/BatchNormGeneDecode1/gamma/m Adam/BatchNormGeneDecode1/beta/mAdam/gene_decoder_2/kernel/mAdam/gene_decoder_2/bias/mAdam/protein_decoder_1/kernel/mAdam/protein_decoder_1/bias/m!Adam/BatchNormGeneDecode2/gamma/m Adam/BatchNormGeneDecode2/beta/m$Adam/BatchNormProteinDecode1/gamma/m#Adam/BatchNormProteinDecode1/beta/mAdam/gene_decoder_last/kernel/mAdam/gene_decoder_last/bias/m"Adam/protein_decoder_last/kernel/m Adam/protein_decoder_last/bias/mAdam/gene_encoder_1/kernel/vAdam/gene_encoder_1/bias/v!Adam/BatchNormGeneEncode1/gamma/v Adam/BatchNormGeneEncode1/beta/vAdam/gene_encoder_2/kernel/vAdam/gene_encoder_2/bias/vAdam/protein_encoder_1/kernel/vAdam/protein_encoder_1/bias/v!Adam/BatchNormGeneEncode2/gamma/v Adam/BatchNormGeneEncode2/beta/v$Adam/BatchNormProteinEncode1/gamma/v#Adam/BatchNormProteinEncode1/beta/vAdam/EmbeddingDimDense/kernel/vAdam/EmbeddingDimDense/bias/vAdam/gene_decoder_1/kernel/vAdam/gene_decoder_1/bias/v!Adam/BatchNormGeneDecode1/gamma/v Adam/BatchNormGeneDecode1/beta/vAdam/gene_decoder_2/kernel/vAdam/gene_decoder_2/bias/vAdam/protein_decoder_1/kernel/vAdam/protein_decoder_1/bias/v!Adam/BatchNormGeneDecode2/gamma/v Adam/BatchNormGeneDecode2/beta/v$Adam/BatchNormProteinDecode1/gamma/v#Adam/BatchNormProteinDecode1/beta/vAdam/gene_decoder_last/kernel/vAdam/gene_decoder_last/bias/v"Adam/protein_decoder_last/kernel/v Adam/protein_decoder_last/bias/v*}
Tinv
t2r*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *+
f&R$
"__inference__traced_restore_189604��
�

�
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_188551

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������@Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�%
�
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_186334

inputs5
'assignmovingavg_readvariableop_resource:}7
)assignmovingavg_1_readvariableop_resource:}3
%batchnorm_mul_readvariableop_resource:}/
!batchnorm_readvariableop_resource:}
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:}*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:}�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������}l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:}*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:}*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:}*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:}*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:}x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:}�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:}*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:}~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:}�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:}P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:}~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������}h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:}v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:}r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������}�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������}: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������}
 
_user_specified_nameinputs
��
�6
__inference__traced_save_189255
file_prefix4
0savev2_gene_encoder_1_kernel_read_readvariableop2
.savev2_gene_encoder_1_bias_read_readvariableop9
5savev2_batchnormgeneencode1_gamma_read_readvariableop8
4savev2_batchnormgeneencode1_beta_read_readvariableop?
;savev2_batchnormgeneencode1_moving_mean_read_readvariableopC
?savev2_batchnormgeneencode1_moving_variance_read_readvariableop4
0savev2_gene_encoder_2_kernel_read_readvariableop2
.savev2_gene_encoder_2_bias_read_readvariableop7
3savev2_protein_encoder_1_kernel_read_readvariableop5
1savev2_protein_encoder_1_bias_read_readvariableop9
5savev2_batchnormgeneencode2_gamma_read_readvariableop8
4savev2_batchnormgeneencode2_beta_read_readvariableop?
;savev2_batchnormgeneencode2_moving_mean_read_readvariableopC
?savev2_batchnormgeneencode2_moving_variance_read_readvariableop<
8savev2_batchnormproteinencode1_gamma_read_readvariableop;
7savev2_batchnormproteinencode1_beta_read_readvariableopB
>savev2_batchnormproteinencode1_moving_mean_read_readvariableopF
Bsavev2_batchnormproteinencode1_moving_variance_read_readvariableop7
3savev2_embeddingdimdense_kernel_read_readvariableop5
1savev2_embeddingdimdense_bias_read_readvariableop4
0savev2_gene_decoder_1_kernel_read_readvariableop2
.savev2_gene_decoder_1_bias_read_readvariableop9
5savev2_batchnormgenedecode1_gamma_read_readvariableop8
4savev2_batchnormgenedecode1_beta_read_readvariableop?
;savev2_batchnormgenedecode1_moving_mean_read_readvariableopC
?savev2_batchnormgenedecode1_moving_variance_read_readvariableop4
0savev2_gene_decoder_2_kernel_read_readvariableop2
.savev2_gene_decoder_2_bias_read_readvariableop7
3savev2_protein_decoder_1_kernel_read_readvariableop5
1savev2_protein_decoder_1_bias_read_readvariableop9
5savev2_batchnormgenedecode2_gamma_read_readvariableop8
4savev2_batchnormgenedecode2_beta_read_readvariableop?
;savev2_batchnormgenedecode2_moving_mean_read_readvariableopC
?savev2_batchnormgenedecode2_moving_variance_read_readvariableop<
8savev2_batchnormproteindecode1_gamma_read_readvariableop;
7savev2_batchnormproteindecode1_beta_read_readvariableopB
>savev2_batchnormproteindecode1_moving_mean_read_readvariableopF
Bsavev2_batchnormproteindecode1_moving_variance_read_readvariableop7
3savev2_gene_decoder_last_kernel_read_readvariableop5
1savev2_gene_decoder_last_bias_read_readvariableop:
6savev2_protein_decoder_last_kernel_read_readvariableop8
4savev2_protein_decoder_last_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop;
7savev2_adam_gene_encoder_1_kernel_m_read_readvariableop9
5savev2_adam_gene_encoder_1_bias_m_read_readvariableop@
<savev2_adam_batchnormgeneencode1_gamma_m_read_readvariableop?
;savev2_adam_batchnormgeneencode1_beta_m_read_readvariableop;
7savev2_adam_gene_encoder_2_kernel_m_read_readvariableop9
5savev2_adam_gene_encoder_2_bias_m_read_readvariableop>
:savev2_adam_protein_encoder_1_kernel_m_read_readvariableop<
8savev2_adam_protein_encoder_1_bias_m_read_readvariableop@
<savev2_adam_batchnormgeneencode2_gamma_m_read_readvariableop?
;savev2_adam_batchnormgeneencode2_beta_m_read_readvariableopC
?savev2_adam_batchnormproteinencode1_gamma_m_read_readvariableopB
>savev2_adam_batchnormproteinencode1_beta_m_read_readvariableop>
:savev2_adam_embeddingdimdense_kernel_m_read_readvariableop<
8savev2_adam_embeddingdimdense_bias_m_read_readvariableop;
7savev2_adam_gene_decoder_1_kernel_m_read_readvariableop9
5savev2_adam_gene_decoder_1_bias_m_read_readvariableop@
<savev2_adam_batchnormgenedecode1_gamma_m_read_readvariableop?
;savev2_adam_batchnormgenedecode1_beta_m_read_readvariableop;
7savev2_adam_gene_decoder_2_kernel_m_read_readvariableop9
5savev2_adam_gene_decoder_2_bias_m_read_readvariableop>
:savev2_adam_protein_decoder_1_kernel_m_read_readvariableop<
8savev2_adam_protein_decoder_1_bias_m_read_readvariableop@
<savev2_adam_batchnormgenedecode2_gamma_m_read_readvariableop?
;savev2_adam_batchnormgenedecode2_beta_m_read_readvariableopC
?savev2_adam_batchnormproteindecode1_gamma_m_read_readvariableopB
>savev2_adam_batchnormproteindecode1_beta_m_read_readvariableop>
:savev2_adam_gene_decoder_last_kernel_m_read_readvariableop<
8savev2_adam_gene_decoder_last_bias_m_read_readvariableopA
=savev2_adam_protein_decoder_last_kernel_m_read_readvariableop?
;savev2_adam_protein_decoder_last_bias_m_read_readvariableop;
7savev2_adam_gene_encoder_1_kernel_v_read_readvariableop9
5savev2_adam_gene_encoder_1_bias_v_read_readvariableop@
<savev2_adam_batchnormgeneencode1_gamma_v_read_readvariableop?
;savev2_adam_batchnormgeneencode1_beta_v_read_readvariableop;
7savev2_adam_gene_encoder_2_kernel_v_read_readvariableop9
5savev2_adam_gene_encoder_2_bias_v_read_readvariableop>
:savev2_adam_protein_encoder_1_kernel_v_read_readvariableop<
8savev2_adam_protein_encoder_1_bias_v_read_readvariableop@
<savev2_adam_batchnormgeneencode2_gamma_v_read_readvariableop?
;savev2_adam_batchnormgeneencode2_beta_v_read_readvariableopC
?savev2_adam_batchnormproteinencode1_gamma_v_read_readvariableopB
>savev2_adam_batchnormproteinencode1_beta_v_read_readvariableop>
:savev2_adam_embeddingdimdense_kernel_v_read_readvariableop<
8savev2_adam_embeddingdimdense_bias_v_read_readvariableop;
7savev2_adam_gene_decoder_1_kernel_v_read_readvariableop9
5savev2_adam_gene_decoder_1_bias_v_read_readvariableop@
<savev2_adam_batchnormgenedecode1_gamma_v_read_readvariableop?
;savev2_adam_batchnormgenedecode1_beta_v_read_readvariableop;
7savev2_adam_gene_decoder_2_kernel_v_read_readvariableop9
5savev2_adam_gene_decoder_2_bias_v_read_readvariableop>
:savev2_adam_protein_decoder_1_kernel_v_read_readvariableop<
8savev2_adam_protein_decoder_1_bias_v_read_readvariableop@
<savev2_adam_batchnormgenedecode2_gamma_v_read_readvariableop?
;savev2_adam_batchnormgenedecode2_beta_v_read_readvariableopC
?savev2_adam_batchnormproteindecode1_gamma_v_read_readvariableopB
>savev2_adam_batchnormproteindecode1_beta_v_read_readvariableop>
:savev2_adam_gene_decoder_last_kernel_v_read_readvariableop<
8savev2_adam_gene_decoder_last_bias_v_read_readvariableopA
=savev2_adam_protein_decoder_last_kernel_v_read_readvariableop?
;savev2_adam_protein_decoder_last_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:r*
dtype0*�>
value�>B�>rB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:r*
dtype0*�
value�B�rB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �3
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_gene_encoder_1_kernel_read_readvariableop.savev2_gene_encoder_1_bias_read_readvariableop5savev2_batchnormgeneencode1_gamma_read_readvariableop4savev2_batchnormgeneencode1_beta_read_readvariableop;savev2_batchnormgeneencode1_moving_mean_read_readvariableop?savev2_batchnormgeneencode1_moving_variance_read_readvariableop0savev2_gene_encoder_2_kernel_read_readvariableop.savev2_gene_encoder_2_bias_read_readvariableop3savev2_protein_encoder_1_kernel_read_readvariableop1savev2_protein_encoder_1_bias_read_readvariableop5savev2_batchnormgeneencode2_gamma_read_readvariableop4savev2_batchnormgeneencode2_beta_read_readvariableop;savev2_batchnormgeneencode2_moving_mean_read_readvariableop?savev2_batchnormgeneencode2_moving_variance_read_readvariableop8savev2_batchnormproteinencode1_gamma_read_readvariableop7savev2_batchnormproteinencode1_beta_read_readvariableop>savev2_batchnormproteinencode1_moving_mean_read_readvariableopBsavev2_batchnormproteinencode1_moving_variance_read_readvariableop3savev2_embeddingdimdense_kernel_read_readvariableop1savev2_embeddingdimdense_bias_read_readvariableop0savev2_gene_decoder_1_kernel_read_readvariableop.savev2_gene_decoder_1_bias_read_readvariableop5savev2_batchnormgenedecode1_gamma_read_readvariableop4savev2_batchnormgenedecode1_beta_read_readvariableop;savev2_batchnormgenedecode1_moving_mean_read_readvariableop?savev2_batchnormgenedecode1_moving_variance_read_readvariableop0savev2_gene_decoder_2_kernel_read_readvariableop.savev2_gene_decoder_2_bias_read_readvariableop3savev2_protein_decoder_1_kernel_read_readvariableop1savev2_protein_decoder_1_bias_read_readvariableop5savev2_batchnormgenedecode2_gamma_read_readvariableop4savev2_batchnormgenedecode2_beta_read_readvariableop;savev2_batchnormgenedecode2_moving_mean_read_readvariableop?savev2_batchnormgenedecode2_moving_variance_read_readvariableop8savev2_batchnormproteindecode1_gamma_read_readvariableop7savev2_batchnormproteindecode1_beta_read_readvariableop>savev2_batchnormproteindecode1_moving_mean_read_readvariableopBsavev2_batchnormproteindecode1_moving_variance_read_readvariableop3savev2_gene_decoder_last_kernel_read_readvariableop1savev2_gene_decoder_last_bias_read_readvariableop6savev2_protein_decoder_last_kernel_read_readvariableop4savev2_protein_decoder_last_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop7savev2_adam_gene_encoder_1_kernel_m_read_readvariableop5savev2_adam_gene_encoder_1_bias_m_read_readvariableop<savev2_adam_batchnormgeneencode1_gamma_m_read_readvariableop;savev2_adam_batchnormgeneencode1_beta_m_read_readvariableop7savev2_adam_gene_encoder_2_kernel_m_read_readvariableop5savev2_adam_gene_encoder_2_bias_m_read_readvariableop:savev2_adam_protein_encoder_1_kernel_m_read_readvariableop8savev2_adam_protein_encoder_1_bias_m_read_readvariableop<savev2_adam_batchnormgeneencode2_gamma_m_read_readvariableop;savev2_adam_batchnormgeneencode2_beta_m_read_readvariableop?savev2_adam_batchnormproteinencode1_gamma_m_read_readvariableop>savev2_adam_batchnormproteinencode1_beta_m_read_readvariableop:savev2_adam_embeddingdimdense_kernel_m_read_readvariableop8savev2_adam_embeddingdimdense_bias_m_read_readvariableop7savev2_adam_gene_decoder_1_kernel_m_read_readvariableop5savev2_adam_gene_decoder_1_bias_m_read_readvariableop<savev2_adam_batchnormgenedecode1_gamma_m_read_readvariableop;savev2_adam_batchnormgenedecode1_beta_m_read_readvariableop7savev2_adam_gene_decoder_2_kernel_m_read_readvariableop5savev2_adam_gene_decoder_2_bias_m_read_readvariableop:savev2_adam_protein_decoder_1_kernel_m_read_readvariableop8savev2_adam_protein_decoder_1_bias_m_read_readvariableop<savev2_adam_batchnormgenedecode2_gamma_m_read_readvariableop;savev2_adam_batchnormgenedecode2_beta_m_read_readvariableop?savev2_adam_batchnormproteindecode1_gamma_m_read_readvariableop>savev2_adam_batchnormproteindecode1_beta_m_read_readvariableop:savev2_adam_gene_decoder_last_kernel_m_read_readvariableop8savev2_adam_gene_decoder_last_bias_m_read_readvariableop=savev2_adam_protein_decoder_last_kernel_m_read_readvariableop;savev2_adam_protein_decoder_last_bias_m_read_readvariableop7savev2_adam_gene_encoder_1_kernel_v_read_readvariableop5savev2_adam_gene_encoder_1_bias_v_read_readvariableop<savev2_adam_batchnormgeneencode1_gamma_v_read_readvariableop;savev2_adam_batchnormgeneencode1_beta_v_read_readvariableop7savev2_adam_gene_encoder_2_kernel_v_read_readvariableop5savev2_adam_gene_encoder_2_bias_v_read_readvariableop:savev2_adam_protein_encoder_1_kernel_v_read_readvariableop8savev2_adam_protein_encoder_1_bias_v_read_readvariableop<savev2_adam_batchnormgeneencode2_gamma_v_read_readvariableop;savev2_adam_batchnormgeneencode2_beta_v_read_readvariableop?savev2_adam_batchnormproteinencode1_gamma_v_read_readvariableop>savev2_adam_batchnormproteinencode1_beta_v_read_readvariableop:savev2_adam_embeddingdimdense_kernel_v_read_readvariableop8savev2_adam_embeddingdimdense_bias_v_read_readvariableop7savev2_adam_gene_decoder_1_kernel_v_read_readvariableop5savev2_adam_gene_decoder_1_bias_v_read_readvariableop<savev2_adam_batchnormgenedecode1_gamma_v_read_readvariableop;savev2_adam_batchnormgenedecode1_beta_v_read_readvariableop7savev2_adam_gene_decoder_2_kernel_v_read_readvariableop5savev2_adam_gene_decoder_2_bias_v_read_readvariableop:savev2_adam_protein_decoder_1_kernel_v_read_readvariableop8savev2_adam_protein_decoder_1_bias_v_read_readvariableop<savev2_adam_batchnormgenedecode2_gamma_v_read_readvariableop;savev2_adam_batchnormgenedecode2_beta_v_read_readvariableop?savev2_adam_batchnormproteindecode1_gamma_v_read_readvariableop>savev2_adam_batchnormproteindecode1_beta_v_read_readvariableop:savev2_adam_gene_decoder_last_kernel_v_read_readvariableop8savev2_adam_gene_decoder_last_bias_v_read_readvariableop=savev2_adam_protein_decoder_last_kernel_v_read_readvariableop;savev2_adam_protein_decoder_last_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *�
dtypesv
t2r	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��:�:�:�:�:�:	�}:}:	�9:9:}:}:}:}:9:9:9:9:	�@:@:@}:}:}:}:}:}:	}�:�:@9:9:�:�:�:�:9:9:9:9:
��:�:	9�:�: : : : : : : : : : : :
��:�:�:�:	�}:}:	�9:9:}:}:9:9:	�@:@:@}:}:}:}:	}�:�:@9:9:�:�:9:9:
��:�:	9�:�:
��:�:�:�:	�}:}:	�9:9:}:}:9:9:	�@:@:@}:}:}:}:	}�:�:@9:9:�:�:9:9:
��:�:	9�:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:%!

_output_shapes
:	�}: 

_output_shapes
:}:%	!

_output_shapes
:	�9: 


_output_shapes
:9: 

_output_shapes
:}: 

_output_shapes
:}: 

_output_shapes
:}: 

_output_shapes
:}: 

_output_shapes
:9: 

_output_shapes
:9: 

_output_shapes
:9: 

_output_shapes
:9:%!

_output_shapes
:	�@: 

_output_shapes
:@:$ 

_output_shapes

:@}: 

_output_shapes
:}: 

_output_shapes
:}: 

_output_shapes
:}: 

_output_shapes
:}: 

_output_shapes
:}:%!

_output_shapes
:	}�:!

_output_shapes	
:�:$ 

_output_shapes

:@9: 

_output_shapes
:9:!

_output_shapes	
:�:! 

_output_shapes	
:�:!!

_output_shapes	
:�:!"

_output_shapes	
:�: #

_output_shapes
:9: $

_output_shapes
:9: %

_output_shapes
:9: &

_output_shapes
:9:&'"
 
_output_shapes
:
��:!(

_output_shapes	
:�:%)!

_output_shapes
:	9�:!*

_output_shapes	
:�:+

_output_shapes
: :,

_output_shapes
: :-

_output_shapes
: :.

_output_shapes
: :/

_output_shapes
: :0

_output_shapes
: :1

_output_shapes
: :2

_output_shapes
: :3

_output_shapes
: :4

_output_shapes
: :5

_output_shapes
: :&6"
 
_output_shapes
:
��:!7

_output_shapes	
:�:!8

_output_shapes	
:�:!9

_output_shapes	
:�:%:!

_output_shapes
:	�}: ;

_output_shapes
:}:%<!

_output_shapes
:	�9: =

_output_shapes
:9: >

_output_shapes
:}: ?

_output_shapes
:}: @

_output_shapes
:9: A

_output_shapes
:9:%B!

_output_shapes
:	�@: C

_output_shapes
:@:$D 

_output_shapes

:@}: E

_output_shapes
:}: F

_output_shapes
:}: G

_output_shapes
:}:%H!

_output_shapes
:	}�:!I

_output_shapes	
:�:$J 

_output_shapes

:@9: K

_output_shapes
:9:!L

_output_shapes	
:�:!M

_output_shapes	
:�: N

_output_shapes
:9: O

_output_shapes
:9:&P"
 
_output_shapes
:
��:!Q

_output_shapes	
:�:%R!

_output_shapes
:	9�:!S

_output_shapes	
:�:&T"
 
_output_shapes
:
��:!U

_output_shapes	
:�:!V

_output_shapes	
:�:!W

_output_shapes	
:�:%X!

_output_shapes
:	�}: Y

_output_shapes
:}:%Z!

_output_shapes
:	�9: [

_output_shapes
:9: \

_output_shapes
:}: ]

_output_shapes
:}: ^

_output_shapes
:9: _

_output_shapes
:9:%`!

_output_shapes
:	�@: a

_output_shapes
:@:$b 

_output_shapes

:@}: c

_output_shapes
:}: d

_output_shapes
:}: e

_output_shapes
:}:%f!

_output_shapes
:	}�:!g

_output_shapes	
:�:$h 

_output_shapes

:@9: i

_output_shapes
:9:!j

_output_shapes	
:�:!k

_output_shapes	
:�: l

_output_shapes
:9: m

_output_shapes
:9:&n"
 
_output_shapes
:
��:!o

_output_shapes	
:�:%p!

_output_shapes
:	9�:!q

_output_shapes	
:�:r

_output_shapes
: 
�
�
5__inference_BatchNormGeneEncode1_layer_call_fn_188264

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_186088p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
5__inference_BatchNormGeneDecode1_layer_call_fn_188597

inputs
unknown:}
	unknown_0:}
	unknown_1:}
	unknown_2:}
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_186334o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������}`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������}: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������}
 
_user_specified_nameinputs
�
�
5__inference_BatchNormGeneDecode1_layer_call_fn_188584

inputs
unknown:}
	unknown_0:}
	unknown_1:}
	unknown_2:}
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_186287o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������}`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������}: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������}
 
_user_specified_nameinputs
�%
�
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_188651

inputs5
'assignmovingavg_readvariableop_resource:}7
)assignmovingavg_1_readvariableop_resource:}3
%batchnorm_mul_readvariableop_resource:}/
!batchnorm_readvariableop_resource:}
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:}*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:}�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������}l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:}*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:}*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:}*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:}*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:}x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:}�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:}*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:}~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:}�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:}P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:}~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������}h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:}v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:}r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������}�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������}: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������}
 
_user_specified_nameinputs
�%
�
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_186416

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
5__inference_BatchNormGeneEncode1_layer_call_fn_188251

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_186041p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
8__inference_BatchNormProteinDecode1_layer_call_fn_188784

inputs
unknown:9
	unknown_0:9
	unknown_1:9
	unknown_2:9
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������9*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_186451o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������9`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������9: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������9
 
_user_specified_nameinputs
�
�

&__inference_model_layer_call_fn_187304
gene_input_layer
protein_input_layer
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�9
	unknown_6:9
	unknown_7:	�}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:9

unknown_14:9

unknown_15:9

unknown_16:9

unknown_17:	�@

unknown_18:@

unknown_19:@}

unknown_20:}

unknown_21:}

unknown_22:}

unknown_23:}

unknown_24:}

unknown_25:@9

unknown_26:9

unknown_27:	}�

unknown_28:	�

unknown_29:9

unknown_30:9

unknown_31:9

unknown_32:9

unknown_33:	�

unknown_34:	�

unknown_35:	�

unknown_36:	�

unknown_37:	9�

unknown_38:	�

unknown_39:
��

unknown_40:	�
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgene_input_layerprotein_input_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:����������:����������*@
_read_only_resource_inputs"
 	
"#&'()*+*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_187123p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:����������
*
_user_specified_nameGene_Input_Layer:]Y
(
_output_shapes
:����������
-
_user_specified_nameProtein_Input_Layer
�

�
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_188238

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�%
�
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_188771

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
J__inference_gene_decoder_2_layer_call_and_return_conditional_losses_188671

inputs1
matmul_readvariableop_resource:	}�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	}�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������}: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������}
 
_user_specified_nameinputs
�
�
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_186451

inputs/
!batchnorm_readvariableop_resource:93
%batchnorm_mul_readvariableop_resource:91
#batchnorm_readvariableop_1_resource:91
#batchnorm_readvariableop_2_resource:9
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:9*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:9P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:9~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:9*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:9c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������9z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:9*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:9z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:9*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:9r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������9b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������9�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������9: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������9
 
_user_specified_nameinputs
�
�
8__inference_BatchNormProteinDecode1_layer_call_fn_188797

inputs
unknown:9
	unknown_0:9
	unknown_1:9
	unknown_2:9
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������9*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_186498o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������9`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������9: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������9
 
_user_specified_nameinputs
�

�
J__inference_gene_decoder_1_layer_call_and_return_conditional_losses_186633

inputs0
matmul_readvariableop_resource:@}-
biasadd_readvariableop_resource:}
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@}*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:}*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������}Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������}w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
/__inference_gene_encoder_2_layer_call_fn_188327

inputs
unknown:	�}
	unknown_0:}
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_186572o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������}`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�%
�
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_188318

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
M__inference_protein_decoder_1_layer_call_and_return_conditional_losses_188691

inputs0
matmul_readvariableop_resource:@9-
biasadd_readvariableop_resource:9
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@9*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������9r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:9*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������9V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������9Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������9w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
8__inference_BatchNormProteinEncode1_layer_call_fn_188464

inputs
unknown:9
	unknown_0:9
	unknown_1:9
	unknown_2:9
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������9*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_186252o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������9`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������9: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������9
 
_user_specified_nameinputs
��
�L
"__inference__traced_restore_189604
file_prefix:
&assignvariableop_gene_encoder_1_kernel:
��5
&assignvariableop_1_gene_encoder_1_bias:	�<
-assignvariableop_2_batchnormgeneencode1_gamma:	�;
,assignvariableop_3_batchnormgeneencode1_beta:	�B
3assignvariableop_4_batchnormgeneencode1_moving_mean:	�F
7assignvariableop_5_batchnormgeneencode1_moving_variance:	�;
(assignvariableop_6_gene_encoder_2_kernel:	�}4
&assignvariableop_7_gene_encoder_2_bias:}>
+assignvariableop_8_protein_encoder_1_kernel:	�97
)assignvariableop_9_protein_encoder_1_bias:9<
.assignvariableop_10_batchnormgeneencode2_gamma:};
-assignvariableop_11_batchnormgeneencode2_beta:}B
4assignvariableop_12_batchnormgeneencode2_moving_mean:}F
8assignvariableop_13_batchnormgeneencode2_moving_variance:}?
1assignvariableop_14_batchnormproteinencode1_gamma:9>
0assignvariableop_15_batchnormproteinencode1_beta:9E
7assignvariableop_16_batchnormproteinencode1_moving_mean:9I
;assignvariableop_17_batchnormproteinencode1_moving_variance:9?
,assignvariableop_18_embeddingdimdense_kernel:	�@8
*assignvariableop_19_embeddingdimdense_bias:@;
)assignvariableop_20_gene_decoder_1_kernel:@}5
'assignvariableop_21_gene_decoder_1_bias:}<
.assignvariableop_22_batchnormgenedecode1_gamma:};
-assignvariableop_23_batchnormgenedecode1_beta:}B
4assignvariableop_24_batchnormgenedecode1_moving_mean:}F
8assignvariableop_25_batchnormgenedecode1_moving_variance:}<
)assignvariableop_26_gene_decoder_2_kernel:	}�6
'assignvariableop_27_gene_decoder_2_bias:	�>
,assignvariableop_28_protein_decoder_1_kernel:@98
*assignvariableop_29_protein_decoder_1_bias:9=
.assignvariableop_30_batchnormgenedecode2_gamma:	�<
-assignvariableop_31_batchnormgenedecode2_beta:	�C
4assignvariableop_32_batchnormgenedecode2_moving_mean:	�G
8assignvariableop_33_batchnormgenedecode2_moving_variance:	�?
1assignvariableop_34_batchnormproteindecode1_gamma:9>
0assignvariableop_35_batchnormproteindecode1_beta:9E
7assignvariableop_36_batchnormproteindecode1_moving_mean:9I
;assignvariableop_37_batchnormproteindecode1_moving_variance:9@
,assignvariableop_38_gene_decoder_last_kernel:
��9
*assignvariableop_39_gene_decoder_last_bias:	�B
/assignvariableop_40_protein_decoder_last_kernel:	9�<
-assignvariableop_41_protein_decoder_last_bias:	�'
assignvariableop_42_adam_iter:	 )
assignvariableop_43_adam_beta_1: )
assignvariableop_44_adam_beta_2: (
assignvariableop_45_adam_decay: 0
&assignvariableop_46_adam_learning_rate: #
assignvariableop_47_total: #
assignvariableop_48_count: %
assignvariableop_49_total_1: %
assignvariableop_50_count_1: %
assignvariableop_51_total_2: %
assignvariableop_52_count_2: D
0assignvariableop_53_adam_gene_encoder_1_kernel_m:
��=
.assignvariableop_54_adam_gene_encoder_1_bias_m:	�D
5assignvariableop_55_adam_batchnormgeneencode1_gamma_m:	�C
4assignvariableop_56_adam_batchnormgeneencode1_beta_m:	�C
0assignvariableop_57_adam_gene_encoder_2_kernel_m:	�}<
.assignvariableop_58_adam_gene_encoder_2_bias_m:}F
3assignvariableop_59_adam_protein_encoder_1_kernel_m:	�9?
1assignvariableop_60_adam_protein_encoder_1_bias_m:9C
5assignvariableop_61_adam_batchnormgeneencode2_gamma_m:}B
4assignvariableop_62_adam_batchnormgeneencode2_beta_m:}F
8assignvariableop_63_adam_batchnormproteinencode1_gamma_m:9E
7assignvariableop_64_adam_batchnormproteinencode1_beta_m:9F
3assignvariableop_65_adam_embeddingdimdense_kernel_m:	�@?
1assignvariableop_66_adam_embeddingdimdense_bias_m:@B
0assignvariableop_67_adam_gene_decoder_1_kernel_m:@}<
.assignvariableop_68_adam_gene_decoder_1_bias_m:}C
5assignvariableop_69_adam_batchnormgenedecode1_gamma_m:}B
4assignvariableop_70_adam_batchnormgenedecode1_beta_m:}C
0assignvariableop_71_adam_gene_decoder_2_kernel_m:	}�=
.assignvariableop_72_adam_gene_decoder_2_bias_m:	�E
3assignvariableop_73_adam_protein_decoder_1_kernel_m:@9?
1assignvariableop_74_adam_protein_decoder_1_bias_m:9D
5assignvariableop_75_adam_batchnormgenedecode2_gamma_m:	�C
4assignvariableop_76_adam_batchnormgenedecode2_beta_m:	�F
8assignvariableop_77_adam_batchnormproteindecode1_gamma_m:9E
7assignvariableop_78_adam_batchnormproteindecode1_beta_m:9G
3assignvariableop_79_adam_gene_decoder_last_kernel_m:
��@
1assignvariableop_80_adam_gene_decoder_last_bias_m:	�I
6assignvariableop_81_adam_protein_decoder_last_kernel_m:	9�C
4assignvariableop_82_adam_protein_decoder_last_bias_m:	�D
0assignvariableop_83_adam_gene_encoder_1_kernel_v:
��=
.assignvariableop_84_adam_gene_encoder_1_bias_v:	�D
5assignvariableop_85_adam_batchnormgeneencode1_gamma_v:	�C
4assignvariableop_86_adam_batchnormgeneencode1_beta_v:	�C
0assignvariableop_87_adam_gene_encoder_2_kernel_v:	�}<
.assignvariableop_88_adam_gene_encoder_2_bias_v:}F
3assignvariableop_89_adam_protein_encoder_1_kernel_v:	�9?
1assignvariableop_90_adam_protein_encoder_1_bias_v:9C
5assignvariableop_91_adam_batchnormgeneencode2_gamma_v:}B
4assignvariableop_92_adam_batchnormgeneencode2_beta_v:}F
8assignvariableop_93_adam_batchnormproteinencode1_gamma_v:9E
7assignvariableop_94_adam_batchnormproteinencode1_beta_v:9F
3assignvariableop_95_adam_embeddingdimdense_kernel_v:	�@?
1assignvariableop_96_adam_embeddingdimdense_bias_v:@B
0assignvariableop_97_adam_gene_decoder_1_kernel_v:@}<
.assignvariableop_98_adam_gene_decoder_1_bias_v:}C
5assignvariableop_99_adam_batchnormgenedecode1_gamma_v:}C
5assignvariableop_100_adam_batchnormgenedecode1_beta_v:}D
1assignvariableop_101_adam_gene_decoder_2_kernel_v:	}�>
/assignvariableop_102_adam_gene_decoder_2_bias_v:	�F
4assignvariableop_103_adam_protein_decoder_1_kernel_v:@9@
2assignvariableop_104_adam_protein_decoder_1_bias_v:9E
6assignvariableop_105_adam_batchnormgenedecode2_gamma_v:	�D
5assignvariableop_106_adam_batchnormgenedecode2_beta_v:	�G
9assignvariableop_107_adam_batchnormproteindecode1_gamma_v:9F
8assignvariableop_108_adam_batchnormproteindecode1_beta_v:9H
4assignvariableop_109_adam_gene_decoder_last_kernel_v:
��A
2assignvariableop_110_adam_gene_decoder_last_bias_v:	�J
7assignvariableop_111_adam_protein_decoder_last_kernel_v:	9�D
5assignvariableop_112_adam_protein_decoder_last_bias_v:	�
identity_114��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_103�AssignVariableOp_104�AssignVariableOp_105�AssignVariableOp_106�AssignVariableOp_107�AssignVariableOp_108�AssignVariableOp_109�AssignVariableOp_11�AssignVariableOp_110�AssignVariableOp_111�AssignVariableOp_112�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:r*
dtype0*�>
value�>B�>rB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-4/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-8/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:r*
dtype0*�
value�B�rB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*�
dtypesv
t2r	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp&assignvariableop_gene_encoder_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp&assignvariableop_1_gene_encoder_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp-assignvariableop_2_batchnormgeneencode1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp,assignvariableop_3_batchnormgeneencode1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp3assignvariableop_4_batchnormgeneencode1_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp7assignvariableop_5_batchnormgeneencode1_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp(assignvariableop_6_gene_encoder_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp&assignvariableop_7_gene_encoder_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp+assignvariableop_8_protein_encoder_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp)assignvariableop_9_protein_encoder_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp.assignvariableop_10_batchnormgeneencode2_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp-assignvariableop_11_batchnormgeneencode2_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp4assignvariableop_12_batchnormgeneencode2_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp8assignvariableop_13_batchnormgeneencode2_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp1assignvariableop_14_batchnormproteinencode1_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp0assignvariableop_15_batchnormproteinencode1_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp7assignvariableop_16_batchnormproteinencode1_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp;assignvariableop_17_batchnormproteinencode1_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp,assignvariableop_18_embeddingdimdense_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp*assignvariableop_19_embeddingdimdense_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp)assignvariableop_20_gene_decoder_1_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp'assignvariableop_21_gene_decoder_1_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp.assignvariableop_22_batchnormgenedecode1_gammaIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp-assignvariableop_23_batchnormgenedecode1_betaIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp4assignvariableop_24_batchnormgenedecode1_moving_meanIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp8assignvariableop_25_batchnormgenedecode1_moving_varianceIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp)assignvariableop_26_gene_decoder_2_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp'assignvariableop_27_gene_decoder_2_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp,assignvariableop_28_protein_decoder_1_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp*assignvariableop_29_protein_decoder_1_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp.assignvariableop_30_batchnormgenedecode2_gammaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp-assignvariableop_31_batchnormgenedecode2_betaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp4assignvariableop_32_batchnormgenedecode2_moving_meanIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp8assignvariableop_33_batchnormgenedecode2_moving_varianceIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp1assignvariableop_34_batchnormproteindecode1_gammaIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp0assignvariableop_35_batchnormproteindecode1_betaIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp7assignvariableop_36_batchnormproteindecode1_moving_meanIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp;assignvariableop_37_batchnormproteindecode1_moving_varianceIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp,assignvariableop_38_gene_decoder_last_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp*assignvariableop_39_gene_decoder_last_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp/assignvariableop_40_protein_decoder_last_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp-assignvariableop_41_protein_decoder_last_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_42AssignVariableOpassignvariableop_42_adam_iterIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOpassignvariableop_43_adam_beta_1Identity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOpassignvariableop_44_adam_beta_2Identity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOpassignvariableop_45_adam_decayIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp&assignvariableop_46_adam_learning_rateIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOpassignvariableop_47_totalIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOpassignvariableop_48_countIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOpassignvariableop_49_total_1Identity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOpassignvariableop_50_count_1Identity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOpassignvariableop_51_total_2Identity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOpassignvariableop_52_count_2Identity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp0assignvariableop_53_adam_gene_encoder_1_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp.assignvariableop_54_adam_gene_encoder_1_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp5assignvariableop_55_adam_batchnormgeneencode1_gamma_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp4assignvariableop_56_adam_batchnormgeneencode1_beta_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp0assignvariableop_57_adam_gene_encoder_2_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp.assignvariableop_58_adam_gene_encoder_2_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp3assignvariableop_59_adam_protein_encoder_1_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp1assignvariableop_60_adam_protein_encoder_1_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp5assignvariableop_61_adam_batchnormgeneencode2_gamma_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp4assignvariableop_62_adam_batchnormgeneencode2_beta_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp8assignvariableop_63_adam_batchnormproteinencode1_gamma_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp7assignvariableop_64_adam_batchnormproteinencode1_beta_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp3assignvariableop_65_adam_embeddingdimdense_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp1assignvariableop_66_adam_embeddingdimdense_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp0assignvariableop_67_adam_gene_decoder_1_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp.assignvariableop_68_adam_gene_decoder_1_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp5assignvariableop_69_adam_batchnormgenedecode1_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp4assignvariableop_70_adam_batchnormgenedecode1_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp0assignvariableop_71_adam_gene_decoder_2_kernel_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp.assignvariableop_72_adam_gene_decoder_2_bias_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp3assignvariableop_73_adam_protein_decoder_1_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp1assignvariableop_74_adam_protein_decoder_1_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp5assignvariableop_75_adam_batchnormgenedecode2_gamma_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp4assignvariableop_76_adam_batchnormgenedecode2_beta_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp8assignvariableop_77_adam_batchnormproteindecode1_gamma_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp7assignvariableop_78_adam_batchnormproteindecode1_beta_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp3assignvariableop_79_adam_gene_decoder_last_kernel_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp1assignvariableop_80_adam_gene_decoder_last_bias_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp6assignvariableop_81_adam_protein_decoder_last_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp4assignvariableop_82_adam_protein_decoder_last_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp0assignvariableop_83_adam_gene_encoder_1_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp.assignvariableop_84_adam_gene_encoder_1_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp5assignvariableop_85_adam_batchnormgeneencode1_gamma_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOp4assignvariableop_86_adam_batchnormgeneencode1_beta_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp0assignvariableop_87_adam_gene_encoder_2_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp.assignvariableop_88_adam_gene_encoder_2_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp3assignvariableop_89_adam_protein_encoder_1_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp1assignvariableop_90_adam_protein_encoder_1_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOp5assignvariableop_91_adam_batchnormgeneencode2_gamma_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOp4assignvariableop_92_adam_batchnormgeneencode2_beta_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOp8assignvariableop_93_adam_batchnormproteinencode1_gamma_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOp7assignvariableop_94_adam_batchnormproteinencode1_beta_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOp3assignvariableop_95_adam_embeddingdimdense_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOp1assignvariableop_96_adam_embeddingdimdense_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOp0assignvariableop_97_adam_gene_decoder_1_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOp.assignvariableop_98_adam_gene_decoder_1_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_99AssignVariableOp5assignvariableop_99_adam_batchnormgenedecode1_gamma_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_100AssignVariableOp5assignvariableop_100_adam_batchnormgenedecode1_beta_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_101AssignVariableOp1assignvariableop_101_adam_gene_decoder_2_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_102AssignVariableOp/assignvariableop_102_adam_gene_decoder_2_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_103AssignVariableOp4assignvariableop_103_adam_protein_decoder_1_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_104AssignVariableOp2assignvariableop_104_adam_protein_decoder_1_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_105AssignVariableOp6assignvariableop_105_adam_batchnormgenedecode2_gamma_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_106AssignVariableOp5assignvariableop_106_adam_batchnormgenedecode2_beta_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_107AssignVariableOp9assignvariableop_107_adam_batchnormproteindecode1_gamma_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_108AssignVariableOp8assignvariableop_108_adam_batchnormproteindecode1_beta_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_109AssignVariableOp4assignvariableop_109_adam_gene_decoder_last_kernel_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_110AssignVariableOp2assignvariableop_110_adam_gene_decoder_last_bias_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_111AssignVariableOp7assignvariableop_111_adam_protein_decoder_last_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_112AssignVariableOp5assignvariableop_112_adam_protein_decoder_last_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_113Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_114IdentityIdentity_113:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_114Identity_114:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022,
AssignVariableOp_103AssignVariableOp_1032,
AssignVariableOp_104AssignVariableOp_1042,
AssignVariableOp_105AssignVariableOp_1052,
AssignVariableOp_106AssignVariableOp_1062,
AssignVariableOp_107AssignVariableOp_1072,
AssignVariableOp_108AssignVariableOp_1082,
AssignVariableOp_109AssignVariableOp_1092*
AssignVariableOp_11AssignVariableOp_112,
AssignVariableOp_110AssignVariableOp_1102,
AssignVariableOp_111AssignVariableOp_1112,
AssignVariableOp_112AssignVariableOp_1122*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
v
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_186603

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :v
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������}:���������9:O K
'
_output_shapes
:���������}
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������9
 
_user_specified_nameinputs
�
�

&__inference_model_layer_call_fn_186825
gene_input_layer
protein_input_layer
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�9
	unknown_6:9
	unknown_7:	�}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:9

unknown_14:9

unknown_15:9

unknown_16:9

unknown_17:	�@

unknown_18:@

unknown_19:@}

unknown_20:}

unknown_21:}

unknown_22:}

unknown_23:}

unknown_24:}

unknown_25:@9

unknown_26:9

unknown_27:	}�

unknown_28:	�

unknown_29:9

unknown_30:9

unknown_31:9

unknown_32:9

unknown_33:	�

unknown_34:	�

unknown_35:	�

unknown_36:	�

unknown_37:	9�

unknown_38:	�

unknown_39:
��

unknown_40:	�
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgene_input_layerprotein_input_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:����������:����������*L
_read_only_resource_inputs.
,*	
 !"#$%&'()*+*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_186736p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:����������
*
_user_specified_nameGene_Input_Layer:]Y
(
_output_shapes
:����������
-
_user_specified_nameProtein_Input_Layer
�%
�
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_188438

inputs5
'assignmovingavg_readvariableop_resource:}7
)assignmovingavg_1_readvariableop_resource:}3
%batchnorm_mul_readvariableop_resource:}/
!batchnorm_readvariableop_resource:}
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:}*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:}�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������}l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:}*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:}*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:}*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:}*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:}x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:}�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:}*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:}~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:}�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:}P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:}~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������}h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:}v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:}r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������}�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������}: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������}
 
_user_specified_nameinputs
�
�
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_186205

inputs/
!batchnorm_readvariableop_resource:93
%batchnorm_mul_readvariableop_resource:91
#batchnorm_readvariableop_1_resource:91
#batchnorm_readvariableop_2_resource:9
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:9*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:9P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:9~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:9*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:9c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������9z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:9*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:9z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:9*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:9r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������9b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������9�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������9: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������9
 
_user_specified_nameinputs
��
�'
A__inference_model_layer_call_and_return_conditional_losses_187873
inputs_0
inputs_1A
-gene_encoder_1_matmul_readvariableop_resource:
��=
.gene_encoder_1_biasadd_readvariableop_resource:	�E
6batchnormgeneencode1_batchnorm_readvariableop_resource:	�I
:batchnormgeneencode1_batchnorm_mul_readvariableop_resource:	�G
8batchnormgeneencode1_batchnorm_readvariableop_1_resource:	�G
8batchnormgeneencode1_batchnorm_readvariableop_2_resource:	�C
0protein_encoder_1_matmul_readvariableop_resource:	�9?
1protein_encoder_1_biasadd_readvariableop_resource:9@
-gene_encoder_2_matmul_readvariableop_resource:	�}<
.gene_encoder_2_biasadd_readvariableop_resource:}D
6batchnormgeneencode2_batchnorm_readvariableop_resource:}H
:batchnormgeneencode2_batchnorm_mul_readvariableop_resource:}F
8batchnormgeneencode2_batchnorm_readvariableop_1_resource:}F
8batchnormgeneencode2_batchnorm_readvariableop_2_resource:}G
9batchnormproteinencode1_batchnorm_readvariableop_resource:9K
=batchnormproteinencode1_batchnorm_mul_readvariableop_resource:9I
;batchnormproteinencode1_batchnorm_readvariableop_1_resource:9I
;batchnormproteinencode1_batchnorm_readvariableop_2_resource:9C
0embeddingdimdense_matmul_readvariableop_resource:	�@?
1embeddingdimdense_biasadd_readvariableop_resource:@?
-gene_decoder_1_matmul_readvariableop_resource:@}<
.gene_decoder_1_biasadd_readvariableop_resource:}D
6batchnormgenedecode1_batchnorm_readvariableop_resource:}H
:batchnormgenedecode1_batchnorm_mul_readvariableop_resource:}F
8batchnormgenedecode1_batchnorm_readvariableop_1_resource:}F
8batchnormgenedecode1_batchnorm_readvariableop_2_resource:}B
0protein_decoder_1_matmul_readvariableop_resource:@9?
1protein_decoder_1_biasadd_readvariableop_resource:9@
-gene_decoder_2_matmul_readvariableop_resource:	}�=
.gene_decoder_2_biasadd_readvariableop_resource:	�G
9batchnormproteindecode1_batchnorm_readvariableop_resource:9K
=batchnormproteindecode1_batchnorm_mul_readvariableop_resource:9I
;batchnormproteindecode1_batchnorm_readvariableop_1_resource:9I
;batchnormproteindecode1_batchnorm_readvariableop_2_resource:9E
6batchnormgenedecode2_batchnorm_readvariableop_resource:	�I
:batchnormgenedecode2_batchnorm_mul_readvariableop_resource:	�G
8batchnormgenedecode2_batchnorm_readvariableop_1_resource:	�G
8batchnormgenedecode2_batchnorm_readvariableop_2_resource:	�F
3protein_decoder_last_matmul_readvariableop_resource:	9�C
4protein_decoder_last_biasadd_readvariableop_resource:	�D
0gene_decoder_last_matmul_readvariableop_resource:
��@
1gene_decoder_last_biasadd_readvariableop_resource:	�
identity

identity_1��-BatchNormGeneDecode1/batchnorm/ReadVariableOp�/BatchNormGeneDecode1/batchnorm/ReadVariableOp_1�/BatchNormGeneDecode1/batchnorm/ReadVariableOp_2�1BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp�-BatchNormGeneDecode2/batchnorm/ReadVariableOp�/BatchNormGeneDecode2/batchnorm/ReadVariableOp_1�/BatchNormGeneDecode2/batchnorm/ReadVariableOp_2�1BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp�-BatchNormGeneEncode1/batchnorm/ReadVariableOp�/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1�/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2�1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp�-BatchNormGeneEncode2/batchnorm/ReadVariableOp�/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1�/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2�1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp�0BatchNormProteinDecode1/batchnorm/ReadVariableOp�2BatchNormProteinDecode1/batchnorm/ReadVariableOp_1�2BatchNormProteinDecode1/batchnorm/ReadVariableOp_2�4BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp�0BatchNormProteinEncode1/batchnorm/ReadVariableOp�2BatchNormProteinEncode1/batchnorm/ReadVariableOp_1�2BatchNormProteinEncode1/batchnorm/ReadVariableOp_2�4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp�(EmbeddingDimDense/BiasAdd/ReadVariableOp�'EmbeddingDimDense/MatMul/ReadVariableOp�%gene_decoder_1/BiasAdd/ReadVariableOp�$gene_decoder_1/MatMul/ReadVariableOp�%gene_decoder_2/BiasAdd/ReadVariableOp�$gene_decoder_2/MatMul/ReadVariableOp�(gene_decoder_last/BiasAdd/ReadVariableOp�'gene_decoder_last/MatMul/ReadVariableOp�%gene_encoder_1/BiasAdd/ReadVariableOp�$gene_encoder_1/MatMul/ReadVariableOp�%gene_encoder_2/BiasAdd/ReadVariableOp�$gene_encoder_2/MatMul/ReadVariableOp�(protein_decoder_1/BiasAdd/ReadVariableOp�'protein_decoder_1/MatMul/ReadVariableOp�+protein_decoder_last/BiasAdd/ReadVariableOp�*protein_decoder_last/MatMul/ReadVariableOp�(protein_encoder_1/BiasAdd/ReadVariableOp�'protein_encoder_1/MatMul/ReadVariableOp�
$gene_encoder_1/MatMul/ReadVariableOpReadVariableOp-gene_encoder_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gene_encoder_1/MatMulMatMulinputs_0,gene_encoder_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%gene_encoder_1/BiasAdd/ReadVariableOpReadVariableOp.gene_encoder_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
gene_encoder_1/BiasAddBiasAddgene_encoder_1/MatMul:product:0-gene_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
gene_encoder_1/SigmoidSigmoidgene_encoder_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
-BatchNormGeneEncode1/batchnorm/ReadVariableOpReadVariableOp6batchnormgeneencode1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0i
$BatchNormGeneEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
"BatchNormGeneEncode1/batchnorm/addAddV25BatchNormGeneEncode1/batchnorm/ReadVariableOp:value:0-BatchNormGeneEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�{
$BatchNormGeneEncode1/batchnorm/RsqrtRsqrt&BatchNormGeneEncode1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgeneencode1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"BatchNormGeneEncode1/batchnorm/mulMul(BatchNormGeneEncode1/batchnorm/Rsqrt:y:09BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
$BatchNormGeneEncode1/batchnorm/mul_1Mulgene_encoder_1/Sigmoid:y:0&BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1ReadVariableOp8batchnormgeneencode1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
$BatchNormGeneEncode1/batchnorm/mul_2Mul7BatchNormGeneEncode1/batchnorm/ReadVariableOp_1:value:0&BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2ReadVariableOp8batchnormgeneencode1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
"BatchNormGeneEncode1/batchnorm/subSub7BatchNormGeneEncode1/batchnorm/ReadVariableOp_2:value:0(BatchNormGeneEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
$BatchNormGeneEncode1/batchnorm/add_1AddV2(BatchNormGeneEncode1/batchnorm/mul_1:z:0&BatchNormGeneEncode1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
'protein_encoder_1/MatMul/ReadVariableOpReadVariableOp0protein_encoder_1_matmul_readvariableop_resource*
_output_shapes
:	�9*
dtype0�
protein_encoder_1/MatMulMatMulinputs_1/protein_encoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������9�
(protein_encoder_1/BiasAdd/ReadVariableOpReadVariableOp1protein_encoder_1_biasadd_readvariableop_resource*
_output_shapes
:9*
dtype0�
protein_encoder_1/BiasAddBiasAdd"protein_encoder_1/MatMul:product:00protein_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������9z
protein_encoder_1/SigmoidSigmoid"protein_encoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������9�
$gene_encoder_2/MatMul/ReadVariableOpReadVariableOp-gene_encoder_2_matmul_readvariableop_resource*
_output_shapes
:	�}*
dtype0�
gene_encoder_2/MatMulMatMul(BatchNormGeneEncode1/batchnorm/add_1:z:0,gene_encoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}�
%gene_encoder_2/BiasAdd/ReadVariableOpReadVariableOp.gene_encoder_2_biasadd_readvariableop_resource*
_output_shapes
:}*
dtype0�
gene_encoder_2/BiasAddBiasAddgene_encoder_2/MatMul:product:0-gene_encoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}t
gene_encoder_2/SigmoidSigmoidgene_encoder_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������}�
-BatchNormGeneEncode2/batchnorm/ReadVariableOpReadVariableOp6batchnormgeneencode2_batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0i
$BatchNormGeneEncode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
"BatchNormGeneEncode2/batchnorm/addAddV25BatchNormGeneEncode2/batchnorm/ReadVariableOp:value:0-BatchNormGeneEncode2/batchnorm/add/y:output:0*
T0*
_output_shapes
:}z
$BatchNormGeneEncode2/batchnorm/RsqrtRsqrt&BatchNormGeneEncode2/batchnorm/add:z:0*
T0*
_output_shapes
:}�
1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgeneencode2_batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0�
"BatchNormGeneEncode2/batchnorm/mulMul(BatchNormGeneEncode2/batchnorm/Rsqrt:y:09BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}�
$BatchNormGeneEncode2/batchnorm/mul_1Mulgene_encoder_2/Sigmoid:y:0&BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������}�
/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1ReadVariableOp8batchnormgeneencode2_batchnorm_readvariableop_1_resource*
_output_shapes
:}*
dtype0�
$BatchNormGeneEncode2/batchnorm/mul_2Mul7BatchNormGeneEncode2/batchnorm/ReadVariableOp_1:value:0&BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*
_output_shapes
:}�
/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2ReadVariableOp8batchnormgeneencode2_batchnorm_readvariableop_2_resource*
_output_shapes
:}*
dtype0�
"BatchNormGeneEncode2/batchnorm/subSub7BatchNormGeneEncode2/batchnorm/ReadVariableOp_2:value:0(BatchNormGeneEncode2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:}�
$BatchNormGeneEncode2/batchnorm/add_1AddV2(BatchNormGeneEncode2/batchnorm/mul_1:z:0&BatchNormGeneEncode2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}�
0BatchNormProteinEncode1/batchnorm/ReadVariableOpReadVariableOp9batchnormproteinencode1_batchnorm_readvariableop_resource*
_output_shapes
:9*
dtype0l
'BatchNormProteinEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
%BatchNormProteinEncode1/batchnorm/addAddV28BatchNormProteinEncode1/batchnorm/ReadVariableOp:value:00BatchNormProteinEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:9�
'BatchNormProteinEncode1/batchnorm/RsqrtRsqrt)BatchNormProteinEncode1/batchnorm/add:z:0*
T0*
_output_shapes
:9�
4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOpReadVariableOp=batchnormproteinencode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:9*
dtype0�
%BatchNormProteinEncode1/batchnorm/mulMul+BatchNormProteinEncode1/batchnorm/Rsqrt:y:0<BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:9�
'BatchNormProteinEncode1/batchnorm/mul_1Mulprotein_encoder_1/Sigmoid:y:0)BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������9�
2BatchNormProteinEncode1/batchnorm/ReadVariableOp_1ReadVariableOp;batchnormproteinencode1_batchnorm_readvariableop_1_resource*
_output_shapes
:9*
dtype0�
'BatchNormProteinEncode1/batchnorm/mul_2Mul:BatchNormProteinEncode1/batchnorm/ReadVariableOp_1:value:0)BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*
_output_shapes
:9�
2BatchNormProteinEncode1/batchnorm/ReadVariableOp_2ReadVariableOp;batchnormproteinencode1_batchnorm_readvariableop_2_resource*
_output_shapes
:9*
dtype0�
%BatchNormProteinEncode1/batchnorm/subSub:BatchNormProteinEncode1/batchnorm/ReadVariableOp_2:value:0+BatchNormProteinEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:9�
'BatchNormProteinEncode1/batchnorm/add_1AddV2+BatchNormProteinEncode1/batchnorm/mul_1:z:0)BatchNormProteinEncode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������9^
ConcatenateLayer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
ConcatenateLayer/concatConcatV2(BatchNormGeneEncode2/batchnorm/add_1:z:0+BatchNormProteinEncode1/batchnorm/add_1:z:0%ConcatenateLayer/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
'EmbeddingDimDense/MatMul/ReadVariableOpReadVariableOp0embeddingdimdense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
EmbeddingDimDense/MatMulMatMul ConcatenateLayer/concat:output:0/EmbeddingDimDense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
(EmbeddingDimDense/BiasAdd/ReadVariableOpReadVariableOp1embeddingdimdense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
EmbeddingDimDense/BiasAddBiasAdd"EmbeddingDimDense/MatMul:product:00EmbeddingDimDense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@z
EmbeddingDimDense/SigmoidSigmoid"EmbeddingDimDense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
$gene_decoder_1/MatMul/ReadVariableOpReadVariableOp-gene_decoder_1_matmul_readvariableop_resource*
_output_shapes

:@}*
dtype0�
gene_decoder_1/MatMulMatMulEmbeddingDimDense/Sigmoid:y:0,gene_decoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}�
%gene_decoder_1/BiasAdd/ReadVariableOpReadVariableOp.gene_decoder_1_biasadd_readvariableop_resource*
_output_shapes
:}*
dtype0�
gene_decoder_1/BiasAddBiasAddgene_decoder_1/MatMul:product:0-gene_decoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}t
gene_decoder_1/SigmoidSigmoidgene_decoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������}�
-BatchNormGeneDecode1/batchnorm/ReadVariableOpReadVariableOp6batchnormgenedecode1_batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0i
$BatchNormGeneDecode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
"BatchNormGeneDecode1/batchnorm/addAddV25BatchNormGeneDecode1/batchnorm/ReadVariableOp:value:0-BatchNormGeneDecode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:}z
$BatchNormGeneDecode1/batchnorm/RsqrtRsqrt&BatchNormGeneDecode1/batchnorm/add:z:0*
T0*
_output_shapes
:}�
1BatchNormGeneDecode1/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgenedecode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0�
"BatchNormGeneDecode1/batchnorm/mulMul(BatchNormGeneDecode1/batchnorm/Rsqrt:y:09BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}�
$BatchNormGeneDecode1/batchnorm/mul_1Mulgene_decoder_1/Sigmoid:y:0&BatchNormGeneDecode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������}�
/BatchNormGeneDecode1/batchnorm/ReadVariableOp_1ReadVariableOp8batchnormgenedecode1_batchnorm_readvariableop_1_resource*
_output_shapes
:}*
dtype0�
$BatchNormGeneDecode1/batchnorm/mul_2Mul7BatchNormGeneDecode1/batchnorm/ReadVariableOp_1:value:0&BatchNormGeneDecode1/batchnorm/mul:z:0*
T0*
_output_shapes
:}�
/BatchNormGeneDecode1/batchnorm/ReadVariableOp_2ReadVariableOp8batchnormgenedecode1_batchnorm_readvariableop_2_resource*
_output_shapes
:}*
dtype0�
"BatchNormGeneDecode1/batchnorm/subSub7BatchNormGeneDecode1/batchnorm/ReadVariableOp_2:value:0(BatchNormGeneDecode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:}�
$BatchNormGeneDecode1/batchnorm/add_1AddV2(BatchNormGeneDecode1/batchnorm/mul_1:z:0&BatchNormGeneDecode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}�
'protein_decoder_1/MatMul/ReadVariableOpReadVariableOp0protein_decoder_1_matmul_readvariableop_resource*
_output_shapes

:@9*
dtype0�
protein_decoder_1/MatMulMatMulEmbeddingDimDense/Sigmoid:y:0/protein_decoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������9�
(protein_decoder_1/BiasAdd/ReadVariableOpReadVariableOp1protein_decoder_1_biasadd_readvariableop_resource*
_output_shapes
:9*
dtype0�
protein_decoder_1/BiasAddBiasAdd"protein_decoder_1/MatMul:product:00protein_decoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������9z
protein_decoder_1/SigmoidSigmoid"protein_decoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������9�
$gene_decoder_2/MatMul/ReadVariableOpReadVariableOp-gene_decoder_2_matmul_readvariableop_resource*
_output_shapes
:	}�*
dtype0�
gene_decoder_2/MatMulMatMul(BatchNormGeneDecode1/batchnorm/add_1:z:0,gene_decoder_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%gene_decoder_2/BiasAdd/ReadVariableOpReadVariableOp.gene_decoder_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
gene_decoder_2/BiasAddBiasAddgene_decoder_2/MatMul:product:0-gene_decoder_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
gene_decoder_2/SigmoidSigmoidgene_decoder_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
0BatchNormProteinDecode1/batchnorm/ReadVariableOpReadVariableOp9batchnormproteindecode1_batchnorm_readvariableop_resource*
_output_shapes
:9*
dtype0l
'BatchNormProteinDecode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
%BatchNormProteinDecode1/batchnorm/addAddV28BatchNormProteinDecode1/batchnorm/ReadVariableOp:value:00BatchNormProteinDecode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:9�
'BatchNormProteinDecode1/batchnorm/RsqrtRsqrt)BatchNormProteinDecode1/batchnorm/add:z:0*
T0*
_output_shapes
:9�
4BatchNormProteinDecode1/batchnorm/mul/ReadVariableOpReadVariableOp=batchnormproteindecode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:9*
dtype0�
%BatchNormProteinDecode1/batchnorm/mulMul+BatchNormProteinDecode1/batchnorm/Rsqrt:y:0<BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:9�
'BatchNormProteinDecode1/batchnorm/mul_1Mulprotein_decoder_1/Sigmoid:y:0)BatchNormProteinDecode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������9�
2BatchNormProteinDecode1/batchnorm/ReadVariableOp_1ReadVariableOp;batchnormproteindecode1_batchnorm_readvariableop_1_resource*
_output_shapes
:9*
dtype0�
'BatchNormProteinDecode1/batchnorm/mul_2Mul:BatchNormProteinDecode1/batchnorm/ReadVariableOp_1:value:0)BatchNormProteinDecode1/batchnorm/mul:z:0*
T0*
_output_shapes
:9�
2BatchNormProteinDecode1/batchnorm/ReadVariableOp_2ReadVariableOp;batchnormproteindecode1_batchnorm_readvariableop_2_resource*
_output_shapes
:9*
dtype0�
%BatchNormProteinDecode1/batchnorm/subSub:BatchNormProteinDecode1/batchnorm/ReadVariableOp_2:value:0+BatchNormProteinDecode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:9�
'BatchNormProteinDecode1/batchnorm/add_1AddV2+BatchNormProteinDecode1/batchnorm/mul_1:z:0)BatchNormProteinDecode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������9�
-BatchNormGeneDecode2/batchnorm/ReadVariableOpReadVariableOp6batchnormgenedecode2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0i
$BatchNormGeneDecode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
"BatchNormGeneDecode2/batchnorm/addAddV25BatchNormGeneDecode2/batchnorm/ReadVariableOp:value:0-BatchNormGeneDecode2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�{
$BatchNormGeneDecode2/batchnorm/RsqrtRsqrt&BatchNormGeneDecode2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
1BatchNormGeneDecode2/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgenedecode2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"BatchNormGeneDecode2/batchnorm/mulMul(BatchNormGeneDecode2/batchnorm/Rsqrt:y:09BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
$BatchNormGeneDecode2/batchnorm/mul_1Mulgene_decoder_2/Sigmoid:y:0&BatchNormGeneDecode2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
/BatchNormGeneDecode2/batchnorm/ReadVariableOp_1ReadVariableOp8batchnormgenedecode2_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
$BatchNormGeneDecode2/batchnorm/mul_2Mul7BatchNormGeneDecode2/batchnorm/ReadVariableOp_1:value:0&BatchNormGeneDecode2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
/BatchNormGeneDecode2/batchnorm/ReadVariableOp_2ReadVariableOp8batchnormgenedecode2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
"BatchNormGeneDecode2/batchnorm/subSub7BatchNormGeneDecode2/batchnorm/ReadVariableOp_2:value:0(BatchNormGeneDecode2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
$BatchNormGeneDecode2/batchnorm/add_1AddV2(BatchNormGeneDecode2/batchnorm/mul_1:z:0&BatchNormGeneDecode2/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
*protein_decoder_last/MatMul/ReadVariableOpReadVariableOp3protein_decoder_last_matmul_readvariableop_resource*
_output_shapes
:	9�*
dtype0�
protein_decoder_last/MatMulMatMul+BatchNormProteinDecode1/batchnorm/add_1:z:02protein_decoder_last/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+protein_decoder_last/BiasAdd/ReadVariableOpReadVariableOp4protein_decoder_last_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
protein_decoder_last/BiasAddBiasAdd%protein_decoder_last/MatMul:product:03protein_decoder_last/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
protein_decoder_last/SigmoidSigmoid%protein_decoder_last/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
'gene_decoder_last/MatMul/ReadVariableOpReadVariableOp0gene_decoder_last_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gene_decoder_last/MatMulMatMul(BatchNormGeneDecode2/batchnorm/add_1:z:0/gene_decoder_last/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
(gene_decoder_last/BiasAdd/ReadVariableOpReadVariableOp1gene_decoder_last_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
gene_decoder_last/BiasAddBiasAdd"gene_decoder_last/MatMul:product:00gene_decoder_last/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
gene_decoder_last/SigmoidSigmoid"gene_decoder_last/BiasAdd:output:0*
T0*(
_output_shapes
:����������m
IdentityIdentitygene_decoder_last/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������r

Identity_1Identity protein_decoder_last/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp.^BatchNormGeneDecode1/batchnorm/ReadVariableOp0^BatchNormGeneDecode1/batchnorm/ReadVariableOp_10^BatchNormGeneDecode1/batchnorm/ReadVariableOp_22^BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp.^BatchNormGeneDecode2/batchnorm/ReadVariableOp0^BatchNormGeneDecode2/batchnorm/ReadVariableOp_10^BatchNormGeneDecode2/batchnorm/ReadVariableOp_22^BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp.^BatchNormGeneEncode1/batchnorm/ReadVariableOp0^BatchNormGeneEncode1/batchnorm/ReadVariableOp_10^BatchNormGeneEncode1/batchnorm/ReadVariableOp_22^BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp.^BatchNormGeneEncode2/batchnorm/ReadVariableOp0^BatchNormGeneEncode2/batchnorm/ReadVariableOp_10^BatchNormGeneEncode2/batchnorm/ReadVariableOp_22^BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp1^BatchNormProteinDecode1/batchnorm/ReadVariableOp3^BatchNormProteinDecode1/batchnorm/ReadVariableOp_13^BatchNormProteinDecode1/batchnorm/ReadVariableOp_25^BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp1^BatchNormProteinEncode1/batchnorm/ReadVariableOp3^BatchNormProteinEncode1/batchnorm/ReadVariableOp_13^BatchNormProteinEncode1/batchnorm/ReadVariableOp_25^BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp)^EmbeddingDimDense/BiasAdd/ReadVariableOp(^EmbeddingDimDense/MatMul/ReadVariableOp&^gene_decoder_1/BiasAdd/ReadVariableOp%^gene_decoder_1/MatMul/ReadVariableOp&^gene_decoder_2/BiasAdd/ReadVariableOp%^gene_decoder_2/MatMul/ReadVariableOp)^gene_decoder_last/BiasAdd/ReadVariableOp(^gene_decoder_last/MatMul/ReadVariableOp&^gene_encoder_1/BiasAdd/ReadVariableOp%^gene_encoder_1/MatMul/ReadVariableOp&^gene_encoder_2/BiasAdd/ReadVariableOp%^gene_encoder_2/MatMul/ReadVariableOp)^protein_decoder_1/BiasAdd/ReadVariableOp(^protein_decoder_1/MatMul/ReadVariableOp,^protein_decoder_last/BiasAdd/ReadVariableOp+^protein_decoder_last/MatMul/ReadVariableOp)^protein_encoder_1/BiasAdd/ReadVariableOp(^protein_encoder_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
-BatchNormGeneDecode1/batchnorm/ReadVariableOp-BatchNormGeneDecode1/batchnorm/ReadVariableOp2b
/BatchNormGeneDecode1/batchnorm/ReadVariableOp_1/BatchNormGeneDecode1/batchnorm/ReadVariableOp_12b
/BatchNormGeneDecode1/batchnorm/ReadVariableOp_2/BatchNormGeneDecode1/batchnorm/ReadVariableOp_22f
1BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp1BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp2^
-BatchNormGeneDecode2/batchnorm/ReadVariableOp-BatchNormGeneDecode2/batchnorm/ReadVariableOp2b
/BatchNormGeneDecode2/batchnorm/ReadVariableOp_1/BatchNormGeneDecode2/batchnorm/ReadVariableOp_12b
/BatchNormGeneDecode2/batchnorm/ReadVariableOp_2/BatchNormGeneDecode2/batchnorm/ReadVariableOp_22f
1BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp1BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp2^
-BatchNormGeneEncode1/batchnorm/ReadVariableOp-BatchNormGeneEncode1/batchnorm/ReadVariableOp2b
/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1/BatchNormGeneEncode1/batchnorm/ReadVariableOp_12b
/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2/BatchNormGeneEncode1/batchnorm/ReadVariableOp_22f
1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp2^
-BatchNormGeneEncode2/batchnorm/ReadVariableOp-BatchNormGeneEncode2/batchnorm/ReadVariableOp2b
/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1/BatchNormGeneEncode2/batchnorm/ReadVariableOp_12b
/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2/BatchNormGeneEncode2/batchnorm/ReadVariableOp_22f
1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp2d
0BatchNormProteinDecode1/batchnorm/ReadVariableOp0BatchNormProteinDecode1/batchnorm/ReadVariableOp2h
2BatchNormProteinDecode1/batchnorm/ReadVariableOp_12BatchNormProteinDecode1/batchnorm/ReadVariableOp_12h
2BatchNormProteinDecode1/batchnorm/ReadVariableOp_22BatchNormProteinDecode1/batchnorm/ReadVariableOp_22l
4BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp4BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp2d
0BatchNormProteinEncode1/batchnorm/ReadVariableOp0BatchNormProteinEncode1/batchnorm/ReadVariableOp2h
2BatchNormProteinEncode1/batchnorm/ReadVariableOp_12BatchNormProteinEncode1/batchnorm/ReadVariableOp_12h
2BatchNormProteinEncode1/batchnorm/ReadVariableOp_22BatchNormProteinEncode1/batchnorm/ReadVariableOp_22l
4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp2T
(EmbeddingDimDense/BiasAdd/ReadVariableOp(EmbeddingDimDense/BiasAdd/ReadVariableOp2R
'EmbeddingDimDense/MatMul/ReadVariableOp'EmbeddingDimDense/MatMul/ReadVariableOp2N
%gene_decoder_1/BiasAdd/ReadVariableOp%gene_decoder_1/BiasAdd/ReadVariableOp2L
$gene_decoder_1/MatMul/ReadVariableOp$gene_decoder_1/MatMul/ReadVariableOp2N
%gene_decoder_2/BiasAdd/ReadVariableOp%gene_decoder_2/BiasAdd/ReadVariableOp2L
$gene_decoder_2/MatMul/ReadVariableOp$gene_decoder_2/MatMul/ReadVariableOp2T
(gene_decoder_last/BiasAdd/ReadVariableOp(gene_decoder_last/BiasAdd/ReadVariableOp2R
'gene_decoder_last/MatMul/ReadVariableOp'gene_decoder_last/MatMul/ReadVariableOp2N
%gene_encoder_1/BiasAdd/ReadVariableOp%gene_encoder_1/BiasAdd/ReadVariableOp2L
$gene_encoder_1/MatMul/ReadVariableOp$gene_encoder_1/MatMul/ReadVariableOp2N
%gene_encoder_2/BiasAdd/ReadVariableOp%gene_encoder_2/BiasAdd/ReadVariableOp2L
$gene_encoder_2/MatMul/ReadVariableOp$gene_encoder_2/MatMul/ReadVariableOp2T
(protein_decoder_1/BiasAdd/ReadVariableOp(protein_decoder_1/BiasAdd/ReadVariableOp2R
'protein_decoder_1/MatMul/ReadVariableOp'protein_decoder_1/MatMul/ReadVariableOp2Z
+protein_decoder_last/BiasAdd/ReadVariableOp+protein_decoder_last/BiasAdd/ReadVariableOp2X
*protein_decoder_last/MatMul/ReadVariableOp*protein_decoder_last/MatMul/ReadVariableOp2T
(protein_encoder_1/BiasAdd/ReadVariableOp(protein_encoder_1/BiasAdd/ReadVariableOp2R
'protein_encoder_1/MatMul/ReadVariableOp'protein_encoder_1/MatMul/ReadVariableOp:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�

�
J__inference_gene_decoder_2_layer_call_and_return_conditional_losses_186676

inputs1
matmul_readvariableop_resource:	}�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	}�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������}: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������}
 
_user_specified_nameinputs
�
�
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_188404

inputs/
!batchnorm_readvariableop_resource:}3
%batchnorm_mul_readvariableop_resource:}1
#batchnorm_readvariableop_1_resource:}1
#batchnorm_readvariableop_2_resource:}
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:}P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:}~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������}z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:}*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:}z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:}*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:}r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������}�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������}: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������}
 
_user_specified_nameinputs
�
�
5__inference_BatchNormGeneDecode2_layer_call_fn_188704

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_186369p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
2__inference_protein_decoder_1_layer_call_fn_188680

inputs
unknown:@9
	unknown_0:9
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������9*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_decoder_1_layer_call_and_return_conditional_losses_186659o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������9`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_188817

inputs/
!batchnorm_readvariableop_resource:93
%batchnorm_mul_readvariableop_resource:91
#batchnorm_readvariableop_1_resource:91
#batchnorm_readvariableop_2_resource:9
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:9*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:9P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:9~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:9*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:9c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������9z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:9*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:9z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:9*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:9r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������9b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������9�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������9: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������9
 
_user_specified_nameinputs
�
�	
&__inference_model_layer_call_fn_187706
inputs_0
inputs_1
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�9
	unknown_6:9
	unknown_7:	�}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:9

unknown_14:9

unknown_15:9

unknown_16:9

unknown_17:	�@

unknown_18:@

unknown_19:@}

unknown_20:}

unknown_21:}

unknown_22:}

unknown_23:}

unknown_24:}

unknown_25:@9

unknown_26:9

unknown_27:	}�

unknown_28:	�

unknown_29:9

unknown_30:9

unknown_31:9

unknown_32:9

unknown_33:	�

unknown_34:	�

unknown_35:	�

unknown_36:	�

unknown_37:	9�

unknown_38:	�

unknown_39:
��

unknown_40:	�
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:����������:����������*@
_read_only_resource_inputs"
 	
"#&'()*+*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_187123p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�f
�
A__inference_model_layer_call_and_return_conditional_losses_187516
gene_input_layer
protein_input_layer)
gene_encoder_1_187414:
��$
gene_encoder_1_187416:	�*
batchnormgeneencode1_187419:	�*
batchnormgeneencode1_187421:	�*
batchnormgeneencode1_187423:	�*
batchnormgeneencode1_187425:	�+
protein_encoder_1_187428:	�9&
protein_encoder_1_187430:9(
gene_encoder_2_187433:	�}#
gene_encoder_2_187435:})
batchnormgeneencode2_187438:})
batchnormgeneencode2_187440:})
batchnormgeneencode2_187442:})
batchnormgeneencode2_187444:},
batchnormproteinencode1_187447:9,
batchnormproteinencode1_187449:9,
batchnormproteinencode1_187451:9,
batchnormproteinencode1_187453:9+
embeddingdimdense_187457:	�@&
embeddingdimdense_187459:@'
gene_decoder_1_187462:@}#
gene_decoder_1_187464:})
batchnormgenedecode1_187467:})
batchnormgenedecode1_187469:})
batchnormgenedecode1_187471:})
batchnormgenedecode1_187473:}*
protein_decoder_1_187476:@9&
protein_decoder_1_187478:9(
gene_decoder_2_187481:	}�$
gene_decoder_2_187483:	�,
batchnormproteindecode1_187486:9,
batchnormproteindecode1_187488:9,
batchnormproteindecode1_187490:9,
batchnormproteindecode1_187492:9*
batchnormgenedecode2_187495:	�*
batchnormgenedecode2_187497:	�*
batchnormgenedecode2_187499:	�*
batchnormgenedecode2_187501:	�.
protein_decoder_last_187504:	9�*
protein_decoder_last_187506:	�,
gene_decoder_last_187509:
��'
gene_decoder_last_187511:	�
identity

identity_1��,BatchNormGeneDecode1/StatefulPartitionedCall�,BatchNormGeneDecode2/StatefulPartitionedCall�,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�/BatchNormProteinDecode1/StatefulPartitionedCall�/BatchNormProteinEncode1/StatefulPartitionedCall�)EmbeddingDimDense/StatefulPartitionedCall�&gene_decoder_1/StatefulPartitionedCall�&gene_decoder_2/StatefulPartitionedCall�)gene_decoder_last/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�)protein_decoder_1/StatefulPartitionedCall�,protein_decoder_last/StatefulPartitionedCall�)protein_encoder_1/StatefulPartitionedCall�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallgene_input_layergene_encoder_1_187414gene_encoder_1_187416*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_186529�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_187419batchnormgeneencode1_187421batchnormgeneencode1_187423batchnormgeneencode1_187425*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_186088�
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallprotein_input_layerprotein_encoder_1_187428protein_encoder_1_187430*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������9*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_186555�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_187433gene_encoder_2_187435*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_186572�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_187438batchnormgeneencode2_187440batchnormgeneencode2_187442batchnormgeneencode2_187444*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_186170�
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_187447batchnormproteinencode1_187449batchnormproteinencode1_187451batchnormproteinencode1_187453*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������9*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_186252�
 ConcatenateLayer/PartitionedCallPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:08BatchNormProteinEncode1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_186603�
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_187457embeddingdimdense_187459*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_186616�
&gene_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0gene_decoder_1_187462gene_decoder_1_187464*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_decoder_1_layer_call_and_return_conditional_losses_186633�
,BatchNormGeneDecode1/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_1/StatefulPartitionedCall:output:0batchnormgenedecode1_187467batchnormgenedecode1_187469batchnormgenedecode1_187471batchnormgenedecode1_187473*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_186334�
)protein_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0protein_decoder_1_187476protein_decoder_1_187478*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������9*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_decoder_1_layer_call_and_return_conditional_losses_186659�
&gene_decoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode1/StatefulPartitionedCall:output:0gene_decoder_2_187481gene_decoder_2_187483*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_decoder_2_layer_call_and_return_conditional_losses_186676�
/BatchNormProteinDecode1/StatefulPartitionedCallStatefulPartitionedCall2protein_decoder_1/StatefulPartitionedCall:output:0batchnormproteindecode1_187486batchnormproteindecode1_187488batchnormproteindecode1_187490batchnormproteindecode1_187492*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������9*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_186498�
,BatchNormGeneDecode2/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_2/StatefulPartitionedCall:output:0batchnormgenedecode2_187495batchnormgenedecode2_187497batchnormgenedecode2_187499batchnormgenedecode2_187501*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_186416�
,protein_decoder_last/StatefulPartitionedCallStatefulPartitionedCall8BatchNormProteinDecode1/StatefulPartitionedCall:output:0protein_decoder_last_187504protein_decoder_last_187506*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_protein_decoder_last_layer_call_and_return_conditional_losses_186711�
)gene_decoder_last/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode2/StatefulPartitionedCall:output:0gene_decoder_last_187509gene_decoder_last_187511*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_gene_decoder_last_layer_call_and_return_conditional_losses_186728�
IdentityIdentity2gene_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������

Identity_1Identity5protein_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp-^BatchNormGeneDecode1/StatefulPartitionedCall-^BatchNormGeneDecode2/StatefulPartitionedCall-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinDecode1/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_decoder_1/StatefulPartitionedCall'^gene_decoder_2/StatefulPartitionedCall*^gene_decoder_last/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_decoder_1/StatefulPartitionedCall-^protein_decoder_last/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneDecode1/StatefulPartitionedCall,BatchNormGeneDecode1/StatefulPartitionedCall2\
,BatchNormGeneDecode2/StatefulPartitionedCall,BatchNormGeneDecode2/StatefulPartitionedCall2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2b
/BatchNormProteinDecode1/StatefulPartitionedCall/BatchNormProteinDecode1/StatefulPartitionedCall2b
/BatchNormProteinEncode1/StatefulPartitionedCall/BatchNormProteinEncode1/StatefulPartitionedCall2V
)EmbeddingDimDense/StatefulPartitionedCall)EmbeddingDimDense/StatefulPartitionedCall2P
&gene_decoder_1/StatefulPartitionedCall&gene_decoder_1/StatefulPartitionedCall2P
&gene_decoder_2/StatefulPartitionedCall&gene_decoder_2/StatefulPartitionedCall2V
)gene_decoder_last/StatefulPartitionedCall)gene_decoder_last/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall2V
)protein_decoder_1/StatefulPartitionedCall)protein_decoder_1/StatefulPartitionedCall2\
,protein_decoder_last/StatefulPartitionedCall,protein_decoder_last/StatefulPartitionedCall2V
)protein_encoder_1/StatefulPartitionedCall)protein_encoder_1/StatefulPartitionedCall:Z V
(
_output_shapes
:����������
*
_user_specified_nameGene_Input_Layer:]Y
(
_output_shapes
:����������
-
_user_specified_nameProtein_Input_Layer
�
�
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_188284

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_188484

inputs/
!batchnorm_readvariableop_resource:93
%batchnorm_mul_readvariableop_resource:91
#batchnorm_readvariableop_1_resource:91
#batchnorm_readvariableop_2_resource:9
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:9*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:9P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:9~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:9*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:9c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������9z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:9*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:9z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:9*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:9r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������9b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������9�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������9: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������9
 
_user_specified_nameinputs
�
�
5__inference_BatchNormGeneDecode2_layer_call_fn_188717

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_186416p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�

$__inference_signature_wrapper_188218
gene_input_layer
protein_input_layer
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�9
	unknown_6:9
	unknown_7:	�}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:9

unknown_14:9

unknown_15:9

unknown_16:9

unknown_17:	�@

unknown_18:@

unknown_19:@}

unknown_20:}

unknown_21:}

unknown_22:}

unknown_23:}

unknown_24:}

unknown_25:@9

unknown_26:9

unknown_27:	}�

unknown_28:	�

unknown_29:9

unknown_30:9

unknown_31:9

unknown_32:9

unknown_33:	�

unknown_34:	�

unknown_35:	�

unknown_36:	�

unknown_37:	9�

unknown_38:	�

unknown_39:
��

unknown_40:	�
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgene_input_layerprotein_input_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:����������:����������*L
_read_only_resource_inputs.
,*	
 !"#$%&'()*+*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_186017p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:����������
*
_user_specified_nameGene_Input_Layer:]Y
(
_output_shapes
:����������
-
_user_specified_nameProtein_Input_Layer
�%
�
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_186498

inputs5
'assignmovingavg_readvariableop_resource:97
)assignmovingavg_1_readvariableop_resource:93
%batchnorm_mul_readvariableop_resource:9/
!batchnorm_readvariableop_resource:9
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:9*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:9�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������9l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:9*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:9*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:9*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:9*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:9x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:9�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:9*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:9~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:9�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:9P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:9~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:9*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:9c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������9h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:9v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:9*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:9r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������9b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������9�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������9: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������9
 
_user_specified_nameinputs
�
�
2__inference_protein_encoder_1_layer_call_fn_188347

inputs
unknown:	�9
	unknown_0:9
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������9*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_186555o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������9`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_188358

inputs1
matmul_readvariableop_resource:	�9-
biasadd_readvariableop_resource:9
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�9*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������9r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:9*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������9V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������9Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������9w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
/__inference_gene_encoder_1_layer_call_fn_188227

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_186529p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_186529

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
̋
�,
A__inference_model_layer_call_and_return_conditional_losses_188124
inputs_0
inputs_1A
-gene_encoder_1_matmul_readvariableop_resource:
��=
.gene_encoder_1_biasadd_readvariableop_resource:	�K
<batchnormgeneencode1_assignmovingavg_readvariableop_resource:	�M
>batchnormgeneencode1_assignmovingavg_1_readvariableop_resource:	�I
:batchnormgeneencode1_batchnorm_mul_readvariableop_resource:	�E
6batchnormgeneencode1_batchnorm_readvariableop_resource:	�C
0protein_encoder_1_matmul_readvariableop_resource:	�9?
1protein_encoder_1_biasadd_readvariableop_resource:9@
-gene_encoder_2_matmul_readvariableop_resource:	�}<
.gene_encoder_2_biasadd_readvariableop_resource:}J
<batchnormgeneencode2_assignmovingavg_readvariableop_resource:}L
>batchnormgeneencode2_assignmovingavg_1_readvariableop_resource:}H
:batchnormgeneencode2_batchnorm_mul_readvariableop_resource:}D
6batchnormgeneencode2_batchnorm_readvariableop_resource:}M
?batchnormproteinencode1_assignmovingavg_readvariableop_resource:9O
Abatchnormproteinencode1_assignmovingavg_1_readvariableop_resource:9K
=batchnormproteinencode1_batchnorm_mul_readvariableop_resource:9G
9batchnormproteinencode1_batchnorm_readvariableop_resource:9C
0embeddingdimdense_matmul_readvariableop_resource:	�@?
1embeddingdimdense_biasadd_readvariableop_resource:@?
-gene_decoder_1_matmul_readvariableop_resource:@}<
.gene_decoder_1_biasadd_readvariableop_resource:}J
<batchnormgenedecode1_assignmovingavg_readvariableop_resource:}L
>batchnormgenedecode1_assignmovingavg_1_readvariableop_resource:}H
:batchnormgenedecode1_batchnorm_mul_readvariableop_resource:}D
6batchnormgenedecode1_batchnorm_readvariableop_resource:}B
0protein_decoder_1_matmul_readvariableop_resource:@9?
1protein_decoder_1_biasadd_readvariableop_resource:9@
-gene_decoder_2_matmul_readvariableop_resource:	}�=
.gene_decoder_2_biasadd_readvariableop_resource:	�M
?batchnormproteindecode1_assignmovingavg_readvariableop_resource:9O
Abatchnormproteindecode1_assignmovingavg_1_readvariableop_resource:9K
=batchnormproteindecode1_batchnorm_mul_readvariableop_resource:9G
9batchnormproteindecode1_batchnorm_readvariableop_resource:9K
<batchnormgenedecode2_assignmovingavg_readvariableop_resource:	�M
>batchnormgenedecode2_assignmovingavg_1_readvariableop_resource:	�I
:batchnormgenedecode2_batchnorm_mul_readvariableop_resource:	�E
6batchnormgenedecode2_batchnorm_readvariableop_resource:	�F
3protein_decoder_last_matmul_readvariableop_resource:	9�C
4protein_decoder_last_biasadd_readvariableop_resource:	�D
0gene_decoder_last_matmul_readvariableop_resource:
��@
1gene_decoder_last_biasadd_readvariableop_resource:	�
identity

identity_1��$BatchNormGeneDecode1/AssignMovingAvg�3BatchNormGeneDecode1/AssignMovingAvg/ReadVariableOp�&BatchNormGeneDecode1/AssignMovingAvg_1�5BatchNormGeneDecode1/AssignMovingAvg_1/ReadVariableOp�-BatchNormGeneDecode1/batchnorm/ReadVariableOp�1BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp�$BatchNormGeneDecode2/AssignMovingAvg�3BatchNormGeneDecode2/AssignMovingAvg/ReadVariableOp�&BatchNormGeneDecode2/AssignMovingAvg_1�5BatchNormGeneDecode2/AssignMovingAvg_1/ReadVariableOp�-BatchNormGeneDecode2/batchnorm/ReadVariableOp�1BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp�$BatchNormGeneEncode1/AssignMovingAvg�3BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOp�&BatchNormGeneEncode1/AssignMovingAvg_1�5BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOp�-BatchNormGeneEncode1/batchnorm/ReadVariableOp�1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp�$BatchNormGeneEncode2/AssignMovingAvg�3BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOp�&BatchNormGeneEncode2/AssignMovingAvg_1�5BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOp�-BatchNormGeneEncode2/batchnorm/ReadVariableOp�1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp�'BatchNormProteinDecode1/AssignMovingAvg�6BatchNormProteinDecode1/AssignMovingAvg/ReadVariableOp�)BatchNormProteinDecode1/AssignMovingAvg_1�8BatchNormProteinDecode1/AssignMovingAvg_1/ReadVariableOp�0BatchNormProteinDecode1/batchnorm/ReadVariableOp�4BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp�'BatchNormProteinEncode1/AssignMovingAvg�6BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOp�)BatchNormProteinEncode1/AssignMovingAvg_1�8BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp�0BatchNormProteinEncode1/batchnorm/ReadVariableOp�4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp�(EmbeddingDimDense/BiasAdd/ReadVariableOp�'EmbeddingDimDense/MatMul/ReadVariableOp�%gene_decoder_1/BiasAdd/ReadVariableOp�$gene_decoder_1/MatMul/ReadVariableOp�%gene_decoder_2/BiasAdd/ReadVariableOp�$gene_decoder_2/MatMul/ReadVariableOp�(gene_decoder_last/BiasAdd/ReadVariableOp�'gene_decoder_last/MatMul/ReadVariableOp�%gene_encoder_1/BiasAdd/ReadVariableOp�$gene_encoder_1/MatMul/ReadVariableOp�%gene_encoder_2/BiasAdd/ReadVariableOp�$gene_encoder_2/MatMul/ReadVariableOp�(protein_decoder_1/BiasAdd/ReadVariableOp�'protein_decoder_1/MatMul/ReadVariableOp�+protein_decoder_last/BiasAdd/ReadVariableOp�*protein_decoder_last/MatMul/ReadVariableOp�(protein_encoder_1/BiasAdd/ReadVariableOp�'protein_encoder_1/MatMul/ReadVariableOp�
$gene_encoder_1/MatMul/ReadVariableOpReadVariableOp-gene_encoder_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gene_encoder_1/MatMulMatMulinputs_0,gene_encoder_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%gene_encoder_1/BiasAdd/ReadVariableOpReadVariableOp.gene_encoder_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
gene_encoder_1/BiasAddBiasAddgene_encoder_1/MatMul:product:0-gene_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
gene_encoder_1/SigmoidSigmoidgene_encoder_1/BiasAdd:output:0*
T0*(
_output_shapes
:����������}
3BatchNormGeneEncode1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
!BatchNormGeneEncode1/moments/meanMeangene_encoder_1/Sigmoid:y:0<BatchNormGeneEncode1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
)BatchNormGeneEncode1/moments/StopGradientStopGradient*BatchNormGeneEncode1/moments/mean:output:0*
T0*
_output_shapes
:	��
.BatchNormGeneEncode1/moments/SquaredDifferenceSquaredDifferencegene_encoder_1/Sigmoid:y:02BatchNormGeneEncode1/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
7BatchNormGeneEncode1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
%BatchNormGeneEncode1/moments/varianceMean2BatchNormGeneEncode1/moments/SquaredDifference:z:0@BatchNormGeneEncode1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
$BatchNormGeneEncode1/moments/SqueezeSqueeze*BatchNormGeneEncode1/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
&BatchNormGeneEncode1/moments/Squeeze_1Squeeze.BatchNormGeneEncode1/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 o
*BatchNormGeneEncode1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
3BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOpReadVariableOp<batchnormgeneencode1_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(BatchNormGeneEncode1/AssignMovingAvg/subSub;BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOp:value:0-BatchNormGeneEncode1/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
(BatchNormGeneEncode1/AssignMovingAvg/mulMul,BatchNormGeneEncode1/AssignMovingAvg/sub:z:03BatchNormGeneEncode1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
$BatchNormGeneEncode1/AssignMovingAvgAssignSubVariableOp<batchnormgeneencode1_assignmovingavg_readvariableop_resource,BatchNormGeneEncode1/AssignMovingAvg/mul:z:04^BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0q
,BatchNormGeneEncode1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOpReadVariableOp>batchnormgeneencode1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*BatchNormGeneEncode1/AssignMovingAvg_1/subSub=BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOp:value:0/BatchNormGeneEncode1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
*BatchNormGeneEncode1/AssignMovingAvg_1/mulMul.BatchNormGeneEncode1/AssignMovingAvg_1/sub:z:05BatchNormGeneEncode1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
&BatchNormGeneEncode1/AssignMovingAvg_1AssignSubVariableOp>batchnormgeneencode1_assignmovingavg_1_readvariableop_resource.BatchNormGeneEncode1/AssignMovingAvg_1/mul:z:06^BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0i
$BatchNormGeneEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
"BatchNormGeneEncode1/batchnorm/addAddV2/BatchNormGeneEncode1/moments/Squeeze_1:output:0-BatchNormGeneEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�{
$BatchNormGeneEncode1/batchnorm/RsqrtRsqrt&BatchNormGeneEncode1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgeneencode1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"BatchNormGeneEncode1/batchnorm/mulMul(BatchNormGeneEncode1/batchnorm/Rsqrt:y:09BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
$BatchNormGeneEncode1/batchnorm/mul_1Mulgene_encoder_1/Sigmoid:y:0&BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
$BatchNormGeneEncode1/batchnorm/mul_2Mul-BatchNormGeneEncode1/moments/Squeeze:output:0&BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
-BatchNormGeneEncode1/batchnorm/ReadVariableOpReadVariableOp6batchnormgeneencode1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"BatchNormGeneEncode1/batchnorm/subSub5BatchNormGeneEncode1/batchnorm/ReadVariableOp:value:0(BatchNormGeneEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
$BatchNormGeneEncode1/batchnorm/add_1AddV2(BatchNormGeneEncode1/batchnorm/mul_1:z:0&BatchNormGeneEncode1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
'protein_encoder_1/MatMul/ReadVariableOpReadVariableOp0protein_encoder_1_matmul_readvariableop_resource*
_output_shapes
:	�9*
dtype0�
protein_encoder_1/MatMulMatMulinputs_1/protein_encoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������9�
(protein_encoder_1/BiasAdd/ReadVariableOpReadVariableOp1protein_encoder_1_biasadd_readvariableop_resource*
_output_shapes
:9*
dtype0�
protein_encoder_1/BiasAddBiasAdd"protein_encoder_1/MatMul:product:00protein_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������9z
protein_encoder_1/SigmoidSigmoid"protein_encoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������9�
$gene_encoder_2/MatMul/ReadVariableOpReadVariableOp-gene_encoder_2_matmul_readvariableop_resource*
_output_shapes
:	�}*
dtype0�
gene_encoder_2/MatMulMatMul(BatchNormGeneEncode1/batchnorm/add_1:z:0,gene_encoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}�
%gene_encoder_2/BiasAdd/ReadVariableOpReadVariableOp.gene_encoder_2_biasadd_readvariableop_resource*
_output_shapes
:}*
dtype0�
gene_encoder_2/BiasAddBiasAddgene_encoder_2/MatMul:product:0-gene_encoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}t
gene_encoder_2/SigmoidSigmoidgene_encoder_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������}}
3BatchNormGeneEncode2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
!BatchNormGeneEncode2/moments/meanMeangene_encoder_2/Sigmoid:y:0<BatchNormGeneEncode2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:}*
	keep_dims(�
)BatchNormGeneEncode2/moments/StopGradientStopGradient*BatchNormGeneEncode2/moments/mean:output:0*
T0*
_output_shapes

:}�
.BatchNormGeneEncode2/moments/SquaredDifferenceSquaredDifferencegene_encoder_2/Sigmoid:y:02BatchNormGeneEncode2/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������}�
7BatchNormGeneEncode2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
%BatchNormGeneEncode2/moments/varianceMean2BatchNormGeneEncode2/moments/SquaredDifference:z:0@BatchNormGeneEncode2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:}*
	keep_dims(�
$BatchNormGeneEncode2/moments/SqueezeSqueeze*BatchNormGeneEncode2/moments/mean:output:0*
T0*
_output_shapes
:}*
squeeze_dims
 �
&BatchNormGeneEncode2/moments/Squeeze_1Squeeze.BatchNormGeneEncode2/moments/variance:output:0*
T0*
_output_shapes
:}*
squeeze_dims
 o
*BatchNormGeneEncode2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
3BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOpReadVariableOp<batchnormgeneencode2_assignmovingavg_readvariableop_resource*
_output_shapes
:}*
dtype0�
(BatchNormGeneEncode2/AssignMovingAvg/subSub;BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOp:value:0-BatchNormGeneEncode2/moments/Squeeze:output:0*
T0*
_output_shapes
:}�
(BatchNormGeneEncode2/AssignMovingAvg/mulMul,BatchNormGeneEncode2/AssignMovingAvg/sub:z:03BatchNormGeneEncode2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:}�
$BatchNormGeneEncode2/AssignMovingAvgAssignSubVariableOp<batchnormgeneencode2_assignmovingavg_readvariableop_resource,BatchNormGeneEncode2/AssignMovingAvg/mul:z:04^BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0q
,BatchNormGeneEncode2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOpReadVariableOp>batchnormgeneencode2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:}*
dtype0�
*BatchNormGeneEncode2/AssignMovingAvg_1/subSub=BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOp:value:0/BatchNormGeneEncode2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:}�
*BatchNormGeneEncode2/AssignMovingAvg_1/mulMul.BatchNormGeneEncode2/AssignMovingAvg_1/sub:z:05BatchNormGeneEncode2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:}�
&BatchNormGeneEncode2/AssignMovingAvg_1AssignSubVariableOp>batchnormgeneencode2_assignmovingavg_1_readvariableop_resource.BatchNormGeneEncode2/AssignMovingAvg_1/mul:z:06^BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0i
$BatchNormGeneEncode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
"BatchNormGeneEncode2/batchnorm/addAddV2/BatchNormGeneEncode2/moments/Squeeze_1:output:0-BatchNormGeneEncode2/batchnorm/add/y:output:0*
T0*
_output_shapes
:}z
$BatchNormGeneEncode2/batchnorm/RsqrtRsqrt&BatchNormGeneEncode2/batchnorm/add:z:0*
T0*
_output_shapes
:}�
1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgeneencode2_batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0�
"BatchNormGeneEncode2/batchnorm/mulMul(BatchNormGeneEncode2/batchnorm/Rsqrt:y:09BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}�
$BatchNormGeneEncode2/batchnorm/mul_1Mulgene_encoder_2/Sigmoid:y:0&BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������}�
$BatchNormGeneEncode2/batchnorm/mul_2Mul-BatchNormGeneEncode2/moments/Squeeze:output:0&BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*
_output_shapes
:}�
-BatchNormGeneEncode2/batchnorm/ReadVariableOpReadVariableOp6batchnormgeneencode2_batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0�
"BatchNormGeneEncode2/batchnorm/subSub5BatchNormGeneEncode2/batchnorm/ReadVariableOp:value:0(BatchNormGeneEncode2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:}�
$BatchNormGeneEncode2/batchnorm/add_1AddV2(BatchNormGeneEncode2/batchnorm/mul_1:z:0&BatchNormGeneEncode2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}�
6BatchNormProteinEncode1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
$BatchNormProteinEncode1/moments/meanMeanprotein_encoder_1/Sigmoid:y:0?BatchNormProteinEncode1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:9*
	keep_dims(�
,BatchNormProteinEncode1/moments/StopGradientStopGradient-BatchNormProteinEncode1/moments/mean:output:0*
T0*
_output_shapes

:9�
1BatchNormProteinEncode1/moments/SquaredDifferenceSquaredDifferenceprotein_encoder_1/Sigmoid:y:05BatchNormProteinEncode1/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������9�
:BatchNormProteinEncode1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
(BatchNormProteinEncode1/moments/varianceMean5BatchNormProteinEncode1/moments/SquaredDifference:z:0CBatchNormProteinEncode1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:9*
	keep_dims(�
'BatchNormProteinEncode1/moments/SqueezeSqueeze-BatchNormProteinEncode1/moments/mean:output:0*
T0*
_output_shapes
:9*
squeeze_dims
 �
)BatchNormProteinEncode1/moments/Squeeze_1Squeeze1BatchNormProteinEncode1/moments/variance:output:0*
T0*
_output_shapes
:9*
squeeze_dims
 r
-BatchNormProteinEncode1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOpReadVariableOp?batchnormproteinencode1_assignmovingavg_readvariableop_resource*
_output_shapes
:9*
dtype0�
+BatchNormProteinEncode1/AssignMovingAvg/subSub>BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOp:value:00BatchNormProteinEncode1/moments/Squeeze:output:0*
T0*
_output_shapes
:9�
+BatchNormProteinEncode1/AssignMovingAvg/mulMul/BatchNormProteinEncode1/AssignMovingAvg/sub:z:06BatchNormProteinEncode1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:9�
'BatchNormProteinEncode1/AssignMovingAvgAssignSubVariableOp?batchnormproteinencode1_assignmovingavg_readvariableop_resource/BatchNormProteinEncode1/AssignMovingAvg/mul:z:07^BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/BatchNormProteinEncode1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
8BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatchnormproteinencode1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:9*
dtype0�
-BatchNormProteinEncode1/AssignMovingAvg_1/subSub@BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp:value:02BatchNormProteinEncode1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:9�
-BatchNormProteinEncode1/AssignMovingAvg_1/mulMul1BatchNormProteinEncode1/AssignMovingAvg_1/sub:z:08BatchNormProteinEncode1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:9�
)BatchNormProteinEncode1/AssignMovingAvg_1AssignSubVariableOpAbatchnormproteinencode1_assignmovingavg_1_readvariableop_resource1BatchNormProteinEncode1/AssignMovingAvg_1/mul:z:09^BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'BatchNormProteinEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
%BatchNormProteinEncode1/batchnorm/addAddV22BatchNormProteinEncode1/moments/Squeeze_1:output:00BatchNormProteinEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:9�
'BatchNormProteinEncode1/batchnorm/RsqrtRsqrt)BatchNormProteinEncode1/batchnorm/add:z:0*
T0*
_output_shapes
:9�
4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOpReadVariableOp=batchnormproteinencode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:9*
dtype0�
%BatchNormProteinEncode1/batchnorm/mulMul+BatchNormProteinEncode1/batchnorm/Rsqrt:y:0<BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:9�
'BatchNormProteinEncode1/batchnorm/mul_1Mulprotein_encoder_1/Sigmoid:y:0)BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������9�
'BatchNormProteinEncode1/batchnorm/mul_2Mul0BatchNormProteinEncode1/moments/Squeeze:output:0)BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*
_output_shapes
:9�
0BatchNormProteinEncode1/batchnorm/ReadVariableOpReadVariableOp9batchnormproteinencode1_batchnorm_readvariableop_resource*
_output_shapes
:9*
dtype0�
%BatchNormProteinEncode1/batchnorm/subSub8BatchNormProteinEncode1/batchnorm/ReadVariableOp:value:0+BatchNormProteinEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:9�
'BatchNormProteinEncode1/batchnorm/add_1AddV2+BatchNormProteinEncode1/batchnorm/mul_1:z:0)BatchNormProteinEncode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������9^
ConcatenateLayer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
ConcatenateLayer/concatConcatV2(BatchNormGeneEncode2/batchnorm/add_1:z:0+BatchNormProteinEncode1/batchnorm/add_1:z:0%ConcatenateLayer/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
'EmbeddingDimDense/MatMul/ReadVariableOpReadVariableOp0embeddingdimdense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
EmbeddingDimDense/MatMulMatMul ConcatenateLayer/concat:output:0/EmbeddingDimDense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
(EmbeddingDimDense/BiasAdd/ReadVariableOpReadVariableOp1embeddingdimdense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
EmbeddingDimDense/BiasAddBiasAdd"EmbeddingDimDense/MatMul:product:00EmbeddingDimDense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@z
EmbeddingDimDense/SigmoidSigmoid"EmbeddingDimDense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
$gene_decoder_1/MatMul/ReadVariableOpReadVariableOp-gene_decoder_1_matmul_readvariableop_resource*
_output_shapes

:@}*
dtype0�
gene_decoder_1/MatMulMatMulEmbeddingDimDense/Sigmoid:y:0,gene_decoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}�
%gene_decoder_1/BiasAdd/ReadVariableOpReadVariableOp.gene_decoder_1_biasadd_readvariableop_resource*
_output_shapes
:}*
dtype0�
gene_decoder_1/BiasAddBiasAddgene_decoder_1/MatMul:product:0-gene_decoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}t
gene_decoder_1/SigmoidSigmoidgene_decoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������}}
3BatchNormGeneDecode1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
!BatchNormGeneDecode1/moments/meanMeangene_decoder_1/Sigmoid:y:0<BatchNormGeneDecode1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:}*
	keep_dims(�
)BatchNormGeneDecode1/moments/StopGradientStopGradient*BatchNormGeneDecode1/moments/mean:output:0*
T0*
_output_shapes

:}�
.BatchNormGeneDecode1/moments/SquaredDifferenceSquaredDifferencegene_decoder_1/Sigmoid:y:02BatchNormGeneDecode1/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������}�
7BatchNormGeneDecode1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
%BatchNormGeneDecode1/moments/varianceMean2BatchNormGeneDecode1/moments/SquaredDifference:z:0@BatchNormGeneDecode1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:}*
	keep_dims(�
$BatchNormGeneDecode1/moments/SqueezeSqueeze*BatchNormGeneDecode1/moments/mean:output:0*
T0*
_output_shapes
:}*
squeeze_dims
 �
&BatchNormGeneDecode1/moments/Squeeze_1Squeeze.BatchNormGeneDecode1/moments/variance:output:0*
T0*
_output_shapes
:}*
squeeze_dims
 o
*BatchNormGeneDecode1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
3BatchNormGeneDecode1/AssignMovingAvg/ReadVariableOpReadVariableOp<batchnormgenedecode1_assignmovingavg_readvariableop_resource*
_output_shapes
:}*
dtype0�
(BatchNormGeneDecode1/AssignMovingAvg/subSub;BatchNormGeneDecode1/AssignMovingAvg/ReadVariableOp:value:0-BatchNormGeneDecode1/moments/Squeeze:output:0*
T0*
_output_shapes
:}�
(BatchNormGeneDecode1/AssignMovingAvg/mulMul,BatchNormGeneDecode1/AssignMovingAvg/sub:z:03BatchNormGeneDecode1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:}�
$BatchNormGeneDecode1/AssignMovingAvgAssignSubVariableOp<batchnormgenedecode1_assignmovingavg_readvariableop_resource,BatchNormGeneDecode1/AssignMovingAvg/mul:z:04^BatchNormGeneDecode1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0q
,BatchNormGeneDecode1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5BatchNormGeneDecode1/AssignMovingAvg_1/ReadVariableOpReadVariableOp>batchnormgenedecode1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:}*
dtype0�
*BatchNormGeneDecode1/AssignMovingAvg_1/subSub=BatchNormGeneDecode1/AssignMovingAvg_1/ReadVariableOp:value:0/BatchNormGeneDecode1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:}�
*BatchNormGeneDecode1/AssignMovingAvg_1/mulMul.BatchNormGeneDecode1/AssignMovingAvg_1/sub:z:05BatchNormGeneDecode1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:}�
&BatchNormGeneDecode1/AssignMovingAvg_1AssignSubVariableOp>batchnormgenedecode1_assignmovingavg_1_readvariableop_resource.BatchNormGeneDecode1/AssignMovingAvg_1/mul:z:06^BatchNormGeneDecode1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0i
$BatchNormGeneDecode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
"BatchNormGeneDecode1/batchnorm/addAddV2/BatchNormGeneDecode1/moments/Squeeze_1:output:0-BatchNormGeneDecode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:}z
$BatchNormGeneDecode1/batchnorm/RsqrtRsqrt&BatchNormGeneDecode1/batchnorm/add:z:0*
T0*
_output_shapes
:}�
1BatchNormGeneDecode1/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgenedecode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0�
"BatchNormGeneDecode1/batchnorm/mulMul(BatchNormGeneDecode1/batchnorm/Rsqrt:y:09BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}�
$BatchNormGeneDecode1/batchnorm/mul_1Mulgene_decoder_1/Sigmoid:y:0&BatchNormGeneDecode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������}�
$BatchNormGeneDecode1/batchnorm/mul_2Mul-BatchNormGeneDecode1/moments/Squeeze:output:0&BatchNormGeneDecode1/batchnorm/mul:z:0*
T0*
_output_shapes
:}�
-BatchNormGeneDecode1/batchnorm/ReadVariableOpReadVariableOp6batchnormgenedecode1_batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0�
"BatchNormGeneDecode1/batchnorm/subSub5BatchNormGeneDecode1/batchnorm/ReadVariableOp:value:0(BatchNormGeneDecode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:}�
$BatchNormGeneDecode1/batchnorm/add_1AddV2(BatchNormGeneDecode1/batchnorm/mul_1:z:0&BatchNormGeneDecode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}�
'protein_decoder_1/MatMul/ReadVariableOpReadVariableOp0protein_decoder_1_matmul_readvariableop_resource*
_output_shapes

:@9*
dtype0�
protein_decoder_1/MatMulMatMulEmbeddingDimDense/Sigmoid:y:0/protein_decoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������9�
(protein_decoder_1/BiasAdd/ReadVariableOpReadVariableOp1protein_decoder_1_biasadd_readvariableop_resource*
_output_shapes
:9*
dtype0�
protein_decoder_1/BiasAddBiasAdd"protein_decoder_1/MatMul:product:00protein_decoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������9z
protein_decoder_1/SigmoidSigmoid"protein_decoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������9�
$gene_decoder_2/MatMul/ReadVariableOpReadVariableOp-gene_decoder_2_matmul_readvariableop_resource*
_output_shapes
:	}�*
dtype0�
gene_decoder_2/MatMulMatMul(BatchNormGeneDecode1/batchnorm/add_1:z:0,gene_decoder_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%gene_decoder_2/BiasAdd/ReadVariableOpReadVariableOp.gene_decoder_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
gene_decoder_2/BiasAddBiasAddgene_decoder_2/MatMul:product:0-gene_decoder_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
gene_decoder_2/SigmoidSigmoidgene_decoder_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
6BatchNormProteinDecode1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
$BatchNormProteinDecode1/moments/meanMeanprotein_decoder_1/Sigmoid:y:0?BatchNormProteinDecode1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:9*
	keep_dims(�
,BatchNormProteinDecode1/moments/StopGradientStopGradient-BatchNormProteinDecode1/moments/mean:output:0*
T0*
_output_shapes

:9�
1BatchNormProteinDecode1/moments/SquaredDifferenceSquaredDifferenceprotein_decoder_1/Sigmoid:y:05BatchNormProteinDecode1/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������9�
:BatchNormProteinDecode1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
(BatchNormProteinDecode1/moments/varianceMean5BatchNormProteinDecode1/moments/SquaredDifference:z:0CBatchNormProteinDecode1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:9*
	keep_dims(�
'BatchNormProteinDecode1/moments/SqueezeSqueeze-BatchNormProteinDecode1/moments/mean:output:0*
T0*
_output_shapes
:9*
squeeze_dims
 �
)BatchNormProteinDecode1/moments/Squeeze_1Squeeze1BatchNormProteinDecode1/moments/variance:output:0*
T0*
_output_shapes
:9*
squeeze_dims
 r
-BatchNormProteinDecode1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6BatchNormProteinDecode1/AssignMovingAvg/ReadVariableOpReadVariableOp?batchnormproteindecode1_assignmovingavg_readvariableop_resource*
_output_shapes
:9*
dtype0�
+BatchNormProteinDecode1/AssignMovingAvg/subSub>BatchNormProteinDecode1/AssignMovingAvg/ReadVariableOp:value:00BatchNormProteinDecode1/moments/Squeeze:output:0*
T0*
_output_shapes
:9�
+BatchNormProteinDecode1/AssignMovingAvg/mulMul/BatchNormProteinDecode1/AssignMovingAvg/sub:z:06BatchNormProteinDecode1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:9�
'BatchNormProteinDecode1/AssignMovingAvgAssignSubVariableOp?batchnormproteindecode1_assignmovingavg_readvariableop_resource/BatchNormProteinDecode1/AssignMovingAvg/mul:z:07^BatchNormProteinDecode1/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/BatchNormProteinDecode1/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
8BatchNormProteinDecode1/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatchnormproteindecode1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:9*
dtype0�
-BatchNormProteinDecode1/AssignMovingAvg_1/subSub@BatchNormProteinDecode1/AssignMovingAvg_1/ReadVariableOp:value:02BatchNormProteinDecode1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:9�
-BatchNormProteinDecode1/AssignMovingAvg_1/mulMul1BatchNormProteinDecode1/AssignMovingAvg_1/sub:z:08BatchNormProteinDecode1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:9�
)BatchNormProteinDecode1/AssignMovingAvg_1AssignSubVariableOpAbatchnormproteindecode1_assignmovingavg_1_readvariableop_resource1BatchNormProteinDecode1/AssignMovingAvg_1/mul:z:09^BatchNormProteinDecode1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'BatchNormProteinDecode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
%BatchNormProteinDecode1/batchnorm/addAddV22BatchNormProteinDecode1/moments/Squeeze_1:output:00BatchNormProteinDecode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:9�
'BatchNormProteinDecode1/batchnorm/RsqrtRsqrt)BatchNormProteinDecode1/batchnorm/add:z:0*
T0*
_output_shapes
:9�
4BatchNormProteinDecode1/batchnorm/mul/ReadVariableOpReadVariableOp=batchnormproteindecode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:9*
dtype0�
%BatchNormProteinDecode1/batchnorm/mulMul+BatchNormProteinDecode1/batchnorm/Rsqrt:y:0<BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:9�
'BatchNormProteinDecode1/batchnorm/mul_1Mulprotein_decoder_1/Sigmoid:y:0)BatchNormProteinDecode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������9�
'BatchNormProteinDecode1/batchnorm/mul_2Mul0BatchNormProteinDecode1/moments/Squeeze:output:0)BatchNormProteinDecode1/batchnorm/mul:z:0*
T0*
_output_shapes
:9�
0BatchNormProteinDecode1/batchnorm/ReadVariableOpReadVariableOp9batchnormproteindecode1_batchnorm_readvariableop_resource*
_output_shapes
:9*
dtype0�
%BatchNormProteinDecode1/batchnorm/subSub8BatchNormProteinDecode1/batchnorm/ReadVariableOp:value:0+BatchNormProteinDecode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:9�
'BatchNormProteinDecode1/batchnorm/add_1AddV2+BatchNormProteinDecode1/batchnorm/mul_1:z:0)BatchNormProteinDecode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������9}
3BatchNormGeneDecode2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
!BatchNormGeneDecode2/moments/meanMeangene_decoder_2/Sigmoid:y:0<BatchNormGeneDecode2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
)BatchNormGeneDecode2/moments/StopGradientStopGradient*BatchNormGeneDecode2/moments/mean:output:0*
T0*
_output_shapes
:	��
.BatchNormGeneDecode2/moments/SquaredDifferenceSquaredDifferencegene_decoder_2/Sigmoid:y:02BatchNormGeneDecode2/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
7BatchNormGeneDecode2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
%BatchNormGeneDecode2/moments/varianceMean2BatchNormGeneDecode2/moments/SquaredDifference:z:0@BatchNormGeneDecode2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
$BatchNormGeneDecode2/moments/SqueezeSqueeze*BatchNormGeneDecode2/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
&BatchNormGeneDecode2/moments/Squeeze_1Squeeze.BatchNormGeneDecode2/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 o
*BatchNormGeneDecode2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
3BatchNormGeneDecode2/AssignMovingAvg/ReadVariableOpReadVariableOp<batchnormgenedecode2_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(BatchNormGeneDecode2/AssignMovingAvg/subSub;BatchNormGeneDecode2/AssignMovingAvg/ReadVariableOp:value:0-BatchNormGeneDecode2/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
(BatchNormGeneDecode2/AssignMovingAvg/mulMul,BatchNormGeneDecode2/AssignMovingAvg/sub:z:03BatchNormGeneDecode2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
$BatchNormGeneDecode2/AssignMovingAvgAssignSubVariableOp<batchnormgenedecode2_assignmovingavg_readvariableop_resource,BatchNormGeneDecode2/AssignMovingAvg/mul:z:04^BatchNormGeneDecode2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0q
,BatchNormGeneDecode2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
5BatchNormGeneDecode2/AssignMovingAvg_1/ReadVariableOpReadVariableOp>batchnormgenedecode2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*BatchNormGeneDecode2/AssignMovingAvg_1/subSub=BatchNormGeneDecode2/AssignMovingAvg_1/ReadVariableOp:value:0/BatchNormGeneDecode2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
*BatchNormGeneDecode2/AssignMovingAvg_1/mulMul.BatchNormGeneDecode2/AssignMovingAvg_1/sub:z:05BatchNormGeneDecode2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
&BatchNormGeneDecode2/AssignMovingAvg_1AssignSubVariableOp>batchnormgenedecode2_assignmovingavg_1_readvariableop_resource.BatchNormGeneDecode2/AssignMovingAvg_1/mul:z:06^BatchNormGeneDecode2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0i
$BatchNormGeneDecode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
"BatchNormGeneDecode2/batchnorm/addAddV2/BatchNormGeneDecode2/moments/Squeeze_1:output:0-BatchNormGeneDecode2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�{
$BatchNormGeneDecode2/batchnorm/RsqrtRsqrt&BatchNormGeneDecode2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
1BatchNormGeneDecode2/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgenedecode2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"BatchNormGeneDecode2/batchnorm/mulMul(BatchNormGeneDecode2/batchnorm/Rsqrt:y:09BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
$BatchNormGeneDecode2/batchnorm/mul_1Mulgene_decoder_2/Sigmoid:y:0&BatchNormGeneDecode2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
$BatchNormGeneDecode2/batchnorm/mul_2Mul-BatchNormGeneDecode2/moments/Squeeze:output:0&BatchNormGeneDecode2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
-BatchNormGeneDecode2/batchnorm/ReadVariableOpReadVariableOp6batchnormgenedecode2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"BatchNormGeneDecode2/batchnorm/subSub5BatchNormGeneDecode2/batchnorm/ReadVariableOp:value:0(BatchNormGeneDecode2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
$BatchNormGeneDecode2/batchnorm/add_1AddV2(BatchNormGeneDecode2/batchnorm/mul_1:z:0&BatchNormGeneDecode2/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
*protein_decoder_last/MatMul/ReadVariableOpReadVariableOp3protein_decoder_last_matmul_readvariableop_resource*
_output_shapes
:	9�*
dtype0�
protein_decoder_last/MatMulMatMul+BatchNormProteinDecode1/batchnorm/add_1:z:02protein_decoder_last/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+protein_decoder_last/BiasAdd/ReadVariableOpReadVariableOp4protein_decoder_last_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
protein_decoder_last/BiasAddBiasAdd%protein_decoder_last/MatMul:product:03protein_decoder_last/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
protein_decoder_last/SigmoidSigmoid%protein_decoder_last/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
'gene_decoder_last/MatMul/ReadVariableOpReadVariableOp0gene_decoder_last_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gene_decoder_last/MatMulMatMul(BatchNormGeneDecode2/batchnorm/add_1:z:0/gene_decoder_last/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
(gene_decoder_last/BiasAdd/ReadVariableOpReadVariableOp1gene_decoder_last_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
gene_decoder_last/BiasAddBiasAdd"gene_decoder_last/MatMul:product:00gene_decoder_last/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������{
gene_decoder_last/SigmoidSigmoid"gene_decoder_last/BiasAdd:output:0*
T0*(
_output_shapes
:����������m
IdentityIdentitygene_decoder_last/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������r

Identity_1Identity protein_decoder_last/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp%^BatchNormGeneDecode1/AssignMovingAvg4^BatchNormGeneDecode1/AssignMovingAvg/ReadVariableOp'^BatchNormGeneDecode1/AssignMovingAvg_16^BatchNormGeneDecode1/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneDecode1/batchnorm/ReadVariableOp2^BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp%^BatchNormGeneDecode2/AssignMovingAvg4^BatchNormGeneDecode2/AssignMovingAvg/ReadVariableOp'^BatchNormGeneDecode2/AssignMovingAvg_16^BatchNormGeneDecode2/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneDecode2/batchnorm/ReadVariableOp2^BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp%^BatchNormGeneEncode1/AssignMovingAvg4^BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOp'^BatchNormGeneEncode1/AssignMovingAvg_16^BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneEncode1/batchnorm/ReadVariableOp2^BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp%^BatchNormGeneEncode2/AssignMovingAvg4^BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOp'^BatchNormGeneEncode2/AssignMovingAvg_16^BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneEncode2/batchnorm/ReadVariableOp2^BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp(^BatchNormProteinDecode1/AssignMovingAvg7^BatchNormProteinDecode1/AssignMovingAvg/ReadVariableOp*^BatchNormProteinDecode1/AssignMovingAvg_19^BatchNormProteinDecode1/AssignMovingAvg_1/ReadVariableOp1^BatchNormProteinDecode1/batchnorm/ReadVariableOp5^BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp(^BatchNormProteinEncode1/AssignMovingAvg7^BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOp*^BatchNormProteinEncode1/AssignMovingAvg_19^BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp1^BatchNormProteinEncode1/batchnorm/ReadVariableOp5^BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp)^EmbeddingDimDense/BiasAdd/ReadVariableOp(^EmbeddingDimDense/MatMul/ReadVariableOp&^gene_decoder_1/BiasAdd/ReadVariableOp%^gene_decoder_1/MatMul/ReadVariableOp&^gene_decoder_2/BiasAdd/ReadVariableOp%^gene_decoder_2/MatMul/ReadVariableOp)^gene_decoder_last/BiasAdd/ReadVariableOp(^gene_decoder_last/MatMul/ReadVariableOp&^gene_encoder_1/BiasAdd/ReadVariableOp%^gene_encoder_1/MatMul/ReadVariableOp&^gene_encoder_2/BiasAdd/ReadVariableOp%^gene_encoder_2/MatMul/ReadVariableOp)^protein_decoder_1/BiasAdd/ReadVariableOp(^protein_decoder_1/MatMul/ReadVariableOp,^protein_decoder_last/BiasAdd/ReadVariableOp+^protein_decoder_last/MatMul/ReadVariableOp)^protein_encoder_1/BiasAdd/ReadVariableOp(^protein_encoder_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$BatchNormGeneDecode1/AssignMovingAvg$BatchNormGeneDecode1/AssignMovingAvg2j
3BatchNormGeneDecode1/AssignMovingAvg/ReadVariableOp3BatchNormGeneDecode1/AssignMovingAvg/ReadVariableOp2P
&BatchNormGeneDecode1/AssignMovingAvg_1&BatchNormGeneDecode1/AssignMovingAvg_12n
5BatchNormGeneDecode1/AssignMovingAvg_1/ReadVariableOp5BatchNormGeneDecode1/AssignMovingAvg_1/ReadVariableOp2^
-BatchNormGeneDecode1/batchnorm/ReadVariableOp-BatchNormGeneDecode1/batchnorm/ReadVariableOp2f
1BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp1BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp2L
$BatchNormGeneDecode2/AssignMovingAvg$BatchNormGeneDecode2/AssignMovingAvg2j
3BatchNormGeneDecode2/AssignMovingAvg/ReadVariableOp3BatchNormGeneDecode2/AssignMovingAvg/ReadVariableOp2P
&BatchNormGeneDecode2/AssignMovingAvg_1&BatchNormGeneDecode2/AssignMovingAvg_12n
5BatchNormGeneDecode2/AssignMovingAvg_1/ReadVariableOp5BatchNormGeneDecode2/AssignMovingAvg_1/ReadVariableOp2^
-BatchNormGeneDecode2/batchnorm/ReadVariableOp-BatchNormGeneDecode2/batchnorm/ReadVariableOp2f
1BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp1BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp2L
$BatchNormGeneEncode1/AssignMovingAvg$BatchNormGeneEncode1/AssignMovingAvg2j
3BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOp3BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOp2P
&BatchNormGeneEncode1/AssignMovingAvg_1&BatchNormGeneEncode1/AssignMovingAvg_12n
5BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOp5BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOp2^
-BatchNormGeneEncode1/batchnorm/ReadVariableOp-BatchNormGeneEncode1/batchnorm/ReadVariableOp2f
1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp2L
$BatchNormGeneEncode2/AssignMovingAvg$BatchNormGeneEncode2/AssignMovingAvg2j
3BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOp3BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOp2P
&BatchNormGeneEncode2/AssignMovingAvg_1&BatchNormGeneEncode2/AssignMovingAvg_12n
5BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOp5BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOp2^
-BatchNormGeneEncode2/batchnorm/ReadVariableOp-BatchNormGeneEncode2/batchnorm/ReadVariableOp2f
1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp2R
'BatchNormProteinDecode1/AssignMovingAvg'BatchNormProteinDecode1/AssignMovingAvg2p
6BatchNormProteinDecode1/AssignMovingAvg/ReadVariableOp6BatchNormProteinDecode1/AssignMovingAvg/ReadVariableOp2V
)BatchNormProteinDecode1/AssignMovingAvg_1)BatchNormProteinDecode1/AssignMovingAvg_12t
8BatchNormProteinDecode1/AssignMovingAvg_1/ReadVariableOp8BatchNormProteinDecode1/AssignMovingAvg_1/ReadVariableOp2d
0BatchNormProteinDecode1/batchnorm/ReadVariableOp0BatchNormProteinDecode1/batchnorm/ReadVariableOp2l
4BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp4BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp2R
'BatchNormProteinEncode1/AssignMovingAvg'BatchNormProteinEncode1/AssignMovingAvg2p
6BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOp6BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOp2V
)BatchNormProteinEncode1/AssignMovingAvg_1)BatchNormProteinEncode1/AssignMovingAvg_12t
8BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp8BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp2d
0BatchNormProteinEncode1/batchnorm/ReadVariableOp0BatchNormProteinEncode1/batchnorm/ReadVariableOp2l
4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp2T
(EmbeddingDimDense/BiasAdd/ReadVariableOp(EmbeddingDimDense/BiasAdd/ReadVariableOp2R
'EmbeddingDimDense/MatMul/ReadVariableOp'EmbeddingDimDense/MatMul/ReadVariableOp2N
%gene_decoder_1/BiasAdd/ReadVariableOp%gene_decoder_1/BiasAdd/ReadVariableOp2L
$gene_decoder_1/MatMul/ReadVariableOp$gene_decoder_1/MatMul/ReadVariableOp2N
%gene_decoder_2/BiasAdd/ReadVariableOp%gene_decoder_2/BiasAdd/ReadVariableOp2L
$gene_decoder_2/MatMul/ReadVariableOp$gene_decoder_2/MatMul/ReadVariableOp2T
(gene_decoder_last/BiasAdd/ReadVariableOp(gene_decoder_last/BiasAdd/ReadVariableOp2R
'gene_decoder_last/MatMul/ReadVariableOp'gene_decoder_last/MatMul/ReadVariableOp2N
%gene_encoder_1/BiasAdd/ReadVariableOp%gene_encoder_1/BiasAdd/ReadVariableOp2L
$gene_encoder_1/MatMul/ReadVariableOp$gene_encoder_1/MatMul/ReadVariableOp2N
%gene_encoder_2/BiasAdd/ReadVariableOp%gene_encoder_2/BiasAdd/ReadVariableOp2L
$gene_encoder_2/MatMul/ReadVariableOp$gene_encoder_2/MatMul/ReadVariableOp2T
(protein_decoder_1/BiasAdd/ReadVariableOp(protein_decoder_1/BiasAdd/ReadVariableOp2R
'protein_decoder_1/MatMul/ReadVariableOp'protein_decoder_1/MatMul/ReadVariableOp2Z
+protein_decoder_last/BiasAdd/ReadVariableOp+protein_decoder_last/BiasAdd/ReadVariableOp2X
*protein_decoder_last/MatMul/ReadVariableOp*protein_decoder_last/MatMul/ReadVariableOp2T
(protein_encoder_1/BiasAdd/ReadVariableOp(protein_encoder_1/BiasAdd/ReadVariableOp2R
'protein_encoder_1/MatMul/ReadVariableOp'protein_encoder_1/MatMul/ReadVariableOp:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�%
�
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_188518

inputs5
'assignmovingavg_readvariableop_resource:97
)assignmovingavg_1_readvariableop_resource:93
%batchnorm_mul_readvariableop_resource:9/
!batchnorm_readvariableop_resource:9
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:9*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:9�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������9l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:9*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:9*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:9*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:9*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:9x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:9�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:9*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:9~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:9�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:9P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:9~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:9*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:9c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������9h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:9v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:9*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:9r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������9b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������9�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������9: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������9
 
_user_specified_nameinputs
�

�
J__inference_gene_decoder_1_layer_call_and_return_conditional_losses_188571

inputs0
matmul_readvariableop_resource:@}-
biasadd_readvariableop_resource:}
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@}*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:}*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������}Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������}w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
]
1__inference_ConcatenateLayer_layer_call_fn_188524
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_186603a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������}:���������9:Q M
'
_output_shapes
:���������}
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������9
"
_user_specified_name
inputs/1
�
�
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_188737

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�+
!__inference__wrapped_model_186017
gene_input_layer
protein_input_layerG
3model_gene_encoder_1_matmul_readvariableop_resource:
��C
4model_gene_encoder_1_biasadd_readvariableop_resource:	�K
<model_batchnormgeneencode1_batchnorm_readvariableop_resource:	�O
@model_batchnormgeneencode1_batchnorm_mul_readvariableop_resource:	�M
>model_batchnormgeneencode1_batchnorm_readvariableop_1_resource:	�M
>model_batchnormgeneencode1_batchnorm_readvariableop_2_resource:	�I
6model_protein_encoder_1_matmul_readvariableop_resource:	�9E
7model_protein_encoder_1_biasadd_readvariableop_resource:9F
3model_gene_encoder_2_matmul_readvariableop_resource:	�}B
4model_gene_encoder_2_biasadd_readvariableop_resource:}J
<model_batchnormgeneencode2_batchnorm_readvariableop_resource:}N
@model_batchnormgeneencode2_batchnorm_mul_readvariableop_resource:}L
>model_batchnormgeneencode2_batchnorm_readvariableop_1_resource:}L
>model_batchnormgeneencode2_batchnorm_readvariableop_2_resource:}M
?model_batchnormproteinencode1_batchnorm_readvariableop_resource:9Q
Cmodel_batchnormproteinencode1_batchnorm_mul_readvariableop_resource:9O
Amodel_batchnormproteinencode1_batchnorm_readvariableop_1_resource:9O
Amodel_batchnormproteinencode1_batchnorm_readvariableop_2_resource:9I
6model_embeddingdimdense_matmul_readvariableop_resource:	�@E
7model_embeddingdimdense_biasadd_readvariableop_resource:@E
3model_gene_decoder_1_matmul_readvariableop_resource:@}B
4model_gene_decoder_1_biasadd_readvariableop_resource:}J
<model_batchnormgenedecode1_batchnorm_readvariableop_resource:}N
@model_batchnormgenedecode1_batchnorm_mul_readvariableop_resource:}L
>model_batchnormgenedecode1_batchnorm_readvariableop_1_resource:}L
>model_batchnormgenedecode1_batchnorm_readvariableop_2_resource:}H
6model_protein_decoder_1_matmul_readvariableop_resource:@9E
7model_protein_decoder_1_biasadd_readvariableop_resource:9F
3model_gene_decoder_2_matmul_readvariableop_resource:	}�C
4model_gene_decoder_2_biasadd_readvariableop_resource:	�M
?model_batchnormproteindecode1_batchnorm_readvariableop_resource:9Q
Cmodel_batchnormproteindecode1_batchnorm_mul_readvariableop_resource:9O
Amodel_batchnormproteindecode1_batchnorm_readvariableop_1_resource:9O
Amodel_batchnormproteindecode1_batchnorm_readvariableop_2_resource:9K
<model_batchnormgenedecode2_batchnorm_readvariableop_resource:	�O
@model_batchnormgenedecode2_batchnorm_mul_readvariableop_resource:	�M
>model_batchnormgenedecode2_batchnorm_readvariableop_1_resource:	�M
>model_batchnormgenedecode2_batchnorm_readvariableop_2_resource:	�L
9model_protein_decoder_last_matmul_readvariableop_resource:	9�I
:model_protein_decoder_last_biasadd_readvariableop_resource:	�J
6model_gene_decoder_last_matmul_readvariableop_resource:
��F
7model_gene_decoder_last_biasadd_readvariableop_resource:	�
identity

identity_1��3model/BatchNormGeneDecode1/batchnorm/ReadVariableOp�5model/BatchNormGeneDecode1/batchnorm/ReadVariableOp_1�5model/BatchNormGeneDecode1/batchnorm/ReadVariableOp_2�7model/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp�3model/BatchNormGeneDecode2/batchnorm/ReadVariableOp�5model/BatchNormGeneDecode2/batchnorm/ReadVariableOp_1�5model/BatchNormGeneDecode2/batchnorm/ReadVariableOp_2�7model/BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp�3model/BatchNormGeneEncode1/batchnorm/ReadVariableOp�5model/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1�5model/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2�7model/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp�3model/BatchNormGeneEncode2/batchnorm/ReadVariableOp�5model/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1�5model/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2�7model/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp�6model/BatchNormProteinDecode1/batchnorm/ReadVariableOp�8model/BatchNormProteinDecode1/batchnorm/ReadVariableOp_1�8model/BatchNormProteinDecode1/batchnorm/ReadVariableOp_2�:model/BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp�6model/BatchNormProteinEncode1/batchnorm/ReadVariableOp�8model/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1�8model/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2�:model/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp�.model/EmbeddingDimDense/BiasAdd/ReadVariableOp�-model/EmbeddingDimDense/MatMul/ReadVariableOp�+model/gene_decoder_1/BiasAdd/ReadVariableOp�*model/gene_decoder_1/MatMul/ReadVariableOp�+model/gene_decoder_2/BiasAdd/ReadVariableOp�*model/gene_decoder_2/MatMul/ReadVariableOp�.model/gene_decoder_last/BiasAdd/ReadVariableOp�-model/gene_decoder_last/MatMul/ReadVariableOp�+model/gene_encoder_1/BiasAdd/ReadVariableOp�*model/gene_encoder_1/MatMul/ReadVariableOp�+model/gene_encoder_2/BiasAdd/ReadVariableOp�*model/gene_encoder_2/MatMul/ReadVariableOp�.model/protein_decoder_1/BiasAdd/ReadVariableOp�-model/protein_decoder_1/MatMul/ReadVariableOp�1model/protein_decoder_last/BiasAdd/ReadVariableOp�0model/protein_decoder_last/MatMul/ReadVariableOp�.model/protein_encoder_1/BiasAdd/ReadVariableOp�-model/protein_encoder_1/MatMul/ReadVariableOp�
*model/gene_encoder_1/MatMul/ReadVariableOpReadVariableOp3model_gene_encoder_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/gene_encoder_1/MatMulMatMulgene_input_layer2model/gene_encoder_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+model/gene_encoder_1/BiasAdd/ReadVariableOpReadVariableOp4model_gene_encoder_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/gene_encoder_1/BiasAddBiasAdd%model/gene_encoder_1/MatMul:product:03model/gene_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
model/gene_encoder_1/SigmoidSigmoid%model/gene_encoder_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
3model/BatchNormGeneEncode1/batchnorm/ReadVariableOpReadVariableOp<model_batchnormgeneencode1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0o
*model/BatchNormGeneEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
(model/BatchNormGeneEncode1/batchnorm/addAddV2;model/BatchNormGeneEncode1/batchnorm/ReadVariableOp:value:03model/BatchNormGeneEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
*model/BatchNormGeneEncode1/batchnorm/RsqrtRsqrt,model/BatchNormGeneEncode1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
7model/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOpReadVariableOp@model_batchnormgeneencode1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(model/BatchNormGeneEncode1/batchnorm/mulMul.model/BatchNormGeneEncode1/batchnorm/Rsqrt:y:0?model/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
*model/BatchNormGeneEncode1/batchnorm/mul_1Mul model/gene_encoder_1/Sigmoid:y:0,model/BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
5model/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1ReadVariableOp>model_batchnormgeneencode1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
*model/BatchNormGeneEncode1/batchnorm/mul_2Mul=model/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1:value:0,model/BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
5model/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2ReadVariableOp>model_batchnormgeneencode1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
(model/BatchNormGeneEncode1/batchnorm/subSub=model/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2:value:0.model/BatchNormGeneEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
*model/BatchNormGeneEncode1/batchnorm/add_1AddV2.model/BatchNormGeneEncode1/batchnorm/mul_1:z:0,model/BatchNormGeneEncode1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
-model/protein_encoder_1/MatMul/ReadVariableOpReadVariableOp6model_protein_encoder_1_matmul_readvariableop_resource*
_output_shapes
:	�9*
dtype0�
model/protein_encoder_1/MatMulMatMulprotein_input_layer5model/protein_encoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������9�
.model/protein_encoder_1/BiasAdd/ReadVariableOpReadVariableOp7model_protein_encoder_1_biasadd_readvariableop_resource*
_output_shapes
:9*
dtype0�
model/protein_encoder_1/BiasAddBiasAdd(model/protein_encoder_1/MatMul:product:06model/protein_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������9�
model/protein_encoder_1/SigmoidSigmoid(model/protein_encoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������9�
*model/gene_encoder_2/MatMul/ReadVariableOpReadVariableOp3model_gene_encoder_2_matmul_readvariableop_resource*
_output_shapes
:	�}*
dtype0�
model/gene_encoder_2/MatMulMatMul.model/BatchNormGeneEncode1/batchnorm/add_1:z:02model/gene_encoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}�
+model/gene_encoder_2/BiasAdd/ReadVariableOpReadVariableOp4model_gene_encoder_2_biasadd_readvariableop_resource*
_output_shapes
:}*
dtype0�
model/gene_encoder_2/BiasAddBiasAdd%model/gene_encoder_2/MatMul:product:03model/gene_encoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}�
model/gene_encoder_2/SigmoidSigmoid%model/gene_encoder_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������}�
3model/BatchNormGeneEncode2/batchnorm/ReadVariableOpReadVariableOp<model_batchnormgeneencode2_batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0o
*model/BatchNormGeneEncode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
(model/BatchNormGeneEncode2/batchnorm/addAddV2;model/BatchNormGeneEncode2/batchnorm/ReadVariableOp:value:03model/BatchNormGeneEncode2/batchnorm/add/y:output:0*
T0*
_output_shapes
:}�
*model/BatchNormGeneEncode2/batchnorm/RsqrtRsqrt,model/BatchNormGeneEncode2/batchnorm/add:z:0*
T0*
_output_shapes
:}�
7model/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOpReadVariableOp@model_batchnormgeneencode2_batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0�
(model/BatchNormGeneEncode2/batchnorm/mulMul.model/BatchNormGeneEncode2/batchnorm/Rsqrt:y:0?model/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}�
*model/BatchNormGeneEncode2/batchnorm/mul_1Mul model/gene_encoder_2/Sigmoid:y:0,model/BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������}�
5model/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1ReadVariableOp>model_batchnormgeneencode2_batchnorm_readvariableop_1_resource*
_output_shapes
:}*
dtype0�
*model/BatchNormGeneEncode2/batchnorm/mul_2Mul=model/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1:value:0,model/BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*
_output_shapes
:}�
5model/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2ReadVariableOp>model_batchnormgeneencode2_batchnorm_readvariableop_2_resource*
_output_shapes
:}*
dtype0�
(model/BatchNormGeneEncode2/batchnorm/subSub=model/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2:value:0.model/BatchNormGeneEncode2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:}�
*model/BatchNormGeneEncode2/batchnorm/add_1AddV2.model/BatchNormGeneEncode2/batchnorm/mul_1:z:0,model/BatchNormGeneEncode2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}�
6model/BatchNormProteinEncode1/batchnorm/ReadVariableOpReadVariableOp?model_batchnormproteinencode1_batchnorm_readvariableop_resource*
_output_shapes
:9*
dtype0r
-model/BatchNormProteinEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
+model/BatchNormProteinEncode1/batchnorm/addAddV2>model/BatchNormProteinEncode1/batchnorm/ReadVariableOp:value:06model/BatchNormProteinEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:9�
-model/BatchNormProteinEncode1/batchnorm/RsqrtRsqrt/model/BatchNormProteinEncode1/batchnorm/add:z:0*
T0*
_output_shapes
:9�
:model/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_batchnormproteinencode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:9*
dtype0�
+model/BatchNormProteinEncode1/batchnorm/mulMul1model/BatchNormProteinEncode1/batchnorm/Rsqrt:y:0Bmodel/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:9�
-model/BatchNormProteinEncode1/batchnorm/mul_1Mul#model/protein_encoder_1/Sigmoid:y:0/model/BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������9�
8model/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_batchnormproteinencode1_batchnorm_readvariableop_1_resource*
_output_shapes
:9*
dtype0�
-model/BatchNormProteinEncode1/batchnorm/mul_2Mul@model/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1:value:0/model/BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*
_output_shapes
:9�
8model/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_batchnormproteinencode1_batchnorm_readvariableop_2_resource*
_output_shapes
:9*
dtype0�
+model/BatchNormProteinEncode1/batchnorm/subSub@model/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2:value:01model/BatchNormProteinEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:9�
-model/BatchNormProteinEncode1/batchnorm/add_1AddV21model/BatchNormProteinEncode1/batchnorm/mul_1:z:0/model/BatchNormProteinEncode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������9d
"model/ConcatenateLayer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/ConcatenateLayer/concatConcatV2.model/BatchNormGeneEncode2/batchnorm/add_1:z:01model/BatchNormProteinEncode1/batchnorm/add_1:z:0+model/ConcatenateLayer/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
-model/EmbeddingDimDense/MatMul/ReadVariableOpReadVariableOp6model_embeddingdimdense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
model/EmbeddingDimDense/MatMulMatMul&model/ConcatenateLayer/concat:output:05model/EmbeddingDimDense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
.model/EmbeddingDimDense/BiasAdd/ReadVariableOpReadVariableOp7model_embeddingdimdense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/EmbeddingDimDense/BiasAddBiasAdd(model/EmbeddingDimDense/MatMul:product:06model/EmbeddingDimDense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
model/EmbeddingDimDense/SigmoidSigmoid(model/EmbeddingDimDense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
*model/gene_decoder_1/MatMul/ReadVariableOpReadVariableOp3model_gene_decoder_1_matmul_readvariableop_resource*
_output_shapes

:@}*
dtype0�
model/gene_decoder_1/MatMulMatMul#model/EmbeddingDimDense/Sigmoid:y:02model/gene_decoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}�
+model/gene_decoder_1/BiasAdd/ReadVariableOpReadVariableOp4model_gene_decoder_1_biasadd_readvariableop_resource*
_output_shapes
:}*
dtype0�
model/gene_decoder_1/BiasAddBiasAdd%model/gene_decoder_1/MatMul:product:03model/gene_decoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}�
model/gene_decoder_1/SigmoidSigmoid%model/gene_decoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������}�
3model/BatchNormGeneDecode1/batchnorm/ReadVariableOpReadVariableOp<model_batchnormgenedecode1_batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0o
*model/BatchNormGeneDecode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
(model/BatchNormGeneDecode1/batchnorm/addAddV2;model/BatchNormGeneDecode1/batchnorm/ReadVariableOp:value:03model/BatchNormGeneDecode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:}�
*model/BatchNormGeneDecode1/batchnorm/RsqrtRsqrt,model/BatchNormGeneDecode1/batchnorm/add:z:0*
T0*
_output_shapes
:}�
7model/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOpReadVariableOp@model_batchnormgenedecode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0�
(model/BatchNormGeneDecode1/batchnorm/mulMul.model/BatchNormGeneDecode1/batchnorm/Rsqrt:y:0?model/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}�
*model/BatchNormGeneDecode1/batchnorm/mul_1Mul model/gene_decoder_1/Sigmoid:y:0,model/BatchNormGeneDecode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������}�
5model/BatchNormGeneDecode1/batchnorm/ReadVariableOp_1ReadVariableOp>model_batchnormgenedecode1_batchnorm_readvariableop_1_resource*
_output_shapes
:}*
dtype0�
*model/BatchNormGeneDecode1/batchnorm/mul_2Mul=model/BatchNormGeneDecode1/batchnorm/ReadVariableOp_1:value:0,model/BatchNormGeneDecode1/batchnorm/mul:z:0*
T0*
_output_shapes
:}�
5model/BatchNormGeneDecode1/batchnorm/ReadVariableOp_2ReadVariableOp>model_batchnormgenedecode1_batchnorm_readvariableop_2_resource*
_output_shapes
:}*
dtype0�
(model/BatchNormGeneDecode1/batchnorm/subSub=model/BatchNormGeneDecode1/batchnorm/ReadVariableOp_2:value:0.model/BatchNormGeneDecode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:}�
*model/BatchNormGeneDecode1/batchnorm/add_1AddV2.model/BatchNormGeneDecode1/batchnorm/mul_1:z:0,model/BatchNormGeneDecode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}�
-model/protein_decoder_1/MatMul/ReadVariableOpReadVariableOp6model_protein_decoder_1_matmul_readvariableop_resource*
_output_shapes

:@9*
dtype0�
model/protein_decoder_1/MatMulMatMul#model/EmbeddingDimDense/Sigmoid:y:05model/protein_decoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������9�
.model/protein_decoder_1/BiasAdd/ReadVariableOpReadVariableOp7model_protein_decoder_1_biasadd_readvariableop_resource*
_output_shapes
:9*
dtype0�
model/protein_decoder_1/BiasAddBiasAdd(model/protein_decoder_1/MatMul:product:06model/protein_decoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������9�
model/protein_decoder_1/SigmoidSigmoid(model/protein_decoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������9�
*model/gene_decoder_2/MatMul/ReadVariableOpReadVariableOp3model_gene_decoder_2_matmul_readvariableop_resource*
_output_shapes
:	}�*
dtype0�
model/gene_decoder_2/MatMulMatMul.model/BatchNormGeneDecode1/batchnorm/add_1:z:02model/gene_decoder_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
+model/gene_decoder_2/BiasAdd/ReadVariableOpReadVariableOp4model_gene_decoder_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/gene_decoder_2/BiasAddBiasAdd%model/gene_decoder_2/MatMul:product:03model/gene_decoder_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
model/gene_decoder_2/SigmoidSigmoid%model/gene_decoder_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
6model/BatchNormProteinDecode1/batchnorm/ReadVariableOpReadVariableOp?model_batchnormproteindecode1_batchnorm_readvariableop_resource*
_output_shapes
:9*
dtype0r
-model/BatchNormProteinDecode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
+model/BatchNormProteinDecode1/batchnorm/addAddV2>model/BatchNormProteinDecode1/batchnorm/ReadVariableOp:value:06model/BatchNormProteinDecode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:9�
-model/BatchNormProteinDecode1/batchnorm/RsqrtRsqrt/model/BatchNormProteinDecode1/batchnorm/add:z:0*
T0*
_output_shapes
:9�
:model/BatchNormProteinDecode1/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_batchnormproteindecode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:9*
dtype0�
+model/BatchNormProteinDecode1/batchnorm/mulMul1model/BatchNormProteinDecode1/batchnorm/Rsqrt:y:0Bmodel/BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:9�
-model/BatchNormProteinDecode1/batchnorm/mul_1Mul#model/protein_decoder_1/Sigmoid:y:0/model/BatchNormProteinDecode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������9�
8model/BatchNormProteinDecode1/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_batchnormproteindecode1_batchnorm_readvariableop_1_resource*
_output_shapes
:9*
dtype0�
-model/BatchNormProteinDecode1/batchnorm/mul_2Mul@model/BatchNormProteinDecode1/batchnorm/ReadVariableOp_1:value:0/model/BatchNormProteinDecode1/batchnorm/mul:z:0*
T0*
_output_shapes
:9�
8model/BatchNormProteinDecode1/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_batchnormproteindecode1_batchnorm_readvariableop_2_resource*
_output_shapes
:9*
dtype0�
+model/BatchNormProteinDecode1/batchnorm/subSub@model/BatchNormProteinDecode1/batchnorm/ReadVariableOp_2:value:01model/BatchNormProteinDecode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:9�
-model/BatchNormProteinDecode1/batchnorm/add_1AddV21model/BatchNormProteinDecode1/batchnorm/mul_1:z:0/model/BatchNormProteinDecode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������9�
3model/BatchNormGeneDecode2/batchnorm/ReadVariableOpReadVariableOp<model_batchnormgenedecode2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0o
*model/BatchNormGeneDecode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
(model/BatchNormGeneDecode2/batchnorm/addAddV2;model/BatchNormGeneDecode2/batchnorm/ReadVariableOp:value:03model/BatchNormGeneDecode2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
*model/BatchNormGeneDecode2/batchnorm/RsqrtRsqrt,model/BatchNormGeneDecode2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
7model/BatchNormGeneDecode2/batchnorm/mul/ReadVariableOpReadVariableOp@model_batchnormgenedecode2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
(model/BatchNormGeneDecode2/batchnorm/mulMul.model/BatchNormGeneDecode2/batchnorm/Rsqrt:y:0?model/BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
*model/BatchNormGeneDecode2/batchnorm/mul_1Mul model/gene_decoder_2/Sigmoid:y:0,model/BatchNormGeneDecode2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
5model/BatchNormGeneDecode2/batchnorm/ReadVariableOp_1ReadVariableOp>model_batchnormgenedecode2_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
*model/BatchNormGeneDecode2/batchnorm/mul_2Mul=model/BatchNormGeneDecode2/batchnorm/ReadVariableOp_1:value:0,model/BatchNormGeneDecode2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
5model/BatchNormGeneDecode2/batchnorm/ReadVariableOp_2ReadVariableOp>model_batchnormgenedecode2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
(model/BatchNormGeneDecode2/batchnorm/subSub=model/BatchNormGeneDecode2/batchnorm/ReadVariableOp_2:value:0.model/BatchNormGeneDecode2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
*model/BatchNormGeneDecode2/batchnorm/add_1AddV2.model/BatchNormGeneDecode2/batchnorm/mul_1:z:0,model/BatchNormGeneDecode2/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
0model/protein_decoder_last/MatMul/ReadVariableOpReadVariableOp9model_protein_decoder_last_matmul_readvariableop_resource*
_output_shapes
:	9�*
dtype0�
!model/protein_decoder_last/MatMulMatMul1model/BatchNormProteinDecode1/batchnorm/add_1:z:08model/protein_decoder_last/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
1model/protein_decoder_last/BiasAdd/ReadVariableOpReadVariableOp:model_protein_decoder_last_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"model/protein_decoder_last/BiasAddBiasAdd+model/protein_decoder_last/MatMul:product:09model/protein_decoder_last/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"model/protein_decoder_last/SigmoidSigmoid+model/protein_decoder_last/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
-model/gene_decoder_last/MatMul/ReadVariableOpReadVariableOp6model_gene_decoder_last_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model/gene_decoder_last/MatMulMatMul.model/BatchNormGeneDecode2/batchnorm/add_1:z:05model/gene_decoder_last/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.model/gene_decoder_last/BiasAdd/ReadVariableOpReadVariableOp7model_gene_decoder_last_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model/gene_decoder_last/BiasAddBiasAdd(model/gene_decoder_last/MatMul:product:06model/gene_decoder_last/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
model/gene_decoder_last/SigmoidSigmoid(model/gene_decoder_last/BiasAdd:output:0*
T0*(
_output_shapes
:����������s
IdentityIdentity#model/gene_decoder_last/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������x

Identity_1Identity&model/protein_decoder_last/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp4^model/BatchNormGeneDecode1/batchnorm/ReadVariableOp6^model/BatchNormGeneDecode1/batchnorm/ReadVariableOp_16^model/BatchNormGeneDecode1/batchnorm/ReadVariableOp_28^model/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp4^model/BatchNormGeneDecode2/batchnorm/ReadVariableOp6^model/BatchNormGeneDecode2/batchnorm/ReadVariableOp_16^model/BatchNormGeneDecode2/batchnorm/ReadVariableOp_28^model/BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp4^model/BatchNormGeneEncode1/batchnorm/ReadVariableOp6^model/BatchNormGeneEncode1/batchnorm/ReadVariableOp_16^model/BatchNormGeneEncode1/batchnorm/ReadVariableOp_28^model/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp4^model/BatchNormGeneEncode2/batchnorm/ReadVariableOp6^model/BatchNormGeneEncode2/batchnorm/ReadVariableOp_16^model/BatchNormGeneEncode2/batchnorm/ReadVariableOp_28^model/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp7^model/BatchNormProteinDecode1/batchnorm/ReadVariableOp9^model/BatchNormProteinDecode1/batchnorm/ReadVariableOp_19^model/BatchNormProteinDecode1/batchnorm/ReadVariableOp_2;^model/BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp7^model/BatchNormProteinEncode1/batchnorm/ReadVariableOp9^model/BatchNormProteinEncode1/batchnorm/ReadVariableOp_19^model/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2;^model/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp/^model/EmbeddingDimDense/BiasAdd/ReadVariableOp.^model/EmbeddingDimDense/MatMul/ReadVariableOp,^model/gene_decoder_1/BiasAdd/ReadVariableOp+^model/gene_decoder_1/MatMul/ReadVariableOp,^model/gene_decoder_2/BiasAdd/ReadVariableOp+^model/gene_decoder_2/MatMul/ReadVariableOp/^model/gene_decoder_last/BiasAdd/ReadVariableOp.^model/gene_decoder_last/MatMul/ReadVariableOp,^model/gene_encoder_1/BiasAdd/ReadVariableOp+^model/gene_encoder_1/MatMul/ReadVariableOp,^model/gene_encoder_2/BiasAdd/ReadVariableOp+^model/gene_encoder_2/MatMul/ReadVariableOp/^model/protein_decoder_1/BiasAdd/ReadVariableOp.^model/protein_decoder_1/MatMul/ReadVariableOp2^model/protein_decoder_last/BiasAdd/ReadVariableOp1^model/protein_decoder_last/MatMul/ReadVariableOp/^model/protein_encoder_1/BiasAdd/ReadVariableOp.^model/protein_encoder_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2j
3model/BatchNormGeneDecode1/batchnorm/ReadVariableOp3model/BatchNormGeneDecode1/batchnorm/ReadVariableOp2n
5model/BatchNormGeneDecode1/batchnorm/ReadVariableOp_15model/BatchNormGeneDecode1/batchnorm/ReadVariableOp_12n
5model/BatchNormGeneDecode1/batchnorm/ReadVariableOp_25model/BatchNormGeneDecode1/batchnorm/ReadVariableOp_22r
7model/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp7model/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp2j
3model/BatchNormGeneDecode2/batchnorm/ReadVariableOp3model/BatchNormGeneDecode2/batchnorm/ReadVariableOp2n
5model/BatchNormGeneDecode2/batchnorm/ReadVariableOp_15model/BatchNormGeneDecode2/batchnorm/ReadVariableOp_12n
5model/BatchNormGeneDecode2/batchnorm/ReadVariableOp_25model/BatchNormGeneDecode2/batchnorm/ReadVariableOp_22r
7model/BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp7model/BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp2j
3model/BatchNormGeneEncode1/batchnorm/ReadVariableOp3model/BatchNormGeneEncode1/batchnorm/ReadVariableOp2n
5model/BatchNormGeneEncode1/batchnorm/ReadVariableOp_15model/BatchNormGeneEncode1/batchnorm/ReadVariableOp_12n
5model/BatchNormGeneEncode1/batchnorm/ReadVariableOp_25model/BatchNormGeneEncode1/batchnorm/ReadVariableOp_22r
7model/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp7model/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp2j
3model/BatchNormGeneEncode2/batchnorm/ReadVariableOp3model/BatchNormGeneEncode2/batchnorm/ReadVariableOp2n
5model/BatchNormGeneEncode2/batchnorm/ReadVariableOp_15model/BatchNormGeneEncode2/batchnorm/ReadVariableOp_12n
5model/BatchNormGeneEncode2/batchnorm/ReadVariableOp_25model/BatchNormGeneEncode2/batchnorm/ReadVariableOp_22r
7model/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp7model/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp2p
6model/BatchNormProteinDecode1/batchnorm/ReadVariableOp6model/BatchNormProteinDecode1/batchnorm/ReadVariableOp2t
8model/BatchNormProteinDecode1/batchnorm/ReadVariableOp_18model/BatchNormProteinDecode1/batchnorm/ReadVariableOp_12t
8model/BatchNormProteinDecode1/batchnorm/ReadVariableOp_28model/BatchNormProteinDecode1/batchnorm/ReadVariableOp_22x
:model/BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp:model/BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp2p
6model/BatchNormProteinEncode1/batchnorm/ReadVariableOp6model/BatchNormProteinEncode1/batchnorm/ReadVariableOp2t
8model/BatchNormProteinEncode1/batchnorm/ReadVariableOp_18model/BatchNormProteinEncode1/batchnorm/ReadVariableOp_12t
8model/BatchNormProteinEncode1/batchnorm/ReadVariableOp_28model/BatchNormProteinEncode1/batchnorm/ReadVariableOp_22x
:model/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp:model/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp2`
.model/EmbeddingDimDense/BiasAdd/ReadVariableOp.model/EmbeddingDimDense/BiasAdd/ReadVariableOp2^
-model/EmbeddingDimDense/MatMul/ReadVariableOp-model/EmbeddingDimDense/MatMul/ReadVariableOp2Z
+model/gene_decoder_1/BiasAdd/ReadVariableOp+model/gene_decoder_1/BiasAdd/ReadVariableOp2X
*model/gene_decoder_1/MatMul/ReadVariableOp*model/gene_decoder_1/MatMul/ReadVariableOp2Z
+model/gene_decoder_2/BiasAdd/ReadVariableOp+model/gene_decoder_2/BiasAdd/ReadVariableOp2X
*model/gene_decoder_2/MatMul/ReadVariableOp*model/gene_decoder_2/MatMul/ReadVariableOp2`
.model/gene_decoder_last/BiasAdd/ReadVariableOp.model/gene_decoder_last/BiasAdd/ReadVariableOp2^
-model/gene_decoder_last/MatMul/ReadVariableOp-model/gene_decoder_last/MatMul/ReadVariableOp2Z
+model/gene_encoder_1/BiasAdd/ReadVariableOp+model/gene_encoder_1/BiasAdd/ReadVariableOp2X
*model/gene_encoder_1/MatMul/ReadVariableOp*model/gene_encoder_1/MatMul/ReadVariableOp2Z
+model/gene_encoder_2/BiasAdd/ReadVariableOp+model/gene_encoder_2/BiasAdd/ReadVariableOp2X
*model/gene_encoder_2/MatMul/ReadVariableOp*model/gene_encoder_2/MatMul/ReadVariableOp2`
.model/protein_decoder_1/BiasAdd/ReadVariableOp.model/protein_decoder_1/BiasAdd/ReadVariableOp2^
-model/protein_decoder_1/MatMul/ReadVariableOp-model/protein_decoder_1/MatMul/ReadVariableOp2f
1model/protein_decoder_last/BiasAdd/ReadVariableOp1model/protein_decoder_last/BiasAdd/ReadVariableOp2d
0model/protein_decoder_last/MatMul/ReadVariableOp0model/protein_decoder_last/MatMul/ReadVariableOp2`
.model/protein_encoder_1/BiasAdd/ReadVariableOp.model/protein_encoder_1/BiasAdd/ReadVariableOp2^
-model/protein_encoder_1/MatMul/ReadVariableOp-model/protein_encoder_1/MatMul/ReadVariableOp:Z V
(
_output_shapes
:����������
*
_user_specified_nameGene_Input_Layer:]Y
(
_output_shapes
:����������
-
_user_specified_nameProtein_Input_Layer
�%
�
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_186252

inputs5
'assignmovingavg_readvariableop_resource:97
)assignmovingavg_1_readvariableop_resource:93
%batchnorm_mul_readvariableop_resource:9/
!batchnorm_readvariableop_resource:9
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:9*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:9�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������9l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:9*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:9*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:9*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:9*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:9x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:9�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:9*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:9~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:9�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:9P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:9~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:9*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:9c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������9h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:9v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:9*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:9r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������9b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������9�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������9: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������9
 
_user_specified_nameinputs
�

�
P__inference_protein_decoder_last_layer_call_and_return_conditional_losses_188891

inputs1
matmul_readvariableop_resource:	9�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	9�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������9: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������9
 
_user_specified_nameinputs
�

�
M__inference_gene_decoder_last_layer_call_and_return_conditional_losses_188871

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_186123

inputs/
!batchnorm_readvariableop_resource:}3
%batchnorm_mul_readvariableop_resource:}1
#batchnorm_readvariableop_1_resource:}1
#batchnorm_readvariableop_2_resource:}
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:}P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:}~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������}z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:}*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:}z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:}*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:}r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������}�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������}: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������}
 
_user_specified_nameinputs
�%
�
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_186170

inputs5
'assignmovingavg_readvariableop_resource:}7
)assignmovingavg_1_readvariableop_resource:}3
%batchnorm_mul_readvariableop_resource:}/
!batchnorm_readvariableop_resource:}
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:}*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:}�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������}l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:}*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:}*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:}*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:}*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:}x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:}�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:}*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:}~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:}�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:}P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:}~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������}h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:}v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:}r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������}�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������}: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������}
 
_user_specified_nameinputs
�e
�
A__inference_model_layer_call_and_return_conditional_losses_187123

inputs
inputs_1)
gene_encoder_1_187021:
��$
gene_encoder_1_187023:	�*
batchnormgeneencode1_187026:	�*
batchnormgeneencode1_187028:	�*
batchnormgeneencode1_187030:	�*
batchnormgeneencode1_187032:	�+
protein_encoder_1_187035:	�9&
protein_encoder_1_187037:9(
gene_encoder_2_187040:	�}#
gene_encoder_2_187042:})
batchnormgeneencode2_187045:})
batchnormgeneencode2_187047:})
batchnormgeneencode2_187049:})
batchnormgeneencode2_187051:},
batchnormproteinencode1_187054:9,
batchnormproteinencode1_187056:9,
batchnormproteinencode1_187058:9,
batchnormproteinencode1_187060:9+
embeddingdimdense_187064:	�@&
embeddingdimdense_187066:@'
gene_decoder_1_187069:@}#
gene_decoder_1_187071:})
batchnormgenedecode1_187074:})
batchnormgenedecode1_187076:})
batchnormgenedecode1_187078:})
batchnormgenedecode1_187080:}*
protein_decoder_1_187083:@9&
protein_decoder_1_187085:9(
gene_decoder_2_187088:	}�$
gene_decoder_2_187090:	�,
batchnormproteindecode1_187093:9,
batchnormproteindecode1_187095:9,
batchnormproteindecode1_187097:9,
batchnormproteindecode1_187099:9*
batchnormgenedecode2_187102:	�*
batchnormgenedecode2_187104:	�*
batchnormgenedecode2_187106:	�*
batchnormgenedecode2_187108:	�.
protein_decoder_last_187111:	9�*
protein_decoder_last_187113:	�,
gene_decoder_last_187116:
��'
gene_decoder_last_187118:	�
identity

identity_1��,BatchNormGeneDecode1/StatefulPartitionedCall�,BatchNormGeneDecode2/StatefulPartitionedCall�,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�/BatchNormProteinDecode1/StatefulPartitionedCall�/BatchNormProteinEncode1/StatefulPartitionedCall�)EmbeddingDimDense/StatefulPartitionedCall�&gene_decoder_1/StatefulPartitionedCall�&gene_decoder_2/StatefulPartitionedCall�)gene_decoder_last/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�)protein_decoder_1/StatefulPartitionedCall�,protein_decoder_last/StatefulPartitionedCall�)protein_encoder_1/StatefulPartitionedCall�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputsgene_encoder_1_187021gene_encoder_1_187023*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_186529�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_187026batchnormgeneencode1_187028batchnormgeneencode1_187030batchnormgeneencode1_187032*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_186088�
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1protein_encoder_1_187035protein_encoder_1_187037*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������9*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_186555�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_187040gene_encoder_2_187042*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_186572�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_187045batchnormgeneencode2_187047batchnormgeneencode2_187049batchnormgeneencode2_187051*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_186170�
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_187054batchnormproteinencode1_187056batchnormproteinencode1_187058batchnormproteinencode1_187060*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������9*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_186252�
 ConcatenateLayer/PartitionedCallPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:08BatchNormProteinEncode1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_186603�
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_187064embeddingdimdense_187066*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_186616�
&gene_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0gene_decoder_1_187069gene_decoder_1_187071*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_decoder_1_layer_call_and_return_conditional_losses_186633�
,BatchNormGeneDecode1/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_1/StatefulPartitionedCall:output:0batchnormgenedecode1_187074batchnormgenedecode1_187076batchnormgenedecode1_187078batchnormgenedecode1_187080*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_186334�
)protein_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0protein_decoder_1_187083protein_decoder_1_187085*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������9*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_decoder_1_layer_call_and_return_conditional_losses_186659�
&gene_decoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode1/StatefulPartitionedCall:output:0gene_decoder_2_187088gene_decoder_2_187090*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_decoder_2_layer_call_and_return_conditional_losses_186676�
/BatchNormProteinDecode1/StatefulPartitionedCallStatefulPartitionedCall2protein_decoder_1/StatefulPartitionedCall:output:0batchnormproteindecode1_187093batchnormproteindecode1_187095batchnormproteindecode1_187097batchnormproteindecode1_187099*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������9*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_186498�
,BatchNormGeneDecode2/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_2/StatefulPartitionedCall:output:0batchnormgenedecode2_187102batchnormgenedecode2_187104batchnormgenedecode2_187106batchnormgenedecode2_187108*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_186416�
,protein_decoder_last/StatefulPartitionedCallStatefulPartitionedCall8BatchNormProteinDecode1/StatefulPartitionedCall:output:0protein_decoder_last_187111protein_decoder_last_187113*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_protein_decoder_last_layer_call_and_return_conditional_losses_186711�
)gene_decoder_last/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode2/StatefulPartitionedCall:output:0gene_decoder_last_187116gene_decoder_last_187118*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_gene_decoder_last_layer_call_and_return_conditional_losses_186728�
IdentityIdentity2gene_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������

Identity_1Identity5protein_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp-^BatchNormGeneDecode1/StatefulPartitionedCall-^BatchNormGeneDecode2/StatefulPartitionedCall-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinDecode1/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_decoder_1/StatefulPartitionedCall'^gene_decoder_2/StatefulPartitionedCall*^gene_decoder_last/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_decoder_1/StatefulPartitionedCall-^protein_decoder_last/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneDecode1/StatefulPartitionedCall,BatchNormGeneDecode1/StatefulPartitionedCall2\
,BatchNormGeneDecode2/StatefulPartitionedCall,BatchNormGeneDecode2/StatefulPartitionedCall2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2b
/BatchNormProteinDecode1/StatefulPartitionedCall/BatchNormProteinDecode1/StatefulPartitionedCall2b
/BatchNormProteinEncode1/StatefulPartitionedCall/BatchNormProteinEncode1/StatefulPartitionedCall2V
)EmbeddingDimDense/StatefulPartitionedCall)EmbeddingDimDense/StatefulPartitionedCall2P
&gene_decoder_1/StatefulPartitionedCall&gene_decoder_1/StatefulPartitionedCall2P
&gene_decoder_2/StatefulPartitionedCall&gene_decoder_2/StatefulPartitionedCall2V
)gene_decoder_last/StatefulPartitionedCall)gene_decoder_last/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall2V
)protein_decoder_1/StatefulPartitionedCall)protein_decoder_1/StatefulPartitionedCall2\
,protein_decoder_last/StatefulPartitionedCall,protein_decoder_last/StatefulPartitionedCall2V
)protein_encoder_1/StatefulPartitionedCall)protein_encoder_1/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_188617

inputs/
!batchnorm_readvariableop_resource:}3
%batchnorm_mul_readvariableop_resource:}1
#batchnorm_readvariableop_1_resource:}1
#batchnorm_readvariableop_2_resource:}
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:}P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:}~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������}z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:}*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:}z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:}*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:}r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������}�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������}: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������}
 
_user_specified_nameinputs
�f
�
A__inference_model_layer_call_and_return_conditional_losses_187410
gene_input_layer
protein_input_layer)
gene_encoder_1_187308:
��$
gene_encoder_1_187310:	�*
batchnormgeneencode1_187313:	�*
batchnormgeneencode1_187315:	�*
batchnormgeneencode1_187317:	�*
batchnormgeneencode1_187319:	�+
protein_encoder_1_187322:	�9&
protein_encoder_1_187324:9(
gene_encoder_2_187327:	�}#
gene_encoder_2_187329:})
batchnormgeneencode2_187332:})
batchnormgeneencode2_187334:})
batchnormgeneencode2_187336:})
batchnormgeneencode2_187338:},
batchnormproteinencode1_187341:9,
batchnormproteinencode1_187343:9,
batchnormproteinencode1_187345:9,
batchnormproteinencode1_187347:9+
embeddingdimdense_187351:	�@&
embeddingdimdense_187353:@'
gene_decoder_1_187356:@}#
gene_decoder_1_187358:})
batchnormgenedecode1_187361:})
batchnormgenedecode1_187363:})
batchnormgenedecode1_187365:})
batchnormgenedecode1_187367:}*
protein_decoder_1_187370:@9&
protein_decoder_1_187372:9(
gene_decoder_2_187375:	}�$
gene_decoder_2_187377:	�,
batchnormproteindecode1_187380:9,
batchnormproteindecode1_187382:9,
batchnormproteindecode1_187384:9,
batchnormproteindecode1_187386:9*
batchnormgenedecode2_187389:	�*
batchnormgenedecode2_187391:	�*
batchnormgenedecode2_187393:	�*
batchnormgenedecode2_187395:	�.
protein_decoder_last_187398:	9�*
protein_decoder_last_187400:	�,
gene_decoder_last_187403:
��'
gene_decoder_last_187405:	�
identity

identity_1��,BatchNormGeneDecode1/StatefulPartitionedCall�,BatchNormGeneDecode2/StatefulPartitionedCall�,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�/BatchNormProteinDecode1/StatefulPartitionedCall�/BatchNormProteinEncode1/StatefulPartitionedCall�)EmbeddingDimDense/StatefulPartitionedCall�&gene_decoder_1/StatefulPartitionedCall�&gene_decoder_2/StatefulPartitionedCall�)gene_decoder_last/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�)protein_decoder_1/StatefulPartitionedCall�,protein_decoder_last/StatefulPartitionedCall�)protein_encoder_1/StatefulPartitionedCall�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallgene_input_layergene_encoder_1_187308gene_encoder_1_187310*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_186529�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_187313batchnormgeneencode1_187315batchnormgeneencode1_187317batchnormgeneencode1_187319*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_186041�
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallprotein_input_layerprotein_encoder_1_187322protein_encoder_1_187324*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������9*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_186555�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_187327gene_encoder_2_187329*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_186572�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_187332batchnormgeneencode2_187334batchnormgeneencode2_187336batchnormgeneencode2_187338*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_186123�
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_187341batchnormproteinencode1_187343batchnormproteinencode1_187345batchnormproteinencode1_187347*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������9*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_186205�
 ConcatenateLayer/PartitionedCallPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:08BatchNormProteinEncode1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_186603�
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_187351embeddingdimdense_187353*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_186616�
&gene_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0gene_decoder_1_187356gene_decoder_1_187358*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_decoder_1_layer_call_and_return_conditional_losses_186633�
,BatchNormGeneDecode1/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_1/StatefulPartitionedCall:output:0batchnormgenedecode1_187361batchnormgenedecode1_187363batchnormgenedecode1_187365batchnormgenedecode1_187367*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_186287�
)protein_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0protein_decoder_1_187370protein_decoder_1_187372*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������9*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_decoder_1_layer_call_and_return_conditional_losses_186659�
&gene_decoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode1/StatefulPartitionedCall:output:0gene_decoder_2_187375gene_decoder_2_187377*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_decoder_2_layer_call_and_return_conditional_losses_186676�
/BatchNormProteinDecode1/StatefulPartitionedCallStatefulPartitionedCall2protein_decoder_1/StatefulPartitionedCall:output:0batchnormproteindecode1_187380batchnormproteindecode1_187382batchnormproteindecode1_187384batchnormproteindecode1_187386*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������9*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_186451�
,BatchNormGeneDecode2/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_2/StatefulPartitionedCall:output:0batchnormgenedecode2_187389batchnormgenedecode2_187391batchnormgenedecode2_187393batchnormgenedecode2_187395*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_186369�
,protein_decoder_last/StatefulPartitionedCallStatefulPartitionedCall8BatchNormProteinDecode1/StatefulPartitionedCall:output:0protein_decoder_last_187398protein_decoder_last_187400*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_protein_decoder_last_layer_call_and_return_conditional_losses_186711�
)gene_decoder_last/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode2/StatefulPartitionedCall:output:0gene_decoder_last_187403gene_decoder_last_187405*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_gene_decoder_last_layer_call_and_return_conditional_losses_186728�
IdentityIdentity2gene_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������

Identity_1Identity5protein_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp-^BatchNormGeneDecode1/StatefulPartitionedCall-^BatchNormGeneDecode2/StatefulPartitionedCall-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinDecode1/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_decoder_1/StatefulPartitionedCall'^gene_decoder_2/StatefulPartitionedCall*^gene_decoder_last/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_decoder_1/StatefulPartitionedCall-^protein_decoder_last/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneDecode1/StatefulPartitionedCall,BatchNormGeneDecode1/StatefulPartitionedCall2\
,BatchNormGeneDecode2/StatefulPartitionedCall,BatchNormGeneDecode2/StatefulPartitionedCall2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2b
/BatchNormProteinDecode1/StatefulPartitionedCall/BatchNormProteinDecode1/StatefulPartitionedCall2b
/BatchNormProteinEncode1/StatefulPartitionedCall/BatchNormProteinEncode1/StatefulPartitionedCall2V
)EmbeddingDimDense/StatefulPartitionedCall)EmbeddingDimDense/StatefulPartitionedCall2P
&gene_decoder_1/StatefulPartitionedCall&gene_decoder_1/StatefulPartitionedCall2P
&gene_decoder_2/StatefulPartitionedCall&gene_decoder_2/StatefulPartitionedCall2V
)gene_decoder_last/StatefulPartitionedCall)gene_decoder_last/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall2V
)protein_decoder_1/StatefulPartitionedCall)protein_decoder_1/StatefulPartitionedCall2\
,protein_decoder_last/StatefulPartitionedCall,protein_decoder_last/StatefulPartitionedCall2V
)protein_encoder_1/StatefulPartitionedCall)protein_encoder_1/StatefulPartitionedCall:Z V
(
_output_shapes
:����������
*
_user_specified_nameGene_Input_Layer:]Y
(
_output_shapes
:����������
-
_user_specified_nameProtein_Input_Layer
�%
�
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_188851

inputs5
'assignmovingavg_readvariableop_resource:97
)assignmovingavg_1_readvariableop_resource:93
%batchnorm_mul_readvariableop_resource:9/
!batchnorm_readvariableop_resource:9
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:9*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:9�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������9l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:9*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:9*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:9*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:9*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:9x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:9�
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:9*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:9~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:9�
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:9P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:9~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:9*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:9c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������9h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:9v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:9*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:9r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������9b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������9�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������9: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������9
 
_user_specified_nameinputs
�

�
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_188338

inputs1
matmul_readvariableop_resource:	�}-
biasadd_readvariableop_resource:}
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�}*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:}*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������}Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������}w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
/__inference_gene_decoder_2_layer_call_fn_188660

inputs
unknown:	}�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_decoder_2_layer_call_and_return_conditional_losses_186676p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������}: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������}
 
_user_specified_nameinputs
�
x
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_188531
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :x
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*(
_output_shapes
:����������X
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������}:���������9:Q M
'
_output_shapes
:���������}
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������9
"
_user_specified_name
inputs/1
�

�
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_186616

inputs1
matmul_readvariableop_resource:	�@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������@Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������@w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
M__inference_protein_decoder_1_layer_call_and_return_conditional_losses_186659

inputs0
matmul_readvariableop_resource:@9-
biasadd_readvariableop_resource:9
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@9*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������9r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:9*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������9V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������9Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������9w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
M__inference_gene_decoder_last_layer_call_and_return_conditional_losses_186728

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
2__inference_EmbeddingDimDense_layer_call_fn_188540

inputs
unknown:	�@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_186616o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�	
&__inference_model_layer_call_fn_187614
inputs_0
inputs_1
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�9
	unknown_6:9
	unknown_7:	�}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:9

unknown_14:9

unknown_15:9

unknown_16:9

unknown_17:	�@

unknown_18:@

unknown_19:@}

unknown_20:}

unknown_21:}

unknown_22:}

unknown_23:}

unknown_24:}

unknown_25:@9

unknown_26:9

unknown_27:	}�

unknown_28:	�

unknown_29:9

unknown_30:9

unknown_31:9

unknown_32:9

unknown_33:	�

unknown_34:	�

unknown_35:	�

unknown_36:	�

unknown_37:	9�

unknown_38:	�

unknown_39:
��

unknown_40:	�
identity

identity_1��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38
unknown_39
unknown_40*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:����������:����������*L
_read_only_resource_inputs.
,*	
 !"#$%&'()*+*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_186736p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������r

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
�
8__inference_BatchNormProteinEncode1_layer_call_fn_188451

inputs
unknown:9
	unknown_0:9
	unknown_1:9
	unknown_2:9
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������9*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_186205o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������9`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������9: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������9
 
_user_specified_nameinputs
�
�
5__inference_protein_decoder_last_layer_call_fn_188880

inputs
unknown:	9�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_protein_decoder_last_layer_call_and_return_conditional_losses_186711p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������9: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������9
 
_user_specified_nameinputs
�

�
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_186555

inputs1
matmul_readvariableop_resource:	�9-
biasadd_readvariableop_resource:9
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�9*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������9r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:9*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������9V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������9Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������9w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_186287

inputs/
!batchnorm_readvariableop_resource:}3
%batchnorm_mul_readvariableop_resource:}1
#batchnorm_readvariableop_1_resource:}1
#batchnorm_readvariableop_2_resource:}
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:}P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:}~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������}z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:}*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:}z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:}*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:}r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������}�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������}: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������}
 
_user_specified_nameinputs
�
�
5__inference_BatchNormGeneEncode2_layer_call_fn_188371

inputs
unknown:}
	unknown_0:}
	unknown_1:}
	unknown_2:}
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_186123o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������}`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������}: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������}
 
_user_specified_nameinputs
�

�
P__inference_protein_decoder_last_layer_call_and_return_conditional_losses_186711

inputs1
matmul_readvariableop_resource:	9�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	9�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������9: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������9
 
_user_specified_nameinputs
�

�
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_186572

inputs1
matmul_readvariableop_resource:	�}-
biasadd_readvariableop_resource:}
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�}*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:}*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������}Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������}w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
/__inference_gene_decoder_1_layer_call_fn_188560

inputs
unknown:@}
	unknown_0:}
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_decoder_1_layer_call_and_return_conditional_losses_186633o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������}`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
2__inference_gene_decoder_last_layer_call_fn_188860

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_gene_decoder_last_layer_call_and_return_conditional_losses_186728p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�e
�
A__inference_model_layer_call_and_return_conditional_losses_186736

inputs
inputs_1)
gene_encoder_1_186530:
��$
gene_encoder_1_186532:	�*
batchnormgeneencode1_186535:	�*
batchnormgeneencode1_186537:	�*
batchnormgeneencode1_186539:	�*
batchnormgeneencode1_186541:	�+
protein_encoder_1_186556:	�9&
protein_encoder_1_186558:9(
gene_encoder_2_186573:	�}#
gene_encoder_2_186575:})
batchnormgeneencode2_186578:})
batchnormgeneencode2_186580:})
batchnormgeneencode2_186582:})
batchnormgeneencode2_186584:},
batchnormproteinencode1_186587:9,
batchnormproteinencode1_186589:9,
batchnormproteinencode1_186591:9,
batchnormproteinencode1_186593:9+
embeddingdimdense_186617:	�@&
embeddingdimdense_186619:@'
gene_decoder_1_186634:@}#
gene_decoder_1_186636:})
batchnormgenedecode1_186639:})
batchnormgenedecode1_186641:})
batchnormgenedecode1_186643:})
batchnormgenedecode1_186645:}*
protein_decoder_1_186660:@9&
protein_decoder_1_186662:9(
gene_decoder_2_186677:	}�$
gene_decoder_2_186679:	�,
batchnormproteindecode1_186682:9,
batchnormproteindecode1_186684:9,
batchnormproteindecode1_186686:9,
batchnormproteindecode1_186688:9*
batchnormgenedecode2_186691:	�*
batchnormgenedecode2_186693:	�*
batchnormgenedecode2_186695:	�*
batchnormgenedecode2_186697:	�.
protein_decoder_last_186712:	9�*
protein_decoder_last_186714:	�,
gene_decoder_last_186729:
��'
gene_decoder_last_186731:	�
identity

identity_1��,BatchNormGeneDecode1/StatefulPartitionedCall�,BatchNormGeneDecode2/StatefulPartitionedCall�,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�/BatchNormProteinDecode1/StatefulPartitionedCall�/BatchNormProteinEncode1/StatefulPartitionedCall�)EmbeddingDimDense/StatefulPartitionedCall�&gene_decoder_1/StatefulPartitionedCall�&gene_decoder_2/StatefulPartitionedCall�)gene_decoder_last/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�)protein_decoder_1/StatefulPartitionedCall�,protein_decoder_last/StatefulPartitionedCall�)protein_encoder_1/StatefulPartitionedCall�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputsgene_encoder_1_186530gene_encoder_1_186532*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_186529�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_186535batchnormgeneencode1_186537batchnormgeneencode1_186539batchnormgeneencode1_186541*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_186041�
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1protein_encoder_1_186556protein_encoder_1_186558*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������9*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_186555�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_186573gene_encoder_2_186575*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_186572�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_186578batchnormgeneencode2_186580batchnormgeneencode2_186582batchnormgeneencode2_186584*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_186123�
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_186587batchnormproteinencode1_186589batchnormproteinencode1_186591batchnormproteinencode1_186593*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������9*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_186205�
 ConcatenateLayer/PartitionedCallPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:08BatchNormProteinEncode1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_186603�
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_186617embeddingdimdense_186619*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_186616�
&gene_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0gene_decoder_1_186634gene_decoder_1_186636*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_decoder_1_layer_call_and_return_conditional_losses_186633�
,BatchNormGeneDecode1/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_1/StatefulPartitionedCall:output:0batchnormgenedecode1_186639batchnormgenedecode1_186641batchnormgenedecode1_186643batchnormgenedecode1_186645*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_186287�
)protein_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0protein_decoder_1_186660protein_decoder_1_186662*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������9*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_decoder_1_layer_call_and_return_conditional_losses_186659�
&gene_decoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode1/StatefulPartitionedCall:output:0gene_decoder_2_186677gene_decoder_2_186679*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_decoder_2_layer_call_and_return_conditional_losses_186676�
/BatchNormProteinDecode1/StatefulPartitionedCallStatefulPartitionedCall2protein_decoder_1/StatefulPartitionedCall:output:0batchnormproteindecode1_186682batchnormproteindecode1_186684batchnormproteindecode1_186686batchnormproteindecode1_186688*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������9*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_186451�
,BatchNormGeneDecode2/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_2/StatefulPartitionedCall:output:0batchnormgenedecode2_186691batchnormgenedecode2_186693batchnormgenedecode2_186695batchnormgenedecode2_186697*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_186369�
,protein_decoder_last/StatefulPartitionedCallStatefulPartitionedCall8BatchNormProteinDecode1/StatefulPartitionedCall:output:0protein_decoder_last_186712protein_decoder_last_186714*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_protein_decoder_last_layer_call_and_return_conditional_losses_186711�
)gene_decoder_last/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode2/StatefulPartitionedCall:output:0gene_decoder_last_186729gene_decoder_last_186731*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_gene_decoder_last_layer_call_and_return_conditional_losses_186728�
IdentityIdentity2gene_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������

Identity_1Identity5protein_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp-^BatchNormGeneDecode1/StatefulPartitionedCall-^BatchNormGeneDecode2/StatefulPartitionedCall-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinDecode1/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_decoder_1/StatefulPartitionedCall'^gene_decoder_2/StatefulPartitionedCall*^gene_decoder_last/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_decoder_1/StatefulPartitionedCall-^protein_decoder_last/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|:����������:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneDecode1/StatefulPartitionedCall,BatchNormGeneDecode1/StatefulPartitionedCall2\
,BatchNormGeneDecode2/StatefulPartitionedCall,BatchNormGeneDecode2/StatefulPartitionedCall2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2b
/BatchNormProteinDecode1/StatefulPartitionedCall/BatchNormProteinDecode1/StatefulPartitionedCall2b
/BatchNormProteinEncode1/StatefulPartitionedCall/BatchNormProteinEncode1/StatefulPartitionedCall2V
)EmbeddingDimDense/StatefulPartitionedCall)EmbeddingDimDense/StatefulPartitionedCall2P
&gene_decoder_1/StatefulPartitionedCall&gene_decoder_1/StatefulPartitionedCall2P
&gene_decoder_2/StatefulPartitionedCall&gene_decoder_2/StatefulPartitionedCall2V
)gene_decoder_last/StatefulPartitionedCall)gene_decoder_last/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall2V
)protein_decoder_1/StatefulPartitionedCall)protein_decoder_1/StatefulPartitionedCall2\
,protein_decoder_last/StatefulPartitionedCall,protein_decoder_last/StatefulPartitionedCall2V
)protein_encoder_1/StatefulPartitionedCall)protein_encoder_1/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_186369

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�%
�
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_186088

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvgAssignSubVariableOp'assignmovingavg_readvariableop_resourceAssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0\
AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_186041

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
5__inference_BatchNormGeneEncode2_layer_call_fn_188384

inputs
unknown:}
	unknown_0:}
	unknown_1:}
	unknown_2:}
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_186170o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������}`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������}: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������}
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
N
Gene_Input_Layer:
"serving_default_Gene_Input_Layer:0����������
T
Protein_Input_Layer=
%serving_default_Protein_Input_Layer:0����������F
gene_decoder_last1
StatefulPartitionedCall:0����������I
protein_decoder_last1
StatefulPartitionedCall:1����������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer-8

layer_with_weights-6

layer-9
layer_with_weights-7
layer-10
layer_with_weights-8
layer-11
layer_with_weights-9
layer-12
layer_with_weights-10
layer-13
layer_with_weights-11
layer-14
layer_with_weights-12
layer-15
layer_with_weights-13
layer-16
layer_with_weights-14
layer-17
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
�
$axis
	%gamma
&beta
'moving_mean
(moving_variance
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
�

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_layer
�

7kernel
8bias
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_layer
�
?axis
	@gamma
Abeta
Bmoving_mean
Cmoving_variance
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses"
_tf_keras_layer
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"
_tf_keras_layer
�

[kernel
\bias
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses"
_tf_keras_layer
�

ckernel
dbias
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses"
_tf_keras_layer
�
kaxis
	lgamma
mbeta
nmoving_mean
omoving_variance
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
�

vkernel
wbias
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses"
_tf_keras_layer
�

~kernel
bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�kernel
	�bias
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	�iter
�beta_1
�beta_2

�decay
�learning_ratem�m�%m�&m�/m�0m�7m�8m�@m�Am�Km�Lm�[m�\m�cm�dm�lm�mm�vm�wm�~m�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�v�v�%v�&v�/v�0v�7v�8v�@v�Av�Kv�Lv�[v�\v�cv�dv�lv�mv�vv�wv�~v�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�"
	optimizer
�
0
1
%2
&3
'4
(5
/6
07
78
89
@10
A11
B12
C13
K14
L15
M16
N17
[18
\19
c20
d21
l22
m23
n24
o25
v26
w27
~28
29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41"
trackable_list_wrapper
�
0
1
%2
&3
/4
05
76
87
@8
A9
K10
L11
[12
\13
c14
d15
l16
m17
v18
w19
~20
21
�22
�23
�24
�25
�26
�27
�28
�29"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
&__inference_model_layer_call_fn_186825
&__inference_model_layer_call_fn_187614
&__inference_model_layer_call_fn_187706
&__inference_model_layer_call_fn_187304�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
A__inference_model_layer_call_and_return_conditional_losses_187873
A__inference_model_layer_call_and_return_conditional_losses_188124
A__inference_model_layer_call_and_return_conditional_losses_187410
A__inference_model_layer_call_and_return_conditional_losses_187516�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
!__inference__wrapped_model_186017Gene_Input_LayerProtein_Input_Layer"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
-
�serving_default"
signature_map
):'
��2gene_encoder_1/kernel
": �2gene_encoder_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�2�
/__inference_gene_encoder_1_layer_call_fn_188227�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_188238�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
):'�2BatchNormGeneEncode1/gamma
(:&�2BatchNormGeneEncode1/beta
1:/� (2 BatchNormGeneEncode1/moving_mean
5:3� (2$BatchNormGeneEncode1/moving_variance
<
%0
&1
'2
(3"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�2�
5__inference_BatchNormGeneEncode1_layer_call_fn_188251
5__inference_BatchNormGeneEncode1_layer_call_fn_188264�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_188284
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_188318�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
(:&	�}2gene_encoder_2/kernel
!:}2gene_encoder_2/bias
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
�2�
/__inference_gene_encoder_2_layer_call_fn_188327�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_188338�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
+:)	�92protein_encoder_1/kernel
$:"92protein_encoder_1/bias
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�2�
2__inference_protein_encoder_1_layer_call_fn_188347�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_188358�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
(:&}2BatchNormGeneEncode2/gamma
':%}2BatchNormGeneEncode2/beta
0:.} (2 BatchNormGeneEncode2/moving_mean
4:2} (2$BatchNormGeneEncode2/moving_variance
<
@0
A1
B2
C3"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
�2�
5__inference_BatchNormGeneEncode2_layer_call_fn_188371
5__inference_BatchNormGeneEncode2_layer_call_fn_188384�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_188404
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_188438�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
+:)92BatchNormProteinEncode1/gamma
*:(92BatchNormProteinEncode1/beta
3:19 (2#BatchNormProteinEncode1/moving_mean
7:59 (2'BatchNormProteinEncode1/moving_variance
<
K0
L1
M2
N3"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
O	variables
Ptrainable_variables
Qregularization_losses
S__call__
*T&call_and_return_all_conditional_losses
&T"call_and_return_conditional_losses"
_generic_user_object
�2�
8__inference_BatchNormProteinEncode1_layer_call_fn_188451
8__inference_BatchNormProteinEncode1_layer_call_fn_188464�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_188484
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_188518�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�2�
1__inference_ConcatenateLayer_layer_call_fn_188524�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_188531�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
+:)	�@2EmbeddingDimDense/kernel
$:"@2EmbeddingDimDense/bias
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
�2�
2__inference_EmbeddingDimDense_layer_call_fn_188540�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_188551�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
':%@}2gene_decoder_1/kernel
!:}2gene_decoder_1/bias
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
�2�
/__inference_gene_decoder_1_layer_call_fn_188560�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_gene_decoder_1_layer_call_and_return_conditional_losses_188571�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
(:&}2BatchNormGeneDecode1/gamma
':%}2BatchNormGeneDecode1/beta
0:.} (2 BatchNormGeneDecode1/moving_mean
4:2} (2$BatchNormGeneDecode1/moving_variance
<
l0
m1
n2
o3"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
�2�
5__inference_BatchNormGeneDecode1_layer_call_fn_188584
5__inference_BatchNormGeneDecode1_layer_call_fn_188597�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_188617
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_188651�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
(:&	}�2gene_decoder_2/kernel
": �2gene_decoder_2/bias
.
v0
w1"
trackable_list_wrapper
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
�2�
/__inference_gene_decoder_2_layer_call_fn_188660�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_gene_decoder_2_layer_call_and_return_conditional_losses_188671�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
*:(@92protein_decoder_1/kernel
$:"92protein_decoder_1/bias
.
~0
1"
trackable_list_wrapper
.
~0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
2__inference_protein_decoder_1_layer_call_fn_188680�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_protein_decoder_1_layer_call_and_return_conditional_losses_188691�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
):'�2BatchNormGeneDecode2/gamma
(:&�2BatchNormGeneDecode2/beta
1:/� (2 BatchNormGeneDecode2/moving_mean
5:3� (2$BatchNormGeneDecode2/moving_variance
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
5__inference_BatchNormGeneDecode2_layer_call_fn_188704
5__inference_BatchNormGeneDecode2_layer_call_fn_188717�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_188737
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_188771�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
 "
trackable_list_wrapper
+:)92BatchNormProteinDecode1/gamma
*:(92BatchNormProteinDecode1/beta
3:19 (2#BatchNormProteinDecode1/moving_mean
7:59 (2'BatchNormProteinDecode1/moving_variance
@
�0
�1
�2
�3"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
8__inference_BatchNormProteinDecode1_layer_call_fn_188784
8__inference_BatchNormProteinDecode1_layer_call_fn_188797�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_188817
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_188851�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
,:*
��2gene_decoder_last/kernel
%:#�2gene_decoder_last/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
2__inference_gene_decoder_last_layer_call_fn_188860�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
M__inference_gene_decoder_last_layer_call_and_return_conditional_losses_188871�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.:,	9�2protein_decoder_last/kernel
(:&�2protein_decoder_last/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
5__inference_protein_decoder_last_layer_call_fn_188880�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
P__inference_protein_decoder_last_layer_call_and_return_conditional_losses_188891�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
z
'0
(1
B2
C3
M4
N5
n6
o7
�8
�9
�10
�11"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
8
�0
�1
�2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_signature_wrapper_188218Gene_Input_LayerProtein_Input_Layer"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
n0
o1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
R

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
R

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
.:,
��2Adam/gene_encoder_1/kernel/m
':%�2Adam/gene_encoder_1/bias/m
.:,�2!Adam/BatchNormGeneEncode1/gamma/m
-:+�2 Adam/BatchNormGeneEncode1/beta/m
-:+	�}2Adam/gene_encoder_2/kernel/m
&:$}2Adam/gene_encoder_2/bias/m
0:.	�92Adam/protein_encoder_1/kernel/m
):'92Adam/protein_encoder_1/bias/m
-:+}2!Adam/BatchNormGeneEncode2/gamma/m
,:*}2 Adam/BatchNormGeneEncode2/beta/m
0:.92$Adam/BatchNormProteinEncode1/gamma/m
/:-92#Adam/BatchNormProteinEncode1/beta/m
0:.	�@2Adam/EmbeddingDimDense/kernel/m
):'@2Adam/EmbeddingDimDense/bias/m
,:*@}2Adam/gene_decoder_1/kernel/m
&:$}2Adam/gene_decoder_1/bias/m
-:+}2!Adam/BatchNormGeneDecode1/gamma/m
,:*}2 Adam/BatchNormGeneDecode1/beta/m
-:+	}�2Adam/gene_decoder_2/kernel/m
':%�2Adam/gene_decoder_2/bias/m
/:-@92Adam/protein_decoder_1/kernel/m
):'92Adam/protein_decoder_1/bias/m
.:,�2!Adam/BatchNormGeneDecode2/gamma/m
-:+�2 Adam/BatchNormGeneDecode2/beta/m
0:.92$Adam/BatchNormProteinDecode1/gamma/m
/:-92#Adam/BatchNormProteinDecode1/beta/m
1:/
��2Adam/gene_decoder_last/kernel/m
*:(�2Adam/gene_decoder_last/bias/m
3:1	9�2"Adam/protein_decoder_last/kernel/m
-:+�2 Adam/protein_decoder_last/bias/m
.:,
��2Adam/gene_encoder_1/kernel/v
':%�2Adam/gene_encoder_1/bias/v
.:,�2!Adam/BatchNormGeneEncode1/gamma/v
-:+�2 Adam/BatchNormGeneEncode1/beta/v
-:+	�}2Adam/gene_encoder_2/kernel/v
&:$}2Adam/gene_encoder_2/bias/v
0:.	�92Adam/protein_encoder_1/kernel/v
):'92Adam/protein_encoder_1/bias/v
-:+}2!Adam/BatchNormGeneEncode2/gamma/v
,:*}2 Adam/BatchNormGeneEncode2/beta/v
0:.92$Adam/BatchNormProteinEncode1/gamma/v
/:-92#Adam/BatchNormProteinEncode1/beta/v
0:.	�@2Adam/EmbeddingDimDense/kernel/v
):'@2Adam/EmbeddingDimDense/bias/v
,:*@}2Adam/gene_decoder_1/kernel/v
&:$}2Adam/gene_decoder_1/bias/v
-:+}2!Adam/BatchNormGeneDecode1/gamma/v
,:*}2 Adam/BatchNormGeneDecode1/beta/v
-:+	}�2Adam/gene_decoder_2/kernel/v
':%�2Adam/gene_decoder_2/bias/v
/:-@92Adam/protein_decoder_1/kernel/v
):'92Adam/protein_decoder_1/bias/v
.:,�2!Adam/BatchNormGeneDecode2/gamma/v
-:+�2 Adam/BatchNormGeneDecode2/beta/v
0:.92$Adam/BatchNormProteinDecode1/gamma/v
/:-92#Adam/BatchNormProteinDecode1/beta/v
1:/
��2Adam/gene_decoder_last/kernel/v
*:(�2Adam/gene_decoder_last/bias/v
3:1	9�2"Adam/protein_decoder_last/kernel/v
-:+�2 Adam/protein_decoder_last/bias/v�
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_188617bolnm3�0
)�&
 �
inputs���������}
p 
� "%�"
�
0���������}
� �
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_188651bnolm3�0
)�&
 �
inputs���������}
p
� "%�"
�
0���������}
� �
5__inference_BatchNormGeneDecode1_layer_call_fn_188584Uolnm3�0
)�&
 �
inputs���������}
p 
� "����������}�
5__inference_BatchNormGeneDecode1_layer_call_fn_188597Unolm3�0
)�&
 �
inputs���������}
p
� "����������}�
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_188737h����4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_188771h����4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
5__inference_BatchNormGeneDecode2_layer_call_fn_188704[����4�1
*�'
!�
inputs����������
p 
� "������������
5__inference_BatchNormGeneDecode2_layer_call_fn_188717[����4�1
*�'
!�
inputs����������
p
� "������������
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_188284d(%'&4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_188318d'(%&4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
5__inference_BatchNormGeneEncode1_layer_call_fn_188251W(%'&4�1
*�'
!�
inputs����������
p 
� "������������
5__inference_BatchNormGeneEncode1_layer_call_fn_188264W'(%&4�1
*�'
!�
inputs����������
p
� "������������
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_188404bC@BA3�0
)�&
 �
inputs���������}
p 
� "%�"
�
0���������}
� �
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_188438bBC@A3�0
)�&
 �
inputs���������}
p
� "%�"
�
0���������}
� �
5__inference_BatchNormGeneEncode2_layer_call_fn_188371UC@BA3�0
)�&
 �
inputs���������}
p 
� "����������}�
5__inference_BatchNormGeneEncode2_layer_call_fn_188384UBC@A3�0
)�&
 �
inputs���������}
p
� "����������}�
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_188817f����3�0
)�&
 �
inputs���������9
p 
� "%�"
�
0���������9
� �
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_188851f����3�0
)�&
 �
inputs���������9
p
� "%�"
�
0���������9
� �
8__inference_BatchNormProteinDecode1_layer_call_fn_188784Y����3�0
)�&
 �
inputs���������9
p 
� "����������9�
8__inference_BatchNormProteinDecode1_layer_call_fn_188797Y����3�0
)�&
 �
inputs���������9
p
� "����������9�
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_188484bNKML3�0
)�&
 �
inputs���������9
p 
� "%�"
�
0���������9
� �
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_188518bMNKL3�0
)�&
 �
inputs���������9
p
� "%�"
�
0���������9
� �
8__inference_BatchNormProteinEncode1_layer_call_fn_188451UNKML3�0
)�&
 �
inputs���������9
p 
� "����������9�
8__inference_BatchNormProteinEncode1_layer_call_fn_188464UMNKL3�0
)�&
 �
inputs���������9
p
� "����������9�
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_188531�Z�W
P�M
K�H
"�
inputs/0���������}
"�
inputs/1���������9
� "&�#
�
0����������
� �
1__inference_ConcatenateLayer_layer_call_fn_188524wZ�W
P�M
K�H
"�
inputs/0���������}
"�
inputs/1���������9
� "������������
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_188551][\0�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� �
2__inference_EmbeddingDimDense_layer_call_fn_188540P[\0�-
&�#
!�
inputs����������
� "����������@�
!__inference__wrapped_model_186017�6(%'&78/0C@BANKML[\cdolnm~vw������������o�l
e�b
`�]
+�(
Gene_Input_Layer����������
.�+
Protein_Input_Layer����������
� "���
A
gene_decoder_last,�)
gene_decoder_last����������
G
protein_decoder_last/�,
protein_decoder_last�����������
J__inference_gene_decoder_1_layer_call_and_return_conditional_losses_188571\cd/�,
%�"
 �
inputs���������@
� "%�"
�
0���������}
� �
/__inference_gene_decoder_1_layer_call_fn_188560Ocd/�,
%�"
 �
inputs���������@
� "����������}�
J__inference_gene_decoder_2_layer_call_and_return_conditional_losses_188671]vw/�,
%�"
 �
inputs���������}
� "&�#
�
0����������
� �
/__inference_gene_decoder_2_layer_call_fn_188660Pvw/�,
%�"
 �
inputs���������}
� "������������
M__inference_gene_decoder_last_layer_call_and_return_conditional_losses_188871`��0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
2__inference_gene_decoder_last_layer_call_fn_188860S��0�-
&�#
!�
inputs����������
� "������������
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_188238^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
/__inference_gene_encoder_1_layer_call_fn_188227Q0�-
&�#
!�
inputs����������
� "������������
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_188338]/00�-
&�#
!�
inputs����������
� "%�"
�
0���������}
� �
/__inference_gene_encoder_2_layer_call_fn_188327P/00�-
&�#
!�
inputs����������
� "����������}�
A__inference_model_layer_call_and_return_conditional_losses_187410�6(%'&78/0C@BANKML[\cdolnm~vw������������w�t
m�j
`�]
+�(
Gene_Input_Layer����������
.�+
Protein_Input_Layer����������
p 

 
� "M�J
C�@
�
0/0����������
�
0/1����������
� �
A__inference_model_layer_call_and_return_conditional_losses_187516�6'(%&78/0BC@AMNKL[\cdnolm~vw������������w�t
m�j
`�]
+�(
Gene_Input_Layer����������
.�+
Protein_Input_Layer����������
p

 
� "M�J
C�@
�
0/0����������
�
0/1����������
� �
A__inference_model_layer_call_and_return_conditional_losses_187873�6(%'&78/0C@BANKML[\cdolnm~vw������������d�a
Z�W
M�J
#� 
inputs/0����������
#� 
inputs/1����������
p 

 
� "M�J
C�@
�
0/0����������
�
0/1����������
� �
A__inference_model_layer_call_and_return_conditional_losses_188124�6'(%&78/0BC@AMNKL[\cdnolm~vw������������d�a
Z�W
M�J
#� 
inputs/0����������
#� 
inputs/1����������
p

 
� "M�J
C�@
�
0/0����������
�
0/1����������
� �
&__inference_model_layer_call_fn_186825�6(%'&78/0C@BANKML[\cdolnm~vw������������w�t
m�j
`�]
+�(
Gene_Input_Layer����������
.�+
Protein_Input_Layer����������
p 

 
� "?�<
�
0����������
�
1�����������
&__inference_model_layer_call_fn_187304�6'(%&78/0BC@AMNKL[\cdnolm~vw������������w�t
m�j
`�]
+�(
Gene_Input_Layer����������
.�+
Protein_Input_Layer����������
p

 
� "?�<
�
0����������
�
1�����������
&__inference_model_layer_call_fn_187614�6(%'&78/0C@BANKML[\cdolnm~vw������������d�a
Z�W
M�J
#� 
inputs/0����������
#� 
inputs/1����������
p 

 
� "?�<
�
0����������
�
1�����������
&__inference_model_layer_call_fn_187706�6'(%&78/0BC@AMNKL[\cdnolm~vw������������d�a
Z�W
M�J
#� 
inputs/0����������
#� 
inputs/1����������
p

 
� "?�<
�
0����������
�
1�����������
M__inference_protein_decoder_1_layer_call_and_return_conditional_losses_188691\~/�,
%�"
 �
inputs���������@
� "%�"
�
0���������9
� �
2__inference_protein_decoder_1_layer_call_fn_188680O~/�,
%�"
 �
inputs���������@
� "����������9�
P__inference_protein_decoder_last_layer_call_and_return_conditional_losses_188891_��/�,
%�"
 �
inputs���������9
� "&�#
�
0����������
� �
5__inference_protein_decoder_last_layer_call_fn_188880R��/�,
%�"
 �
inputs���������9
� "������������
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_188358]780�-
&�#
!�
inputs����������
� "%�"
�
0���������9
� �
2__inference_protein_encoder_1_layer_call_fn_188347P780�-
&�#
!�
inputs����������
� "����������9�
$__inference_signature_wrapper_188218�6(%'&78/0C@BANKML[\cdolnm~vw���������������
� 
���
?
Gene_Input_Layer+�(
Gene_Input_Layer����������
E
Protein_Input_Layer.�+
Protein_Input_Layer����������"���
A
gene_decoder_last,�)
gene_decoder_last����������
G
protein_decoder_last/�,
protein_decoder_last����������