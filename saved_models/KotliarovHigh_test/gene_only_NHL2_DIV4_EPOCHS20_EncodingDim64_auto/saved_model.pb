��
��
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
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
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
 �"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8��
�
Adam/gene_decoder_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameAdam/gene_decoder_3/bias/v
�
.Adam/gene_decoder_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_3/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/gene_decoder_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*-
shared_nameAdam/gene_decoder_3/kernel/v
�
0Adam/gene_decoder_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_3/kernel/v* 
_output_shapes
:
��*
dtype0
�
Adam/BatchNormGene2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameAdam/BatchNormGene2/beta/v
�
.Adam/BatchNormGene2/beta/v/Read/ReadVariableOpReadVariableOpAdam/BatchNormGene2/beta/v*
_output_shapes	
:�*
dtype0
�
Adam/BatchNormGene2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_nameAdam/BatchNormGene2/gamma/v
�
/Adam/BatchNormGene2/gamma/v/Read/ReadVariableOpReadVariableOpAdam/BatchNormGene2/gamma/v*
_output_shapes	
:�*
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
Adam/EmbeddingDimGene3/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/EmbeddingDimGene3/beta/v
�
1Adam/EmbeddingDimGene3/beta/v/Read/ReadVariableOpReadVariableOpAdam/EmbeddingDimGene3/beta/v*
_output_shapes
:@*
dtype0
�
Adam/EmbeddingDimGene3/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/EmbeddingDimGene3/gamma/v
�
2Adam/EmbeddingDimGene3/gamma/v/Read/ReadVariableOpReadVariableOpAdam/EmbeddingDimGene3/gamma/v*
_output_shapes
:@*
dtype0
�
Adam/EmbeddingDimGene/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameAdam/EmbeddingDimGene/bias/v
�
0Adam/EmbeddingDimGene/bias/v/Read/ReadVariableOpReadVariableOpAdam/EmbeddingDimGene/bias/v*
_output_shapes
:@*
dtype0
�
Adam/EmbeddingDimGene/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:}@*/
shared_name Adam/EmbeddingDimGene/kernel/v
�
2Adam/EmbeddingDimGene/kernel/v/Read/ReadVariableOpReadVariableOpAdam/EmbeddingDimGene/kernel/v*
_output_shapes

:}@*
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
Adam/gene_decoder_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameAdam/gene_decoder_3/bias/m
�
.Adam/gene_decoder_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_3/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/gene_decoder_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*-
shared_nameAdam/gene_decoder_3/kernel/m
�
0Adam/gene_decoder_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_3/kernel/m* 
_output_shapes
:
��*
dtype0
�
Adam/BatchNormGene2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameAdam/BatchNormGene2/beta/m
�
.Adam/BatchNormGene2/beta/m/Read/ReadVariableOpReadVariableOpAdam/BatchNormGene2/beta/m*
_output_shapes	
:�*
dtype0
�
Adam/BatchNormGene2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*,
shared_nameAdam/BatchNormGene2/gamma/m
�
/Adam/BatchNormGene2/gamma/m/Read/ReadVariableOpReadVariableOpAdam/BatchNormGene2/gamma/m*
_output_shapes	
:�*
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
Adam/EmbeddingDimGene3/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameAdam/EmbeddingDimGene3/beta/m
�
1Adam/EmbeddingDimGene3/beta/m/Read/ReadVariableOpReadVariableOpAdam/EmbeddingDimGene3/beta/m*
_output_shapes
:@*
dtype0
�
Adam/EmbeddingDimGene3/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*/
shared_name Adam/EmbeddingDimGene3/gamma/m
�
2Adam/EmbeddingDimGene3/gamma/m/Read/ReadVariableOpReadVariableOpAdam/EmbeddingDimGene3/gamma/m*
_output_shapes
:@*
dtype0
�
Adam/EmbeddingDimGene/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameAdam/EmbeddingDimGene/bias/m
�
0Adam/EmbeddingDimGene/bias/m/Read/ReadVariableOpReadVariableOpAdam/EmbeddingDimGene/bias/m*
_output_shapes
:@*
dtype0
�
Adam/EmbeddingDimGene/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:}@*/
shared_name Adam/EmbeddingDimGene/kernel/m
�
2Adam/EmbeddingDimGene/kernel/m/Read/ReadVariableOpReadVariableOpAdam/EmbeddingDimGene/kernel/m*
_output_shapes

:}@*
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

gene_decoder_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_namegene_decoder_3/bias
x
'gene_decoder_3/bias/Read/ReadVariableOpReadVariableOpgene_decoder_3/bias*
_output_shapes	
:�*
dtype0
�
gene_decoder_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_namegene_decoder_3/kernel
�
)gene_decoder_3/kernel/Read/ReadVariableOpReadVariableOpgene_decoder_3/kernel* 
_output_shapes
:
��*
dtype0
�
BatchNormGene2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*/
shared_name BatchNormGene2/moving_variance
�
2BatchNormGene2/moving_variance/Read/ReadVariableOpReadVariableOpBatchNormGene2/moving_variance*
_output_shapes	
:�*
dtype0
�
BatchNormGene2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameBatchNormGene2/moving_mean
�
.BatchNormGene2/moving_mean/Read/ReadVariableOpReadVariableOpBatchNormGene2/moving_mean*
_output_shapes	
:�*
dtype0

BatchNormGene2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameBatchNormGene2/beta
x
'BatchNormGene2/beta/Read/ReadVariableOpReadVariableOpBatchNormGene2/beta*
_output_shapes	
:�*
dtype0
�
BatchNormGene2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*%
shared_nameBatchNormGene2/gamma
z
(BatchNormGene2/gamma/Read/ReadVariableOpReadVariableOpBatchNormGene2/gamma*
_output_shapes	
:�*
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
�
!EmbeddingDimGene3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!EmbeddingDimGene3/moving_variance
�
5EmbeddingDimGene3/moving_variance/Read/ReadVariableOpReadVariableOp!EmbeddingDimGene3/moving_variance*
_output_shapes
:@*
dtype0
�
EmbeddingDimGene3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*.
shared_nameEmbeddingDimGene3/moving_mean
�
1EmbeddingDimGene3/moving_mean/Read/ReadVariableOpReadVariableOpEmbeddingDimGene3/moving_mean*
_output_shapes
:@*
dtype0
�
EmbeddingDimGene3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameEmbeddingDimGene3/beta
}
*EmbeddingDimGene3/beta/Read/ReadVariableOpReadVariableOpEmbeddingDimGene3/beta*
_output_shapes
:@*
dtype0
�
EmbeddingDimGene3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameEmbeddingDimGene3/gamma

+EmbeddingDimGene3/gamma/Read/ReadVariableOpReadVariableOpEmbeddingDimGene3/gamma*
_output_shapes
:@*
dtype0
�
EmbeddingDimGene/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameEmbeddingDimGene/bias
{
)EmbeddingDimGene/bias/Read/ReadVariableOpReadVariableOpEmbeddingDimGene/bias*
_output_shapes
:@*
dtype0
�
EmbeddingDimGene/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:}@*(
shared_nameEmbeddingDimGene/kernel
�
+EmbeddingDimGene/kernel/Read/ReadVariableOpReadVariableOpEmbeddingDimGene/kernel*
_output_shapes

:}@*
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
�
 serving_default_gene_input_layerPlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�	
StatefulPartitionedCallStatefulPartitionedCall serving_default_gene_input_layergene_encoder_1/kernelgene_encoder_1/bias$BatchNormGeneEncode1/moving_varianceBatchNormGeneEncode1/gamma BatchNormGeneEncode1/moving_meanBatchNormGeneEncode1/betagene_encoder_2/kernelgene_encoder_2/bias$BatchNormGeneEncode2/moving_varianceBatchNormGeneEncode2/gamma BatchNormGeneEncode2/moving_meanBatchNormGeneEncode2/betaEmbeddingDimGene/kernelEmbeddingDimGene/bias!EmbeddingDimGene3/moving_varianceEmbeddingDimGene3/gammaEmbeddingDimGene3/moving_meanEmbeddingDimGene3/betagene_decoder_1/kernelgene_decoder_1/bias$BatchNormGeneDecode1/moving_varianceBatchNormGeneDecode1/gamma BatchNormGeneDecode1/moving_meanBatchNormGeneDecode1/betagene_decoder_2/kernelgene_decoder_2/biasBatchNormGene2/moving_varianceBatchNormGene2/gammaBatchNormGene2/moving_meanBatchNormGene2/betagene_decoder_3/kernelgene_decoder_3/bias*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_2020971

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer_with_weights-10
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$axis
	%gamma
&beta
'moving_mean
(moving_variance*
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias*
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
7axis
	8gamma
9beta
:moving_mean
;moving_variance*
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

Bkernel
Cbias*
�
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance*
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

Ukernel
Vbias*
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]axis
	^gamma
_beta
`moving_mean
amoving_variance*
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias*
�
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses
paxis
	qgamma
rbeta
smoving_mean
tmoving_variance*
�
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses

{kernel
|bias*
�
0
1
%2
&3
'4
(5
/6
07
88
99
:10
;11
B12
C13
K14
L15
M16
N17
U18
V19
^20
_21
`22
a23
h24
i25
q26
r27
s28
t29
{30
|31*
�
0
1
%2
&3
/4
05
86
97
B8
C9
K10
L11
U12
V13
^14
_15
h16
i17
q18
r19
{20
|21*
* 
�
}non_trainable_variables

~layers
metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
�trace_0
�trace_1
�trace_2
�trace_3* 
:
�trace_0
�trace_1
�trace_2
�trace_3* 
* 
�
	�iter
�beta_1
�beta_2

�decay
�learning_ratem�m�%m�&m�/m�0m�8m�9m�Bm�Cm�Km�Lm�Um�Vm�^m�_m�hm�im�qm�rm�{m�|m�v�v�%v�&v�/v�0v�8v�9v�Bv�Cv�Kv�Lv�Uv�Vv�^v�_v�hv�iv�qv�rv�{v�|v�*

�serving_default* 
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
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
e_
VARIABLE_VALUEgene_encoder_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEgene_encoder_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
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
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ic
VARIABLE_VALUEBatchNormGeneEncode1/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEBatchNormGeneEncode1/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE BatchNormGeneEncode1/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE$BatchNormGeneEncode1/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
e_
VARIABLE_VALUEgene_encoder_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEgene_encoder_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
80
91
:2
;3*

80
91*
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

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ic
VARIABLE_VALUEBatchNormGeneEncode2/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEBatchNormGeneEncode2/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE BatchNormGeneEncode2/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE$BatchNormGeneEncode2/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

B0
C1*

B0
C1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ga
VARIABLE_VALUEEmbeddingDimGene/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEEmbeddingDimGene/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
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
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
f`
VARIABLE_VALUEEmbeddingDimGene3/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEEmbeddingDimGene3/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEEmbeddingDimGene3/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE!EmbeddingDimGene3/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

U0
V1*

U0
V1*
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

�trace_0* 

�trace_0* 
e_
VARIABLE_VALUEgene_decoder_1/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEgene_decoder_1/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
^0
_1
`2
a3*

^0
_1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ic
VARIABLE_VALUEBatchNormGeneDecode1/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEBatchNormGeneDecode1/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE BatchNormGeneDecode1/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE$BatchNormGeneDecode1/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

h0
i1*

h0
i1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
e_
VARIABLE_VALUEgene_decoder_2/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEgene_decoder_2/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
q0
r1
s2
t3*

q0
r1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
c]
VARIABLE_VALUEBatchNormGene2/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEBatchNormGene2/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUEBatchNormGene2/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUEBatchNormGene2/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

{0
|1*

{0
|1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
f`
VARIABLE_VALUEgene_decoder_3/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEgene_decoder_3/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
J
'0
(1
:2
;3
M4
N5
`6
a7
s8
t9*
Z
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
11*

�0*
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
* 

:0
;1*
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

`0
a1*
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
s0
t1*
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
<
�	variables
�	keras_api

�total

�count*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUE!Adam/BatchNormGeneEncode2/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/BatchNormGeneEncode2/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/EmbeddingDimGene/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/EmbeddingDimGene/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/EmbeddingDimGene3/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/EmbeddingDimGene3/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_decoder_1/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/gene_decoder_1/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/BatchNormGeneDecode1/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/BatchNormGeneDecode1/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_decoder_2/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/gene_decoder_2/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/BatchNormGene2/gamma/mQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/BatchNormGene2/beta/mPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_decoder_3/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEAdam/gene_decoder_3/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUE!Adam/BatchNormGeneEncode2/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/BatchNormGeneEncode2/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/EmbeddingDimGene/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/EmbeddingDimGene/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/EmbeddingDimGene3/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/EmbeddingDimGene3/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_decoder_1/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/gene_decoder_1/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/BatchNormGeneDecode1/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/BatchNormGeneDecode1/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_decoder_2/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/gene_decoder_2/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/BatchNormGene2/gamma/vQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/BatchNormGene2/beta/vPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_decoder_3/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEAdam/gene_decoder_3/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�!
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)gene_encoder_1/kernel/Read/ReadVariableOp'gene_encoder_1/bias/Read/ReadVariableOp.BatchNormGeneEncode1/gamma/Read/ReadVariableOp-BatchNormGeneEncode1/beta/Read/ReadVariableOp4BatchNormGeneEncode1/moving_mean/Read/ReadVariableOp8BatchNormGeneEncode1/moving_variance/Read/ReadVariableOp)gene_encoder_2/kernel/Read/ReadVariableOp'gene_encoder_2/bias/Read/ReadVariableOp.BatchNormGeneEncode2/gamma/Read/ReadVariableOp-BatchNormGeneEncode2/beta/Read/ReadVariableOp4BatchNormGeneEncode2/moving_mean/Read/ReadVariableOp8BatchNormGeneEncode2/moving_variance/Read/ReadVariableOp+EmbeddingDimGene/kernel/Read/ReadVariableOp)EmbeddingDimGene/bias/Read/ReadVariableOp+EmbeddingDimGene3/gamma/Read/ReadVariableOp*EmbeddingDimGene3/beta/Read/ReadVariableOp1EmbeddingDimGene3/moving_mean/Read/ReadVariableOp5EmbeddingDimGene3/moving_variance/Read/ReadVariableOp)gene_decoder_1/kernel/Read/ReadVariableOp'gene_decoder_1/bias/Read/ReadVariableOp.BatchNormGeneDecode1/gamma/Read/ReadVariableOp-BatchNormGeneDecode1/beta/Read/ReadVariableOp4BatchNormGeneDecode1/moving_mean/Read/ReadVariableOp8BatchNormGeneDecode1/moving_variance/Read/ReadVariableOp)gene_decoder_2/kernel/Read/ReadVariableOp'gene_decoder_2/bias/Read/ReadVariableOp(BatchNormGene2/gamma/Read/ReadVariableOp'BatchNormGene2/beta/Read/ReadVariableOp.BatchNormGene2/moving_mean/Read/ReadVariableOp2BatchNormGene2/moving_variance/Read/ReadVariableOp)gene_decoder_3/kernel/Read/ReadVariableOp'gene_decoder_3/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp0Adam/gene_encoder_1/kernel/m/Read/ReadVariableOp.Adam/gene_encoder_1/bias/m/Read/ReadVariableOp5Adam/BatchNormGeneEncode1/gamma/m/Read/ReadVariableOp4Adam/BatchNormGeneEncode1/beta/m/Read/ReadVariableOp0Adam/gene_encoder_2/kernel/m/Read/ReadVariableOp.Adam/gene_encoder_2/bias/m/Read/ReadVariableOp5Adam/BatchNormGeneEncode2/gamma/m/Read/ReadVariableOp4Adam/BatchNormGeneEncode2/beta/m/Read/ReadVariableOp2Adam/EmbeddingDimGene/kernel/m/Read/ReadVariableOp0Adam/EmbeddingDimGene/bias/m/Read/ReadVariableOp2Adam/EmbeddingDimGene3/gamma/m/Read/ReadVariableOp1Adam/EmbeddingDimGene3/beta/m/Read/ReadVariableOp0Adam/gene_decoder_1/kernel/m/Read/ReadVariableOp.Adam/gene_decoder_1/bias/m/Read/ReadVariableOp5Adam/BatchNormGeneDecode1/gamma/m/Read/ReadVariableOp4Adam/BatchNormGeneDecode1/beta/m/Read/ReadVariableOp0Adam/gene_decoder_2/kernel/m/Read/ReadVariableOp.Adam/gene_decoder_2/bias/m/Read/ReadVariableOp/Adam/BatchNormGene2/gamma/m/Read/ReadVariableOp.Adam/BatchNormGene2/beta/m/Read/ReadVariableOp0Adam/gene_decoder_3/kernel/m/Read/ReadVariableOp.Adam/gene_decoder_3/bias/m/Read/ReadVariableOp0Adam/gene_encoder_1/kernel/v/Read/ReadVariableOp.Adam/gene_encoder_1/bias/v/Read/ReadVariableOp5Adam/BatchNormGeneEncode1/gamma/v/Read/ReadVariableOp4Adam/BatchNormGeneEncode1/beta/v/Read/ReadVariableOp0Adam/gene_encoder_2/kernel/v/Read/ReadVariableOp.Adam/gene_encoder_2/bias/v/Read/ReadVariableOp5Adam/BatchNormGeneEncode2/gamma/v/Read/ReadVariableOp4Adam/BatchNormGeneEncode2/beta/v/Read/ReadVariableOp2Adam/EmbeddingDimGene/kernel/v/Read/ReadVariableOp0Adam/EmbeddingDimGene/bias/v/Read/ReadVariableOp2Adam/EmbeddingDimGene3/gamma/v/Read/ReadVariableOp1Adam/EmbeddingDimGene3/beta/v/Read/ReadVariableOp0Adam/gene_decoder_1/kernel/v/Read/ReadVariableOp.Adam/gene_decoder_1/bias/v/Read/ReadVariableOp5Adam/BatchNormGeneDecode1/gamma/v/Read/ReadVariableOp4Adam/BatchNormGeneDecode1/beta/v/Read/ReadVariableOp0Adam/gene_decoder_2/kernel/v/Read/ReadVariableOp.Adam/gene_decoder_2/bias/v/Read/ReadVariableOp/Adam/BatchNormGene2/gamma/v/Read/ReadVariableOp.Adam/BatchNormGene2/beta/v/Read/ReadVariableOp0Adam/gene_decoder_3/kernel/v/Read/ReadVariableOp.Adam/gene_decoder_3/bias/v/Read/ReadVariableOpConst*`
TinY
W2U	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_save_2022223
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegene_encoder_1/kernelgene_encoder_1/biasBatchNormGeneEncode1/gammaBatchNormGeneEncode1/beta BatchNormGeneEncode1/moving_mean$BatchNormGeneEncode1/moving_variancegene_encoder_2/kernelgene_encoder_2/biasBatchNormGeneEncode2/gammaBatchNormGeneEncode2/beta BatchNormGeneEncode2/moving_mean$BatchNormGeneEncode2/moving_varianceEmbeddingDimGene/kernelEmbeddingDimGene/biasEmbeddingDimGene3/gammaEmbeddingDimGene3/betaEmbeddingDimGene3/moving_mean!EmbeddingDimGene3/moving_variancegene_decoder_1/kernelgene_decoder_1/biasBatchNormGeneDecode1/gammaBatchNormGeneDecode1/beta BatchNormGeneDecode1/moving_mean$BatchNormGeneDecode1/moving_variancegene_decoder_2/kernelgene_decoder_2/biasBatchNormGene2/gammaBatchNormGene2/betaBatchNormGene2/moving_meanBatchNormGene2/moving_variancegene_decoder_3/kernelgene_decoder_3/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/gene_encoder_1/kernel/mAdam/gene_encoder_1/bias/m!Adam/BatchNormGeneEncode1/gamma/m Adam/BatchNormGeneEncode1/beta/mAdam/gene_encoder_2/kernel/mAdam/gene_encoder_2/bias/m!Adam/BatchNormGeneEncode2/gamma/m Adam/BatchNormGeneEncode2/beta/mAdam/EmbeddingDimGene/kernel/mAdam/EmbeddingDimGene/bias/mAdam/EmbeddingDimGene3/gamma/mAdam/EmbeddingDimGene3/beta/mAdam/gene_decoder_1/kernel/mAdam/gene_decoder_1/bias/m!Adam/BatchNormGeneDecode1/gamma/m Adam/BatchNormGeneDecode1/beta/mAdam/gene_decoder_2/kernel/mAdam/gene_decoder_2/bias/mAdam/BatchNormGene2/gamma/mAdam/BatchNormGene2/beta/mAdam/gene_decoder_3/kernel/mAdam/gene_decoder_3/bias/mAdam/gene_encoder_1/kernel/vAdam/gene_encoder_1/bias/v!Adam/BatchNormGeneEncode1/gamma/v Adam/BatchNormGeneEncode1/beta/vAdam/gene_encoder_2/kernel/vAdam/gene_encoder_2/bias/v!Adam/BatchNormGeneEncode2/gamma/v Adam/BatchNormGeneEncode2/beta/vAdam/EmbeddingDimGene/kernel/vAdam/EmbeddingDimGene/bias/vAdam/EmbeddingDimGene3/gamma/vAdam/EmbeddingDimGene3/beta/vAdam/gene_decoder_1/kernel/vAdam/gene_decoder_1/bias/v!Adam/BatchNormGeneDecode1/gamma/v Adam/BatchNormGeneDecode1/beta/vAdam/gene_decoder_2/kernel/vAdam/gene_decoder_2/bias/vAdam/BatchNormGene2/gamma/vAdam/BatchNormGene2/beta/vAdam/gene_decoder_3/kernel/vAdam/gene_decoder_3/bias/v*_
TinX
V2T*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__traced_restore_2022482��
�
�
3__inference_EmbeddingDimGene3_layer_call_fn_2021677

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_2019993o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�%
�
N__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_2021731

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
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
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
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
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
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
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_2021551

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
�F
�
E__inference_model_42_layer_call_and_return_conditional_losses_2020815
gene_input_layer*
gene_encoder_1_2020739:
��%
gene_encoder_1_2020741:	�+
batchnormgeneencode1_2020744:	�+
batchnormgeneencode1_2020746:	�+
batchnormgeneencode1_2020748:	�+
batchnormgeneencode1_2020750:	�)
gene_encoder_2_2020753:	�}$
gene_encoder_2_2020755:}*
batchnormgeneencode2_2020758:}*
batchnormgeneencode2_2020760:}*
batchnormgeneencode2_2020762:}*
batchnormgeneencode2_2020764:}*
embeddingdimgene_2020767:}@&
embeddingdimgene_2020769:@'
embeddingdimgene3_2020772:@'
embeddingdimgene3_2020774:@'
embeddingdimgene3_2020776:@'
embeddingdimgene3_2020778:@(
gene_decoder_1_2020781:@}$
gene_decoder_1_2020783:}*
batchnormgenedecode1_2020786:}*
batchnormgenedecode1_2020788:}*
batchnormgenedecode1_2020790:}*
batchnormgenedecode1_2020792:})
gene_decoder_2_2020795:	}�%
gene_decoder_2_2020797:	�%
batchnormgene2_2020800:	�%
batchnormgene2_2020802:	�%
batchnormgene2_2020804:	�%
batchnormgene2_2020806:	�*
gene_decoder_3_2020809:
��%
gene_decoder_3_2020811:	�
identity��&BatchNormGene2/StatefulPartitionedCall�,BatchNormGeneDecode1/StatefulPartitionedCall�,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�(EmbeddingDimGene/StatefulPartitionedCall�)EmbeddingDimGene3/StatefulPartitionedCall�&gene_decoder_1/StatefulPartitionedCall�&gene_decoder_2/StatefulPartitionedCall�&gene_decoder_3/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallgene_input_layergene_encoder_1_2020739gene_encoder_1_2020741*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_2020186�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_2020744batchnormgeneencode1_2020746batchnormgeneencode1_2020748batchnormgeneencode1_2020750*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_2019782�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_2020753gene_encoder_2_2020755*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_2020212�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_2020758batchnormgeneencode2_2020760batchnormgeneencode2_2020762batchnormgeneencode2_2020764*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_2019864�
(EmbeddingDimGene/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:0embeddingdimgene_2020767embeddingdimgene_2020769*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_EmbeddingDimGene_layer_call_and_return_conditional_losses_2020238�
)EmbeddingDimGene3/StatefulPartitionedCallStatefulPartitionedCall1EmbeddingDimGene/StatefulPartitionedCall:output:0embeddingdimgene3_2020772embeddingdimgene3_2020774embeddingdimgene3_2020776embeddingdimgene3_2020778*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_2019946�
&gene_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimGene3/StatefulPartitionedCall:output:0gene_decoder_1_2020781gene_decoder_1_2020783*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_decoder_1_layer_call_and_return_conditional_losses_2020264�
,BatchNormGeneDecode1/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_1/StatefulPartitionedCall:output:0batchnormgenedecode1_2020786batchnormgenedecode1_2020788batchnormgenedecode1_2020790batchnormgenedecode1_2020792*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_2020028�
&gene_decoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode1/StatefulPartitionedCall:output:0gene_decoder_2_2020795gene_decoder_2_2020797*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_decoder_2_layer_call_and_return_conditional_losses_2020290�
&BatchNormGene2/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_2/StatefulPartitionedCall:output:0batchnormgene2_2020800batchnormgene2_2020802batchnormgene2_2020804batchnormgene2_2020806*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_BatchNormGene2_layer_call_and_return_conditional_losses_2020110�
&gene_decoder_3/StatefulPartitionedCallStatefulPartitionedCall/BatchNormGene2/StatefulPartitionedCall:output:0gene_decoder_3_2020809gene_decoder_3_2020811*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_decoder_3_layer_call_and_return_conditional_losses_2020316
IdentityIdentity/gene_decoder_3/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp'^BatchNormGene2/StatefulPartitionedCall-^BatchNormGeneDecode1/StatefulPartitionedCall-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall)^EmbeddingDimGene/StatefulPartitionedCall*^EmbeddingDimGene3/StatefulPartitionedCall'^gene_decoder_1/StatefulPartitionedCall'^gene_decoder_2/StatefulPartitionedCall'^gene_decoder_3/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&BatchNormGene2/StatefulPartitionedCall&BatchNormGene2/StatefulPartitionedCall2\
,BatchNormGeneDecode1/StatefulPartitionedCall,BatchNormGeneDecode1/StatefulPartitionedCall2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2T
(EmbeddingDimGene/StatefulPartitionedCall(EmbeddingDimGene/StatefulPartitionedCall2V
)EmbeddingDimGene3/StatefulPartitionedCall)EmbeddingDimGene3/StatefulPartitionedCall2P
&gene_decoder_1/StatefulPartitionedCall&gene_decoder_1/StatefulPartitionedCall2P
&gene_decoder_2/StatefulPartitionedCall&gene_decoder_2/StatefulPartitionedCall2P
&gene_decoder_3/StatefulPartitionedCall&gene_decoder_3/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall:Z V
(
_output_shapes
:����������
*
_user_specified_namegene_input_layer
�
�
6__inference_BatchNormGeneEncode2_layer_call_fn_2021577

inputs
unknown:}
	unknown_0:}
	unknown_1:}
	unknown_2:}
identity��StatefulPartitionedCall�
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
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_2019911o
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
6__inference_BatchNormGeneEncode1_layer_call_fn_2021464

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
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_2019782p
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
�
�
N__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_2019946

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
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
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
6__inference_BatchNormGeneDecode1_layer_call_fn_2021764

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
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_2020028o
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
�
�
0__inference_gene_decoder_3_layer_call_fn_2021940

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
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_decoder_3_layer_call_and_return_conditional_losses_2020316p
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
�%
�
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_2019829

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
�
�
*__inference_model_42_layer_call_fn_2020390
gene_input_layer
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�}
	unknown_6:}
	unknown_7:}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@}

unknown_18:}

unknown_19:}

unknown_20:}

unknown_21:}

unknown_22:}

unknown_23:	}�

unknown_24:	�

unknown_25:	�

unknown_26:	�

unknown_27:	�

unknown_28:	�

unknown_29:
��

unknown_30:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgene_input_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_42_layer_call_and_return_conditional_losses_2020323p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:����������
*
_user_specified_namegene_input_layer
�
�
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_2020028

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
6__inference_BatchNormGeneEncode2_layer_call_fn_2021564

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
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_2019864o
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
�
�
0__inference_gene_decoder_1_layer_call_fn_2021740

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
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_decoder_1_layer_call_and_return_conditional_losses_2020264o
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
�

�
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_2020186

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
�
�
*__inference_model_42_layer_call_fn_2021040

inputs
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�}
	unknown_6:}
	unknown_7:}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@}

unknown_18:}

unknown_19:}

unknown_20:}

unknown_21:}

unknown_22:}

unknown_23:	}�

unknown_24:	�

unknown_25:	�

unknown_26:	�

unknown_27:	�

unknown_28:	�

unknown_29:
��

unknown_30:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_42_layer_call_and_return_conditional_losses_2020323p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
6__inference_BatchNormGeneEncode1_layer_call_fn_2021477

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
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
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_2019829p
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
�%
�
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_2021831

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
�

�
K__inference_gene_decoder_3_layer_call_and_return_conditional_losses_2021951

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
�F
�
E__inference_model_42_layer_call_and_return_conditional_losses_2020894
gene_input_layer*
gene_encoder_1_2020818:
��%
gene_encoder_1_2020820:	�+
batchnormgeneencode1_2020823:	�+
batchnormgeneencode1_2020825:	�+
batchnormgeneencode1_2020827:	�+
batchnormgeneencode1_2020829:	�)
gene_encoder_2_2020832:	�}$
gene_encoder_2_2020834:}*
batchnormgeneencode2_2020837:}*
batchnormgeneencode2_2020839:}*
batchnormgeneencode2_2020841:}*
batchnormgeneencode2_2020843:}*
embeddingdimgene_2020846:}@&
embeddingdimgene_2020848:@'
embeddingdimgene3_2020851:@'
embeddingdimgene3_2020853:@'
embeddingdimgene3_2020855:@'
embeddingdimgene3_2020857:@(
gene_decoder_1_2020860:@}$
gene_decoder_1_2020862:}*
batchnormgenedecode1_2020865:}*
batchnormgenedecode1_2020867:}*
batchnormgenedecode1_2020869:}*
batchnormgenedecode1_2020871:})
gene_decoder_2_2020874:	}�%
gene_decoder_2_2020876:	�%
batchnormgene2_2020879:	�%
batchnormgene2_2020881:	�%
batchnormgene2_2020883:	�%
batchnormgene2_2020885:	�*
gene_decoder_3_2020888:
��%
gene_decoder_3_2020890:	�
identity��&BatchNormGene2/StatefulPartitionedCall�,BatchNormGeneDecode1/StatefulPartitionedCall�,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�(EmbeddingDimGene/StatefulPartitionedCall�)EmbeddingDimGene3/StatefulPartitionedCall�&gene_decoder_1/StatefulPartitionedCall�&gene_decoder_2/StatefulPartitionedCall�&gene_decoder_3/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallgene_input_layergene_encoder_1_2020818gene_encoder_1_2020820*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_2020186�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_2020823batchnormgeneencode1_2020825batchnormgeneencode1_2020827batchnormgeneencode1_2020829*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_2019829�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_2020832gene_encoder_2_2020834*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_2020212�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_2020837batchnormgeneencode2_2020839batchnormgeneencode2_2020841batchnormgeneencode2_2020843*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_2019911�
(EmbeddingDimGene/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:0embeddingdimgene_2020846embeddingdimgene_2020848*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_EmbeddingDimGene_layer_call_and_return_conditional_losses_2020238�
)EmbeddingDimGene3/StatefulPartitionedCallStatefulPartitionedCall1EmbeddingDimGene/StatefulPartitionedCall:output:0embeddingdimgene3_2020851embeddingdimgene3_2020853embeddingdimgene3_2020855embeddingdimgene3_2020857*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_2019993�
&gene_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimGene3/StatefulPartitionedCall:output:0gene_decoder_1_2020860gene_decoder_1_2020862*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_decoder_1_layer_call_and_return_conditional_losses_2020264�
,BatchNormGeneDecode1/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_1/StatefulPartitionedCall:output:0batchnormgenedecode1_2020865batchnormgenedecode1_2020867batchnormgenedecode1_2020869batchnormgenedecode1_2020871*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_2020075�
&gene_decoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode1/StatefulPartitionedCall:output:0gene_decoder_2_2020874gene_decoder_2_2020876*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_decoder_2_layer_call_and_return_conditional_losses_2020290�
&BatchNormGene2/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_2/StatefulPartitionedCall:output:0batchnormgene2_2020879batchnormgene2_2020881batchnormgene2_2020883batchnormgene2_2020885*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_BatchNormGene2_layer_call_and_return_conditional_losses_2020157�
&gene_decoder_3/StatefulPartitionedCallStatefulPartitionedCall/BatchNormGene2/StatefulPartitionedCall:output:0gene_decoder_3_2020888gene_decoder_3_2020890*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_decoder_3_layer_call_and_return_conditional_losses_2020316
IdentityIdentity/gene_decoder_3/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp'^BatchNormGene2/StatefulPartitionedCall-^BatchNormGeneDecode1/StatefulPartitionedCall-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall)^EmbeddingDimGene/StatefulPartitionedCall*^EmbeddingDimGene3/StatefulPartitionedCall'^gene_decoder_1/StatefulPartitionedCall'^gene_decoder_2/StatefulPartitionedCall'^gene_decoder_3/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&BatchNormGene2/StatefulPartitionedCall&BatchNormGene2/StatefulPartitionedCall2\
,BatchNormGeneDecode1/StatefulPartitionedCall,BatchNormGeneDecode1/StatefulPartitionedCall2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2T
(EmbeddingDimGene/StatefulPartitionedCall(EmbeddingDimGene/StatefulPartitionedCall2V
)EmbeddingDimGene3/StatefulPartitionedCall)EmbeddingDimGene3/StatefulPartitionedCall2P
&gene_decoder_1/StatefulPartitionedCall&gene_decoder_1/StatefulPartitionedCall2P
&gene_decoder_2/StatefulPartitionedCall&gene_decoder_2/StatefulPartitionedCall2P
&gene_decoder_3/StatefulPartitionedCall&gene_decoder_3/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall:Z V
(
_output_shapes
:����������
*
_user_specified_namegene_input_layer
�
�
N__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_2021697

inputs/
!batchnorm_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@1
#batchnorm_readvariableop_1_resource:@1
#batchnorm_readvariableop_2_resource:@
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
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
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:@z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
K__inference_gene_decoder_2_layer_call_and_return_conditional_losses_2020290

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
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_2019782

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
�

�
K__inference_gene_decoder_2_layer_call_and_return_conditional_losses_2021851

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
K__inference_BatchNormGene2_layer_call_and_return_conditional_losses_2020110

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
�
�
0__inference_gene_decoder_2_layer_call_fn_2021840

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
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_decoder_2_layer_call_and_return_conditional_losses_2020290p
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
�
�
0__inference_BatchNormGene2_layer_call_fn_2021877

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
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
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_BatchNormGene2_layer_call_and_return_conditional_losses_2020157p
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
�
�
*__inference_model_42_layer_call_fn_2020736
gene_input_layer
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�}
	unknown_6:}
	unknown_7:}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@}

unknown_18:}

unknown_19:}

unknown_20:}

unknown_21:}

unknown_22:}

unknown_23:	}�

unknown_24:	�

unknown_25:	�

unknown_26:	�

unknown_27:	�

unknown_28:	�

unknown_29:
��

unknown_30:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgene_input_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*8
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_42_layer_call_and_return_conditional_losses_2020600p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:����������
*
_user_specified_namegene_input_layer
�F
�
E__inference_model_42_layer_call_and_return_conditional_losses_2020600

inputs*
gene_encoder_1_2020524:
��%
gene_encoder_1_2020526:	�+
batchnormgeneencode1_2020529:	�+
batchnormgeneencode1_2020531:	�+
batchnormgeneencode1_2020533:	�+
batchnormgeneencode1_2020535:	�)
gene_encoder_2_2020538:	�}$
gene_encoder_2_2020540:}*
batchnormgeneencode2_2020543:}*
batchnormgeneencode2_2020545:}*
batchnormgeneencode2_2020547:}*
batchnormgeneencode2_2020549:}*
embeddingdimgene_2020552:}@&
embeddingdimgene_2020554:@'
embeddingdimgene3_2020557:@'
embeddingdimgene3_2020559:@'
embeddingdimgene3_2020561:@'
embeddingdimgene3_2020563:@(
gene_decoder_1_2020566:@}$
gene_decoder_1_2020568:}*
batchnormgenedecode1_2020571:}*
batchnormgenedecode1_2020573:}*
batchnormgenedecode1_2020575:}*
batchnormgenedecode1_2020577:})
gene_decoder_2_2020580:	}�%
gene_decoder_2_2020582:	�%
batchnormgene2_2020585:	�%
batchnormgene2_2020587:	�%
batchnormgene2_2020589:	�%
batchnormgene2_2020591:	�*
gene_decoder_3_2020594:
��%
gene_decoder_3_2020596:	�
identity��&BatchNormGene2/StatefulPartitionedCall�,BatchNormGeneDecode1/StatefulPartitionedCall�,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�(EmbeddingDimGene/StatefulPartitionedCall�)EmbeddingDimGene3/StatefulPartitionedCall�&gene_decoder_1/StatefulPartitionedCall�&gene_decoder_2/StatefulPartitionedCall�&gene_decoder_3/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputsgene_encoder_1_2020524gene_encoder_1_2020526*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_2020186�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_2020529batchnormgeneencode1_2020531batchnormgeneencode1_2020533batchnormgeneencode1_2020535*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_2019829�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_2020538gene_encoder_2_2020540*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_2020212�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_2020543batchnormgeneencode2_2020545batchnormgeneencode2_2020547batchnormgeneencode2_2020549*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_2019911�
(EmbeddingDimGene/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:0embeddingdimgene_2020552embeddingdimgene_2020554*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_EmbeddingDimGene_layer_call_and_return_conditional_losses_2020238�
)EmbeddingDimGene3/StatefulPartitionedCallStatefulPartitionedCall1EmbeddingDimGene/StatefulPartitionedCall:output:0embeddingdimgene3_2020557embeddingdimgene3_2020559embeddingdimgene3_2020561embeddingdimgene3_2020563*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_2019993�
&gene_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimGene3/StatefulPartitionedCall:output:0gene_decoder_1_2020566gene_decoder_1_2020568*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_decoder_1_layer_call_and_return_conditional_losses_2020264�
,BatchNormGeneDecode1/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_1/StatefulPartitionedCall:output:0batchnormgenedecode1_2020571batchnormgenedecode1_2020573batchnormgenedecode1_2020575batchnormgenedecode1_2020577*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_2020075�
&gene_decoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode1/StatefulPartitionedCall:output:0gene_decoder_2_2020580gene_decoder_2_2020582*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_decoder_2_layer_call_and_return_conditional_losses_2020290�
&BatchNormGene2/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_2/StatefulPartitionedCall:output:0batchnormgene2_2020585batchnormgene2_2020587batchnormgene2_2020589batchnormgene2_2020591*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_BatchNormGene2_layer_call_and_return_conditional_losses_2020157�
&gene_decoder_3/StatefulPartitionedCallStatefulPartitionedCall/BatchNormGene2/StatefulPartitionedCall:output:0gene_decoder_3_2020594gene_decoder_3_2020596*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_decoder_3_layer_call_and_return_conditional_losses_2020316
IdentityIdentity/gene_decoder_3/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp'^BatchNormGene2/StatefulPartitionedCall-^BatchNormGeneDecode1/StatefulPartitionedCall-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall)^EmbeddingDimGene/StatefulPartitionedCall*^EmbeddingDimGene3/StatefulPartitionedCall'^gene_decoder_1/StatefulPartitionedCall'^gene_decoder_2/StatefulPartitionedCall'^gene_decoder_3/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&BatchNormGene2/StatefulPartitionedCall&BatchNormGene2/StatefulPartitionedCall2\
,BatchNormGeneDecode1/StatefulPartitionedCall,BatchNormGeneDecode1/StatefulPartitionedCall2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2T
(EmbeddingDimGene/StatefulPartitionedCall(EmbeddingDimGene/StatefulPartitionedCall2V
)EmbeddingDimGene3/StatefulPartitionedCall)EmbeddingDimGene3/StatefulPartitionedCall2P
&gene_decoder_1/StatefulPartitionedCall&gene_decoder_1/StatefulPartitionedCall2P
&gene_decoder_2/StatefulPartitionedCall&gene_decoder_2/StatefulPartitionedCall2P
&gene_decoder_3/StatefulPartitionedCall&gene_decoder_3/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_model_42_layer_call_fn_2021109

inputs
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�}
	unknown_6:}
	unknown_7:}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@}

unknown_18:}

unknown_19:}

unknown_20:}

unknown_21:}

unknown_22:}

unknown_23:	}�

unknown_24:	�

unknown_25:	�

unknown_26:	�

unknown_27:	�

unknown_28:	�

unknown_29:
��

unknown_30:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*8
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_42_layer_call_and_return_conditional_losses_2020600p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�%
�
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_2021631

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
�
�
0__inference_gene_encoder_2_layer_call_fn_2021540

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
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_2020212o
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
�
�
%__inference_signature_wrapper_2020971
gene_input_layer
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�}
	unknown_6:}
	unknown_7:}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}@

unknown_12:@

unknown_13:@

unknown_14:@

unknown_15:@

unknown_16:@

unknown_17:@}

unknown_18:}

unknown_19:}

unknown_20:}

unknown_21:}

unknown_22:}

unknown_23:	}�

unknown_24:	�

unknown_25:	�

unknown_26:	�

unknown_27:	�

unknown_28:	�

unknown_29:
��

unknown_30:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgene_input_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_2019758p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:����������
*
_user_specified_namegene_input_layer
�

�
M__inference_EmbeddingDimGene_layer_call_and_return_conditional_losses_2020238

inputs0
matmul_readvariableop_resource:}@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:}@*
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
_construction_contextkEagerRuntime**
_input_shapes
:���������}: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������}
 
_user_specified_nameinputs
�%
�
K__inference_BatchNormGene2_layer_call_and_return_conditional_losses_2020157

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
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_2021451

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
�
�
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_2021597

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
��
�7
#__inference__traced_restore_2022482
file_prefix:
&assignvariableop_gene_encoder_1_kernel:
��5
&assignvariableop_1_gene_encoder_1_bias:	�<
-assignvariableop_2_batchnormgeneencode1_gamma:	�;
,assignvariableop_3_batchnormgeneencode1_beta:	�B
3assignvariableop_4_batchnormgeneencode1_moving_mean:	�F
7assignvariableop_5_batchnormgeneencode1_moving_variance:	�;
(assignvariableop_6_gene_encoder_2_kernel:	�}4
&assignvariableop_7_gene_encoder_2_bias:};
-assignvariableop_8_batchnormgeneencode2_gamma:}:
,assignvariableop_9_batchnormgeneencode2_beta:}B
4assignvariableop_10_batchnormgeneencode2_moving_mean:}F
8assignvariableop_11_batchnormgeneencode2_moving_variance:}=
+assignvariableop_12_embeddingdimgene_kernel:}@7
)assignvariableop_13_embeddingdimgene_bias:@9
+assignvariableop_14_embeddingdimgene3_gamma:@8
*assignvariableop_15_embeddingdimgene3_beta:@?
1assignvariableop_16_embeddingdimgene3_moving_mean:@C
5assignvariableop_17_embeddingdimgene3_moving_variance:@;
)assignvariableop_18_gene_decoder_1_kernel:@}5
'assignvariableop_19_gene_decoder_1_bias:}<
.assignvariableop_20_batchnormgenedecode1_gamma:};
-assignvariableop_21_batchnormgenedecode1_beta:}B
4assignvariableop_22_batchnormgenedecode1_moving_mean:}F
8assignvariableop_23_batchnormgenedecode1_moving_variance:}<
)assignvariableop_24_gene_decoder_2_kernel:	}�6
'assignvariableop_25_gene_decoder_2_bias:	�7
(assignvariableop_26_batchnormgene2_gamma:	�6
'assignvariableop_27_batchnormgene2_beta:	�=
.assignvariableop_28_batchnormgene2_moving_mean:	�A
2assignvariableop_29_batchnormgene2_moving_variance:	�=
)assignvariableop_30_gene_decoder_3_kernel:
��6
'assignvariableop_31_gene_decoder_3_bias:	�'
assignvariableop_32_adam_iter:	 )
assignvariableop_33_adam_beta_1: )
assignvariableop_34_adam_beta_2: (
assignvariableop_35_adam_decay: 0
&assignvariableop_36_adam_learning_rate: #
assignvariableop_37_total: #
assignvariableop_38_count: D
0assignvariableop_39_adam_gene_encoder_1_kernel_m:
��=
.assignvariableop_40_adam_gene_encoder_1_bias_m:	�D
5assignvariableop_41_adam_batchnormgeneencode1_gamma_m:	�C
4assignvariableop_42_adam_batchnormgeneencode1_beta_m:	�C
0assignvariableop_43_adam_gene_encoder_2_kernel_m:	�}<
.assignvariableop_44_adam_gene_encoder_2_bias_m:}C
5assignvariableop_45_adam_batchnormgeneencode2_gamma_m:}B
4assignvariableop_46_adam_batchnormgeneencode2_beta_m:}D
2assignvariableop_47_adam_embeddingdimgene_kernel_m:}@>
0assignvariableop_48_adam_embeddingdimgene_bias_m:@@
2assignvariableop_49_adam_embeddingdimgene3_gamma_m:@?
1assignvariableop_50_adam_embeddingdimgene3_beta_m:@B
0assignvariableop_51_adam_gene_decoder_1_kernel_m:@}<
.assignvariableop_52_adam_gene_decoder_1_bias_m:}C
5assignvariableop_53_adam_batchnormgenedecode1_gamma_m:}B
4assignvariableop_54_adam_batchnormgenedecode1_beta_m:}C
0assignvariableop_55_adam_gene_decoder_2_kernel_m:	}�=
.assignvariableop_56_adam_gene_decoder_2_bias_m:	�>
/assignvariableop_57_adam_batchnormgene2_gamma_m:	�=
.assignvariableop_58_adam_batchnormgene2_beta_m:	�D
0assignvariableop_59_adam_gene_decoder_3_kernel_m:
��=
.assignvariableop_60_adam_gene_decoder_3_bias_m:	�D
0assignvariableop_61_adam_gene_encoder_1_kernel_v:
��=
.assignvariableop_62_adam_gene_encoder_1_bias_v:	�D
5assignvariableop_63_adam_batchnormgeneencode1_gamma_v:	�C
4assignvariableop_64_adam_batchnormgeneencode1_beta_v:	�C
0assignvariableop_65_adam_gene_encoder_2_kernel_v:	�}<
.assignvariableop_66_adam_gene_encoder_2_bias_v:}C
5assignvariableop_67_adam_batchnormgeneencode2_gamma_v:}B
4assignvariableop_68_adam_batchnormgeneencode2_beta_v:}D
2assignvariableop_69_adam_embeddingdimgene_kernel_v:}@>
0assignvariableop_70_adam_embeddingdimgene_bias_v:@@
2assignvariableop_71_adam_embeddingdimgene3_gamma_v:@?
1assignvariableop_72_adam_embeddingdimgene3_beta_v:@B
0assignvariableop_73_adam_gene_decoder_1_kernel_v:@}<
.assignvariableop_74_adam_gene_decoder_1_bias_v:}C
5assignvariableop_75_adam_batchnormgenedecode1_gamma_v:}B
4assignvariableop_76_adam_batchnormgenedecode1_beta_v:}C
0assignvariableop_77_adam_gene_decoder_2_kernel_v:	}�=
.assignvariableop_78_adam_gene_decoder_2_bias_v:	�>
/assignvariableop_79_adam_batchnormgene2_gamma_v:	�=
.assignvariableop_80_adam_batchnormgene2_beta_v:	�D
0assignvariableop_81_adam_gene_decoder_3_kernel_v:
��=
.assignvariableop_82_adam_gene_decoder_3_bias_v:	�
identity_84��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_9�.
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*�-
value�-B�-TB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*�
value�B�TB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*b
dtypesX
V2T	[
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
AssignVariableOp_8AssignVariableOp-assignvariableop_8_batchnormgeneencode2_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp,assignvariableop_9_batchnormgeneencode2_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp4assignvariableop_10_batchnormgeneencode2_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp8assignvariableop_11_batchnormgeneencode2_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp+assignvariableop_12_embeddingdimgene_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp)assignvariableop_13_embeddingdimgene_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp+assignvariableop_14_embeddingdimgene3_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp*assignvariableop_15_embeddingdimgene3_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp1assignvariableop_16_embeddingdimgene3_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp5assignvariableop_17_embeddingdimgene3_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp)assignvariableop_18_gene_decoder_1_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp'assignvariableop_19_gene_decoder_1_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp.assignvariableop_20_batchnormgenedecode1_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp-assignvariableop_21_batchnormgenedecode1_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp4assignvariableop_22_batchnormgenedecode1_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp8assignvariableop_23_batchnormgenedecode1_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp)assignvariableop_24_gene_decoder_2_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp'assignvariableop_25_gene_decoder_2_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp(assignvariableop_26_batchnormgene2_gammaIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp'assignvariableop_27_batchnormgene2_betaIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp.assignvariableop_28_batchnormgene2_moving_meanIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp2assignvariableop_29_batchnormgene2_moving_varianceIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp)assignvariableop_30_gene_decoder_3_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp'assignvariableop_31_gene_decoder_3_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_32AssignVariableOpassignvariableop_32_adam_iterIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOpassignvariableop_33_adam_beta_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOpassignvariableop_34_adam_beta_2Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOpassignvariableop_35_adam_decayIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_learning_rateIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOpassignvariableop_37_totalIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOpassignvariableop_38_countIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp0assignvariableop_39_adam_gene_encoder_1_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp.assignvariableop_40_adam_gene_encoder_1_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp5assignvariableop_41_adam_batchnormgeneencode1_gamma_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp4assignvariableop_42_adam_batchnormgeneencode1_beta_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp0assignvariableop_43_adam_gene_encoder_2_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp.assignvariableop_44_adam_gene_encoder_2_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp5assignvariableop_45_adam_batchnormgeneencode2_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp4assignvariableop_46_adam_batchnormgeneencode2_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp2assignvariableop_47_adam_embeddingdimgene_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp0assignvariableop_48_adam_embeddingdimgene_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp2assignvariableop_49_adam_embeddingdimgene3_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp1assignvariableop_50_adam_embeddingdimgene3_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp0assignvariableop_51_adam_gene_decoder_1_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp.assignvariableop_52_adam_gene_decoder_1_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp5assignvariableop_53_adam_batchnormgenedecode1_gamma_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp4assignvariableop_54_adam_batchnormgenedecode1_beta_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp0assignvariableop_55_adam_gene_decoder_2_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp.assignvariableop_56_adam_gene_decoder_2_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp/assignvariableop_57_adam_batchnormgene2_gamma_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp.assignvariableop_58_adam_batchnormgene2_beta_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp0assignvariableop_59_adam_gene_decoder_3_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp.assignvariableop_60_adam_gene_decoder_3_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp0assignvariableop_61_adam_gene_encoder_1_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp.assignvariableop_62_adam_gene_encoder_1_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp5assignvariableop_63_adam_batchnormgeneencode1_gamma_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp4assignvariableop_64_adam_batchnormgeneencode1_beta_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp0assignvariableop_65_adam_gene_encoder_2_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp.assignvariableop_66_adam_gene_encoder_2_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp5assignvariableop_67_adam_batchnormgeneencode2_gamma_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp4assignvariableop_68_adam_batchnormgeneencode2_beta_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp2assignvariableop_69_adam_embeddingdimgene_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp0assignvariableop_70_adam_embeddingdimgene_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp2assignvariableop_71_adam_embeddingdimgene3_gamma_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp1assignvariableop_72_adam_embeddingdimgene3_beta_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp0assignvariableop_73_adam_gene_decoder_1_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp.assignvariableop_74_adam_gene_decoder_1_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp5assignvariableop_75_adam_batchnormgenedecode1_gamma_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp4assignvariableop_76_adam_batchnormgenedecode1_beta_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp0assignvariableop_77_adam_gene_decoder_2_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp.assignvariableop_78_adam_gene_decoder_2_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp/assignvariableop_79_adam_batchnormgene2_gamma_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp.assignvariableop_80_adam_batchnormgene2_beta_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp0assignvariableop_81_adam_gene_decoder_3_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp.assignvariableop_82_adam_gene_decoder_3_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_83Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_84IdentityIdentity_83:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_84Identity_84:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
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
AssignVariableOp_82AssignVariableOp_822(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
E__inference_model_42_layer_call_and_return_conditional_losses_2021235

inputsA
-gene_encoder_1_matmul_readvariableop_resource:
��=
.gene_encoder_1_biasadd_readvariableop_resource:	�E
6batchnormgeneencode1_batchnorm_readvariableop_resource:	�I
:batchnormgeneencode1_batchnorm_mul_readvariableop_resource:	�G
8batchnormgeneencode1_batchnorm_readvariableop_1_resource:	�G
8batchnormgeneencode1_batchnorm_readvariableop_2_resource:	�@
-gene_encoder_2_matmul_readvariableop_resource:	�}<
.gene_encoder_2_biasadd_readvariableop_resource:}D
6batchnormgeneencode2_batchnorm_readvariableop_resource:}H
:batchnormgeneencode2_batchnorm_mul_readvariableop_resource:}F
8batchnormgeneencode2_batchnorm_readvariableop_1_resource:}F
8batchnormgeneencode2_batchnorm_readvariableop_2_resource:}A
/embeddingdimgene_matmul_readvariableop_resource:}@>
0embeddingdimgene_biasadd_readvariableop_resource:@A
3embeddingdimgene3_batchnorm_readvariableop_resource:@E
7embeddingdimgene3_batchnorm_mul_readvariableop_resource:@C
5embeddingdimgene3_batchnorm_readvariableop_1_resource:@C
5embeddingdimgene3_batchnorm_readvariableop_2_resource:@?
-gene_decoder_1_matmul_readvariableop_resource:@}<
.gene_decoder_1_biasadd_readvariableop_resource:}D
6batchnormgenedecode1_batchnorm_readvariableop_resource:}H
:batchnormgenedecode1_batchnorm_mul_readvariableop_resource:}F
8batchnormgenedecode1_batchnorm_readvariableop_1_resource:}F
8batchnormgenedecode1_batchnorm_readvariableop_2_resource:}@
-gene_decoder_2_matmul_readvariableop_resource:	}�=
.gene_decoder_2_biasadd_readvariableop_resource:	�?
0batchnormgene2_batchnorm_readvariableop_resource:	�C
4batchnormgene2_batchnorm_mul_readvariableop_resource:	�A
2batchnormgene2_batchnorm_readvariableop_1_resource:	�A
2batchnormgene2_batchnorm_readvariableop_2_resource:	�A
-gene_decoder_3_matmul_readvariableop_resource:
��=
.gene_decoder_3_biasadd_readvariableop_resource:	�
identity��'BatchNormGene2/batchnorm/ReadVariableOp�)BatchNormGene2/batchnorm/ReadVariableOp_1�)BatchNormGene2/batchnorm/ReadVariableOp_2�+BatchNormGene2/batchnorm/mul/ReadVariableOp�-BatchNormGeneDecode1/batchnorm/ReadVariableOp�/BatchNormGeneDecode1/batchnorm/ReadVariableOp_1�/BatchNormGeneDecode1/batchnorm/ReadVariableOp_2�1BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp�-BatchNormGeneEncode1/batchnorm/ReadVariableOp�/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1�/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2�1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp�-BatchNormGeneEncode2/batchnorm/ReadVariableOp�/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1�/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2�1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp�'EmbeddingDimGene/BiasAdd/ReadVariableOp�&EmbeddingDimGene/MatMul/ReadVariableOp�*EmbeddingDimGene3/batchnorm/ReadVariableOp�,EmbeddingDimGene3/batchnorm/ReadVariableOp_1�,EmbeddingDimGene3/batchnorm/ReadVariableOp_2�.EmbeddingDimGene3/batchnorm/mul/ReadVariableOp�%gene_decoder_1/BiasAdd/ReadVariableOp�$gene_decoder_1/MatMul/ReadVariableOp�%gene_decoder_2/BiasAdd/ReadVariableOp�$gene_decoder_2/MatMul/ReadVariableOp�%gene_decoder_3/BiasAdd/ReadVariableOp�$gene_decoder_3/MatMul/ReadVariableOp�%gene_encoder_1/BiasAdd/ReadVariableOp�$gene_encoder_1/MatMul/ReadVariableOp�%gene_encoder_2/BiasAdd/ReadVariableOp�$gene_encoder_2/MatMul/ReadVariableOp�
$gene_encoder_1/MatMul/ReadVariableOpReadVariableOp-gene_encoder_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gene_encoder_1/MatMulMatMulinputs,gene_encoder_1/MatMul/ReadVariableOp:value:0*
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
&EmbeddingDimGene/MatMul/ReadVariableOpReadVariableOp/embeddingdimgene_matmul_readvariableop_resource*
_output_shapes

:}@*
dtype0�
EmbeddingDimGene/MatMulMatMul(BatchNormGeneEncode2/batchnorm/add_1:z:0.EmbeddingDimGene/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
'EmbeddingDimGene/BiasAdd/ReadVariableOpReadVariableOp0embeddingdimgene_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
EmbeddingDimGene/BiasAddBiasAdd!EmbeddingDimGene/MatMul:product:0/EmbeddingDimGene/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@x
EmbeddingDimGene/SigmoidSigmoid!EmbeddingDimGene/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
*EmbeddingDimGene3/batchnorm/ReadVariableOpReadVariableOp3embeddingdimgene3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0f
!EmbeddingDimGene3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
EmbeddingDimGene3/batchnorm/addAddV22EmbeddingDimGene3/batchnorm/ReadVariableOp:value:0*EmbeddingDimGene3/batchnorm/add/y:output:0*
T0*
_output_shapes
:@t
!EmbeddingDimGene3/batchnorm/RsqrtRsqrt#EmbeddingDimGene3/batchnorm/add:z:0*
T0*
_output_shapes
:@�
.EmbeddingDimGene3/batchnorm/mul/ReadVariableOpReadVariableOp7embeddingdimgene3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
EmbeddingDimGene3/batchnorm/mulMul%EmbeddingDimGene3/batchnorm/Rsqrt:y:06EmbeddingDimGene3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
!EmbeddingDimGene3/batchnorm/mul_1MulEmbeddingDimGene/Sigmoid:y:0#EmbeddingDimGene3/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
,EmbeddingDimGene3/batchnorm/ReadVariableOp_1ReadVariableOp5embeddingdimgene3_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
!EmbeddingDimGene3/batchnorm/mul_2Mul4EmbeddingDimGene3/batchnorm/ReadVariableOp_1:value:0#EmbeddingDimGene3/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
,EmbeddingDimGene3/batchnorm/ReadVariableOp_2ReadVariableOp5embeddingdimgene3_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
EmbeddingDimGene3/batchnorm/subSub4EmbeddingDimGene3/batchnorm/ReadVariableOp_2:value:0%EmbeddingDimGene3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
!EmbeddingDimGene3/batchnorm/add_1AddV2%EmbeddingDimGene3/batchnorm/mul_1:z:0#EmbeddingDimGene3/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
$gene_decoder_1/MatMul/ReadVariableOpReadVariableOp-gene_decoder_1_matmul_readvariableop_resource*
_output_shapes

:@}*
dtype0�
gene_decoder_1/MatMulMatMul%EmbeddingDimGene3/batchnorm/add_1:z:0,gene_decoder_1/MatMul/ReadVariableOp:value:0*
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
'BatchNormGene2/batchnorm/ReadVariableOpReadVariableOp0batchnormgene2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0c
BatchNormGene2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
BatchNormGene2/batchnorm/addAddV2/BatchNormGene2/batchnorm/ReadVariableOp:value:0'BatchNormGene2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�o
BatchNormGene2/batchnorm/RsqrtRsqrt BatchNormGene2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
+BatchNormGene2/batchnorm/mul/ReadVariableOpReadVariableOp4batchnormgene2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BatchNormGene2/batchnorm/mulMul"BatchNormGene2/batchnorm/Rsqrt:y:03BatchNormGene2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
BatchNormGene2/batchnorm/mul_1Mulgene_decoder_2/Sigmoid:y:0 BatchNormGene2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
)BatchNormGene2/batchnorm/ReadVariableOp_1ReadVariableOp2batchnormgene2_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
BatchNormGene2/batchnorm/mul_2Mul1BatchNormGene2/batchnorm/ReadVariableOp_1:value:0 BatchNormGene2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
)BatchNormGene2/batchnorm/ReadVariableOp_2ReadVariableOp2batchnormgene2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
BatchNormGene2/batchnorm/subSub1BatchNormGene2/batchnorm/ReadVariableOp_2:value:0"BatchNormGene2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
BatchNormGene2/batchnorm/add_1AddV2"BatchNormGene2/batchnorm/mul_1:z:0 BatchNormGene2/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
$gene_decoder_3/MatMul/ReadVariableOpReadVariableOp-gene_decoder_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gene_decoder_3/MatMulMatMul"BatchNormGene2/batchnorm/add_1:z:0,gene_decoder_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%gene_decoder_3/BiasAdd/ReadVariableOpReadVariableOp.gene_decoder_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
gene_decoder_3/BiasAddBiasAddgene_decoder_3/MatMul:product:0-gene_decoder_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
gene_decoder_3/SigmoidSigmoidgene_decoder_3/BiasAdd:output:0*
T0*(
_output_shapes
:����������j
IdentityIdentitygene_decoder_3/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp(^BatchNormGene2/batchnorm/ReadVariableOp*^BatchNormGene2/batchnorm/ReadVariableOp_1*^BatchNormGene2/batchnorm/ReadVariableOp_2,^BatchNormGene2/batchnorm/mul/ReadVariableOp.^BatchNormGeneDecode1/batchnorm/ReadVariableOp0^BatchNormGeneDecode1/batchnorm/ReadVariableOp_10^BatchNormGeneDecode1/batchnorm/ReadVariableOp_22^BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp.^BatchNormGeneEncode1/batchnorm/ReadVariableOp0^BatchNormGeneEncode1/batchnorm/ReadVariableOp_10^BatchNormGeneEncode1/batchnorm/ReadVariableOp_22^BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp.^BatchNormGeneEncode2/batchnorm/ReadVariableOp0^BatchNormGeneEncode2/batchnorm/ReadVariableOp_10^BatchNormGeneEncode2/batchnorm/ReadVariableOp_22^BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp(^EmbeddingDimGene/BiasAdd/ReadVariableOp'^EmbeddingDimGene/MatMul/ReadVariableOp+^EmbeddingDimGene3/batchnorm/ReadVariableOp-^EmbeddingDimGene3/batchnorm/ReadVariableOp_1-^EmbeddingDimGene3/batchnorm/ReadVariableOp_2/^EmbeddingDimGene3/batchnorm/mul/ReadVariableOp&^gene_decoder_1/BiasAdd/ReadVariableOp%^gene_decoder_1/MatMul/ReadVariableOp&^gene_decoder_2/BiasAdd/ReadVariableOp%^gene_decoder_2/MatMul/ReadVariableOp&^gene_decoder_3/BiasAdd/ReadVariableOp%^gene_decoder_3/MatMul/ReadVariableOp&^gene_encoder_1/BiasAdd/ReadVariableOp%^gene_encoder_1/MatMul/ReadVariableOp&^gene_encoder_2/BiasAdd/ReadVariableOp%^gene_encoder_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2R
'BatchNormGene2/batchnorm/ReadVariableOp'BatchNormGene2/batchnorm/ReadVariableOp2V
)BatchNormGene2/batchnorm/ReadVariableOp_1)BatchNormGene2/batchnorm/ReadVariableOp_12V
)BatchNormGene2/batchnorm/ReadVariableOp_2)BatchNormGene2/batchnorm/ReadVariableOp_22Z
+BatchNormGene2/batchnorm/mul/ReadVariableOp+BatchNormGene2/batchnorm/mul/ReadVariableOp2^
-BatchNormGeneDecode1/batchnorm/ReadVariableOp-BatchNormGeneDecode1/batchnorm/ReadVariableOp2b
/BatchNormGeneDecode1/batchnorm/ReadVariableOp_1/BatchNormGeneDecode1/batchnorm/ReadVariableOp_12b
/BatchNormGeneDecode1/batchnorm/ReadVariableOp_2/BatchNormGeneDecode1/batchnorm/ReadVariableOp_22f
1BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp1BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp2^
-BatchNormGeneEncode1/batchnorm/ReadVariableOp-BatchNormGeneEncode1/batchnorm/ReadVariableOp2b
/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1/BatchNormGeneEncode1/batchnorm/ReadVariableOp_12b
/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2/BatchNormGeneEncode1/batchnorm/ReadVariableOp_22f
1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp2^
-BatchNormGeneEncode2/batchnorm/ReadVariableOp-BatchNormGeneEncode2/batchnorm/ReadVariableOp2b
/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1/BatchNormGeneEncode2/batchnorm/ReadVariableOp_12b
/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2/BatchNormGeneEncode2/batchnorm/ReadVariableOp_22f
1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp2R
'EmbeddingDimGene/BiasAdd/ReadVariableOp'EmbeddingDimGene/BiasAdd/ReadVariableOp2P
&EmbeddingDimGene/MatMul/ReadVariableOp&EmbeddingDimGene/MatMul/ReadVariableOp2X
*EmbeddingDimGene3/batchnorm/ReadVariableOp*EmbeddingDimGene3/batchnorm/ReadVariableOp2\
,EmbeddingDimGene3/batchnorm/ReadVariableOp_1,EmbeddingDimGene3/batchnorm/ReadVariableOp_12\
,EmbeddingDimGene3/batchnorm/ReadVariableOp_2,EmbeddingDimGene3/batchnorm/ReadVariableOp_22`
.EmbeddingDimGene3/batchnorm/mul/ReadVariableOp.EmbeddingDimGene3/batchnorm/mul/ReadVariableOp2N
%gene_decoder_1/BiasAdd/ReadVariableOp%gene_decoder_1/BiasAdd/ReadVariableOp2L
$gene_decoder_1/MatMul/ReadVariableOp$gene_decoder_1/MatMul/ReadVariableOp2N
%gene_decoder_2/BiasAdd/ReadVariableOp%gene_decoder_2/BiasAdd/ReadVariableOp2L
$gene_decoder_2/MatMul/ReadVariableOp$gene_decoder_2/MatMul/ReadVariableOp2N
%gene_decoder_3/BiasAdd/ReadVariableOp%gene_decoder_3/BiasAdd/ReadVariableOp2L
$gene_decoder_3/MatMul/ReadVariableOp$gene_decoder_3/MatMul/ReadVariableOp2N
%gene_encoder_1/BiasAdd/ReadVariableOp%gene_encoder_1/BiasAdd/ReadVariableOp2L
$gene_encoder_1/MatMul/ReadVariableOp$gene_encoder_1/MatMul/ReadVariableOp2N
%gene_encoder_2/BiasAdd/ReadVariableOp%gene_encoder_2/BiasAdd/ReadVariableOp2L
$gene_encoder_2/MatMul/ReadVariableOp$gene_encoder_2/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
3__inference_EmbeddingDimGene3_layer_call_fn_2021664

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_2019946o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

�
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_2020212

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
�
�
0__inference_BatchNormGene2_layer_call_fn_2021864

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
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
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_BatchNormGene2_layer_call_and_return_conditional_losses_2020110p
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
�

�
K__inference_gene_decoder_1_layer_call_and_return_conditional_losses_2020264

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
�%
�
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_2020075

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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_2019911

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
�

�
K__inference_gene_decoder_3_layer_call_and_return_conditional_losses_2020316

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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_2019864

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
��
� 
E__inference_model_42_layer_call_and_return_conditional_losses_2021431

inputsA
-gene_encoder_1_matmul_readvariableop_resource:
��=
.gene_encoder_1_biasadd_readvariableop_resource:	�K
<batchnormgeneencode1_assignmovingavg_readvariableop_resource:	�M
>batchnormgeneencode1_assignmovingavg_1_readvariableop_resource:	�I
:batchnormgeneencode1_batchnorm_mul_readvariableop_resource:	�E
6batchnormgeneencode1_batchnorm_readvariableop_resource:	�@
-gene_encoder_2_matmul_readvariableop_resource:	�}<
.gene_encoder_2_biasadd_readvariableop_resource:}J
<batchnormgeneencode2_assignmovingavg_readvariableop_resource:}L
>batchnormgeneencode2_assignmovingavg_1_readvariableop_resource:}H
:batchnormgeneencode2_batchnorm_mul_readvariableop_resource:}D
6batchnormgeneencode2_batchnorm_readvariableop_resource:}A
/embeddingdimgene_matmul_readvariableop_resource:}@>
0embeddingdimgene_biasadd_readvariableop_resource:@G
9embeddingdimgene3_assignmovingavg_readvariableop_resource:@I
;embeddingdimgene3_assignmovingavg_1_readvariableop_resource:@E
7embeddingdimgene3_batchnorm_mul_readvariableop_resource:@A
3embeddingdimgene3_batchnorm_readvariableop_resource:@?
-gene_decoder_1_matmul_readvariableop_resource:@}<
.gene_decoder_1_biasadd_readvariableop_resource:}J
<batchnormgenedecode1_assignmovingavg_readvariableop_resource:}L
>batchnormgenedecode1_assignmovingavg_1_readvariableop_resource:}H
:batchnormgenedecode1_batchnorm_mul_readvariableop_resource:}D
6batchnormgenedecode1_batchnorm_readvariableop_resource:}@
-gene_decoder_2_matmul_readvariableop_resource:	}�=
.gene_decoder_2_biasadd_readvariableop_resource:	�E
6batchnormgene2_assignmovingavg_readvariableop_resource:	�G
8batchnormgene2_assignmovingavg_1_readvariableop_resource:	�C
4batchnormgene2_batchnorm_mul_readvariableop_resource:	�?
0batchnormgene2_batchnorm_readvariableop_resource:	�A
-gene_decoder_3_matmul_readvariableop_resource:
��=
.gene_decoder_3_biasadd_readvariableop_resource:	�
identity��BatchNormGene2/AssignMovingAvg�-BatchNormGene2/AssignMovingAvg/ReadVariableOp� BatchNormGene2/AssignMovingAvg_1�/BatchNormGene2/AssignMovingAvg_1/ReadVariableOp�'BatchNormGene2/batchnorm/ReadVariableOp�+BatchNormGene2/batchnorm/mul/ReadVariableOp�$BatchNormGeneDecode1/AssignMovingAvg�3BatchNormGeneDecode1/AssignMovingAvg/ReadVariableOp�&BatchNormGeneDecode1/AssignMovingAvg_1�5BatchNormGeneDecode1/AssignMovingAvg_1/ReadVariableOp�-BatchNormGeneDecode1/batchnorm/ReadVariableOp�1BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp�$BatchNormGeneEncode1/AssignMovingAvg�3BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOp�&BatchNormGeneEncode1/AssignMovingAvg_1�5BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOp�-BatchNormGeneEncode1/batchnorm/ReadVariableOp�1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp�$BatchNormGeneEncode2/AssignMovingAvg�3BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOp�&BatchNormGeneEncode2/AssignMovingAvg_1�5BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOp�-BatchNormGeneEncode2/batchnorm/ReadVariableOp�1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp�'EmbeddingDimGene/BiasAdd/ReadVariableOp�&EmbeddingDimGene/MatMul/ReadVariableOp�!EmbeddingDimGene3/AssignMovingAvg�0EmbeddingDimGene3/AssignMovingAvg/ReadVariableOp�#EmbeddingDimGene3/AssignMovingAvg_1�2EmbeddingDimGene3/AssignMovingAvg_1/ReadVariableOp�*EmbeddingDimGene3/batchnorm/ReadVariableOp�.EmbeddingDimGene3/batchnorm/mul/ReadVariableOp�%gene_decoder_1/BiasAdd/ReadVariableOp�$gene_decoder_1/MatMul/ReadVariableOp�%gene_decoder_2/BiasAdd/ReadVariableOp�$gene_decoder_2/MatMul/ReadVariableOp�%gene_decoder_3/BiasAdd/ReadVariableOp�$gene_decoder_3/MatMul/ReadVariableOp�%gene_encoder_1/BiasAdd/ReadVariableOp�$gene_encoder_1/MatMul/ReadVariableOp�%gene_encoder_2/BiasAdd/ReadVariableOp�$gene_encoder_2/MatMul/ReadVariableOp�
$gene_encoder_1/MatMul/ReadVariableOpReadVariableOp-gene_encoder_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gene_encoder_1/MatMulMatMulinputs,gene_encoder_1/MatMul/ReadVariableOp:value:0*
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
&EmbeddingDimGene/MatMul/ReadVariableOpReadVariableOp/embeddingdimgene_matmul_readvariableop_resource*
_output_shapes

:}@*
dtype0�
EmbeddingDimGene/MatMulMatMul(BatchNormGeneEncode2/batchnorm/add_1:z:0.EmbeddingDimGene/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
'EmbeddingDimGene/BiasAdd/ReadVariableOpReadVariableOp0embeddingdimgene_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
EmbeddingDimGene/BiasAddBiasAdd!EmbeddingDimGene/MatMul:product:0/EmbeddingDimGene/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@x
EmbeddingDimGene/SigmoidSigmoid!EmbeddingDimGene/BiasAdd:output:0*
T0*'
_output_shapes
:���������@z
0EmbeddingDimGene3/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
EmbeddingDimGene3/moments/meanMeanEmbeddingDimGene/Sigmoid:y:09EmbeddingDimGene3/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
&EmbeddingDimGene3/moments/StopGradientStopGradient'EmbeddingDimGene3/moments/mean:output:0*
T0*
_output_shapes

:@�
+EmbeddingDimGene3/moments/SquaredDifferenceSquaredDifferenceEmbeddingDimGene/Sigmoid:y:0/EmbeddingDimGene3/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������@~
4EmbeddingDimGene3/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
"EmbeddingDimGene3/moments/varianceMean/EmbeddingDimGene3/moments/SquaredDifference:z:0=EmbeddingDimGene3/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(�
!EmbeddingDimGene3/moments/SqueezeSqueeze'EmbeddingDimGene3/moments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 �
#EmbeddingDimGene3/moments/Squeeze_1Squeeze+EmbeddingDimGene3/moments/variance:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 l
'EmbeddingDimGene3/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
0EmbeddingDimGene3/AssignMovingAvg/ReadVariableOpReadVariableOp9embeddingdimgene3_assignmovingavg_readvariableop_resource*
_output_shapes
:@*
dtype0�
%EmbeddingDimGene3/AssignMovingAvg/subSub8EmbeddingDimGene3/AssignMovingAvg/ReadVariableOp:value:0*EmbeddingDimGene3/moments/Squeeze:output:0*
T0*
_output_shapes
:@�
%EmbeddingDimGene3/AssignMovingAvg/mulMul)EmbeddingDimGene3/AssignMovingAvg/sub:z:00EmbeddingDimGene3/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
!EmbeddingDimGene3/AssignMovingAvgAssignSubVariableOp9embeddingdimgene3_assignmovingavg_readvariableop_resource)EmbeddingDimGene3/AssignMovingAvg/mul:z:01^EmbeddingDimGene3/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0n
)EmbeddingDimGene3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
2EmbeddingDimGene3/AssignMovingAvg_1/ReadVariableOpReadVariableOp;embeddingdimgene3_assignmovingavg_1_readvariableop_resource*
_output_shapes
:@*
dtype0�
'EmbeddingDimGene3/AssignMovingAvg_1/subSub:EmbeddingDimGene3/AssignMovingAvg_1/ReadVariableOp:value:0,EmbeddingDimGene3/moments/Squeeze_1:output:0*
T0*
_output_shapes
:@�
'EmbeddingDimGene3/AssignMovingAvg_1/mulMul+EmbeddingDimGene3/AssignMovingAvg_1/sub:z:02EmbeddingDimGene3/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
#EmbeddingDimGene3/AssignMovingAvg_1AssignSubVariableOp;embeddingdimgene3_assignmovingavg_1_readvariableop_resource+EmbeddingDimGene3/AssignMovingAvg_1/mul:z:03^EmbeddingDimGene3/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0f
!EmbeddingDimGene3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
EmbeddingDimGene3/batchnorm/addAddV2,EmbeddingDimGene3/moments/Squeeze_1:output:0*EmbeddingDimGene3/batchnorm/add/y:output:0*
T0*
_output_shapes
:@t
!EmbeddingDimGene3/batchnorm/RsqrtRsqrt#EmbeddingDimGene3/batchnorm/add:z:0*
T0*
_output_shapes
:@�
.EmbeddingDimGene3/batchnorm/mul/ReadVariableOpReadVariableOp7embeddingdimgene3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
EmbeddingDimGene3/batchnorm/mulMul%EmbeddingDimGene3/batchnorm/Rsqrt:y:06EmbeddingDimGene3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
!EmbeddingDimGene3/batchnorm/mul_1MulEmbeddingDimGene/Sigmoid:y:0#EmbeddingDimGene3/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
!EmbeddingDimGene3/batchnorm/mul_2Mul*EmbeddingDimGene3/moments/Squeeze:output:0#EmbeddingDimGene3/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
*EmbeddingDimGene3/batchnorm/ReadVariableOpReadVariableOp3embeddingdimgene3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0�
EmbeddingDimGene3/batchnorm/subSub2EmbeddingDimGene3/batchnorm/ReadVariableOp:value:0%EmbeddingDimGene3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
!EmbeddingDimGene3/batchnorm/add_1AddV2%EmbeddingDimGene3/batchnorm/mul_1:z:0#EmbeddingDimGene3/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
$gene_decoder_1/MatMul/ReadVariableOpReadVariableOp-gene_decoder_1_matmul_readvariableop_resource*
_output_shapes

:@}*
dtype0�
gene_decoder_1/MatMulMatMul%EmbeddingDimGene3/batchnorm/add_1:z:0,gene_decoder_1/MatMul/ReadVariableOp:value:0*
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
:����������w
-BatchNormGene2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
BatchNormGene2/moments/meanMeangene_decoder_2/Sigmoid:y:06BatchNormGene2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
#BatchNormGene2/moments/StopGradientStopGradient$BatchNormGene2/moments/mean:output:0*
T0*
_output_shapes
:	��
(BatchNormGene2/moments/SquaredDifferenceSquaredDifferencegene_decoder_2/Sigmoid:y:0,BatchNormGene2/moments/StopGradient:output:0*
T0*(
_output_shapes
:����������{
1BatchNormGene2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
BatchNormGene2/moments/varianceMean,BatchNormGene2/moments/SquaredDifference:z:0:BatchNormGene2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
BatchNormGene2/moments/SqueezeSqueeze$BatchNormGene2/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
 BatchNormGene2/moments/Squeeze_1Squeeze(BatchNormGene2/moments/variance:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 i
$BatchNormGene2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
-BatchNormGene2/AssignMovingAvg/ReadVariableOpReadVariableOp6batchnormgene2_assignmovingavg_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"BatchNormGene2/AssignMovingAvg/subSub5BatchNormGene2/AssignMovingAvg/ReadVariableOp:value:0'BatchNormGene2/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
"BatchNormGene2/AssignMovingAvg/mulMul&BatchNormGene2/AssignMovingAvg/sub:z:0-BatchNormGene2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
BatchNormGene2/AssignMovingAvgAssignSubVariableOp6batchnormgene2_assignmovingavg_readvariableop_resource&BatchNormGene2/AssignMovingAvg/mul:z:0.^BatchNormGene2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0k
&BatchNormGene2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
/BatchNormGene2/AssignMovingAvg_1/ReadVariableOpReadVariableOp8batchnormgene2_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$BatchNormGene2/AssignMovingAvg_1/subSub7BatchNormGene2/AssignMovingAvg_1/ReadVariableOp:value:0)BatchNormGene2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
$BatchNormGene2/AssignMovingAvg_1/mulMul(BatchNormGene2/AssignMovingAvg_1/sub:z:0/BatchNormGene2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
 BatchNormGene2/AssignMovingAvg_1AssignSubVariableOp8batchnormgene2_assignmovingavg_1_readvariableop_resource(BatchNormGene2/AssignMovingAvg_1/mul:z:00^BatchNormGene2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0c
BatchNormGene2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
BatchNormGene2/batchnorm/addAddV2)BatchNormGene2/moments/Squeeze_1:output:0'BatchNormGene2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:�o
BatchNormGene2/batchnorm/RsqrtRsqrt BatchNormGene2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
+BatchNormGene2/batchnorm/mul/ReadVariableOpReadVariableOp4batchnormgene2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BatchNormGene2/batchnorm/mulMul"BatchNormGene2/batchnorm/Rsqrt:y:03BatchNormGene2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
BatchNormGene2/batchnorm/mul_1Mulgene_decoder_2/Sigmoid:y:0 BatchNormGene2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
BatchNormGene2/batchnorm/mul_2Mul'BatchNormGene2/moments/Squeeze:output:0 BatchNormGene2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
'BatchNormGene2/batchnorm/ReadVariableOpReadVariableOp0batchnormgene2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
BatchNormGene2/batchnorm/subSub/BatchNormGene2/batchnorm/ReadVariableOp:value:0"BatchNormGene2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
BatchNormGene2/batchnorm/add_1AddV2"BatchNormGene2/batchnorm/mul_1:z:0 BatchNormGene2/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
$gene_decoder_3/MatMul/ReadVariableOpReadVariableOp-gene_decoder_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
gene_decoder_3/MatMulMatMul"BatchNormGene2/batchnorm/add_1:z:0,gene_decoder_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%gene_decoder_3/BiasAdd/ReadVariableOpReadVariableOp.gene_decoder_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
gene_decoder_3/BiasAddBiasAddgene_decoder_3/MatMul:product:0-gene_decoder_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
gene_decoder_3/SigmoidSigmoidgene_decoder_3/BiasAdd:output:0*
T0*(
_output_shapes
:����������j
IdentityIdentitygene_decoder_3/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^BatchNormGene2/AssignMovingAvg.^BatchNormGene2/AssignMovingAvg/ReadVariableOp!^BatchNormGene2/AssignMovingAvg_10^BatchNormGene2/AssignMovingAvg_1/ReadVariableOp(^BatchNormGene2/batchnorm/ReadVariableOp,^BatchNormGene2/batchnorm/mul/ReadVariableOp%^BatchNormGeneDecode1/AssignMovingAvg4^BatchNormGeneDecode1/AssignMovingAvg/ReadVariableOp'^BatchNormGeneDecode1/AssignMovingAvg_16^BatchNormGeneDecode1/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneDecode1/batchnorm/ReadVariableOp2^BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp%^BatchNormGeneEncode1/AssignMovingAvg4^BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOp'^BatchNormGeneEncode1/AssignMovingAvg_16^BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneEncode1/batchnorm/ReadVariableOp2^BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp%^BatchNormGeneEncode2/AssignMovingAvg4^BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOp'^BatchNormGeneEncode2/AssignMovingAvg_16^BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneEncode2/batchnorm/ReadVariableOp2^BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp(^EmbeddingDimGene/BiasAdd/ReadVariableOp'^EmbeddingDimGene/MatMul/ReadVariableOp"^EmbeddingDimGene3/AssignMovingAvg1^EmbeddingDimGene3/AssignMovingAvg/ReadVariableOp$^EmbeddingDimGene3/AssignMovingAvg_13^EmbeddingDimGene3/AssignMovingAvg_1/ReadVariableOp+^EmbeddingDimGene3/batchnorm/ReadVariableOp/^EmbeddingDimGene3/batchnorm/mul/ReadVariableOp&^gene_decoder_1/BiasAdd/ReadVariableOp%^gene_decoder_1/MatMul/ReadVariableOp&^gene_decoder_2/BiasAdd/ReadVariableOp%^gene_decoder_2/MatMul/ReadVariableOp&^gene_decoder_3/BiasAdd/ReadVariableOp%^gene_decoder_3/MatMul/ReadVariableOp&^gene_encoder_1/BiasAdd/ReadVariableOp%^gene_encoder_1/MatMul/ReadVariableOp&^gene_encoder_2/BiasAdd/ReadVariableOp%^gene_encoder_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
BatchNormGene2/AssignMovingAvgBatchNormGene2/AssignMovingAvg2^
-BatchNormGene2/AssignMovingAvg/ReadVariableOp-BatchNormGene2/AssignMovingAvg/ReadVariableOp2D
 BatchNormGene2/AssignMovingAvg_1 BatchNormGene2/AssignMovingAvg_12b
/BatchNormGene2/AssignMovingAvg_1/ReadVariableOp/BatchNormGene2/AssignMovingAvg_1/ReadVariableOp2R
'BatchNormGene2/batchnorm/ReadVariableOp'BatchNormGene2/batchnorm/ReadVariableOp2Z
+BatchNormGene2/batchnorm/mul/ReadVariableOp+BatchNormGene2/batchnorm/mul/ReadVariableOp2L
$BatchNormGeneDecode1/AssignMovingAvg$BatchNormGeneDecode1/AssignMovingAvg2j
3BatchNormGeneDecode1/AssignMovingAvg/ReadVariableOp3BatchNormGeneDecode1/AssignMovingAvg/ReadVariableOp2P
&BatchNormGeneDecode1/AssignMovingAvg_1&BatchNormGeneDecode1/AssignMovingAvg_12n
5BatchNormGeneDecode1/AssignMovingAvg_1/ReadVariableOp5BatchNormGeneDecode1/AssignMovingAvg_1/ReadVariableOp2^
-BatchNormGeneDecode1/batchnorm/ReadVariableOp-BatchNormGeneDecode1/batchnorm/ReadVariableOp2f
1BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp1BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp2L
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
'EmbeddingDimGene/BiasAdd/ReadVariableOp'EmbeddingDimGene/BiasAdd/ReadVariableOp2P
&EmbeddingDimGene/MatMul/ReadVariableOp&EmbeddingDimGene/MatMul/ReadVariableOp2F
!EmbeddingDimGene3/AssignMovingAvg!EmbeddingDimGene3/AssignMovingAvg2d
0EmbeddingDimGene3/AssignMovingAvg/ReadVariableOp0EmbeddingDimGene3/AssignMovingAvg/ReadVariableOp2J
#EmbeddingDimGene3/AssignMovingAvg_1#EmbeddingDimGene3/AssignMovingAvg_12h
2EmbeddingDimGene3/AssignMovingAvg_1/ReadVariableOp2EmbeddingDimGene3/AssignMovingAvg_1/ReadVariableOp2X
*EmbeddingDimGene3/batchnorm/ReadVariableOp*EmbeddingDimGene3/batchnorm/ReadVariableOp2`
.EmbeddingDimGene3/batchnorm/mul/ReadVariableOp.EmbeddingDimGene3/batchnorm/mul/ReadVariableOp2N
%gene_decoder_1/BiasAdd/ReadVariableOp%gene_decoder_1/BiasAdd/ReadVariableOp2L
$gene_decoder_1/MatMul/ReadVariableOp$gene_decoder_1/MatMul/ReadVariableOp2N
%gene_decoder_2/BiasAdd/ReadVariableOp%gene_decoder_2/BiasAdd/ReadVariableOp2L
$gene_decoder_2/MatMul/ReadVariableOp$gene_decoder_2/MatMul/ReadVariableOp2N
%gene_decoder_3/BiasAdd/ReadVariableOp%gene_decoder_3/BiasAdd/ReadVariableOp2L
$gene_decoder_3/MatMul/ReadVariableOp$gene_decoder_3/MatMul/ReadVariableOp2N
%gene_encoder_1/BiasAdd/ReadVariableOp%gene_encoder_1/BiasAdd/ReadVariableOp2L
$gene_encoder_1/MatMul/ReadVariableOp$gene_encoder_1/MatMul/ReadVariableOp2N
%gene_encoder_2/BiasAdd/ReadVariableOp%gene_encoder_2/BiasAdd/ReadVariableOp2L
$gene_encoder_2/MatMul/ReadVariableOp$gene_encoder_2/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_2021797

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
N__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_2019993

inputs5
'assignmovingavg_readvariableop_resource:@7
)assignmovingavg_1_readvariableop_resource:@3
%batchnorm_mul_readvariableop_resource:@/
!batchnorm_readvariableop_resource:@
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:@�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������@l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:@*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:@*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:@*
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
:@*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:@x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:@�
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
:@*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:@~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:@�
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
:@P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:@~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������@h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:@v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:@r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�F
�
E__inference_model_42_layer_call_and_return_conditional_losses_2020323

inputs*
gene_encoder_1_2020187:
��%
gene_encoder_1_2020189:	�+
batchnormgeneencode1_2020192:	�+
batchnormgeneencode1_2020194:	�+
batchnormgeneencode1_2020196:	�+
batchnormgeneencode1_2020198:	�)
gene_encoder_2_2020213:	�}$
gene_encoder_2_2020215:}*
batchnormgeneencode2_2020218:}*
batchnormgeneencode2_2020220:}*
batchnormgeneencode2_2020222:}*
batchnormgeneencode2_2020224:}*
embeddingdimgene_2020239:}@&
embeddingdimgene_2020241:@'
embeddingdimgene3_2020244:@'
embeddingdimgene3_2020246:@'
embeddingdimgene3_2020248:@'
embeddingdimgene3_2020250:@(
gene_decoder_1_2020265:@}$
gene_decoder_1_2020267:}*
batchnormgenedecode1_2020270:}*
batchnormgenedecode1_2020272:}*
batchnormgenedecode1_2020274:}*
batchnormgenedecode1_2020276:})
gene_decoder_2_2020291:	}�%
gene_decoder_2_2020293:	�%
batchnormgene2_2020296:	�%
batchnormgene2_2020298:	�%
batchnormgene2_2020300:	�%
batchnormgene2_2020302:	�*
gene_decoder_3_2020317:
��%
gene_decoder_3_2020319:	�
identity��&BatchNormGene2/StatefulPartitionedCall�,BatchNormGeneDecode1/StatefulPartitionedCall�,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�(EmbeddingDimGene/StatefulPartitionedCall�)EmbeddingDimGene3/StatefulPartitionedCall�&gene_decoder_1/StatefulPartitionedCall�&gene_decoder_2/StatefulPartitionedCall�&gene_decoder_3/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputsgene_encoder_1_2020187gene_encoder_1_2020189*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_2020186�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_2020192batchnormgeneencode1_2020194batchnormgeneencode1_2020196batchnormgeneencode1_2020198*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_2019782�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_2020213gene_encoder_2_2020215*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_2020212�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_2020218batchnormgeneencode2_2020220batchnormgeneencode2_2020222batchnormgeneencode2_2020224*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_2019864�
(EmbeddingDimGene/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:0embeddingdimgene_2020239embeddingdimgene_2020241*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_EmbeddingDimGene_layer_call_and_return_conditional_losses_2020238�
)EmbeddingDimGene3/StatefulPartitionedCallStatefulPartitionedCall1EmbeddingDimGene/StatefulPartitionedCall:output:0embeddingdimgene3_2020244embeddingdimgene3_2020246embeddingdimgene3_2020248embeddingdimgene3_2020250*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_2019946�
&gene_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimGene3/StatefulPartitionedCall:output:0gene_decoder_1_2020265gene_decoder_1_2020267*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_decoder_1_layer_call_and_return_conditional_losses_2020264�
,BatchNormGeneDecode1/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_1/StatefulPartitionedCall:output:0batchnormgenedecode1_2020270batchnormgenedecode1_2020272batchnormgenedecode1_2020274batchnormgenedecode1_2020276*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������}*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_2020028�
&gene_decoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode1/StatefulPartitionedCall:output:0gene_decoder_2_2020291gene_decoder_2_2020293*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_decoder_2_layer_call_and_return_conditional_losses_2020290�
&BatchNormGene2/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_2/StatefulPartitionedCall:output:0batchnormgene2_2020296batchnormgene2_2020298batchnormgene2_2020300batchnormgene2_2020302*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_BatchNormGene2_layer_call_and_return_conditional_losses_2020110�
&gene_decoder_3/StatefulPartitionedCallStatefulPartitionedCall/BatchNormGene2/StatefulPartitionedCall:output:0gene_decoder_3_2020317gene_decoder_3_2020319*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_decoder_3_layer_call_and_return_conditional_losses_2020316
IdentityIdentity/gene_decoder_3/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp'^BatchNormGene2/StatefulPartitionedCall-^BatchNormGeneDecode1/StatefulPartitionedCall-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall)^EmbeddingDimGene/StatefulPartitionedCall*^EmbeddingDimGene3/StatefulPartitionedCall'^gene_decoder_1/StatefulPartitionedCall'^gene_decoder_2/StatefulPartitionedCall'^gene_decoder_3/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2P
&BatchNormGene2/StatefulPartitionedCall&BatchNormGene2/StatefulPartitionedCall2\
,BatchNormGeneDecode1/StatefulPartitionedCall,BatchNormGeneDecode1/StatefulPartitionedCall2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2T
(EmbeddingDimGene/StatefulPartitionedCall(EmbeddingDimGene/StatefulPartitionedCall2V
)EmbeddingDimGene3/StatefulPartitionedCall)EmbeddingDimGene3/StatefulPartitionedCall2P
&gene_decoder_1/StatefulPartitionedCall&gene_decoder_1/StatefulPartitionedCall2P
&gene_decoder_2/StatefulPartitionedCall&gene_decoder_2/StatefulPartitionedCall2P
&gene_decoder_3/StatefulPartitionedCall&gene_decoder_3/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
0__inference_gene_encoder_1_layer_call_fn_2021440

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
*-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_2020186p
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
��
�!
"__inference__wrapped_model_2019758
gene_input_layerJ
6model_42_gene_encoder_1_matmul_readvariableop_resource:
��F
7model_42_gene_encoder_1_biasadd_readvariableop_resource:	�N
?model_42_batchnormgeneencode1_batchnorm_readvariableop_resource:	�R
Cmodel_42_batchnormgeneencode1_batchnorm_mul_readvariableop_resource:	�P
Amodel_42_batchnormgeneencode1_batchnorm_readvariableop_1_resource:	�P
Amodel_42_batchnormgeneencode1_batchnorm_readvariableop_2_resource:	�I
6model_42_gene_encoder_2_matmul_readvariableop_resource:	�}E
7model_42_gene_encoder_2_biasadd_readvariableop_resource:}M
?model_42_batchnormgeneencode2_batchnorm_readvariableop_resource:}Q
Cmodel_42_batchnormgeneencode2_batchnorm_mul_readvariableop_resource:}O
Amodel_42_batchnormgeneencode2_batchnorm_readvariableop_1_resource:}O
Amodel_42_batchnormgeneencode2_batchnorm_readvariableop_2_resource:}J
8model_42_embeddingdimgene_matmul_readvariableop_resource:}@G
9model_42_embeddingdimgene_biasadd_readvariableop_resource:@J
<model_42_embeddingdimgene3_batchnorm_readvariableop_resource:@N
@model_42_embeddingdimgene3_batchnorm_mul_readvariableop_resource:@L
>model_42_embeddingdimgene3_batchnorm_readvariableop_1_resource:@L
>model_42_embeddingdimgene3_batchnorm_readvariableop_2_resource:@H
6model_42_gene_decoder_1_matmul_readvariableop_resource:@}E
7model_42_gene_decoder_1_biasadd_readvariableop_resource:}M
?model_42_batchnormgenedecode1_batchnorm_readvariableop_resource:}Q
Cmodel_42_batchnormgenedecode1_batchnorm_mul_readvariableop_resource:}O
Amodel_42_batchnormgenedecode1_batchnorm_readvariableop_1_resource:}O
Amodel_42_batchnormgenedecode1_batchnorm_readvariableop_2_resource:}I
6model_42_gene_decoder_2_matmul_readvariableop_resource:	}�F
7model_42_gene_decoder_2_biasadd_readvariableop_resource:	�H
9model_42_batchnormgene2_batchnorm_readvariableop_resource:	�L
=model_42_batchnormgene2_batchnorm_mul_readvariableop_resource:	�J
;model_42_batchnormgene2_batchnorm_readvariableop_1_resource:	�J
;model_42_batchnormgene2_batchnorm_readvariableop_2_resource:	�J
6model_42_gene_decoder_3_matmul_readvariableop_resource:
��F
7model_42_gene_decoder_3_biasadd_readvariableop_resource:	�
identity��0model_42/BatchNormGene2/batchnorm/ReadVariableOp�2model_42/BatchNormGene2/batchnorm/ReadVariableOp_1�2model_42/BatchNormGene2/batchnorm/ReadVariableOp_2�4model_42/BatchNormGene2/batchnorm/mul/ReadVariableOp�6model_42/BatchNormGeneDecode1/batchnorm/ReadVariableOp�8model_42/BatchNormGeneDecode1/batchnorm/ReadVariableOp_1�8model_42/BatchNormGeneDecode1/batchnorm/ReadVariableOp_2�:model_42/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp�6model_42/BatchNormGeneEncode1/batchnorm/ReadVariableOp�8model_42/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1�8model_42/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2�:model_42/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp�6model_42/BatchNormGeneEncode2/batchnorm/ReadVariableOp�8model_42/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1�8model_42/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2�:model_42/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp�0model_42/EmbeddingDimGene/BiasAdd/ReadVariableOp�/model_42/EmbeddingDimGene/MatMul/ReadVariableOp�3model_42/EmbeddingDimGene3/batchnorm/ReadVariableOp�5model_42/EmbeddingDimGene3/batchnorm/ReadVariableOp_1�5model_42/EmbeddingDimGene3/batchnorm/ReadVariableOp_2�7model_42/EmbeddingDimGene3/batchnorm/mul/ReadVariableOp�.model_42/gene_decoder_1/BiasAdd/ReadVariableOp�-model_42/gene_decoder_1/MatMul/ReadVariableOp�.model_42/gene_decoder_2/BiasAdd/ReadVariableOp�-model_42/gene_decoder_2/MatMul/ReadVariableOp�.model_42/gene_decoder_3/BiasAdd/ReadVariableOp�-model_42/gene_decoder_3/MatMul/ReadVariableOp�.model_42/gene_encoder_1/BiasAdd/ReadVariableOp�-model_42/gene_encoder_1/MatMul/ReadVariableOp�.model_42/gene_encoder_2/BiasAdd/ReadVariableOp�-model_42/gene_encoder_2/MatMul/ReadVariableOp�
-model_42/gene_encoder_1/MatMul/ReadVariableOpReadVariableOp6model_42_gene_encoder_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model_42/gene_encoder_1/MatMulMatMulgene_input_layer5model_42/gene_encoder_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.model_42/gene_encoder_1/BiasAdd/ReadVariableOpReadVariableOp7model_42_gene_encoder_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_42/gene_encoder_1/BiasAddBiasAdd(model_42/gene_encoder_1/MatMul:product:06model_42/gene_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
model_42/gene_encoder_1/SigmoidSigmoid(model_42/gene_encoder_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
6model_42/BatchNormGeneEncode1/batchnorm/ReadVariableOpReadVariableOp?model_42_batchnormgeneencode1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0r
-model_42/BatchNormGeneEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
+model_42/BatchNormGeneEncode1/batchnorm/addAddV2>model_42/BatchNormGeneEncode1/batchnorm/ReadVariableOp:value:06model_42/BatchNormGeneEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
-model_42/BatchNormGeneEncode1/batchnorm/RsqrtRsqrt/model_42/BatchNormGeneEncode1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
:model_42/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_42_batchnormgeneencode1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+model_42/BatchNormGeneEncode1/batchnorm/mulMul1model_42/BatchNormGeneEncode1/batchnorm/Rsqrt:y:0Bmodel_42/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
-model_42/BatchNormGeneEncode1/batchnorm/mul_1Mul#model_42/gene_encoder_1/Sigmoid:y:0/model_42/BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
8model_42/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_42_batchnormgeneencode1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
-model_42/BatchNormGeneEncode1/batchnorm/mul_2Mul@model_42/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1:value:0/model_42/BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
8model_42/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_42_batchnormgeneencode1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
+model_42/BatchNormGeneEncode1/batchnorm/subSub@model_42/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2:value:01model_42/BatchNormGeneEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
-model_42/BatchNormGeneEncode1/batchnorm/add_1AddV21model_42/BatchNormGeneEncode1/batchnorm/mul_1:z:0/model_42/BatchNormGeneEncode1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
-model_42/gene_encoder_2/MatMul/ReadVariableOpReadVariableOp6model_42_gene_encoder_2_matmul_readvariableop_resource*
_output_shapes
:	�}*
dtype0�
model_42/gene_encoder_2/MatMulMatMul1model_42/BatchNormGeneEncode1/batchnorm/add_1:z:05model_42/gene_encoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}�
.model_42/gene_encoder_2/BiasAdd/ReadVariableOpReadVariableOp7model_42_gene_encoder_2_biasadd_readvariableop_resource*
_output_shapes
:}*
dtype0�
model_42/gene_encoder_2/BiasAddBiasAdd(model_42/gene_encoder_2/MatMul:product:06model_42/gene_encoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}�
model_42/gene_encoder_2/SigmoidSigmoid(model_42/gene_encoder_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������}�
6model_42/BatchNormGeneEncode2/batchnorm/ReadVariableOpReadVariableOp?model_42_batchnormgeneencode2_batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0r
-model_42/BatchNormGeneEncode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
+model_42/BatchNormGeneEncode2/batchnorm/addAddV2>model_42/BatchNormGeneEncode2/batchnorm/ReadVariableOp:value:06model_42/BatchNormGeneEncode2/batchnorm/add/y:output:0*
T0*
_output_shapes
:}�
-model_42/BatchNormGeneEncode2/batchnorm/RsqrtRsqrt/model_42/BatchNormGeneEncode2/batchnorm/add:z:0*
T0*
_output_shapes
:}�
:model_42/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_42_batchnormgeneencode2_batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0�
+model_42/BatchNormGeneEncode2/batchnorm/mulMul1model_42/BatchNormGeneEncode2/batchnorm/Rsqrt:y:0Bmodel_42/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}�
-model_42/BatchNormGeneEncode2/batchnorm/mul_1Mul#model_42/gene_encoder_2/Sigmoid:y:0/model_42/BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������}�
8model_42/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_42_batchnormgeneencode2_batchnorm_readvariableop_1_resource*
_output_shapes
:}*
dtype0�
-model_42/BatchNormGeneEncode2/batchnorm/mul_2Mul@model_42/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1:value:0/model_42/BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*
_output_shapes
:}�
8model_42/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_42_batchnormgeneencode2_batchnorm_readvariableop_2_resource*
_output_shapes
:}*
dtype0�
+model_42/BatchNormGeneEncode2/batchnorm/subSub@model_42/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2:value:01model_42/BatchNormGeneEncode2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:}�
-model_42/BatchNormGeneEncode2/batchnorm/add_1AddV21model_42/BatchNormGeneEncode2/batchnorm/mul_1:z:0/model_42/BatchNormGeneEncode2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}�
/model_42/EmbeddingDimGene/MatMul/ReadVariableOpReadVariableOp8model_42_embeddingdimgene_matmul_readvariableop_resource*
_output_shapes

:}@*
dtype0�
 model_42/EmbeddingDimGene/MatMulMatMul1model_42/BatchNormGeneEncode2/batchnorm/add_1:z:07model_42/EmbeddingDimGene/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
0model_42/EmbeddingDimGene/BiasAdd/ReadVariableOpReadVariableOp9model_42_embeddingdimgene_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
!model_42/EmbeddingDimGene/BiasAddBiasAdd*model_42/EmbeddingDimGene/MatMul:product:08model_42/EmbeddingDimGene/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
!model_42/EmbeddingDimGene/SigmoidSigmoid*model_42/EmbeddingDimGene/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
3model_42/EmbeddingDimGene3/batchnorm/ReadVariableOpReadVariableOp<model_42_embeddingdimgene3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0o
*model_42/EmbeddingDimGene3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
(model_42/EmbeddingDimGene3/batchnorm/addAddV2;model_42/EmbeddingDimGene3/batchnorm/ReadVariableOp:value:03model_42/EmbeddingDimGene3/batchnorm/add/y:output:0*
T0*
_output_shapes
:@�
*model_42/EmbeddingDimGene3/batchnorm/RsqrtRsqrt,model_42/EmbeddingDimGene3/batchnorm/add:z:0*
T0*
_output_shapes
:@�
7model_42/EmbeddingDimGene3/batchnorm/mul/ReadVariableOpReadVariableOp@model_42_embeddingdimgene3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
(model_42/EmbeddingDimGene3/batchnorm/mulMul.model_42/EmbeddingDimGene3/batchnorm/Rsqrt:y:0?model_42/EmbeddingDimGene3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
*model_42/EmbeddingDimGene3/batchnorm/mul_1Mul%model_42/EmbeddingDimGene/Sigmoid:y:0,model_42/EmbeddingDimGene3/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
5model_42/EmbeddingDimGene3/batchnorm/ReadVariableOp_1ReadVariableOp>model_42_embeddingdimgene3_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
*model_42/EmbeddingDimGene3/batchnorm/mul_2Mul=model_42/EmbeddingDimGene3/batchnorm/ReadVariableOp_1:value:0,model_42/EmbeddingDimGene3/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
5model_42/EmbeddingDimGene3/batchnorm/ReadVariableOp_2ReadVariableOp>model_42_embeddingdimgene3_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
(model_42/EmbeddingDimGene3/batchnorm/subSub=model_42/EmbeddingDimGene3/batchnorm/ReadVariableOp_2:value:0.model_42/EmbeddingDimGene3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
*model_42/EmbeddingDimGene3/batchnorm/add_1AddV2.model_42/EmbeddingDimGene3/batchnorm/mul_1:z:0,model_42/EmbeddingDimGene3/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@�
-model_42/gene_decoder_1/MatMul/ReadVariableOpReadVariableOp6model_42_gene_decoder_1_matmul_readvariableop_resource*
_output_shapes

:@}*
dtype0�
model_42/gene_decoder_1/MatMulMatMul.model_42/EmbeddingDimGene3/batchnorm/add_1:z:05model_42/gene_decoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}�
.model_42/gene_decoder_1/BiasAdd/ReadVariableOpReadVariableOp7model_42_gene_decoder_1_biasadd_readvariableop_resource*
_output_shapes
:}*
dtype0�
model_42/gene_decoder_1/BiasAddBiasAdd(model_42/gene_decoder_1/MatMul:product:06model_42/gene_decoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}�
model_42/gene_decoder_1/SigmoidSigmoid(model_42/gene_decoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������}�
6model_42/BatchNormGeneDecode1/batchnorm/ReadVariableOpReadVariableOp?model_42_batchnormgenedecode1_batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0r
-model_42/BatchNormGeneDecode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
+model_42/BatchNormGeneDecode1/batchnorm/addAddV2>model_42/BatchNormGeneDecode1/batchnorm/ReadVariableOp:value:06model_42/BatchNormGeneDecode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:}�
-model_42/BatchNormGeneDecode1/batchnorm/RsqrtRsqrt/model_42/BatchNormGeneDecode1/batchnorm/add:z:0*
T0*
_output_shapes
:}�
:model_42/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_42_batchnormgenedecode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0�
+model_42/BatchNormGeneDecode1/batchnorm/mulMul1model_42/BatchNormGeneDecode1/batchnorm/Rsqrt:y:0Bmodel_42/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}�
-model_42/BatchNormGeneDecode1/batchnorm/mul_1Mul#model_42/gene_decoder_1/Sigmoid:y:0/model_42/BatchNormGeneDecode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������}�
8model_42/BatchNormGeneDecode1/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_42_batchnormgenedecode1_batchnorm_readvariableop_1_resource*
_output_shapes
:}*
dtype0�
-model_42/BatchNormGeneDecode1/batchnorm/mul_2Mul@model_42/BatchNormGeneDecode1/batchnorm/ReadVariableOp_1:value:0/model_42/BatchNormGeneDecode1/batchnorm/mul:z:0*
T0*
_output_shapes
:}�
8model_42/BatchNormGeneDecode1/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_42_batchnormgenedecode1_batchnorm_readvariableop_2_resource*
_output_shapes
:}*
dtype0�
+model_42/BatchNormGeneDecode1/batchnorm/subSub@model_42/BatchNormGeneDecode1/batchnorm/ReadVariableOp_2:value:01model_42/BatchNormGeneDecode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:}�
-model_42/BatchNormGeneDecode1/batchnorm/add_1AddV21model_42/BatchNormGeneDecode1/batchnorm/mul_1:z:0/model_42/BatchNormGeneDecode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}�
-model_42/gene_decoder_2/MatMul/ReadVariableOpReadVariableOp6model_42_gene_decoder_2_matmul_readvariableop_resource*
_output_shapes
:	}�*
dtype0�
model_42/gene_decoder_2/MatMulMatMul1model_42/BatchNormGeneDecode1/batchnorm/add_1:z:05model_42/gene_decoder_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.model_42/gene_decoder_2/BiasAdd/ReadVariableOpReadVariableOp7model_42_gene_decoder_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_42/gene_decoder_2/BiasAddBiasAdd(model_42/gene_decoder_2/MatMul:product:06model_42/gene_decoder_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
model_42/gene_decoder_2/SigmoidSigmoid(model_42/gene_decoder_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
0model_42/BatchNormGene2/batchnorm/ReadVariableOpReadVariableOp9model_42_batchnormgene2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0l
'model_42/BatchNormGene2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
%model_42/BatchNormGene2/batchnorm/addAddV28model_42/BatchNormGene2/batchnorm/ReadVariableOp:value:00model_42/BatchNormGene2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
'model_42/BatchNormGene2/batchnorm/RsqrtRsqrt)model_42/BatchNormGene2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
4model_42/BatchNormGene2/batchnorm/mul/ReadVariableOpReadVariableOp=model_42_batchnormgene2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%model_42/BatchNormGene2/batchnorm/mulMul+model_42/BatchNormGene2/batchnorm/Rsqrt:y:0<model_42/BatchNormGene2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
'model_42/BatchNormGene2/batchnorm/mul_1Mul#model_42/gene_decoder_2/Sigmoid:y:0)model_42/BatchNormGene2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
2model_42/BatchNormGene2/batchnorm/ReadVariableOp_1ReadVariableOp;model_42_batchnormgene2_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
'model_42/BatchNormGene2/batchnorm/mul_2Mul:model_42/BatchNormGene2/batchnorm/ReadVariableOp_1:value:0)model_42/BatchNormGene2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
2model_42/BatchNormGene2/batchnorm/ReadVariableOp_2ReadVariableOp;model_42_batchnormgene2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
%model_42/BatchNormGene2/batchnorm/subSub:model_42/BatchNormGene2/batchnorm/ReadVariableOp_2:value:0+model_42/BatchNormGene2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
'model_42/BatchNormGene2/batchnorm/add_1AddV2+model_42/BatchNormGene2/batchnorm/mul_1:z:0)model_42/BatchNormGene2/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
-model_42/gene_decoder_3/MatMul/ReadVariableOpReadVariableOp6model_42_gene_decoder_3_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model_42/gene_decoder_3/MatMulMatMul+model_42/BatchNormGene2/batchnorm/add_1:z:05model_42/gene_decoder_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.model_42/gene_decoder_3/BiasAdd/ReadVariableOpReadVariableOp7model_42_gene_decoder_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_42/gene_decoder_3/BiasAddBiasAdd(model_42/gene_decoder_3/MatMul:product:06model_42/gene_decoder_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
model_42/gene_decoder_3/SigmoidSigmoid(model_42/gene_decoder_3/BiasAdd:output:0*
T0*(
_output_shapes
:����������s
IdentityIdentity#model_42/gene_decoder_3/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp1^model_42/BatchNormGene2/batchnorm/ReadVariableOp3^model_42/BatchNormGene2/batchnorm/ReadVariableOp_13^model_42/BatchNormGene2/batchnorm/ReadVariableOp_25^model_42/BatchNormGene2/batchnorm/mul/ReadVariableOp7^model_42/BatchNormGeneDecode1/batchnorm/ReadVariableOp9^model_42/BatchNormGeneDecode1/batchnorm/ReadVariableOp_19^model_42/BatchNormGeneDecode1/batchnorm/ReadVariableOp_2;^model_42/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp7^model_42/BatchNormGeneEncode1/batchnorm/ReadVariableOp9^model_42/BatchNormGeneEncode1/batchnorm/ReadVariableOp_19^model_42/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2;^model_42/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp7^model_42/BatchNormGeneEncode2/batchnorm/ReadVariableOp9^model_42/BatchNormGeneEncode2/batchnorm/ReadVariableOp_19^model_42/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2;^model_42/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp1^model_42/EmbeddingDimGene/BiasAdd/ReadVariableOp0^model_42/EmbeddingDimGene/MatMul/ReadVariableOp4^model_42/EmbeddingDimGene3/batchnorm/ReadVariableOp6^model_42/EmbeddingDimGene3/batchnorm/ReadVariableOp_16^model_42/EmbeddingDimGene3/batchnorm/ReadVariableOp_28^model_42/EmbeddingDimGene3/batchnorm/mul/ReadVariableOp/^model_42/gene_decoder_1/BiasAdd/ReadVariableOp.^model_42/gene_decoder_1/MatMul/ReadVariableOp/^model_42/gene_decoder_2/BiasAdd/ReadVariableOp.^model_42/gene_decoder_2/MatMul/ReadVariableOp/^model_42/gene_decoder_3/BiasAdd/ReadVariableOp.^model_42/gene_decoder_3/MatMul/ReadVariableOp/^model_42/gene_encoder_1/BiasAdd/ReadVariableOp.^model_42/gene_encoder_1/MatMul/ReadVariableOp/^model_42/gene_encoder_2/BiasAdd/ReadVariableOp.^model_42/gene_encoder_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2d
0model_42/BatchNormGene2/batchnorm/ReadVariableOp0model_42/BatchNormGene2/batchnorm/ReadVariableOp2h
2model_42/BatchNormGene2/batchnorm/ReadVariableOp_12model_42/BatchNormGene2/batchnorm/ReadVariableOp_12h
2model_42/BatchNormGene2/batchnorm/ReadVariableOp_22model_42/BatchNormGene2/batchnorm/ReadVariableOp_22l
4model_42/BatchNormGene2/batchnorm/mul/ReadVariableOp4model_42/BatchNormGene2/batchnorm/mul/ReadVariableOp2p
6model_42/BatchNormGeneDecode1/batchnorm/ReadVariableOp6model_42/BatchNormGeneDecode1/batchnorm/ReadVariableOp2t
8model_42/BatchNormGeneDecode1/batchnorm/ReadVariableOp_18model_42/BatchNormGeneDecode1/batchnorm/ReadVariableOp_12t
8model_42/BatchNormGeneDecode1/batchnorm/ReadVariableOp_28model_42/BatchNormGeneDecode1/batchnorm/ReadVariableOp_22x
:model_42/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp:model_42/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp2p
6model_42/BatchNormGeneEncode1/batchnorm/ReadVariableOp6model_42/BatchNormGeneEncode1/batchnorm/ReadVariableOp2t
8model_42/BatchNormGeneEncode1/batchnorm/ReadVariableOp_18model_42/BatchNormGeneEncode1/batchnorm/ReadVariableOp_12t
8model_42/BatchNormGeneEncode1/batchnorm/ReadVariableOp_28model_42/BatchNormGeneEncode1/batchnorm/ReadVariableOp_22x
:model_42/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp:model_42/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp2p
6model_42/BatchNormGeneEncode2/batchnorm/ReadVariableOp6model_42/BatchNormGeneEncode2/batchnorm/ReadVariableOp2t
8model_42/BatchNormGeneEncode2/batchnorm/ReadVariableOp_18model_42/BatchNormGeneEncode2/batchnorm/ReadVariableOp_12t
8model_42/BatchNormGeneEncode2/batchnorm/ReadVariableOp_28model_42/BatchNormGeneEncode2/batchnorm/ReadVariableOp_22x
:model_42/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:model_42/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp2d
0model_42/EmbeddingDimGene/BiasAdd/ReadVariableOp0model_42/EmbeddingDimGene/BiasAdd/ReadVariableOp2b
/model_42/EmbeddingDimGene/MatMul/ReadVariableOp/model_42/EmbeddingDimGene/MatMul/ReadVariableOp2j
3model_42/EmbeddingDimGene3/batchnorm/ReadVariableOp3model_42/EmbeddingDimGene3/batchnorm/ReadVariableOp2n
5model_42/EmbeddingDimGene3/batchnorm/ReadVariableOp_15model_42/EmbeddingDimGene3/batchnorm/ReadVariableOp_12n
5model_42/EmbeddingDimGene3/batchnorm/ReadVariableOp_25model_42/EmbeddingDimGene3/batchnorm/ReadVariableOp_22r
7model_42/EmbeddingDimGene3/batchnorm/mul/ReadVariableOp7model_42/EmbeddingDimGene3/batchnorm/mul/ReadVariableOp2`
.model_42/gene_decoder_1/BiasAdd/ReadVariableOp.model_42/gene_decoder_1/BiasAdd/ReadVariableOp2^
-model_42/gene_decoder_1/MatMul/ReadVariableOp-model_42/gene_decoder_1/MatMul/ReadVariableOp2`
.model_42/gene_decoder_2/BiasAdd/ReadVariableOp.model_42/gene_decoder_2/BiasAdd/ReadVariableOp2^
-model_42/gene_decoder_2/MatMul/ReadVariableOp-model_42/gene_decoder_2/MatMul/ReadVariableOp2`
.model_42/gene_decoder_3/BiasAdd/ReadVariableOp.model_42/gene_decoder_3/BiasAdd/ReadVariableOp2^
-model_42/gene_decoder_3/MatMul/ReadVariableOp-model_42/gene_decoder_3/MatMul/ReadVariableOp2`
.model_42/gene_encoder_1/BiasAdd/ReadVariableOp.model_42/gene_encoder_1/BiasAdd/ReadVariableOp2^
-model_42/gene_encoder_1/MatMul/ReadVariableOp-model_42/gene_encoder_1/MatMul/ReadVariableOp2`
.model_42/gene_encoder_2/BiasAdd/ReadVariableOp.model_42/gene_encoder_2/BiasAdd/ReadVariableOp2^
-model_42/gene_encoder_2/MatMul/ReadVariableOp-model_42/gene_encoder_2/MatMul/ReadVariableOp:Z V
(
_output_shapes
:����������
*
_user_specified_namegene_input_layer
�%
�
K__inference_BatchNormGene2_layer_call_and_return_conditional_losses_2021931

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
K__inference_gene_decoder_1_layer_call_and_return_conditional_losses_2021751

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
�
�
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_2021497

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
�

�
M__inference_EmbeddingDimGene_layer_call_and_return_conditional_losses_2021651

inputs0
matmul_readvariableop_resource:}@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:}@*
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
_construction_contextkEagerRuntime**
_input_shapes
:���������}: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������}
 
_user_specified_nameinputs
�
�
2__inference_EmbeddingDimGene_layer_call_fn_2021640

inputs
unknown:}@
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
*-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_EmbeddingDimGene_layer_call_and_return_conditional_losses_2020238o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
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
�%
�
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_2021531

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
K__inference_BatchNormGene2_layer_call_and_return_conditional_losses_2021897

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
˥
�&
 __inference__traced_save_2022223
file_prefix4
0savev2_gene_encoder_1_kernel_read_readvariableop2
.savev2_gene_encoder_1_bias_read_readvariableop9
5savev2_batchnormgeneencode1_gamma_read_readvariableop8
4savev2_batchnormgeneencode1_beta_read_readvariableop?
;savev2_batchnormgeneencode1_moving_mean_read_readvariableopC
?savev2_batchnormgeneencode1_moving_variance_read_readvariableop4
0savev2_gene_encoder_2_kernel_read_readvariableop2
.savev2_gene_encoder_2_bias_read_readvariableop9
5savev2_batchnormgeneencode2_gamma_read_readvariableop8
4savev2_batchnormgeneencode2_beta_read_readvariableop?
;savev2_batchnormgeneencode2_moving_mean_read_readvariableopC
?savev2_batchnormgeneencode2_moving_variance_read_readvariableop6
2savev2_embeddingdimgene_kernel_read_readvariableop4
0savev2_embeddingdimgene_bias_read_readvariableop6
2savev2_embeddingdimgene3_gamma_read_readvariableop5
1savev2_embeddingdimgene3_beta_read_readvariableop<
8savev2_embeddingdimgene3_moving_mean_read_readvariableop@
<savev2_embeddingdimgene3_moving_variance_read_readvariableop4
0savev2_gene_decoder_1_kernel_read_readvariableop2
.savev2_gene_decoder_1_bias_read_readvariableop9
5savev2_batchnormgenedecode1_gamma_read_readvariableop8
4savev2_batchnormgenedecode1_beta_read_readvariableop?
;savev2_batchnormgenedecode1_moving_mean_read_readvariableopC
?savev2_batchnormgenedecode1_moving_variance_read_readvariableop4
0savev2_gene_decoder_2_kernel_read_readvariableop2
.savev2_gene_decoder_2_bias_read_readvariableop3
/savev2_batchnormgene2_gamma_read_readvariableop2
.savev2_batchnormgene2_beta_read_readvariableop9
5savev2_batchnormgene2_moving_mean_read_readvariableop=
9savev2_batchnormgene2_moving_variance_read_readvariableop4
0savev2_gene_decoder_3_kernel_read_readvariableop2
.savev2_gene_decoder_3_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop;
7savev2_adam_gene_encoder_1_kernel_m_read_readvariableop9
5savev2_adam_gene_encoder_1_bias_m_read_readvariableop@
<savev2_adam_batchnormgeneencode1_gamma_m_read_readvariableop?
;savev2_adam_batchnormgeneencode1_beta_m_read_readvariableop;
7savev2_adam_gene_encoder_2_kernel_m_read_readvariableop9
5savev2_adam_gene_encoder_2_bias_m_read_readvariableop@
<savev2_adam_batchnormgeneencode2_gamma_m_read_readvariableop?
;savev2_adam_batchnormgeneencode2_beta_m_read_readvariableop=
9savev2_adam_embeddingdimgene_kernel_m_read_readvariableop;
7savev2_adam_embeddingdimgene_bias_m_read_readvariableop=
9savev2_adam_embeddingdimgene3_gamma_m_read_readvariableop<
8savev2_adam_embeddingdimgene3_beta_m_read_readvariableop;
7savev2_adam_gene_decoder_1_kernel_m_read_readvariableop9
5savev2_adam_gene_decoder_1_bias_m_read_readvariableop@
<savev2_adam_batchnormgenedecode1_gamma_m_read_readvariableop?
;savev2_adam_batchnormgenedecode1_beta_m_read_readvariableop;
7savev2_adam_gene_decoder_2_kernel_m_read_readvariableop9
5savev2_adam_gene_decoder_2_bias_m_read_readvariableop:
6savev2_adam_batchnormgene2_gamma_m_read_readvariableop9
5savev2_adam_batchnormgene2_beta_m_read_readvariableop;
7savev2_adam_gene_decoder_3_kernel_m_read_readvariableop9
5savev2_adam_gene_decoder_3_bias_m_read_readvariableop;
7savev2_adam_gene_encoder_1_kernel_v_read_readvariableop9
5savev2_adam_gene_encoder_1_bias_v_read_readvariableop@
<savev2_adam_batchnormgeneencode1_gamma_v_read_readvariableop?
;savev2_adam_batchnormgeneencode1_beta_v_read_readvariableop;
7savev2_adam_gene_encoder_2_kernel_v_read_readvariableop9
5savev2_adam_gene_encoder_2_bias_v_read_readvariableop@
<savev2_adam_batchnormgeneencode2_gamma_v_read_readvariableop?
;savev2_adam_batchnormgeneencode2_beta_v_read_readvariableop=
9savev2_adam_embeddingdimgene_kernel_v_read_readvariableop;
7savev2_adam_embeddingdimgene_bias_v_read_readvariableop=
9savev2_adam_embeddingdimgene3_gamma_v_read_readvariableop<
8savev2_adam_embeddingdimgene3_beta_v_read_readvariableop;
7savev2_adam_gene_decoder_1_kernel_v_read_readvariableop9
5savev2_adam_gene_decoder_1_bias_v_read_readvariableop@
<savev2_adam_batchnormgenedecode1_gamma_v_read_readvariableop?
;savev2_adam_batchnormgenedecode1_beta_v_read_readvariableop;
7savev2_adam_gene_decoder_2_kernel_v_read_readvariableop9
5savev2_adam_gene_decoder_2_bias_v_read_readvariableop:
6savev2_adam_batchnormgene2_gamma_v_read_readvariableop9
5savev2_adam_batchnormgene2_beta_v_read_readvariableop;
7savev2_adam_gene_decoder_3_kernel_v_read_readvariableop9
5savev2_adam_gene_decoder_3_bias_v_read_readvariableop
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
: �.
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*�-
value�-B�-TB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-1/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-9/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:T*
dtype0*�
value�B�TB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �%
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_gene_encoder_1_kernel_read_readvariableop.savev2_gene_encoder_1_bias_read_readvariableop5savev2_batchnormgeneencode1_gamma_read_readvariableop4savev2_batchnormgeneencode1_beta_read_readvariableop;savev2_batchnormgeneencode1_moving_mean_read_readvariableop?savev2_batchnormgeneencode1_moving_variance_read_readvariableop0savev2_gene_encoder_2_kernel_read_readvariableop.savev2_gene_encoder_2_bias_read_readvariableop5savev2_batchnormgeneencode2_gamma_read_readvariableop4savev2_batchnormgeneencode2_beta_read_readvariableop;savev2_batchnormgeneencode2_moving_mean_read_readvariableop?savev2_batchnormgeneencode2_moving_variance_read_readvariableop2savev2_embeddingdimgene_kernel_read_readvariableop0savev2_embeddingdimgene_bias_read_readvariableop2savev2_embeddingdimgene3_gamma_read_readvariableop1savev2_embeddingdimgene3_beta_read_readvariableop8savev2_embeddingdimgene3_moving_mean_read_readvariableop<savev2_embeddingdimgene3_moving_variance_read_readvariableop0savev2_gene_decoder_1_kernel_read_readvariableop.savev2_gene_decoder_1_bias_read_readvariableop5savev2_batchnormgenedecode1_gamma_read_readvariableop4savev2_batchnormgenedecode1_beta_read_readvariableop;savev2_batchnormgenedecode1_moving_mean_read_readvariableop?savev2_batchnormgenedecode1_moving_variance_read_readvariableop0savev2_gene_decoder_2_kernel_read_readvariableop.savev2_gene_decoder_2_bias_read_readvariableop/savev2_batchnormgene2_gamma_read_readvariableop.savev2_batchnormgene2_beta_read_readvariableop5savev2_batchnormgene2_moving_mean_read_readvariableop9savev2_batchnormgene2_moving_variance_read_readvariableop0savev2_gene_decoder_3_kernel_read_readvariableop.savev2_gene_decoder_3_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop7savev2_adam_gene_encoder_1_kernel_m_read_readvariableop5savev2_adam_gene_encoder_1_bias_m_read_readvariableop<savev2_adam_batchnormgeneencode1_gamma_m_read_readvariableop;savev2_adam_batchnormgeneencode1_beta_m_read_readvariableop7savev2_adam_gene_encoder_2_kernel_m_read_readvariableop5savev2_adam_gene_encoder_2_bias_m_read_readvariableop<savev2_adam_batchnormgeneencode2_gamma_m_read_readvariableop;savev2_adam_batchnormgeneencode2_beta_m_read_readvariableop9savev2_adam_embeddingdimgene_kernel_m_read_readvariableop7savev2_adam_embeddingdimgene_bias_m_read_readvariableop9savev2_adam_embeddingdimgene3_gamma_m_read_readvariableop8savev2_adam_embeddingdimgene3_beta_m_read_readvariableop7savev2_adam_gene_decoder_1_kernel_m_read_readvariableop5savev2_adam_gene_decoder_1_bias_m_read_readvariableop<savev2_adam_batchnormgenedecode1_gamma_m_read_readvariableop;savev2_adam_batchnormgenedecode1_beta_m_read_readvariableop7savev2_adam_gene_decoder_2_kernel_m_read_readvariableop5savev2_adam_gene_decoder_2_bias_m_read_readvariableop6savev2_adam_batchnormgene2_gamma_m_read_readvariableop5savev2_adam_batchnormgene2_beta_m_read_readvariableop7savev2_adam_gene_decoder_3_kernel_m_read_readvariableop5savev2_adam_gene_decoder_3_bias_m_read_readvariableop7savev2_adam_gene_encoder_1_kernel_v_read_readvariableop5savev2_adam_gene_encoder_1_bias_v_read_readvariableop<savev2_adam_batchnormgeneencode1_gamma_v_read_readvariableop;savev2_adam_batchnormgeneencode1_beta_v_read_readvariableop7savev2_adam_gene_encoder_2_kernel_v_read_readvariableop5savev2_adam_gene_encoder_2_bias_v_read_readvariableop<savev2_adam_batchnormgeneencode2_gamma_v_read_readvariableop;savev2_adam_batchnormgeneencode2_beta_v_read_readvariableop9savev2_adam_embeddingdimgene_kernel_v_read_readvariableop7savev2_adam_embeddingdimgene_bias_v_read_readvariableop9savev2_adam_embeddingdimgene3_gamma_v_read_readvariableop8savev2_adam_embeddingdimgene3_beta_v_read_readvariableop7savev2_adam_gene_decoder_1_kernel_v_read_readvariableop5savev2_adam_gene_decoder_1_bias_v_read_readvariableop<savev2_adam_batchnormgenedecode1_gamma_v_read_readvariableop;savev2_adam_batchnormgenedecode1_beta_v_read_readvariableop7savev2_adam_gene_decoder_2_kernel_v_read_readvariableop5savev2_adam_gene_decoder_2_bias_v_read_readvariableop6savev2_adam_batchnormgene2_gamma_v_read_readvariableop5savev2_adam_batchnormgene2_beta_v_read_readvariableop7savev2_adam_gene_decoder_3_kernel_v_read_readvariableop5savev2_adam_gene_decoder_3_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *b
dtypesX
V2T	�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��:�:�:�:�:�:	�}:}:}:}:}:}:}@:@:@:@:@:@:@}:}:}:}:}:}:	}�:�:�:�:�:�:
��:�: : : : : : : :
��:�:�:�:	�}:}:}:}:}@:@:@:@:@}:}:}:}:	}�:�:�:�:
��:�:
��:�:�:�:	�}:}:}:}:}@:@:@:@:@}:}:}:}:	}�:�:�:�:
��:�: 2(
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
:}: 	

_output_shapes
:}: 


_output_shapes
:}: 

_output_shapes
:}: 

_output_shapes
:}:$ 

_output_shapes

:}@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:$ 

_output_shapes

:@}: 

_output_shapes
:}: 

_output_shapes
:}: 

_output_shapes
:}: 

_output_shapes
:}: 

_output_shapes
:}:%!

_output_shapes
:	}�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:! 

_output_shapes	
:�:!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :&("
 
_output_shapes
:
��:!)

_output_shapes	
:�:!*

_output_shapes	
:�:!+

_output_shapes	
:�:%,!

_output_shapes
:	�}: -

_output_shapes
:}: .

_output_shapes
:}: /

_output_shapes
:}:$0 

_output_shapes

:}@: 1

_output_shapes
:@: 2

_output_shapes
:@: 3

_output_shapes
:@:$4 

_output_shapes

:@}: 5

_output_shapes
:}: 6

_output_shapes
:}: 7

_output_shapes
:}:%8!

_output_shapes
:	}�:!9

_output_shapes	
:�:!:

_output_shapes	
:�:!;

_output_shapes	
:�:&<"
 
_output_shapes
:
��:!=

_output_shapes	
:�:&>"
 
_output_shapes
:
��:!?

_output_shapes	
:�:!@

_output_shapes	
:�:!A

_output_shapes	
:�:%B!

_output_shapes
:	�}: C

_output_shapes
:}: D

_output_shapes
:}: E

_output_shapes
:}:$F 

_output_shapes

:}@: G

_output_shapes
:@: H

_output_shapes
:@: I

_output_shapes
:@:$J 

_output_shapes

:@}: K

_output_shapes
:}: L

_output_shapes
:}: M

_output_shapes
:}:%N!

_output_shapes
:	}�:!O

_output_shapes	
:�:!P

_output_shapes	
:�:!Q

_output_shapes	
:�:&R"
 
_output_shapes
:
��:!S

_output_shapes	
:�:T

_output_shapes
: 
�
�
6__inference_BatchNormGeneDecode1_layer_call_fn_2021777

inputs
unknown:}
	unknown_0:}
	unknown_1:}
	unknown_2:}
identity��StatefulPartitionedCall�
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
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_2020075o
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
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
N
gene_input_layer:
"serving_default_gene_input_layer:0����������C
gene_decoder_31
StatefulPartitionedCall:0����������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer_with_weights-8

layer-9
layer_with_weights-9
layer-10
layer_with_weights-10
layer-11
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$axis
	%gamma
&beta
'moving_mean
(moving_variance"
_tf_keras_layer
�
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias"
_tf_keras_layer
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
7axis
	8gamma
9beta
:moving_mean
;moving_variance"
_tf_keras_layer
�
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses

Bkernel
Cbias"
_tf_keras_layer
�
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses
Jaxis
	Kgamma
Lbeta
Mmoving_mean
Nmoving_variance"
_tf_keras_layer
�
O	variables
Ptrainable_variables
Qregularization_losses
R	keras_api
S__call__
*T&call_and_return_all_conditional_losses

Ukernel
Vbias"
_tf_keras_layer
�
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]axis
	^gamma
_beta
`moving_mean
amoving_variance"
_tf_keras_layer
�
b	variables
ctrainable_variables
dregularization_losses
e	keras_api
f__call__
*g&call_and_return_all_conditional_losses

hkernel
ibias"
_tf_keras_layer
�
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses
paxis
	qgamma
rbeta
smoving_mean
tmoving_variance"
_tf_keras_layer
�
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses

{kernel
|bias"
_tf_keras_layer
�
0
1
%2
&3
'4
(5
/6
07
88
99
:10
;11
B12
C13
K14
L15
M16
N17
U18
V19
^20
_21
`22
a23
h24
i25
q26
r27
s28
t29
{30
|31"
trackable_list_wrapper
�
0
1
%2
&3
/4
05
86
97
B8
C9
K10
L11
U12
V13
^14
_15
h16
i17
q18
r19
{20
|21"
trackable_list_wrapper
 "
trackable_list_wrapper
�
}non_trainable_variables

~layers
metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
*__inference_model_42_layer_call_fn_2020390
*__inference_model_42_layer_call_fn_2021040
*__inference_model_42_layer_call_fn_2021109
*__inference_model_42_layer_call_fn_2020736�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�
�trace_0
�trace_1
�trace_2
�trace_32�
E__inference_model_42_layer_call_and_return_conditional_losses_2021235
E__inference_model_42_layer_call_and_return_conditional_losses_2021431
E__inference_model_42_layer_call_and_return_conditional_losses_2020815
E__inference_model_42_layer_call_and_return_conditional_losses_2020894�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1z�trace_2z�trace_3
�B�
"__inference__wrapped_model_2019758gene_input_layer"�
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
�
	�iter
�beta_1
�beta_2

�decay
�learning_ratem�m�%m�&m�/m�0m�8m�9m�Bm�Cm�Km�Lm�Um�Vm�^m�_m�hm�im�qm�rm�{m�|m�v�v�%v�&v�/v�0v�8v�9v�Bv�Cv�Kv�Lv�Uv�Vv�^v�_v�hv�iv�qv�rv�{v�|v�"
	optimizer
-
�serving_default"
signature_map
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
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_gene_encoder_1_layer_call_fn_2021440�
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
 z�trace_0
�
�trace_02�
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_2021451�
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
 z�trace_0
):'
��2gene_encoder_1/kernel
": �2gene_encoder_1/bias
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
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_BatchNormGeneEncode1_layer_call_fn_2021464
6__inference_BatchNormGeneEncode1_layer_call_fn_2021477�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_2021497
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_2021531�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
):'�2BatchNormGeneEncode1/gamma
(:&�2BatchNormGeneEncode1/beta
1:/� (2 BatchNormGeneEncode1/moving_mean
5:3� (2$BatchNormGeneEncode1/moving_variance
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
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_gene_encoder_2_layer_call_fn_2021540�
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
 z�trace_0
�
�trace_02�
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_2021551�
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
 z�trace_0
(:&	�}2gene_encoder_2/kernel
!:}2gene_encoder_2/bias
<
80
91
:2
;3"
trackable_list_wrapper
.
80
91"
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
�
�trace_0
�trace_12�
6__inference_BatchNormGeneEncode2_layer_call_fn_2021564
6__inference_BatchNormGeneEncode2_layer_call_fn_2021577�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_2021597
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_2021631�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
(:&}2BatchNormGeneEncode2/gamma
':%}2BatchNormGeneEncode2/beta
0:.} (2 BatchNormGeneEncode2/moving_mean
4:2} (2$BatchNormGeneEncode2/moving_variance
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
2__inference_EmbeddingDimGene_layer_call_fn_2021640�
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
 z�trace_0
�
�trace_02�
M__inference_EmbeddingDimGene_layer_call_and_return_conditional_losses_2021651�
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
 z�trace_0
):'}@2EmbeddingDimGene/kernel
#:!@2EmbeddingDimGene/bias
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
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
3__inference_EmbeddingDimGene3_layer_call_fn_2021664
3__inference_EmbeddingDimGene3_layer_call_fn_2021677�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
N__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_2021697
N__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_2021731�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
%:#@2EmbeddingDimGene3/gamma
$:"@2EmbeddingDimGene3/beta
-:+@ (2EmbeddingDimGene3/moving_mean
1:/@ (2!EmbeddingDimGene3/moving_variance
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
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
�
�trace_02�
0__inference_gene_decoder_1_layer_call_fn_2021740�
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
 z�trace_0
�
�trace_02�
K__inference_gene_decoder_1_layer_call_and_return_conditional_losses_2021751�
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
 z�trace_0
':%@}2gene_decoder_1/kernel
!:}2gene_decoder_1/bias
<
^0
_1
`2
a3"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_BatchNormGeneDecode1_layer_call_fn_2021764
6__inference_BatchNormGeneDecode1_layer_call_fn_2021777�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_2021797
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_2021831�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
(:&}2BatchNormGeneDecode1/gamma
':%}2BatchNormGeneDecode1/beta
0:.} (2 BatchNormGeneDecode1/moving_mean
4:2} (2$BatchNormGeneDecode1/moving_variance
.
h0
i1"
trackable_list_wrapper
.
h0
i1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
b	variables
ctrainable_variables
dregularization_losses
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_gene_decoder_2_layer_call_fn_2021840�
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
 z�trace_0
�
�trace_02�
K__inference_gene_decoder_2_layer_call_and_return_conditional_losses_2021851�
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
 z�trace_0
(:&	}�2gene_decoder_2/kernel
": �2gene_decoder_2/bias
<
q0
r1
s2
t3"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
0__inference_BatchNormGene2_layer_call_fn_2021864
0__inference_BatchNormGene2_layer_call_fn_2021877�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
K__inference_BatchNormGene2_layer_call_and_return_conditional_losses_2021897
K__inference_BatchNormGene2_layer_call_and_return_conditional_losses_2021931�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
#:!�2BatchNormGene2/gamma
": �2BatchNormGene2/beta
+:)� (2BatchNormGene2/moving_mean
/:-� (2BatchNormGene2/moving_variance
.
{0
|1"
trackable_list_wrapper
.
{0
|1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_gene_decoder_3_layer_call_fn_2021940�
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
 z�trace_0
�
�trace_02�
K__inference_gene_decoder_3_layer_call_and_return_conditional_losses_2021951�
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
 z�trace_0
):'
��2gene_decoder_3/kernel
": �2gene_decoder_3/bias
f
'0
(1
:2
;3
M4
N5
`6
a7
s8
t9"
trackable_list_wrapper
v
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
11"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_model_42_layer_call_fn_2020390gene_input_layer"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_model_42_layer_call_fn_2021040inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_model_42_layer_call_fn_2021109inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
*__inference_model_42_layer_call_fn_2020736gene_input_layer"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_model_42_layer_call_and_return_conditional_losses_2021235inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_model_42_layer_call_and_return_conditional_losses_2021431inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_model_42_layer_call_and_return_conditional_losses_2020815gene_input_layer"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_model_42_layer_call_and_return_conditional_losses_2020894gene_input_layer"�
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
�B�
%__inference_signature_wrapper_2020971gene_input_layer"�
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
�B�
0__inference_gene_encoder_1_layer_call_fn_2021440inputs"�
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
�B�
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_2021451inputs"�
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
�B�
6__inference_BatchNormGeneEncode1_layer_call_fn_2021464inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_BatchNormGeneEncode1_layer_call_fn_2021477inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_2021497inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_2021531inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_gene_encoder_2_layer_call_fn_2021540inputs"�
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
�B�
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_2021551inputs"�
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
.
:0
;1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_BatchNormGeneEncode2_layer_call_fn_2021564inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_BatchNormGeneEncode2_layer_call_fn_2021577inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_2021597inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_2021631inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
2__inference_EmbeddingDimGene_layer_call_fn_2021640inputs"�
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
�B�
M__inference_EmbeddingDimGene_layer_call_and_return_conditional_losses_2021651inputs"�
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
�B�
3__inference_EmbeddingDimGene3_layer_call_fn_2021664inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
3__inference_EmbeddingDimGene3_layer_call_fn_2021677inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_2021697inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_2021731inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_gene_decoder_1_layer_call_fn_2021740inputs"�
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
�B�
K__inference_gene_decoder_1_layer_call_and_return_conditional_losses_2021751inputs"�
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
.
`0
a1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_BatchNormGeneDecode1_layer_call_fn_2021764inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
6__inference_BatchNormGeneDecode1_layer_call_fn_2021777inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_2021797inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_2021831inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_gene_decoder_2_layer_call_fn_2021840inputs"�
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
�B�
K__inference_gene_decoder_2_layer_call_and_return_conditional_losses_2021851inputs"�
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
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_BatchNormGene2_layer_call_fn_2021864inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
0__inference_BatchNormGene2_layer_call_fn_2021877inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_BatchNormGene2_layer_call_and_return_conditional_losses_2021897inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_BatchNormGene2_layer_call_and_return_conditional_losses_2021931inputs"�
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

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_gene_decoder_3_layer_call_fn_2021940inputs"�
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
�B�
K__inference_gene_decoder_3_layer_call_and_return_conditional_losses_2021951inputs"�
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
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
.:,
��2Adam/gene_encoder_1/kernel/m
':%�2Adam/gene_encoder_1/bias/m
.:,�2!Adam/BatchNormGeneEncode1/gamma/m
-:+�2 Adam/BatchNormGeneEncode1/beta/m
-:+	�}2Adam/gene_encoder_2/kernel/m
&:$}2Adam/gene_encoder_2/bias/m
-:+}2!Adam/BatchNormGeneEncode2/gamma/m
,:*}2 Adam/BatchNormGeneEncode2/beta/m
.:,}@2Adam/EmbeddingDimGene/kernel/m
(:&@2Adam/EmbeddingDimGene/bias/m
*:(@2Adam/EmbeddingDimGene3/gamma/m
):'@2Adam/EmbeddingDimGene3/beta/m
,:*@}2Adam/gene_decoder_1/kernel/m
&:$}2Adam/gene_decoder_1/bias/m
-:+}2!Adam/BatchNormGeneDecode1/gamma/m
,:*}2 Adam/BatchNormGeneDecode1/beta/m
-:+	}�2Adam/gene_decoder_2/kernel/m
':%�2Adam/gene_decoder_2/bias/m
(:&�2Adam/BatchNormGene2/gamma/m
':%�2Adam/BatchNormGene2/beta/m
.:,
��2Adam/gene_decoder_3/kernel/m
':%�2Adam/gene_decoder_3/bias/m
.:,
��2Adam/gene_encoder_1/kernel/v
':%�2Adam/gene_encoder_1/bias/v
.:,�2!Adam/BatchNormGeneEncode1/gamma/v
-:+�2 Adam/BatchNormGeneEncode1/beta/v
-:+	�}2Adam/gene_encoder_2/kernel/v
&:$}2Adam/gene_encoder_2/bias/v
-:+}2!Adam/BatchNormGeneEncode2/gamma/v
,:*}2 Adam/BatchNormGeneEncode2/beta/v
.:,}@2Adam/EmbeddingDimGene/kernel/v
(:&@2Adam/EmbeddingDimGene/bias/v
*:(@2Adam/EmbeddingDimGene3/gamma/v
):'@2Adam/EmbeddingDimGene3/beta/v
,:*@}2Adam/gene_decoder_1/kernel/v
&:$}2Adam/gene_decoder_1/bias/v
-:+}2!Adam/BatchNormGeneDecode1/gamma/v
,:*}2 Adam/BatchNormGeneDecode1/beta/v
-:+	}�2Adam/gene_decoder_2/kernel/v
':%�2Adam/gene_decoder_2/bias/v
(:&�2Adam/BatchNormGene2/gamma/v
':%�2Adam/BatchNormGene2/beta/v
.:,
��2Adam/gene_decoder_3/kernel/v
':%�2Adam/gene_decoder_3/bias/v�
K__inference_BatchNormGene2_layer_call_and_return_conditional_losses_2021897dtqsr4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
K__inference_BatchNormGene2_layer_call_and_return_conditional_losses_2021931dstqr4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
0__inference_BatchNormGene2_layer_call_fn_2021864Wtqsr4�1
*�'
!�
inputs����������
p 
� "������������
0__inference_BatchNormGene2_layer_call_fn_2021877Wstqr4�1
*�'
!�
inputs����������
p
� "������������
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_2021797ba^`_3�0
)�&
 �
inputs���������}
p 
� "%�"
�
0���������}
� �
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_2021831b`a^_3�0
)�&
 �
inputs���������}
p
� "%�"
�
0���������}
� �
6__inference_BatchNormGeneDecode1_layer_call_fn_2021764Ua^`_3�0
)�&
 �
inputs���������}
p 
� "����������}�
6__inference_BatchNormGeneDecode1_layer_call_fn_2021777U`a^_3�0
)�&
 �
inputs���������}
p
� "����������}�
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_2021497d(%'&4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_2021531d'(%&4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
6__inference_BatchNormGeneEncode1_layer_call_fn_2021464W(%'&4�1
*�'
!�
inputs����������
p 
� "������������
6__inference_BatchNormGeneEncode1_layer_call_fn_2021477W'(%&4�1
*�'
!�
inputs����������
p
� "������������
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_2021597b;8:93�0
)�&
 �
inputs���������}
p 
� "%�"
�
0���������}
� �
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_2021631b:;893�0
)�&
 �
inputs���������}
p
� "%�"
�
0���������}
� �
6__inference_BatchNormGeneEncode2_layer_call_fn_2021564U;8:93�0
)�&
 �
inputs���������}
p 
� "����������}�
6__inference_BatchNormGeneEncode2_layer_call_fn_2021577U:;893�0
)�&
 �
inputs���������}
p
� "����������}�
N__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_2021697bNKML3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
N__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_2021731bMNKL3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� �
3__inference_EmbeddingDimGene3_layer_call_fn_2021664UNKML3�0
)�&
 �
inputs���������@
p 
� "����������@�
3__inference_EmbeddingDimGene3_layer_call_fn_2021677UMNKL3�0
)�&
 �
inputs���������@
p
� "����������@�
M__inference_EmbeddingDimGene_layer_call_and_return_conditional_losses_2021651\BC/�,
%�"
 �
inputs���������}
� "%�"
�
0���������@
� �
2__inference_EmbeddingDimGene_layer_call_fn_2021640OBC/�,
%�"
 �
inputs���������}
� "����������@�
"__inference__wrapped_model_2019758� (%'&/0;8:9BCNKMLUVa^`_hitqsr{|:�7
0�-
+�(
gene_input_layer����������
� "@�=
;
gene_decoder_3)�&
gene_decoder_3�����������
K__inference_gene_decoder_1_layer_call_and_return_conditional_losses_2021751\UV/�,
%�"
 �
inputs���������@
� "%�"
�
0���������}
� �
0__inference_gene_decoder_1_layer_call_fn_2021740OUV/�,
%�"
 �
inputs���������@
� "����������}�
K__inference_gene_decoder_2_layer_call_and_return_conditional_losses_2021851]hi/�,
%�"
 �
inputs���������}
� "&�#
�
0����������
� �
0__inference_gene_decoder_2_layer_call_fn_2021840Phi/�,
%�"
 �
inputs���������}
� "������������
K__inference_gene_decoder_3_layer_call_and_return_conditional_losses_2021951^{|0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
0__inference_gene_decoder_3_layer_call_fn_2021940Q{|0�-
&�#
!�
inputs����������
� "������������
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_2021451^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
0__inference_gene_encoder_1_layer_call_fn_2021440Q0�-
&�#
!�
inputs����������
� "������������
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_2021551]/00�-
&�#
!�
inputs����������
� "%�"
�
0���������}
� �
0__inference_gene_encoder_2_layer_call_fn_2021540P/00�-
&�#
!�
inputs����������
� "����������}�
E__inference_model_42_layer_call_and_return_conditional_losses_2020815� (%'&/0;8:9BCNKMLUVa^`_hitqsr{|B�?
8�5
+�(
gene_input_layer����������
p 

 
� "&�#
�
0����������
� �
E__inference_model_42_layer_call_and_return_conditional_losses_2020894� '(%&/0:;89BCMNKLUV`a^_histqr{|B�?
8�5
+�(
gene_input_layer����������
p

 
� "&�#
�
0����������
� �
E__inference_model_42_layer_call_and_return_conditional_losses_2021235� (%'&/0;8:9BCNKMLUVa^`_hitqsr{|8�5
.�+
!�
inputs����������
p 

 
� "&�#
�
0����������
� �
E__inference_model_42_layer_call_and_return_conditional_losses_2021431� '(%&/0:;89BCMNKLUV`a^_histqr{|8�5
.�+
!�
inputs����������
p

 
� "&�#
�
0����������
� �
*__inference_model_42_layer_call_fn_2020390� (%'&/0;8:9BCNKMLUVa^`_hitqsr{|B�?
8�5
+�(
gene_input_layer����������
p 

 
� "������������
*__inference_model_42_layer_call_fn_2020736� '(%&/0:;89BCMNKLUV`a^_histqr{|B�?
8�5
+�(
gene_input_layer����������
p

 
� "������������
*__inference_model_42_layer_call_fn_2021040w (%'&/0;8:9BCNKMLUVa^`_hitqsr{|8�5
.�+
!�
inputs����������
p 

 
� "������������
*__inference_model_42_layer_call_fn_2021109w '(%&/0:;89BCMNKLUV`a^_histqr{|8�5
.�+
!�
inputs����������
p

 
� "������������
%__inference_signature_wrapper_2020971� (%'&/0;8:9BCNKMLUVa^`_hitqsr{|N�K
� 
D�A
?
gene_input_layer+�(
gene_input_layer����������"@�=
;
gene_decoder_3)�&
gene_decoder_3����������