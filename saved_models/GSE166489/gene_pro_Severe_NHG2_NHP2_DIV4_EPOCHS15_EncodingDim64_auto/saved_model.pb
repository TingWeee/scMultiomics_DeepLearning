��(
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
 �"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68��%
�
gene_encoder_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�
�*&
shared_namegene_encoder_1/kernel
�
)gene_encoder_1/kernel/Read/ReadVariableOpReadVariableOpgene_encoder_1/kernel* 
_output_shapes
:
�
�*
dtype0

gene_encoder_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_namegene_encoder_1/bias
x
'gene_encoder_1/bias/Read/ReadVariableOpReadVariableOpgene_encoder_1/bias*
_output_shapes	
:�*
dtype0
�
protein_encoder_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�/*)
shared_nameprotein_encoder_1/kernel
�
,protein_encoder_1/kernel/Read/ReadVariableOpReadVariableOpprotein_encoder_1/kernel*
_output_shapes
:	�/*
dtype0
�
protein_encoder_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*'
shared_nameprotein_encoder_1/bias
}
*protein_encoder_1/bias/Read/ReadVariableOpReadVariableOpprotein_encoder_1/bias*
_output_shapes
:/*
dtype0
�
BatchNormGeneEncode1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameBatchNormGeneEncode1/gamma
�
.BatchNormGeneEncode1/gamma/Read/ReadVariableOpReadVariableOpBatchNormGeneEncode1/gamma*
_output_shapes	
:�*
dtype0
�
BatchNormGeneEncode1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameBatchNormGeneEncode1/beta
�
-BatchNormGeneEncode1/beta/Read/ReadVariableOpReadVariableOpBatchNormGeneEncode1/beta*
_output_shapes	
:�*
dtype0
�
 BatchNormGeneEncode1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" BatchNormGeneEncode1/moving_mean
�
4BatchNormGeneEncode1/moving_mean/Read/ReadVariableOpReadVariableOp BatchNormGeneEncode1/moving_mean*
_output_shapes	
:�*
dtype0
�
$BatchNormGeneEncode1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*5
shared_name&$BatchNormGeneEncode1/moving_variance
�
8BatchNormGeneEncode1/moving_variance/Read/ReadVariableOpReadVariableOp$BatchNormGeneEncode1/moving_variance*
_output_shapes	
:�*
dtype0
�
BatchNormProteinEncode1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*.
shared_nameBatchNormProteinEncode1/gamma
�
1BatchNormProteinEncode1/gamma/Read/ReadVariableOpReadVariableOpBatchNormProteinEncode1/gamma*
_output_shapes
:/*
dtype0
�
BatchNormProteinEncode1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*-
shared_nameBatchNormProteinEncode1/beta
�
0BatchNormProteinEncode1/beta/Read/ReadVariableOpReadVariableOpBatchNormProteinEncode1/beta*
_output_shapes
:/*
dtype0
�
#BatchNormProteinEncode1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*4
shared_name%#BatchNormProteinEncode1/moving_mean
�
7BatchNormProteinEncode1/moving_mean/Read/ReadVariableOpReadVariableOp#BatchNormProteinEncode1/moving_mean*
_output_shapes
:/*
dtype0
�
'BatchNormProteinEncode1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*8
shared_name)'BatchNormProteinEncode1/moving_variance
�
;BatchNormProteinEncode1/moving_variance/Read/ReadVariableOpReadVariableOp'BatchNormProteinEncode1/moving_variance*
_output_shapes
:/*
dtype0
�
gene_encoder_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�P*&
shared_namegene_encoder_2/kernel
�
)gene_encoder_2/kernel/Read/ReadVariableOpReadVariableOpgene_encoder_2/kernel*
_output_shapes
:	�P*
dtype0
~
gene_encoder_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*$
shared_namegene_encoder_2/bias
w
'gene_encoder_2/bias/Read/ReadVariableOpReadVariableOpgene_encoder_2/bias*
_output_shapes
:P*
dtype0
�
protein_encoder_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/*)
shared_nameprotein_encoder_2/kernel
�
,protein_encoder_2/kernel/Read/ReadVariableOpReadVariableOpprotein_encoder_2/kernel*
_output_shapes

:/*
dtype0
�
protein_encoder_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameprotein_encoder_2/bias
}
*protein_encoder_2/bias/Read/ReadVariableOpReadVariableOpprotein_encoder_2/bias*
_output_shapes
:*
dtype0
�
BatchNormGeneEncode2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_nameBatchNormGeneEncode2/gamma
�
.BatchNormGeneEncode2/gamma/Read/ReadVariableOpReadVariableOpBatchNormGeneEncode2/gamma*
_output_shapes
:P*
dtype0
�
BatchNormGeneEncode2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P**
shared_nameBatchNormGeneEncode2/beta
�
-BatchNormGeneEncode2/beta/Read/ReadVariableOpReadVariableOpBatchNormGeneEncode2/beta*
_output_shapes
:P*
dtype0
�
 BatchNormGeneEncode2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*1
shared_name" BatchNormGeneEncode2/moving_mean
�
4BatchNormGeneEncode2/moving_mean/Read/ReadVariableOpReadVariableOp BatchNormGeneEncode2/moving_mean*
_output_shapes
:P*
dtype0
�
$BatchNormGeneEncode2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*5
shared_name&$BatchNormGeneEncode2/moving_variance
�
8BatchNormGeneEncode2/moving_variance/Read/ReadVariableOpReadVariableOp$BatchNormGeneEncode2/moving_variance*
_output_shapes
:P*
dtype0
�
BatchNormProteinEncode2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameBatchNormProteinEncode2/gamma
�
1BatchNormProteinEncode2/gamma/Read/ReadVariableOpReadVariableOpBatchNormProteinEncode2/gamma*
_output_shapes
:*
dtype0
�
BatchNormProteinEncode2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameBatchNormProteinEncode2/beta
�
0BatchNormProteinEncode2/beta/Read/ReadVariableOpReadVariableOpBatchNormProteinEncode2/beta*
_output_shapes
:*
dtype0
�
#BatchNormProteinEncode2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#BatchNormProteinEncode2/moving_mean
�
7BatchNormProteinEncode2/moving_mean/Read/ReadVariableOpReadVariableOp#BatchNormProteinEncode2/moving_mean*
_output_shapes
:*
dtype0
�
'BatchNormProteinEncode2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'BatchNormProteinEncode2/moving_variance
�
;BatchNormProteinEncode2/moving_variance/Read/ReadVariableOpReadVariableOp'BatchNormProteinEncode2/moving_variance*
_output_shapes
:*
dtype0
�
EmbeddingDimDense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:[@*)
shared_nameEmbeddingDimDense/kernel
�
,EmbeddingDimDense/kernel/Read/ReadVariableOpReadVariableOpEmbeddingDimDense/kernel*
_output_shapes

:[@*
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
:@P*&
shared_namegene_decoder_1/kernel

)gene_decoder_1/kernel/Read/ReadVariableOpReadVariableOpgene_decoder_1/kernel*
_output_shapes

:@P*
dtype0
~
gene_decoder_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*$
shared_namegene_decoder_1/bias
w
'gene_decoder_1/bias/Read/ReadVariableOpReadVariableOpgene_decoder_1/bias*
_output_shapes
:P*
dtype0
�
protein_decoder_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*)
shared_nameprotein_decoder_1/kernel
�
,protein_decoder_1/kernel/Read/ReadVariableOpReadVariableOpprotein_decoder_1/kernel*
_output_shapes

:@*
dtype0
�
protein_decoder_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameprotein_decoder_1/bias
}
*protein_decoder_1/bias/Read/ReadVariableOpReadVariableOpprotein_decoder_1/bias*
_output_shapes
:*
dtype0
�
BatchNormGeneDecode1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_nameBatchNormGeneDecode1/gamma
�
.BatchNormGeneDecode1/gamma/Read/ReadVariableOpReadVariableOpBatchNormGeneDecode1/gamma*
_output_shapes
:P*
dtype0
�
BatchNormGeneDecode1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:P**
shared_nameBatchNormGeneDecode1/beta
�
-BatchNormGeneDecode1/beta/Read/ReadVariableOpReadVariableOpBatchNormGeneDecode1/beta*
_output_shapes
:P*
dtype0
�
 BatchNormGeneDecode1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*1
shared_name" BatchNormGeneDecode1/moving_mean
�
4BatchNormGeneDecode1/moving_mean/Read/ReadVariableOpReadVariableOp BatchNormGeneDecode1/moving_mean*
_output_shapes
:P*
dtype0
�
$BatchNormGeneDecode1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*5
shared_name&$BatchNormGeneDecode1/moving_variance
�
8BatchNormGeneDecode1/moving_variance/Read/ReadVariableOpReadVariableOp$BatchNormGeneDecode1/moving_variance*
_output_shapes
:P*
dtype0
�
BatchNormProteinDecode1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameBatchNormProteinDecode1/gamma
�
1BatchNormProteinDecode1/gamma/Read/ReadVariableOpReadVariableOpBatchNormProteinDecode1/gamma*
_output_shapes
:*
dtype0
�
BatchNormProteinDecode1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameBatchNormProteinDecode1/beta
�
0BatchNormProteinDecode1/beta/Read/ReadVariableOpReadVariableOpBatchNormProteinDecode1/beta*
_output_shapes
:*
dtype0
�
#BatchNormProteinDecode1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#BatchNormProteinDecode1/moving_mean
�
7BatchNormProteinDecode1/moving_mean/Read/ReadVariableOpReadVariableOp#BatchNormProteinDecode1/moving_mean*
_output_shapes
:*
dtype0
�
'BatchNormProteinDecode1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'BatchNormProteinDecode1/moving_variance
�
;BatchNormProteinDecode1/moving_variance/Read/ReadVariableOpReadVariableOp'BatchNormProteinDecode1/moving_variance*
_output_shapes
:*
dtype0
�
gene_decoder_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	P�*&
shared_namegene_decoder_2/kernel
�
)gene_decoder_2/kernel/Read/ReadVariableOpReadVariableOpgene_decoder_2/kernel*
_output_shapes
:	P�*
dtype0

gene_decoder_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_namegene_decoder_2/bias
x
'gene_decoder_2/bias/Read/ReadVariableOpReadVariableOpgene_decoder_2/bias*
_output_shapes	
:�*
dtype0
�
protein_decoder_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/*)
shared_nameprotein_decoder_2/kernel
�
,protein_decoder_2/kernel/Read/ReadVariableOpReadVariableOpprotein_decoder_2/kernel*
_output_shapes

:/*
dtype0
�
protein_decoder_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*'
shared_nameprotein_decoder_2/bias
}
*protein_decoder_2/bias/Read/ReadVariableOpReadVariableOpprotein_decoder_2/bias*
_output_shapes
:/*
dtype0
�
BatchNormGeneDecode2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameBatchNormGeneDecode2/gamma
�
.BatchNormGeneDecode2/gamma/Read/ReadVariableOpReadVariableOpBatchNormGeneDecode2/gamma*
_output_shapes	
:�*
dtype0
�
BatchNormGeneDecode2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:�**
shared_nameBatchNormGeneDecode2/beta
�
-BatchNormGeneDecode2/beta/Read/ReadVariableOpReadVariableOpBatchNormGeneDecode2/beta*
_output_shapes	
:�*
dtype0
�
 BatchNormGeneDecode2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" BatchNormGeneDecode2/moving_mean
�
4BatchNormGeneDecode2/moving_mean/Read/ReadVariableOpReadVariableOp BatchNormGeneDecode2/moving_mean*
_output_shapes	
:�*
dtype0
�
$BatchNormGeneDecode2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*5
shared_name&$BatchNormGeneDecode2/moving_variance
�
8BatchNormGeneDecode2/moving_variance/Read/ReadVariableOpReadVariableOp$BatchNormGeneDecode2/moving_variance*
_output_shapes	
:�*
dtype0
�
BatchNormProteinDecode2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*.
shared_nameBatchNormProteinDecode2/gamma
�
1BatchNormProteinDecode2/gamma/Read/ReadVariableOpReadVariableOpBatchNormProteinDecode2/gamma*
_output_shapes
:/*
dtype0
�
BatchNormProteinDecode2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*-
shared_nameBatchNormProteinDecode2/beta
�
0BatchNormProteinDecode2/beta/Read/ReadVariableOpReadVariableOpBatchNormProteinDecode2/beta*
_output_shapes
:/*
dtype0
�
#BatchNormProteinDecode2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*4
shared_name%#BatchNormProteinDecode2/moving_mean
�
7BatchNormProteinDecode2/moving_mean/Read/ReadVariableOpReadVariableOp#BatchNormProteinDecode2/moving_mean*
_output_shapes
:/*
dtype0
�
'BatchNormProteinDecode2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*8
shared_name)'BatchNormProteinDecode2/moving_variance
�
;BatchNormProteinDecode2/moving_variance/Read/ReadVariableOpReadVariableOp'BatchNormProteinDecode2/moving_variance*
_output_shapes
:/*
dtype0
�
gene_decoder_last/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��
*)
shared_namegene_decoder_last/kernel
�
,gene_decoder_last/kernel/Read/ReadVariableOpReadVariableOpgene_decoder_last/kernel* 
_output_shapes
:
��
*
dtype0
�
gene_decoder_last/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�
*'
shared_namegene_decoder_last/bias
~
*gene_decoder_last/bias/Read/ReadVariableOpReadVariableOpgene_decoder_last/bias*
_output_shapes	
:�
*
dtype0
�
protein_decoder_last/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	/�*,
shared_nameprotein_decoder_last/kernel
�
/protein_decoder_last/kernel/Read/ReadVariableOpReadVariableOpprotein_decoder_last/kernel*
_output_shapes
:	/�*
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
�
�*-
shared_nameAdam/gene_encoder_1/kernel/m
�
0Adam/gene_encoder_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gene_encoder_1/kernel/m* 
_output_shapes
:
�
�*
dtype0
�
Adam/gene_encoder_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameAdam/gene_encoder_1/bias/m
�
.Adam/gene_encoder_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/gene_encoder_1/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/protein_encoder_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�/*0
shared_name!Adam/protein_encoder_1/kernel/m
�
3Adam/protein_encoder_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/protein_encoder_1/kernel/m*
_output_shapes
:	�/*
dtype0
�
Adam/protein_encoder_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*.
shared_nameAdam/protein_encoder_1/bias/m
�
1Adam/protein_encoder_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/protein_encoder_1/bias/m*
_output_shapes
:/*
dtype0
�
!Adam/BatchNormGeneEncode1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/BatchNormGeneEncode1/gamma/m
�
5Adam/BatchNormGeneEncode1/gamma/m/Read/ReadVariableOpReadVariableOp!Adam/BatchNormGeneEncode1/gamma/m*
_output_shapes	
:�*
dtype0
�
 Adam/BatchNormGeneEncode1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/BatchNormGeneEncode1/beta/m
�
4Adam/BatchNormGeneEncode1/beta/m/Read/ReadVariableOpReadVariableOp Adam/BatchNormGeneEncode1/beta/m*
_output_shapes	
:�*
dtype0
�
$Adam/BatchNormProteinEncode1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*5
shared_name&$Adam/BatchNormProteinEncode1/gamma/m
�
8Adam/BatchNormProteinEncode1/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/BatchNormProteinEncode1/gamma/m*
_output_shapes
:/*
dtype0
�
#Adam/BatchNormProteinEncode1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*4
shared_name%#Adam/BatchNormProteinEncode1/beta/m
�
7Adam/BatchNormProteinEncode1/beta/m/Read/ReadVariableOpReadVariableOp#Adam/BatchNormProteinEncode1/beta/m*
_output_shapes
:/*
dtype0
�
Adam/gene_encoder_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�P*-
shared_nameAdam/gene_encoder_2/kernel/m
�
0Adam/gene_encoder_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gene_encoder_2/kernel/m*
_output_shapes
:	�P*
dtype0
�
Adam/gene_encoder_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_nameAdam/gene_encoder_2/bias/m
�
.Adam/gene_encoder_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/gene_encoder_2/bias/m*
_output_shapes
:P*
dtype0
�
Adam/protein_encoder_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/*0
shared_name!Adam/protein_encoder_2/kernel/m
�
3Adam/protein_encoder_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/protein_encoder_2/kernel/m*
_output_shapes

:/*
dtype0
�
Adam/protein_encoder_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/protein_encoder_2/bias/m
�
1Adam/protein_encoder_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/protein_encoder_2/bias/m*
_output_shapes
:*
dtype0
�
!Adam/BatchNormGeneEncode2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*2
shared_name#!Adam/BatchNormGeneEncode2/gamma/m
�
5Adam/BatchNormGeneEncode2/gamma/m/Read/ReadVariableOpReadVariableOp!Adam/BatchNormGeneEncode2/gamma/m*
_output_shapes
:P*
dtype0
�
 Adam/BatchNormGeneEncode2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*1
shared_name" Adam/BatchNormGeneEncode2/beta/m
�
4Adam/BatchNormGeneEncode2/beta/m/Read/ReadVariableOpReadVariableOp Adam/BatchNormGeneEncode2/beta/m*
_output_shapes
:P*
dtype0
�
$Adam/BatchNormProteinEncode2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/BatchNormProteinEncode2/gamma/m
�
8Adam/BatchNormProteinEncode2/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/BatchNormProteinEncode2/gamma/m*
_output_shapes
:*
dtype0
�
#Adam/BatchNormProteinEncode2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/BatchNormProteinEncode2/beta/m
�
7Adam/BatchNormProteinEncode2/beta/m/Read/ReadVariableOpReadVariableOp#Adam/BatchNormProteinEncode2/beta/m*
_output_shapes
:*
dtype0
�
Adam/EmbeddingDimDense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:[@*0
shared_name!Adam/EmbeddingDimDense/kernel/m
�
3Adam/EmbeddingDimDense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/EmbeddingDimDense/kernel/m*
_output_shapes

:[@*
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
:@P*-
shared_nameAdam/gene_decoder_1/kernel/m
�
0Adam/gene_decoder_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_1/kernel/m*
_output_shapes

:@P*
dtype0
�
Adam/gene_decoder_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_nameAdam/gene_decoder_1/bias/m
�
.Adam/gene_decoder_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_1/bias/m*
_output_shapes
:P*
dtype0
�
Adam/protein_decoder_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!Adam/protein_decoder_1/kernel/m
�
3Adam/protein_decoder_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/protein_decoder_1/kernel/m*
_output_shapes

:@*
dtype0
�
Adam/protein_decoder_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/protein_decoder_1/bias/m
�
1Adam/protein_decoder_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/protein_decoder_1/bias/m*
_output_shapes
:*
dtype0
�
!Adam/BatchNormGeneDecode1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*2
shared_name#!Adam/BatchNormGeneDecode1/gamma/m
�
5Adam/BatchNormGeneDecode1/gamma/m/Read/ReadVariableOpReadVariableOp!Adam/BatchNormGeneDecode1/gamma/m*
_output_shapes
:P*
dtype0
�
 Adam/BatchNormGeneDecode1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*1
shared_name" Adam/BatchNormGeneDecode1/beta/m
�
4Adam/BatchNormGeneDecode1/beta/m/Read/ReadVariableOpReadVariableOp Adam/BatchNormGeneDecode1/beta/m*
_output_shapes
:P*
dtype0
�
$Adam/BatchNormProteinDecode1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/BatchNormProteinDecode1/gamma/m
�
8Adam/BatchNormProteinDecode1/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/BatchNormProteinDecode1/gamma/m*
_output_shapes
:*
dtype0
�
#Adam/BatchNormProteinDecode1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/BatchNormProteinDecode1/beta/m
�
7Adam/BatchNormProteinDecode1/beta/m/Read/ReadVariableOpReadVariableOp#Adam/BatchNormProteinDecode1/beta/m*
_output_shapes
:*
dtype0
�
Adam/gene_decoder_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	P�*-
shared_nameAdam/gene_decoder_2/kernel/m
�
0Adam/gene_decoder_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_2/kernel/m*
_output_shapes
:	P�*
dtype0
�
Adam/gene_decoder_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameAdam/gene_decoder_2/bias/m
�
.Adam/gene_decoder_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_2/bias/m*
_output_shapes	
:�*
dtype0
�
Adam/protein_decoder_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/*0
shared_name!Adam/protein_decoder_2/kernel/m
�
3Adam/protein_decoder_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/protein_decoder_2/kernel/m*
_output_shapes

:/*
dtype0
�
Adam/protein_decoder_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*.
shared_nameAdam/protein_decoder_2/bias/m
�
1Adam/protein_decoder_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/protein_decoder_2/bias/m*
_output_shapes
:/*
dtype0
�
!Adam/BatchNormGeneDecode2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/BatchNormGeneDecode2/gamma/m
�
5Adam/BatchNormGeneDecode2/gamma/m/Read/ReadVariableOpReadVariableOp!Adam/BatchNormGeneDecode2/gamma/m*
_output_shapes	
:�*
dtype0
�
 Adam/BatchNormGeneDecode2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/BatchNormGeneDecode2/beta/m
�
4Adam/BatchNormGeneDecode2/beta/m/Read/ReadVariableOpReadVariableOp Adam/BatchNormGeneDecode2/beta/m*
_output_shapes	
:�*
dtype0
�
$Adam/BatchNormProteinDecode2/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*5
shared_name&$Adam/BatchNormProteinDecode2/gamma/m
�
8Adam/BatchNormProteinDecode2/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/BatchNormProteinDecode2/gamma/m*
_output_shapes
:/*
dtype0
�
#Adam/BatchNormProteinDecode2/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*4
shared_name%#Adam/BatchNormProteinDecode2/beta/m
�
7Adam/BatchNormProteinDecode2/beta/m/Read/ReadVariableOpReadVariableOp#Adam/BatchNormProteinDecode2/beta/m*
_output_shapes
:/*
dtype0
�
Adam/gene_decoder_last/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��
*0
shared_name!Adam/gene_decoder_last/kernel/m
�
3Adam/gene_decoder_last/kernel/m/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_last/kernel/m* 
_output_shapes
:
��
*
dtype0
�
Adam/gene_decoder_last/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:�
*.
shared_nameAdam/gene_decoder_last/bias/m
�
1Adam/gene_decoder_last/bias/m/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_last/bias/m*
_output_shapes	
:�
*
dtype0
�
"Adam/protein_decoder_last/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	/�*3
shared_name$"Adam/protein_decoder_last/kernel/m
�
6Adam/protein_decoder_last/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/protein_decoder_last/kernel/m*
_output_shapes
:	/�*
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
�
�*-
shared_nameAdam/gene_encoder_1/kernel/v
�
0Adam/gene_encoder_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gene_encoder_1/kernel/v* 
_output_shapes
:
�
�*
dtype0
�
Adam/gene_encoder_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameAdam/gene_encoder_1/bias/v
�
.Adam/gene_encoder_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/gene_encoder_1/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/protein_encoder_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�/*0
shared_name!Adam/protein_encoder_1/kernel/v
�
3Adam/protein_encoder_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/protein_encoder_1/kernel/v*
_output_shapes
:	�/*
dtype0
�
Adam/protein_encoder_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*.
shared_nameAdam/protein_encoder_1/bias/v
�
1Adam/protein_encoder_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/protein_encoder_1/bias/v*
_output_shapes
:/*
dtype0
�
!Adam/BatchNormGeneEncode1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/BatchNormGeneEncode1/gamma/v
�
5Adam/BatchNormGeneEncode1/gamma/v/Read/ReadVariableOpReadVariableOp!Adam/BatchNormGeneEncode1/gamma/v*
_output_shapes	
:�*
dtype0
�
 Adam/BatchNormGeneEncode1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/BatchNormGeneEncode1/beta/v
�
4Adam/BatchNormGeneEncode1/beta/v/Read/ReadVariableOpReadVariableOp Adam/BatchNormGeneEncode1/beta/v*
_output_shapes	
:�*
dtype0
�
$Adam/BatchNormProteinEncode1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*5
shared_name&$Adam/BatchNormProteinEncode1/gamma/v
�
8Adam/BatchNormProteinEncode1/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/BatchNormProteinEncode1/gamma/v*
_output_shapes
:/*
dtype0
�
#Adam/BatchNormProteinEncode1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*4
shared_name%#Adam/BatchNormProteinEncode1/beta/v
�
7Adam/BatchNormProteinEncode1/beta/v/Read/ReadVariableOpReadVariableOp#Adam/BatchNormProteinEncode1/beta/v*
_output_shapes
:/*
dtype0
�
Adam/gene_encoder_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�P*-
shared_nameAdam/gene_encoder_2/kernel/v
�
0Adam/gene_encoder_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gene_encoder_2/kernel/v*
_output_shapes
:	�P*
dtype0
�
Adam/gene_encoder_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_nameAdam/gene_encoder_2/bias/v
�
.Adam/gene_encoder_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/gene_encoder_2/bias/v*
_output_shapes
:P*
dtype0
�
Adam/protein_encoder_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/*0
shared_name!Adam/protein_encoder_2/kernel/v
�
3Adam/protein_encoder_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/protein_encoder_2/kernel/v*
_output_shapes

:/*
dtype0
�
Adam/protein_encoder_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/protein_encoder_2/bias/v
�
1Adam/protein_encoder_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/protein_encoder_2/bias/v*
_output_shapes
:*
dtype0
�
!Adam/BatchNormGeneEncode2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*2
shared_name#!Adam/BatchNormGeneEncode2/gamma/v
�
5Adam/BatchNormGeneEncode2/gamma/v/Read/ReadVariableOpReadVariableOp!Adam/BatchNormGeneEncode2/gamma/v*
_output_shapes
:P*
dtype0
�
 Adam/BatchNormGeneEncode2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*1
shared_name" Adam/BatchNormGeneEncode2/beta/v
�
4Adam/BatchNormGeneEncode2/beta/v/Read/ReadVariableOpReadVariableOp Adam/BatchNormGeneEncode2/beta/v*
_output_shapes
:P*
dtype0
�
$Adam/BatchNormProteinEncode2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/BatchNormProteinEncode2/gamma/v
�
8Adam/BatchNormProteinEncode2/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/BatchNormProteinEncode2/gamma/v*
_output_shapes
:*
dtype0
�
#Adam/BatchNormProteinEncode2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/BatchNormProteinEncode2/beta/v
�
7Adam/BatchNormProteinEncode2/beta/v/Read/ReadVariableOpReadVariableOp#Adam/BatchNormProteinEncode2/beta/v*
_output_shapes
:*
dtype0
�
Adam/EmbeddingDimDense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:[@*0
shared_name!Adam/EmbeddingDimDense/kernel/v
�
3Adam/EmbeddingDimDense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/EmbeddingDimDense/kernel/v*
_output_shapes

:[@*
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
:@P*-
shared_nameAdam/gene_decoder_1/kernel/v
�
0Adam/gene_decoder_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_1/kernel/v*
_output_shapes

:@P*
dtype0
�
Adam/gene_decoder_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*+
shared_nameAdam/gene_decoder_1/bias/v
�
.Adam/gene_decoder_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_1/bias/v*
_output_shapes
:P*
dtype0
�
Adam/protein_decoder_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!Adam/protein_decoder_1/kernel/v
�
3Adam/protein_decoder_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/protein_decoder_1/kernel/v*
_output_shapes

:@*
dtype0
�
Adam/protein_decoder_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/protein_decoder_1/bias/v
�
1Adam/protein_decoder_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/protein_decoder_1/bias/v*
_output_shapes
:*
dtype0
�
!Adam/BatchNormGeneDecode1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*2
shared_name#!Adam/BatchNormGeneDecode1/gamma/v
�
5Adam/BatchNormGeneDecode1/gamma/v/Read/ReadVariableOpReadVariableOp!Adam/BatchNormGeneDecode1/gamma/v*
_output_shapes
:P*
dtype0
�
 Adam/BatchNormGeneDecode1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*1
shared_name" Adam/BatchNormGeneDecode1/beta/v
�
4Adam/BatchNormGeneDecode1/beta/v/Read/ReadVariableOpReadVariableOp Adam/BatchNormGeneDecode1/beta/v*
_output_shapes
:P*
dtype0
�
$Adam/BatchNormProteinDecode1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/BatchNormProteinDecode1/gamma/v
�
8Adam/BatchNormProteinDecode1/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/BatchNormProteinDecode1/gamma/v*
_output_shapes
:*
dtype0
�
#Adam/BatchNormProteinDecode1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/BatchNormProteinDecode1/beta/v
�
7Adam/BatchNormProteinDecode1/beta/v/Read/ReadVariableOpReadVariableOp#Adam/BatchNormProteinDecode1/beta/v*
_output_shapes
:*
dtype0
�
Adam/gene_decoder_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	P�*-
shared_nameAdam/gene_decoder_2/kernel/v
�
0Adam/gene_decoder_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_2/kernel/v*
_output_shapes
:	P�*
dtype0
�
Adam/gene_decoder_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*+
shared_nameAdam/gene_decoder_2/bias/v
�
.Adam/gene_decoder_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_2/bias/v*
_output_shapes	
:�*
dtype0
�
Adam/protein_decoder_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:/*0
shared_name!Adam/protein_decoder_2/kernel/v
�
3Adam/protein_decoder_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/protein_decoder_2/kernel/v*
_output_shapes

:/*
dtype0
�
Adam/protein_decoder_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*.
shared_nameAdam/protein_decoder_2/bias/v
�
1Adam/protein_decoder_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/protein_decoder_2/bias/v*
_output_shapes
:/*
dtype0
�
!Adam/BatchNormGeneDecode2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*2
shared_name#!Adam/BatchNormGeneDecode2/gamma/v
�
5Adam/BatchNormGeneDecode2/gamma/v/Read/ReadVariableOpReadVariableOp!Adam/BatchNormGeneDecode2/gamma/v*
_output_shapes	
:�*
dtype0
�
 Adam/BatchNormGeneDecode2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*1
shared_name" Adam/BatchNormGeneDecode2/beta/v
�
4Adam/BatchNormGeneDecode2/beta/v/Read/ReadVariableOpReadVariableOp Adam/BatchNormGeneDecode2/beta/v*
_output_shapes	
:�*
dtype0
�
$Adam/BatchNormProteinDecode2/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*5
shared_name&$Adam/BatchNormProteinDecode2/gamma/v
�
8Adam/BatchNormProteinDecode2/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/BatchNormProteinDecode2/gamma/v*
_output_shapes
:/*
dtype0
�
#Adam/BatchNormProteinDecode2/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:/*4
shared_name%#Adam/BatchNormProteinDecode2/beta/v
�
7Adam/BatchNormProteinDecode2/beta/v/Read/ReadVariableOpReadVariableOp#Adam/BatchNormProteinDecode2/beta/v*
_output_shapes
:/*
dtype0
�
Adam/gene_decoder_last/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��
*0
shared_name!Adam/gene_decoder_last/kernel/v
�
3Adam/gene_decoder_last/kernel/v/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_last/kernel/v* 
_output_shapes
:
��
*
dtype0
�
Adam/gene_decoder_last/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:�
*.
shared_nameAdam/gene_decoder_last/bias/v
�
1Adam/gene_decoder_last/bias/v/Read/ReadVariableOpReadVariableOpAdam/gene_decoder_last/bias/v*
_output_shapes	
:�
*
dtype0
�
"Adam/protein_decoder_last/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	/�*3
shared_name$"Adam/protein_decoder_last/kernel/v
�
6Adam/protein_decoder_last/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/protein_decoder_last/kernel/v*
_output_shapes
:	/�*
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
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
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
layer_with_weights-15
layer-18
layer_with_weights-16
layer-19
layer_with_weights-17
layer-20
layer_with_weights-18
layer-21
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
* 
�

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses*
�

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses*
�
0axis
	1gamma
2beta
3moving_mean
4moving_variance
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses*
�
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses*
�

Fkernel
Gbias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses*
�

Nkernel
Obias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses*
�
Vaxis
	Wgamma
Xbeta
Ymoving_mean
Zmoving_variance
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses*
�
aaxis
	bgamma
cbeta
dmoving_mean
emoving_variance
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses*
�
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses* 
�

rkernel
sbias
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses*
�

zkernel
{bias
|	variables
}trainable_variables
~regularization_losses
	keras_api
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
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate m�!m�(m�)m�1m�2m�<m�=m�Fm�Gm�Nm�Om�Wm�Xm�bm�cm�rm�sm�zm�{m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m� v�!v�(v�)v�1v�2v�<v�=v�Fv�Gv�Nv�Ov�Wv�Xv�bv�cv�rv�sv�zv�{v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�*
�
 0
!1
(2
)3
14
25
36
47
<8
=9
>10
?11
F12
G13
N14
O15
W16
X17
Y18
Z19
b20
c21
d22
e23
r24
s25
z26
{27
�28
�29
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
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53*
�
 0
!1
(2
)3
14
25
<6
=7
F8
G9
N10
O11
W12
X13
b14
c15
r16
s17
z18
{19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
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
 0
!1*

 0
!1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*
* 
* 
hb
VARIABLE_VALUEprotein_encoder_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEprotein_encoder_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

(0
)1*

(0
)1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses*
* 
* 
* 
ic
VARIABLE_VALUEBatchNormGeneEncode1/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEBatchNormGeneEncode1/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE BatchNormGeneEncode1/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE$BatchNormGeneEncode1/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
10
21
32
43*

10
21*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEBatchNormProteinEncode1/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEBatchNormProteinEncode1/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#BatchNormProteinEncode1/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE'BatchNormProteinEncode1/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
<0
=1
>2
?3*

<0
=1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*
* 
* 
e_
VARIABLE_VALUEgene_encoder_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEgene_encoder_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

F0
G1*

F0
G1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses*
* 
* 
hb
VARIABLE_VALUEprotein_encoder_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEprotein_encoder_2/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*

N0
O1*

N0
O1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*
* 
* 
* 
ic
VARIABLE_VALUEBatchNormGeneEncode2/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEBatchNormGeneEncode2/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE BatchNormGeneEncode2/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE$BatchNormGeneEncode2/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
W0
X1
Y2
Z3*

W0
X1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEBatchNormProteinEncode2/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEBatchNormProteinEncode2/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#BatchNormProteinEncode2/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE'BatchNormProteinEncode2/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
b0
c1
d2
e3*

b0
c1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses* 
* 
* 
hb
VARIABLE_VALUEEmbeddingDimDense/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEEmbeddingDimDense/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

r0
s1*

r0
s1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses*
* 
* 
e_
VARIABLE_VALUEgene_decoder_1/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEgene_decoder_1/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

z0
{1*

z0
{1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
|	variables
}trainable_variables
~regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
ic
VARIABLE_VALUEprotein_decoder_1/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEprotein_decoder_1/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
* 
jd
VARIABLE_VALUEBatchNormGeneDecode1/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEBatchNormGeneDecode1/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE BatchNormGeneDecode1/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE$BatchNormGeneDecode1/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
f`
VARIABLE_VALUEgene_decoder_2/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEgene_decoder_2/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
ic
VARIABLE_VALUEprotein_decoder_2/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEprotein_decoder_2/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
* 
jd
VARIABLE_VALUEBatchNormGeneDecode2/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEBatchNormGeneDecode2/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE*
vp
VARIABLE_VALUE BatchNormGeneDecode2/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE$BatchNormGeneDecode2/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
* 
mg
VARIABLE_VALUEBatchNormProteinDecode2/gamma6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEBatchNormProteinDecode2/beta5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#BatchNormProteinDecode2/moving_mean<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE'BatchNormProteinDecode2/moving_variance@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
ic
VARIABLE_VALUEgene_decoder_last/kernel7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEgene_decoder_last/bias5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
lf
VARIABLE_VALUEprotein_decoder_last/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEprotein_decoder_last/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
�
30
41
>2
?3
Y4
Z5
d6
e7
�8
�9
�10
�11
�12
�13
�14
�15*
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
17
18
19
20
21*

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
* 
* 
* 
* 
* 

30
41*
* 
* 
* 
* 

>0
?1*
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
Y0
Z1*
* 
* 
* 
* 

d0
e1*
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
VARIABLE_VALUEAdam/protein_encoder_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_encoder_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/BatchNormGeneEncode1/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/BatchNormGeneEncode1/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE$Adam/BatchNormProteinEncode1/gamma/mQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/BatchNormProteinEncode1/beta/mPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_encoder_2/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/gene_encoder_2/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_encoder_2/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_encoder_2/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/BatchNormGeneEncode2/gamma/mQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/BatchNormGeneEncode2/beta/mPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE$Adam/BatchNormProteinEncode2/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/BatchNormProteinEncode2/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/EmbeddingDimDense/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/EmbeddingDimDense/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_decoder_1/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/gene_decoder_1/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_decoder_1/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_decoder_1/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/BatchNormGeneDecode1/gamma/mRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/BatchNormGeneDecode1/beta/mQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE$Adam/BatchNormProteinDecode1/gamma/mRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/BatchNormProteinDecode1/beta/mQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_decoder_2/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEAdam/gene_decoder_2/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_decoder_2/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_decoder_2/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/BatchNormGeneDecode2/gamma/mRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/BatchNormGeneDecode2/beta/mQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE$Adam/BatchNormProteinDecode2/gamma/mRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/BatchNormProteinDecode2/beta/mQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_decoder_last/kernel/mSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_decoder_last/bias/mQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/protein_decoder_last/kernel/mSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/protein_decoder_last/bias/mQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_encoder_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/gene_encoder_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_encoder_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_encoder_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/BatchNormGeneEncode1/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/BatchNormGeneEncode1/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE$Adam/BatchNormProteinEncode1/gamma/vQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/BatchNormProteinEncode1/beta/vPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_encoder_2/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/gene_encoder_2/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_encoder_2/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_encoder_2/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/BatchNormGeneEncode2/gamma/vQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/BatchNormGeneEncode2/beta/vPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE$Adam/BatchNormProteinEncode2/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/BatchNormProteinEncode2/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/EmbeddingDimDense/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/EmbeddingDimDense/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_decoder_1/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�~
VARIABLE_VALUEAdam/gene_decoder_1/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_decoder_1/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_decoder_1/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/BatchNormGeneDecode1/gamma/vRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/BatchNormGeneDecode1/beta/vQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE$Adam/BatchNormProteinDecode1/gamma/vRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/BatchNormProteinDecode1/beta/vQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_decoder_2/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
VARIABLE_VALUEAdam/gene_decoder_2/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_decoder_2/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/protein_decoder_2/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE!Adam/BatchNormGeneDecode2/gamma/vRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/BatchNormGeneDecode2/beta/vQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE$Adam/BatchNormProteinDecode2/gamma/vRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE#Adam/BatchNormProteinDecode2/beta/vQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_decoder_last/kernel/vSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUEAdam/gene_decoder_last/bias/vQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE"Adam/protein_decoder_last/kernel/vSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
��
VARIABLE_VALUE Adam/protein_decoder_last/bias/vQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
�
 serving_default_Gene_Input_LayerPlaceholder*(
_output_shapes
:����������
*
dtype0*
shape:����������

�
#serving_default_Protein_Input_LayerPlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCall serving_default_Gene_Input_Layer#serving_default_Protein_Input_Layerprotein_encoder_1/kernelprotein_encoder_1/biasgene_encoder_1/kernelgene_encoder_1/bias'BatchNormProteinEncode1/moving_varianceBatchNormProteinEncode1/gamma#BatchNormProteinEncode1/moving_meanBatchNormProteinEncode1/beta$BatchNormGeneEncode1/moving_varianceBatchNormGeneEncode1/gamma BatchNormGeneEncode1/moving_meanBatchNormGeneEncode1/betaprotein_encoder_2/kernelprotein_encoder_2/biasgene_encoder_2/kernelgene_encoder_2/bias$BatchNormGeneEncode2/moving_varianceBatchNormGeneEncode2/gamma BatchNormGeneEncode2/moving_meanBatchNormGeneEncode2/beta'BatchNormProteinEncode2/moving_varianceBatchNormProteinEncode2/gamma#BatchNormProteinEncode2/moving_meanBatchNormProteinEncode2/betaEmbeddingDimDense/kernelEmbeddingDimDense/biasprotein_decoder_1/kernelprotein_decoder_1/biasgene_decoder_1/kernelgene_decoder_1/bias'BatchNormProteinDecode1/moving_varianceBatchNormProteinDecode1/gamma#BatchNormProteinDecode1/moving_meanBatchNormProteinDecode1/beta$BatchNormGeneDecode1/moving_varianceBatchNormGeneDecode1/gamma BatchNormGeneDecode1/moving_meanBatchNormGeneDecode1/betaprotein_decoder_2/kernelprotein_decoder_2/biasgene_decoder_2/kernelgene_decoder_2/bias'BatchNormProteinDecode2/moving_varianceBatchNormProteinDecode2/gamma#BatchNormProteinDecode2/moving_meanBatchNormProteinDecode2/beta$BatchNormGeneDecode2/moving_varianceBatchNormGeneDecode2/gamma BatchNormGeneDecode2/moving_meanBatchNormGeneDecode2/betaprotein_decoder_last/kernelprotein_decoder_last/biasgene_decoder_last/kernelgene_decoder_last/bias*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:����������
:����������*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./01234567*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_331111
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�:
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)gene_encoder_1/kernel/Read/ReadVariableOp'gene_encoder_1/bias/Read/ReadVariableOp,protein_encoder_1/kernel/Read/ReadVariableOp*protein_encoder_1/bias/Read/ReadVariableOp.BatchNormGeneEncode1/gamma/Read/ReadVariableOp-BatchNormGeneEncode1/beta/Read/ReadVariableOp4BatchNormGeneEncode1/moving_mean/Read/ReadVariableOp8BatchNormGeneEncode1/moving_variance/Read/ReadVariableOp1BatchNormProteinEncode1/gamma/Read/ReadVariableOp0BatchNormProteinEncode1/beta/Read/ReadVariableOp7BatchNormProteinEncode1/moving_mean/Read/ReadVariableOp;BatchNormProteinEncode1/moving_variance/Read/ReadVariableOp)gene_encoder_2/kernel/Read/ReadVariableOp'gene_encoder_2/bias/Read/ReadVariableOp,protein_encoder_2/kernel/Read/ReadVariableOp*protein_encoder_2/bias/Read/ReadVariableOp.BatchNormGeneEncode2/gamma/Read/ReadVariableOp-BatchNormGeneEncode2/beta/Read/ReadVariableOp4BatchNormGeneEncode2/moving_mean/Read/ReadVariableOp8BatchNormGeneEncode2/moving_variance/Read/ReadVariableOp1BatchNormProteinEncode2/gamma/Read/ReadVariableOp0BatchNormProteinEncode2/beta/Read/ReadVariableOp7BatchNormProteinEncode2/moving_mean/Read/ReadVariableOp;BatchNormProteinEncode2/moving_variance/Read/ReadVariableOp,EmbeddingDimDense/kernel/Read/ReadVariableOp*EmbeddingDimDense/bias/Read/ReadVariableOp)gene_decoder_1/kernel/Read/ReadVariableOp'gene_decoder_1/bias/Read/ReadVariableOp,protein_decoder_1/kernel/Read/ReadVariableOp*protein_decoder_1/bias/Read/ReadVariableOp.BatchNormGeneDecode1/gamma/Read/ReadVariableOp-BatchNormGeneDecode1/beta/Read/ReadVariableOp4BatchNormGeneDecode1/moving_mean/Read/ReadVariableOp8BatchNormGeneDecode1/moving_variance/Read/ReadVariableOp1BatchNormProteinDecode1/gamma/Read/ReadVariableOp0BatchNormProteinDecode1/beta/Read/ReadVariableOp7BatchNormProteinDecode1/moving_mean/Read/ReadVariableOp;BatchNormProteinDecode1/moving_variance/Read/ReadVariableOp)gene_decoder_2/kernel/Read/ReadVariableOp'gene_decoder_2/bias/Read/ReadVariableOp,protein_decoder_2/kernel/Read/ReadVariableOp*protein_decoder_2/bias/Read/ReadVariableOp.BatchNormGeneDecode2/gamma/Read/ReadVariableOp-BatchNormGeneDecode2/beta/Read/ReadVariableOp4BatchNormGeneDecode2/moving_mean/Read/ReadVariableOp8BatchNormGeneDecode2/moving_variance/Read/ReadVariableOp1BatchNormProteinDecode2/gamma/Read/ReadVariableOp0BatchNormProteinDecode2/beta/Read/ReadVariableOp7BatchNormProteinDecode2/moving_mean/Read/ReadVariableOp;BatchNormProteinDecode2/moving_variance/Read/ReadVariableOp,gene_decoder_last/kernel/Read/ReadVariableOp*gene_decoder_last/bias/Read/ReadVariableOp/protein_decoder_last/kernel/Read/ReadVariableOp-protein_decoder_last/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp0Adam/gene_encoder_1/kernel/m/Read/ReadVariableOp.Adam/gene_encoder_1/bias/m/Read/ReadVariableOp3Adam/protein_encoder_1/kernel/m/Read/ReadVariableOp1Adam/protein_encoder_1/bias/m/Read/ReadVariableOp5Adam/BatchNormGeneEncode1/gamma/m/Read/ReadVariableOp4Adam/BatchNormGeneEncode1/beta/m/Read/ReadVariableOp8Adam/BatchNormProteinEncode1/gamma/m/Read/ReadVariableOp7Adam/BatchNormProteinEncode1/beta/m/Read/ReadVariableOp0Adam/gene_encoder_2/kernel/m/Read/ReadVariableOp.Adam/gene_encoder_2/bias/m/Read/ReadVariableOp3Adam/protein_encoder_2/kernel/m/Read/ReadVariableOp1Adam/protein_encoder_2/bias/m/Read/ReadVariableOp5Adam/BatchNormGeneEncode2/gamma/m/Read/ReadVariableOp4Adam/BatchNormGeneEncode2/beta/m/Read/ReadVariableOp8Adam/BatchNormProteinEncode2/gamma/m/Read/ReadVariableOp7Adam/BatchNormProteinEncode2/beta/m/Read/ReadVariableOp3Adam/EmbeddingDimDense/kernel/m/Read/ReadVariableOp1Adam/EmbeddingDimDense/bias/m/Read/ReadVariableOp0Adam/gene_decoder_1/kernel/m/Read/ReadVariableOp.Adam/gene_decoder_1/bias/m/Read/ReadVariableOp3Adam/protein_decoder_1/kernel/m/Read/ReadVariableOp1Adam/protein_decoder_1/bias/m/Read/ReadVariableOp5Adam/BatchNormGeneDecode1/gamma/m/Read/ReadVariableOp4Adam/BatchNormGeneDecode1/beta/m/Read/ReadVariableOp8Adam/BatchNormProteinDecode1/gamma/m/Read/ReadVariableOp7Adam/BatchNormProteinDecode1/beta/m/Read/ReadVariableOp0Adam/gene_decoder_2/kernel/m/Read/ReadVariableOp.Adam/gene_decoder_2/bias/m/Read/ReadVariableOp3Adam/protein_decoder_2/kernel/m/Read/ReadVariableOp1Adam/protein_decoder_2/bias/m/Read/ReadVariableOp5Adam/BatchNormGeneDecode2/gamma/m/Read/ReadVariableOp4Adam/BatchNormGeneDecode2/beta/m/Read/ReadVariableOp8Adam/BatchNormProteinDecode2/gamma/m/Read/ReadVariableOp7Adam/BatchNormProteinDecode2/beta/m/Read/ReadVariableOp3Adam/gene_decoder_last/kernel/m/Read/ReadVariableOp1Adam/gene_decoder_last/bias/m/Read/ReadVariableOp6Adam/protein_decoder_last/kernel/m/Read/ReadVariableOp4Adam/protein_decoder_last/bias/m/Read/ReadVariableOp0Adam/gene_encoder_1/kernel/v/Read/ReadVariableOp.Adam/gene_encoder_1/bias/v/Read/ReadVariableOp3Adam/protein_encoder_1/kernel/v/Read/ReadVariableOp1Adam/protein_encoder_1/bias/v/Read/ReadVariableOp5Adam/BatchNormGeneEncode1/gamma/v/Read/ReadVariableOp4Adam/BatchNormGeneEncode1/beta/v/Read/ReadVariableOp8Adam/BatchNormProteinEncode1/gamma/v/Read/ReadVariableOp7Adam/BatchNormProteinEncode1/beta/v/Read/ReadVariableOp0Adam/gene_encoder_2/kernel/v/Read/ReadVariableOp.Adam/gene_encoder_2/bias/v/Read/ReadVariableOp3Adam/protein_encoder_2/kernel/v/Read/ReadVariableOp1Adam/protein_encoder_2/bias/v/Read/ReadVariableOp5Adam/BatchNormGeneEncode2/gamma/v/Read/ReadVariableOp4Adam/BatchNormGeneEncode2/beta/v/Read/ReadVariableOp8Adam/BatchNormProteinEncode2/gamma/v/Read/ReadVariableOp7Adam/BatchNormProteinEncode2/beta/v/Read/ReadVariableOp3Adam/EmbeddingDimDense/kernel/v/Read/ReadVariableOp1Adam/EmbeddingDimDense/bias/v/Read/ReadVariableOp0Adam/gene_decoder_1/kernel/v/Read/ReadVariableOp.Adam/gene_decoder_1/bias/v/Read/ReadVariableOp3Adam/protein_decoder_1/kernel/v/Read/ReadVariableOp1Adam/protein_decoder_1/bias/v/Read/ReadVariableOp5Adam/BatchNormGeneDecode1/gamma/v/Read/ReadVariableOp4Adam/BatchNormGeneDecode1/beta/v/Read/ReadVariableOp8Adam/BatchNormProteinDecode1/gamma/v/Read/ReadVariableOp7Adam/BatchNormProteinDecode1/beta/v/Read/ReadVariableOp0Adam/gene_decoder_2/kernel/v/Read/ReadVariableOp.Adam/gene_decoder_2/bias/v/Read/ReadVariableOp3Adam/protein_decoder_2/kernel/v/Read/ReadVariableOp1Adam/protein_decoder_2/bias/v/Read/ReadVariableOp5Adam/BatchNormGeneDecode2/gamma/v/Read/ReadVariableOp4Adam/BatchNormGeneDecode2/beta/v/Read/ReadVariableOp8Adam/BatchNormProteinDecode2/gamma/v/Read/ReadVariableOp7Adam/BatchNormProteinDecode2/beta/v/Read/ReadVariableOp3Adam/gene_decoder_last/kernel/v/Read/ReadVariableOp1Adam/gene_decoder_last/bias/v/Read/ReadVariableOp6Adam/protein_decoder_last/kernel/v/Read/ReadVariableOp4Adam/protein_decoder_last/bias/v/Read/ReadVariableOpConst*�
Tin�
�2�	*
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
__inference__traced_save_332432
�$
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegene_encoder_1/kernelgene_encoder_1/biasprotein_encoder_1/kernelprotein_encoder_1/biasBatchNormGeneEncode1/gammaBatchNormGeneEncode1/beta BatchNormGeneEncode1/moving_mean$BatchNormGeneEncode1/moving_varianceBatchNormProteinEncode1/gammaBatchNormProteinEncode1/beta#BatchNormProteinEncode1/moving_mean'BatchNormProteinEncode1/moving_variancegene_encoder_2/kernelgene_encoder_2/biasprotein_encoder_2/kernelprotein_encoder_2/biasBatchNormGeneEncode2/gammaBatchNormGeneEncode2/beta BatchNormGeneEncode2/moving_mean$BatchNormGeneEncode2/moving_varianceBatchNormProteinEncode2/gammaBatchNormProteinEncode2/beta#BatchNormProteinEncode2/moving_mean'BatchNormProteinEncode2/moving_varianceEmbeddingDimDense/kernelEmbeddingDimDense/biasgene_decoder_1/kernelgene_decoder_1/biasprotein_decoder_1/kernelprotein_decoder_1/biasBatchNormGeneDecode1/gammaBatchNormGeneDecode1/beta BatchNormGeneDecode1/moving_mean$BatchNormGeneDecode1/moving_varianceBatchNormProteinDecode1/gammaBatchNormProteinDecode1/beta#BatchNormProteinDecode1/moving_mean'BatchNormProteinDecode1/moving_variancegene_decoder_2/kernelgene_decoder_2/biasprotein_decoder_2/kernelprotein_decoder_2/biasBatchNormGeneDecode2/gammaBatchNormGeneDecode2/beta BatchNormGeneDecode2/moving_mean$BatchNormGeneDecode2/moving_varianceBatchNormProteinDecode2/gammaBatchNormProteinDecode2/beta#BatchNormProteinDecode2/moving_mean'BatchNormProteinDecode2/moving_variancegene_decoder_last/kernelgene_decoder_last/biasprotein_decoder_last/kernelprotein_decoder_last/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/gene_encoder_1/kernel/mAdam/gene_encoder_1/bias/mAdam/protein_encoder_1/kernel/mAdam/protein_encoder_1/bias/m!Adam/BatchNormGeneEncode1/gamma/m Adam/BatchNormGeneEncode1/beta/m$Adam/BatchNormProteinEncode1/gamma/m#Adam/BatchNormProteinEncode1/beta/mAdam/gene_encoder_2/kernel/mAdam/gene_encoder_2/bias/mAdam/protein_encoder_2/kernel/mAdam/protein_encoder_2/bias/m!Adam/BatchNormGeneEncode2/gamma/m Adam/BatchNormGeneEncode2/beta/m$Adam/BatchNormProteinEncode2/gamma/m#Adam/BatchNormProteinEncode2/beta/mAdam/EmbeddingDimDense/kernel/mAdam/EmbeddingDimDense/bias/mAdam/gene_decoder_1/kernel/mAdam/gene_decoder_1/bias/mAdam/protein_decoder_1/kernel/mAdam/protein_decoder_1/bias/m!Adam/BatchNormGeneDecode1/gamma/m Adam/BatchNormGeneDecode1/beta/m$Adam/BatchNormProteinDecode1/gamma/m#Adam/BatchNormProteinDecode1/beta/mAdam/gene_decoder_2/kernel/mAdam/gene_decoder_2/bias/mAdam/protein_decoder_2/kernel/mAdam/protein_decoder_2/bias/m!Adam/BatchNormGeneDecode2/gamma/m Adam/BatchNormGeneDecode2/beta/m$Adam/BatchNormProteinDecode2/gamma/m#Adam/BatchNormProteinDecode2/beta/mAdam/gene_decoder_last/kernel/mAdam/gene_decoder_last/bias/m"Adam/protein_decoder_last/kernel/m Adam/protein_decoder_last/bias/mAdam/gene_encoder_1/kernel/vAdam/gene_encoder_1/bias/vAdam/protein_encoder_1/kernel/vAdam/protein_encoder_1/bias/v!Adam/BatchNormGeneEncode1/gamma/v Adam/BatchNormGeneEncode1/beta/v$Adam/BatchNormProteinEncode1/gamma/v#Adam/BatchNormProteinEncode1/beta/vAdam/gene_encoder_2/kernel/vAdam/gene_encoder_2/bias/vAdam/protein_encoder_2/kernel/vAdam/protein_encoder_2/bias/v!Adam/BatchNormGeneEncode2/gamma/v Adam/BatchNormGeneEncode2/beta/v$Adam/BatchNormProteinEncode2/gamma/v#Adam/BatchNormProteinEncode2/beta/vAdam/EmbeddingDimDense/kernel/vAdam/EmbeddingDimDense/bias/vAdam/gene_decoder_1/kernel/vAdam/gene_decoder_1/bias/vAdam/protein_decoder_1/kernel/vAdam/protein_decoder_1/bias/v!Adam/BatchNormGeneDecode1/gamma/v Adam/BatchNormGeneDecode1/beta/v$Adam/BatchNormProteinDecode1/gamma/v#Adam/BatchNormProteinDecode1/beta/vAdam/gene_decoder_2/kernel/vAdam/gene_decoder_2/bias/vAdam/protein_decoder_2/kernel/vAdam/protein_decoder_2/bias/v!Adam/BatchNormGeneDecode2/gamma/v Adam/BatchNormGeneDecode2/beta/v$Adam/BatchNormProteinDecode2/gamma/v#Adam/BatchNormProteinDecode2/beta/vAdam/gene_decoder_last/kernel/vAdam/gene_decoder_last/bias/v"Adam/protein_decoder_last/kernel/v Adam/protein_decoder_last/bias/v*�
Tin�
�2�*
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
"__inference__traced_restore_332865��
��
�`
"__inference__traced_restore_332865
file_prefix:
&assignvariableop_gene_encoder_1_kernel:
�
�5
&assignvariableop_1_gene_encoder_1_bias:	�>
+assignvariableop_2_protein_encoder_1_kernel:	�/7
)assignvariableop_3_protein_encoder_1_bias:/<
-assignvariableop_4_batchnormgeneencode1_gamma:	�;
,assignvariableop_5_batchnormgeneencode1_beta:	�B
3assignvariableop_6_batchnormgeneencode1_moving_mean:	�F
7assignvariableop_7_batchnormgeneencode1_moving_variance:	�>
0assignvariableop_8_batchnormproteinencode1_gamma:/=
/assignvariableop_9_batchnormproteinencode1_beta:/E
7assignvariableop_10_batchnormproteinencode1_moving_mean:/I
;assignvariableop_11_batchnormproteinencode1_moving_variance:/<
)assignvariableop_12_gene_encoder_2_kernel:	�P5
'assignvariableop_13_gene_encoder_2_bias:P>
,assignvariableop_14_protein_encoder_2_kernel:/8
*assignvariableop_15_protein_encoder_2_bias:<
.assignvariableop_16_batchnormgeneencode2_gamma:P;
-assignvariableop_17_batchnormgeneencode2_beta:PB
4assignvariableop_18_batchnormgeneencode2_moving_mean:PF
8assignvariableop_19_batchnormgeneencode2_moving_variance:P?
1assignvariableop_20_batchnormproteinencode2_gamma:>
0assignvariableop_21_batchnormproteinencode2_beta:E
7assignvariableop_22_batchnormproteinencode2_moving_mean:I
;assignvariableop_23_batchnormproteinencode2_moving_variance:>
,assignvariableop_24_embeddingdimdense_kernel:[@8
*assignvariableop_25_embeddingdimdense_bias:@;
)assignvariableop_26_gene_decoder_1_kernel:@P5
'assignvariableop_27_gene_decoder_1_bias:P>
,assignvariableop_28_protein_decoder_1_kernel:@8
*assignvariableop_29_protein_decoder_1_bias:<
.assignvariableop_30_batchnormgenedecode1_gamma:P;
-assignvariableop_31_batchnormgenedecode1_beta:PB
4assignvariableop_32_batchnormgenedecode1_moving_mean:PF
8assignvariableop_33_batchnormgenedecode1_moving_variance:P?
1assignvariableop_34_batchnormproteindecode1_gamma:>
0assignvariableop_35_batchnormproteindecode1_beta:E
7assignvariableop_36_batchnormproteindecode1_moving_mean:I
;assignvariableop_37_batchnormproteindecode1_moving_variance:<
)assignvariableop_38_gene_decoder_2_kernel:	P�6
'assignvariableop_39_gene_decoder_2_bias:	�>
,assignvariableop_40_protein_decoder_2_kernel:/8
*assignvariableop_41_protein_decoder_2_bias:/=
.assignvariableop_42_batchnormgenedecode2_gamma:	�<
-assignvariableop_43_batchnormgenedecode2_beta:	�C
4assignvariableop_44_batchnormgenedecode2_moving_mean:	�G
8assignvariableop_45_batchnormgenedecode2_moving_variance:	�?
1assignvariableop_46_batchnormproteindecode2_gamma:/>
0assignvariableop_47_batchnormproteindecode2_beta:/E
7assignvariableop_48_batchnormproteindecode2_moving_mean:/I
;assignvariableop_49_batchnormproteindecode2_moving_variance:/@
,assignvariableop_50_gene_decoder_last_kernel:
��
9
*assignvariableop_51_gene_decoder_last_bias:	�
B
/assignvariableop_52_protein_decoder_last_kernel:	/�<
-assignvariableop_53_protein_decoder_last_bias:	�'
assignvariableop_54_adam_iter:	 )
assignvariableop_55_adam_beta_1: )
assignvariableop_56_adam_beta_2: (
assignvariableop_57_adam_decay: 0
&assignvariableop_58_adam_learning_rate: #
assignvariableop_59_total: #
assignvariableop_60_count: %
assignvariableop_61_total_1: %
assignvariableop_62_count_1: %
assignvariableop_63_total_2: %
assignvariableop_64_count_2: D
0assignvariableop_65_adam_gene_encoder_1_kernel_m:
�
�=
.assignvariableop_66_adam_gene_encoder_1_bias_m:	�F
3assignvariableop_67_adam_protein_encoder_1_kernel_m:	�/?
1assignvariableop_68_adam_protein_encoder_1_bias_m:/D
5assignvariableop_69_adam_batchnormgeneencode1_gamma_m:	�C
4assignvariableop_70_adam_batchnormgeneencode1_beta_m:	�F
8assignvariableop_71_adam_batchnormproteinencode1_gamma_m:/E
7assignvariableop_72_adam_batchnormproteinencode1_beta_m:/C
0assignvariableop_73_adam_gene_encoder_2_kernel_m:	�P<
.assignvariableop_74_adam_gene_encoder_2_bias_m:PE
3assignvariableop_75_adam_protein_encoder_2_kernel_m:/?
1assignvariableop_76_adam_protein_encoder_2_bias_m:C
5assignvariableop_77_adam_batchnormgeneencode2_gamma_m:PB
4assignvariableop_78_adam_batchnormgeneencode2_beta_m:PF
8assignvariableop_79_adam_batchnormproteinencode2_gamma_m:E
7assignvariableop_80_adam_batchnormproteinencode2_beta_m:E
3assignvariableop_81_adam_embeddingdimdense_kernel_m:[@?
1assignvariableop_82_adam_embeddingdimdense_bias_m:@B
0assignvariableop_83_adam_gene_decoder_1_kernel_m:@P<
.assignvariableop_84_adam_gene_decoder_1_bias_m:PE
3assignvariableop_85_adam_protein_decoder_1_kernel_m:@?
1assignvariableop_86_adam_protein_decoder_1_bias_m:C
5assignvariableop_87_adam_batchnormgenedecode1_gamma_m:PB
4assignvariableop_88_adam_batchnormgenedecode1_beta_m:PF
8assignvariableop_89_adam_batchnormproteindecode1_gamma_m:E
7assignvariableop_90_adam_batchnormproteindecode1_beta_m:C
0assignvariableop_91_adam_gene_decoder_2_kernel_m:	P�=
.assignvariableop_92_adam_gene_decoder_2_bias_m:	�E
3assignvariableop_93_adam_protein_decoder_2_kernel_m:/?
1assignvariableop_94_adam_protein_decoder_2_bias_m:/D
5assignvariableop_95_adam_batchnormgenedecode2_gamma_m:	�C
4assignvariableop_96_adam_batchnormgenedecode2_beta_m:	�F
8assignvariableop_97_adam_batchnormproteindecode2_gamma_m:/E
7assignvariableop_98_adam_batchnormproteindecode2_beta_m:/G
3assignvariableop_99_adam_gene_decoder_last_kernel_m:
��
A
2assignvariableop_100_adam_gene_decoder_last_bias_m:	�
J
7assignvariableop_101_adam_protein_decoder_last_kernel_m:	/�D
5assignvariableop_102_adam_protein_decoder_last_bias_m:	�E
1assignvariableop_103_adam_gene_encoder_1_kernel_v:
�
�>
/assignvariableop_104_adam_gene_encoder_1_bias_v:	�G
4assignvariableop_105_adam_protein_encoder_1_kernel_v:	�/@
2assignvariableop_106_adam_protein_encoder_1_bias_v:/E
6assignvariableop_107_adam_batchnormgeneencode1_gamma_v:	�D
5assignvariableop_108_adam_batchnormgeneencode1_beta_v:	�G
9assignvariableop_109_adam_batchnormproteinencode1_gamma_v:/F
8assignvariableop_110_adam_batchnormproteinencode1_beta_v:/D
1assignvariableop_111_adam_gene_encoder_2_kernel_v:	�P=
/assignvariableop_112_adam_gene_encoder_2_bias_v:PF
4assignvariableop_113_adam_protein_encoder_2_kernel_v:/@
2assignvariableop_114_adam_protein_encoder_2_bias_v:D
6assignvariableop_115_adam_batchnormgeneencode2_gamma_v:PC
5assignvariableop_116_adam_batchnormgeneencode2_beta_v:PG
9assignvariableop_117_adam_batchnormproteinencode2_gamma_v:F
8assignvariableop_118_adam_batchnormproteinencode2_beta_v:F
4assignvariableop_119_adam_embeddingdimdense_kernel_v:[@@
2assignvariableop_120_adam_embeddingdimdense_bias_v:@C
1assignvariableop_121_adam_gene_decoder_1_kernel_v:@P=
/assignvariableop_122_adam_gene_decoder_1_bias_v:PF
4assignvariableop_123_adam_protein_decoder_1_kernel_v:@@
2assignvariableop_124_adam_protein_decoder_1_bias_v:D
6assignvariableop_125_adam_batchnormgenedecode1_gamma_v:PC
5assignvariableop_126_adam_batchnormgenedecode1_beta_v:PG
9assignvariableop_127_adam_batchnormproteindecode1_gamma_v:F
8assignvariableop_128_adam_batchnormproteindecode1_beta_v:D
1assignvariableop_129_adam_gene_decoder_2_kernel_v:	P�>
/assignvariableop_130_adam_gene_decoder_2_bias_v:	�F
4assignvariableop_131_adam_protein_decoder_2_kernel_v:/@
2assignvariableop_132_adam_protein_decoder_2_bias_v:/E
6assignvariableop_133_adam_batchnormgenedecode2_gamma_v:	�D
5assignvariableop_134_adam_batchnormgenedecode2_beta_v:	�G
9assignvariableop_135_adam_batchnormproteindecode2_gamma_v:/F
8assignvariableop_136_adam_batchnormproteindecode2_beta_v:/H
4assignvariableop_137_adam_gene_decoder_last_kernel_v:
��
A
2assignvariableop_138_adam_gene_decoder_last_bias_v:	�
J
7assignvariableop_139_adam_protein_decoder_last_kernel_v:	/�D
5assignvariableop_140_adam_protein_decoder_last_bias_v:	�
identity_142��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_100�AssignVariableOp_101�AssignVariableOp_102�AssignVariableOp_103�AssignVariableOp_104�AssignVariableOp_105�AssignVariableOp_106�AssignVariableOp_107�AssignVariableOp_108�AssignVariableOp_109�AssignVariableOp_11�AssignVariableOp_110�AssignVariableOp_111�AssignVariableOp_112�AssignVariableOp_113�AssignVariableOp_114�AssignVariableOp_115�AssignVariableOp_116�AssignVariableOp_117�AssignVariableOp_118�AssignVariableOp_119�AssignVariableOp_12�AssignVariableOp_120�AssignVariableOp_121�AssignVariableOp_122�AssignVariableOp_123�AssignVariableOp_124�AssignVariableOp_125�AssignVariableOp_126�AssignVariableOp_127�AssignVariableOp_128�AssignVariableOp_129�AssignVariableOp_13�AssignVariableOp_130�AssignVariableOp_131�AssignVariableOp_132�AssignVariableOp_133�AssignVariableOp_134�AssignVariableOp_135�AssignVariableOp_136�AssignVariableOp_137�AssignVariableOp_138�AssignVariableOp_139�AssignVariableOp_14�AssignVariableOp_140�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�AssignVariableOp_96�AssignVariableOp_97�AssignVariableOp_98�AssignVariableOp_99�O
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�N
value�NB�N�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*�
dtypes�
�2�	[
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
AssignVariableOp_2AssignVariableOp+assignvariableop_2_protein_encoder_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp)assignvariableop_3_protein_encoder_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp-assignvariableop_4_batchnormgeneencode1_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp,assignvariableop_5_batchnormgeneencode1_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp3assignvariableop_6_batchnormgeneencode1_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp7assignvariableop_7_batchnormgeneencode1_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp0assignvariableop_8_batchnormproteinencode1_gammaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp/assignvariableop_9_batchnormproteinencode1_betaIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp7assignvariableop_10_batchnormproteinencode1_moving_meanIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp;assignvariableop_11_batchnormproteinencode1_moving_varianceIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp)assignvariableop_12_gene_encoder_2_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp'assignvariableop_13_gene_encoder_2_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp,assignvariableop_14_protein_encoder_2_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp*assignvariableop_15_protein_encoder_2_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp.assignvariableop_16_batchnormgeneencode2_gammaIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp-assignvariableop_17_batchnormgeneencode2_betaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp4assignvariableop_18_batchnormgeneencode2_moving_meanIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp8assignvariableop_19_batchnormgeneencode2_moving_varianceIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp1assignvariableop_20_batchnormproteinencode2_gammaIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp0assignvariableop_21_batchnormproteinencode2_betaIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp7assignvariableop_22_batchnormproteinencode2_moving_meanIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp;assignvariableop_23_batchnormproteinencode2_moving_varianceIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp,assignvariableop_24_embeddingdimdense_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp*assignvariableop_25_embeddingdimdense_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp)assignvariableop_26_gene_decoder_1_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp'assignvariableop_27_gene_decoder_1_biasIdentity_27:output:0"/device:CPU:0*
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
AssignVariableOp_30AssignVariableOp.assignvariableop_30_batchnormgenedecode1_gammaIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp-assignvariableop_31_batchnormgenedecode1_betaIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp4assignvariableop_32_batchnormgenedecode1_moving_meanIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp8assignvariableop_33_batchnormgenedecode1_moving_varianceIdentity_33:output:0"/device:CPU:0*
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
AssignVariableOp_38AssignVariableOp)assignvariableop_38_gene_decoder_2_kernelIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp'assignvariableop_39_gene_decoder_2_biasIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp,assignvariableop_40_protein_decoder_2_kernelIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp*assignvariableop_41_protein_decoder_2_biasIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp.assignvariableop_42_batchnormgenedecode2_gammaIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp-assignvariableop_43_batchnormgenedecode2_betaIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp4assignvariableop_44_batchnormgenedecode2_moving_meanIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp8assignvariableop_45_batchnormgenedecode2_moving_varianceIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp1assignvariableop_46_batchnormproteindecode2_gammaIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp0assignvariableop_47_batchnormproteindecode2_betaIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp7assignvariableop_48_batchnormproteindecode2_moving_meanIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp;assignvariableop_49_batchnormproteindecode2_moving_varianceIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp,assignvariableop_50_gene_decoder_last_kernelIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp*assignvariableop_51_gene_decoder_last_biasIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp/assignvariableop_52_protein_decoder_last_kernelIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp-assignvariableop_53_protein_decoder_last_biasIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_54AssignVariableOpassignvariableop_54_adam_iterIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOpassignvariableop_55_adam_beta_1Identity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOpassignvariableop_56_adam_beta_2Identity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOpassignvariableop_57_adam_decayIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp&assignvariableop_58_adam_learning_rateIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOpassignvariableop_59_totalIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOpassignvariableop_60_countIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOpassignvariableop_61_total_1Identity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOpassignvariableop_62_count_1Identity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOpassignvariableop_63_total_2Identity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOpassignvariableop_64_count_2Identity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp0assignvariableop_65_adam_gene_encoder_1_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp.assignvariableop_66_adam_gene_encoder_1_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp3assignvariableop_67_adam_protein_encoder_1_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp1assignvariableop_68_adam_protein_encoder_1_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp5assignvariableop_69_adam_batchnormgeneencode1_gamma_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp4assignvariableop_70_adam_batchnormgeneencode1_beta_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp8assignvariableop_71_adam_batchnormproteinencode1_gamma_mIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp7assignvariableop_72_adam_batchnormproteinencode1_beta_mIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp0assignvariableop_73_adam_gene_encoder_2_kernel_mIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp.assignvariableop_74_adam_gene_encoder_2_bias_mIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp3assignvariableop_75_adam_protein_encoder_2_kernel_mIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp1assignvariableop_76_adam_protein_encoder_2_bias_mIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp5assignvariableop_77_adam_batchnormgeneencode2_gamma_mIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp4assignvariableop_78_adam_batchnormgeneencode2_beta_mIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp8assignvariableop_79_adam_batchnormproteinencode2_gamma_mIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp7assignvariableop_80_adam_batchnormproteinencode2_beta_mIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp3assignvariableop_81_adam_embeddingdimdense_kernel_mIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp1assignvariableop_82_adam_embeddingdimdense_bias_mIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp0assignvariableop_83_adam_gene_decoder_1_kernel_mIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp.assignvariableop_84_adam_gene_decoder_1_bias_mIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp3assignvariableop_85_adam_protein_decoder_1_kernel_mIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOp1assignvariableop_86_adam_protein_decoder_1_bias_mIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp5assignvariableop_87_adam_batchnormgenedecode1_gamma_mIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp4assignvariableop_88_adam_batchnormgenedecode1_beta_mIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp8assignvariableop_89_adam_batchnormproteindecode1_gamma_mIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp7assignvariableop_90_adam_batchnormproteindecode1_beta_mIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOp0assignvariableop_91_adam_gene_decoder_2_kernel_mIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOp.assignvariableop_92_adam_gene_decoder_2_bias_mIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOp3assignvariableop_93_adam_protein_decoder_2_kernel_mIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOp1assignvariableop_94_adam_protein_decoder_2_bias_mIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOp5assignvariableop_95_adam_batchnormgenedecode2_gamma_mIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_96AssignVariableOp4assignvariableop_96_adam_batchnormgenedecode2_beta_mIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_97AssignVariableOp8assignvariableop_97_adam_batchnormproteindecode2_gamma_mIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_98AssignVariableOp7assignvariableop_98_adam_batchnormproteindecode2_beta_mIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_99AssignVariableOp3assignvariableop_99_adam_gene_decoder_last_kernel_mIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_100AssignVariableOp2assignvariableop_100_adam_gene_decoder_last_bias_mIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_101AssignVariableOp7assignvariableop_101_adam_protein_decoder_last_kernel_mIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_102AssignVariableOp5assignvariableop_102_adam_protein_decoder_last_bias_mIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_103IdentityRestoreV2:tensors:103"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_103AssignVariableOp1assignvariableop_103_adam_gene_encoder_1_kernel_vIdentity_103:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_104IdentityRestoreV2:tensors:104"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_104AssignVariableOp/assignvariableop_104_adam_gene_encoder_1_bias_vIdentity_104:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_105IdentityRestoreV2:tensors:105"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_105AssignVariableOp4assignvariableop_105_adam_protein_encoder_1_kernel_vIdentity_105:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_106IdentityRestoreV2:tensors:106"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_106AssignVariableOp2assignvariableop_106_adam_protein_encoder_1_bias_vIdentity_106:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_107IdentityRestoreV2:tensors:107"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_107AssignVariableOp6assignvariableop_107_adam_batchnormgeneencode1_gamma_vIdentity_107:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_108IdentityRestoreV2:tensors:108"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_108AssignVariableOp5assignvariableop_108_adam_batchnormgeneencode1_beta_vIdentity_108:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_109IdentityRestoreV2:tensors:109"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_109AssignVariableOp9assignvariableop_109_adam_batchnormproteinencode1_gamma_vIdentity_109:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_110IdentityRestoreV2:tensors:110"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_110AssignVariableOp8assignvariableop_110_adam_batchnormproteinencode1_beta_vIdentity_110:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_111IdentityRestoreV2:tensors:111"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_111AssignVariableOp1assignvariableop_111_adam_gene_encoder_2_kernel_vIdentity_111:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_112IdentityRestoreV2:tensors:112"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_112AssignVariableOp/assignvariableop_112_adam_gene_encoder_2_bias_vIdentity_112:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_113IdentityRestoreV2:tensors:113"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_113AssignVariableOp4assignvariableop_113_adam_protein_encoder_2_kernel_vIdentity_113:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_114IdentityRestoreV2:tensors:114"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_114AssignVariableOp2assignvariableop_114_adam_protein_encoder_2_bias_vIdentity_114:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_115IdentityRestoreV2:tensors:115"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_115AssignVariableOp6assignvariableop_115_adam_batchnormgeneencode2_gamma_vIdentity_115:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_116IdentityRestoreV2:tensors:116"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_116AssignVariableOp5assignvariableop_116_adam_batchnormgeneencode2_beta_vIdentity_116:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_117IdentityRestoreV2:tensors:117"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_117AssignVariableOp9assignvariableop_117_adam_batchnormproteinencode2_gamma_vIdentity_117:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_118IdentityRestoreV2:tensors:118"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_118AssignVariableOp8assignvariableop_118_adam_batchnormproteinencode2_beta_vIdentity_118:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_119IdentityRestoreV2:tensors:119"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_119AssignVariableOp4assignvariableop_119_adam_embeddingdimdense_kernel_vIdentity_119:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_120IdentityRestoreV2:tensors:120"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_120AssignVariableOp2assignvariableop_120_adam_embeddingdimdense_bias_vIdentity_120:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_121IdentityRestoreV2:tensors:121"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_121AssignVariableOp1assignvariableop_121_adam_gene_decoder_1_kernel_vIdentity_121:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_122IdentityRestoreV2:tensors:122"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_122AssignVariableOp/assignvariableop_122_adam_gene_decoder_1_bias_vIdentity_122:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_123IdentityRestoreV2:tensors:123"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_123AssignVariableOp4assignvariableop_123_adam_protein_decoder_1_kernel_vIdentity_123:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_124IdentityRestoreV2:tensors:124"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_124AssignVariableOp2assignvariableop_124_adam_protein_decoder_1_bias_vIdentity_124:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_125IdentityRestoreV2:tensors:125"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_125AssignVariableOp6assignvariableop_125_adam_batchnormgenedecode1_gamma_vIdentity_125:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_126IdentityRestoreV2:tensors:126"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_126AssignVariableOp5assignvariableop_126_adam_batchnormgenedecode1_beta_vIdentity_126:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_127IdentityRestoreV2:tensors:127"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_127AssignVariableOp9assignvariableop_127_adam_batchnormproteindecode1_gamma_vIdentity_127:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_128IdentityRestoreV2:tensors:128"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_128AssignVariableOp8assignvariableop_128_adam_batchnormproteindecode1_beta_vIdentity_128:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_129IdentityRestoreV2:tensors:129"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_129AssignVariableOp1assignvariableop_129_adam_gene_decoder_2_kernel_vIdentity_129:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_130IdentityRestoreV2:tensors:130"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_130AssignVariableOp/assignvariableop_130_adam_gene_decoder_2_bias_vIdentity_130:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_131IdentityRestoreV2:tensors:131"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_131AssignVariableOp4assignvariableop_131_adam_protein_decoder_2_kernel_vIdentity_131:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_132IdentityRestoreV2:tensors:132"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_132AssignVariableOp2assignvariableop_132_adam_protein_decoder_2_bias_vIdentity_132:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_133IdentityRestoreV2:tensors:133"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_133AssignVariableOp6assignvariableop_133_adam_batchnormgenedecode2_gamma_vIdentity_133:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_134IdentityRestoreV2:tensors:134"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_134AssignVariableOp5assignvariableop_134_adam_batchnormgenedecode2_beta_vIdentity_134:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_135IdentityRestoreV2:tensors:135"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_135AssignVariableOp9assignvariableop_135_adam_batchnormproteindecode2_gamma_vIdentity_135:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_136IdentityRestoreV2:tensors:136"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_136AssignVariableOp8assignvariableop_136_adam_batchnormproteindecode2_beta_vIdentity_136:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_137IdentityRestoreV2:tensors:137"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_137AssignVariableOp4assignvariableop_137_adam_gene_decoder_last_kernel_vIdentity_137:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_138IdentityRestoreV2:tensors:138"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_138AssignVariableOp2assignvariableop_138_adam_gene_decoder_last_bias_vIdentity_138:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_139IdentityRestoreV2:tensors:139"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_139AssignVariableOp7assignvariableop_139_adam_protein_decoder_last_kernel_vIdentity_139:output:0"/device:CPU:0*
_output_shapes
 *
dtype0a
Identity_140IdentityRestoreV2:tensors:140"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_140AssignVariableOp5assignvariableop_140_adam_protein_decoder_last_bias_vIdentity_140:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_141Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_142IdentityIdentity_141:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_103^AssignVariableOp_104^AssignVariableOp_105^AssignVariableOp_106^AssignVariableOp_107^AssignVariableOp_108^AssignVariableOp_109^AssignVariableOp_11^AssignVariableOp_110^AssignVariableOp_111^AssignVariableOp_112^AssignVariableOp_113^AssignVariableOp_114^AssignVariableOp_115^AssignVariableOp_116^AssignVariableOp_117^AssignVariableOp_118^AssignVariableOp_119^AssignVariableOp_12^AssignVariableOp_120^AssignVariableOp_121^AssignVariableOp_122^AssignVariableOp_123^AssignVariableOp_124^AssignVariableOp_125^AssignVariableOp_126^AssignVariableOp_127^AssignVariableOp_128^AssignVariableOp_129^AssignVariableOp_13^AssignVariableOp_130^AssignVariableOp_131^AssignVariableOp_132^AssignVariableOp_133^AssignVariableOp_134^AssignVariableOp_135^AssignVariableOp_136^AssignVariableOp_137^AssignVariableOp_138^AssignVariableOp_139^AssignVariableOp_14^AssignVariableOp_140^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_142Identity_142:output:0*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_112AssignVariableOp_1122,
AssignVariableOp_113AssignVariableOp_1132,
AssignVariableOp_114AssignVariableOp_1142,
AssignVariableOp_115AssignVariableOp_1152,
AssignVariableOp_116AssignVariableOp_1162,
AssignVariableOp_117AssignVariableOp_1172,
AssignVariableOp_118AssignVariableOp_1182,
AssignVariableOp_119AssignVariableOp_1192*
AssignVariableOp_12AssignVariableOp_122,
AssignVariableOp_120AssignVariableOp_1202,
AssignVariableOp_121AssignVariableOp_1212,
AssignVariableOp_122AssignVariableOp_1222,
AssignVariableOp_123AssignVariableOp_1232,
AssignVariableOp_124AssignVariableOp_1242,
AssignVariableOp_125AssignVariableOp_1252,
AssignVariableOp_126AssignVariableOp_1262,
AssignVariableOp_127AssignVariableOp_1272,
AssignVariableOp_128AssignVariableOp_1282,
AssignVariableOp_129AssignVariableOp_1292*
AssignVariableOp_13AssignVariableOp_132,
AssignVariableOp_130AssignVariableOp_1302,
AssignVariableOp_131AssignVariableOp_1312,
AssignVariableOp_132AssignVariableOp_1322,
AssignVariableOp_133AssignVariableOp_1332,
AssignVariableOp_134AssignVariableOp_1342,
AssignVariableOp_135AssignVariableOp_1352,
AssignVariableOp_136AssignVariableOp_1362,
AssignVariableOp_137AssignVariableOp_1372,
AssignVariableOp_138AssignVariableOp_1382,
AssignVariableOp_139AssignVariableOp_1392*
AssignVariableOp_14AssignVariableOp_142,
AssignVariableOp_140AssignVariableOp_1402*
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
�
�
2__inference_EmbeddingDimDense_layer_call_fn_331533

inputs
unknown:[@
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
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_329091o
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
:���������[: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������[
 
_user_specified_nameinputs
�

�
M__inference_protein_decoder_1_layer_call_and_return_conditional_losses_331584

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
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
8__inference_BatchNormProteinDecode1_layer_call_fn_331677

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_328736o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_328408

inputs/
!batchnorm_readvariableop_resource:/3
%batchnorm_mul_readvariableop_resource:/1
#batchnorm_readvariableop_1_resource:/1
#batchnorm_readvariableop_2_resource:/
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:/*
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
:/P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:/~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������/z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:/*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:/z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:/*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:/r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������/b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������/�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������/: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������/
 
_user_specified_nameinputs
��
�9
C__inference_model_8_layer_call_and_return_conditional_losses_330993
inputs_0
inputs_1C
0protein_encoder_1_matmul_readvariableop_resource:	�/?
1protein_encoder_1_biasadd_readvariableop_resource:/A
-gene_encoder_1_matmul_readvariableop_resource:
�
�=
.gene_encoder_1_biasadd_readvariableop_resource:	�M
?batchnormproteinencode1_assignmovingavg_readvariableop_resource:/O
Abatchnormproteinencode1_assignmovingavg_1_readvariableop_resource:/K
=batchnormproteinencode1_batchnorm_mul_readvariableop_resource:/G
9batchnormproteinencode1_batchnorm_readvariableop_resource:/K
<batchnormgeneencode1_assignmovingavg_readvariableop_resource:	�M
>batchnormgeneencode1_assignmovingavg_1_readvariableop_resource:	�I
:batchnormgeneencode1_batchnorm_mul_readvariableop_resource:	�E
6batchnormgeneencode1_batchnorm_readvariableop_resource:	�B
0protein_encoder_2_matmul_readvariableop_resource:/?
1protein_encoder_2_biasadd_readvariableop_resource:@
-gene_encoder_2_matmul_readvariableop_resource:	�P<
.gene_encoder_2_biasadd_readvariableop_resource:PJ
<batchnormgeneencode2_assignmovingavg_readvariableop_resource:PL
>batchnormgeneencode2_assignmovingavg_1_readvariableop_resource:PH
:batchnormgeneencode2_batchnorm_mul_readvariableop_resource:PD
6batchnormgeneencode2_batchnorm_readvariableop_resource:PM
?batchnormproteinencode2_assignmovingavg_readvariableop_resource:O
Abatchnormproteinencode2_assignmovingavg_1_readvariableop_resource:K
=batchnormproteinencode2_batchnorm_mul_readvariableop_resource:G
9batchnormproteinencode2_batchnorm_readvariableop_resource:B
0embeddingdimdense_matmul_readvariableop_resource:[@?
1embeddingdimdense_biasadd_readvariableop_resource:@B
0protein_decoder_1_matmul_readvariableop_resource:@?
1protein_decoder_1_biasadd_readvariableop_resource:?
-gene_decoder_1_matmul_readvariableop_resource:@P<
.gene_decoder_1_biasadd_readvariableop_resource:PM
?batchnormproteindecode1_assignmovingavg_readvariableop_resource:O
Abatchnormproteindecode1_assignmovingavg_1_readvariableop_resource:K
=batchnormproteindecode1_batchnorm_mul_readvariableop_resource:G
9batchnormproteindecode1_batchnorm_readvariableop_resource:J
<batchnormgenedecode1_assignmovingavg_readvariableop_resource:PL
>batchnormgenedecode1_assignmovingavg_1_readvariableop_resource:PH
:batchnormgenedecode1_batchnorm_mul_readvariableop_resource:PD
6batchnormgenedecode1_batchnorm_readvariableop_resource:PB
0protein_decoder_2_matmul_readvariableop_resource:/?
1protein_decoder_2_biasadd_readvariableop_resource:/@
-gene_decoder_2_matmul_readvariableop_resource:	P�=
.gene_decoder_2_biasadd_readvariableop_resource:	�M
?batchnormproteindecode2_assignmovingavg_readvariableop_resource:/O
Abatchnormproteindecode2_assignmovingavg_1_readvariableop_resource:/K
=batchnormproteindecode2_batchnorm_mul_readvariableop_resource:/G
9batchnormproteindecode2_batchnorm_readvariableop_resource:/K
<batchnormgenedecode2_assignmovingavg_readvariableop_resource:	�M
>batchnormgenedecode2_assignmovingavg_1_readvariableop_resource:	�I
:batchnormgenedecode2_batchnorm_mul_readvariableop_resource:	�E
6batchnormgenedecode2_batchnorm_readvariableop_resource:	�F
3protein_decoder_last_matmul_readvariableop_resource:	/�C
4protein_decoder_last_biasadd_readvariableop_resource:	�D
0gene_decoder_last_matmul_readvariableop_resource:
��
@
1gene_decoder_last_biasadd_readvariableop_resource:	�

identity

identity_1��$BatchNormGeneDecode1/AssignMovingAvg�3BatchNormGeneDecode1/AssignMovingAvg/ReadVariableOp�&BatchNormGeneDecode1/AssignMovingAvg_1�5BatchNormGeneDecode1/AssignMovingAvg_1/ReadVariableOp�-BatchNormGeneDecode1/batchnorm/ReadVariableOp�1BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp�$BatchNormGeneDecode2/AssignMovingAvg�3BatchNormGeneDecode2/AssignMovingAvg/ReadVariableOp�&BatchNormGeneDecode2/AssignMovingAvg_1�5BatchNormGeneDecode2/AssignMovingAvg_1/ReadVariableOp�-BatchNormGeneDecode2/batchnorm/ReadVariableOp�1BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp�$BatchNormGeneEncode1/AssignMovingAvg�3BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOp�&BatchNormGeneEncode1/AssignMovingAvg_1�5BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOp�-BatchNormGeneEncode1/batchnorm/ReadVariableOp�1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp�$BatchNormGeneEncode2/AssignMovingAvg�3BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOp�&BatchNormGeneEncode2/AssignMovingAvg_1�5BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOp�-BatchNormGeneEncode2/batchnorm/ReadVariableOp�1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp�'BatchNormProteinDecode1/AssignMovingAvg�6BatchNormProteinDecode1/AssignMovingAvg/ReadVariableOp�)BatchNormProteinDecode1/AssignMovingAvg_1�8BatchNormProteinDecode1/AssignMovingAvg_1/ReadVariableOp�0BatchNormProteinDecode1/batchnorm/ReadVariableOp�4BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp�'BatchNormProteinDecode2/AssignMovingAvg�6BatchNormProteinDecode2/AssignMovingAvg/ReadVariableOp�)BatchNormProteinDecode2/AssignMovingAvg_1�8BatchNormProteinDecode2/AssignMovingAvg_1/ReadVariableOp�0BatchNormProteinDecode2/batchnorm/ReadVariableOp�4BatchNormProteinDecode2/batchnorm/mul/ReadVariableOp�'BatchNormProteinEncode1/AssignMovingAvg�6BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOp�)BatchNormProteinEncode1/AssignMovingAvg_1�8BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp�0BatchNormProteinEncode1/batchnorm/ReadVariableOp�4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp�'BatchNormProteinEncode2/AssignMovingAvg�6BatchNormProteinEncode2/AssignMovingAvg/ReadVariableOp�)BatchNormProteinEncode2/AssignMovingAvg_1�8BatchNormProteinEncode2/AssignMovingAvg_1/ReadVariableOp�0BatchNormProteinEncode2/batchnorm/ReadVariableOp�4BatchNormProteinEncode2/batchnorm/mul/ReadVariableOp�(EmbeddingDimDense/BiasAdd/ReadVariableOp�'EmbeddingDimDense/MatMul/ReadVariableOp�%gene_decoder_1/BiasAdd/ReadVariableOp�$gene_decoder_1/MatMul/ReadVariableOp�%gene_decoder_2/BiasAdd/ReadVariableOp�$gene_decoder_2/MatMul/ReadVariableOp�(gene_decoder_last/BiasAdd/ReadVariableOp�'gene_decoder_last/MatMul/ReadVariableOp�%gene_encoder_1/BiasAdd/ReadVariableOp�$gene_encoder_1/MatMul/ReadVariableOp�%gene_encoder_2/BiasAdd/ReadVariableOp�$gene_encoder_2/MatMul/ReadVariableOp�(protein_decoder_1/BiasAdd/ReadVariableOp�'protein_decoder_1/MatMul/ReadVariableOp�(protein_decoder_2/BiasAdd/ReadVariableOp�'protein_decoder_2/MatMul/ReadVariableOp�+protein_decoder_last/BiasAdd/ReadVariableOp�*protein_decoder_last/MatMul/ReadVariableOp�(protein_encoder_1/BiasAdd/ReadVariableOp�'protein_encoder_1/MatMul/ReadVariableOp�(protein_encoder_2/BiasAdd/ReadVariableOp�'protein_encoder_2/MatMul/ReadVariableOp�
'protein_encoder_1/MatMul/ReadVariableOpReadVariableOp0protein_encoder_1_matmul_readvariableop_resource*
_output_shapes
:	�/*
dtype0�
protein_encoder_1/MatMulMatMulinputs_1/protein_encoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������/�
(protein_encoder_1/BiasAdd/ReadVariableOpReadVariableOp1protein_encoder_1_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0�
protein_encoder_1/BiasAddBiasAdd"protein_encoder_1/MatMul:product:00protein_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������/z
protein_encoder_1/SigmoidSigmoid"protein_encoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������/�
$gene_encoder_1/MatMul/ReadVariableOpReadVariableOp-gene_encoder_1_matmul_readvariableop_resource* 
_output_shapes
:
�
�*
dtype0�
gene_encoder_1/MatMulMatMulinputs_0,gene_encoder_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%gene_encoder_1/BiasAdd/ReadVariableOpReadVariableOp.gene_encoder_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
gene_encoder_1/BiasAddBiasAddgene_encoder_1/MatMul:product:0-gene_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
gene_encoder_1/SigmoidSigmoidgene_encoder_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
6BatchNormProteinEncode1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
$BatchNormProteinEncode1/moments/meanMeanprotein_encoder_1/Sigmoid:y:0?BatchNormProteinEncode1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(�
,BatchNormProteinEncode1/moments/StopGradientStopGradient-BatchNormProteinEncode1/moments/mean:output:0*
T0*
_output_shapes

:/�
1BatchNormProteinEncode1/moments/SquaredDifferenceSquaredDifferenceprotein_encoder_1/Sigmoid:y:05BatchNormProteinEncode1/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������/�
:BatchNormProteinEncode1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
(BatchNormProteinEncode1/moments/varianceMean5BatchNormProteinEncode1/moments/SquaredDifference:z:0CBatchNormProteinEncode1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(�
'BatchNormProteinEncode1/moments/SqueezeSqueeze-BatchNormProteinEncode1/moments/mean:output:0*
T0*
_output_shapes
:/*
squeeze_dims
 �
)BatchNormProteinEncode1/moments/Squeeze_1Squeeze1BatchNormProteinEncode1/moments/variance:output:0*
T0*
_output_shapes
:/*
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
:/*
dtype0�
+BatchNormProteinEncode1/AssignMovingAvg/subSub>BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOp:value:00BatchNormProteinEncode1/moments/Squeeze:output:0*
T0*
_output_shapes
:/�
+BatchNormProteinEncode1/AssignMovingAvg/mulMul/BatchNormProteinEncode1/AssignMovingAvg/sub:z:06BatchNormProteinEncode1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:/�
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
:/*
dtype0�
-BatchNormProteinEncode1/AssignMovingAvg_1/subSub@BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp:value:02BatchNormProteinEncode1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:/�
-BatchNormProteinEncode1/AssignMovingAvg_1/mulMul1BatchNormProteinEncode1/AssignMovingAvg_1/sub:z:08BatchNormProteinEncode1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:/�
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
:/�
'BatchNormProteinEncode1/batchnorm/RsqrtRsqrt)BatchNormProteinEncode1/batchnorm/add:z:0*
T0*
_output_shapes
:/�
4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOpReadVariableOp=batchnormproteinencode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0�
%BatchNormProteinEncode1/batchnorm/mulMul+BatchNormProteinEncode1/batchnorm/Rsqrt:y:0<BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/�
'BatchNormProteinEncode1/batchnorm/mul_1Mulprotein_encoder_1/Sigmoid:y:0)BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������/�
'BatchNormProteinEncode1/batchnorm/mul_2Mul0BatchNormProteinEncode1/moments/Squeeze:output:0)BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*
_output_shapes
:/�
0BatchNormProteinEncode1/batchnorm/ReadVariableOpReadVariableOp9batchnormproteinencode1_batchnorm_readvariableop_resource*
_output_shapes
:/*
dtype0�
%BatchNormProteinEncode1/batchnorm/subSub8BatchNormProteinEncode1/batchnorm/ReadVariableOp:value:0+BatchNormProteinEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:/�
'BatchNormProteinEncode1/batchnorm/add_1AddV2+BatchNormProteinEncode1/batchnorm/mul_1:z:0)BatchNormProteinEncode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������/}
3BatchNormGeneEncode1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
!BatchNormGeneEncode1/moments/meanMeangene_encoder_1/Sigmoid:y:0<BatchNormGeneEncode1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
)BatchNormGeneEncode1/moments/StopGradientStopGradient*BatchNormGeneEncode1/moments/mean:output:0*
T0*
_output_shapes
:	��
.BatchNormGeneEncode1/moments/SquaredDifferenceSquaredDifferencegene_encoder_1/Sigmoid:y:02BatchNormGeneEncode1/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
7BatchNormGeneEncode1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
%BatchNormGeneEncode1/moments/varianceMean2BatchNormGeneEncode1/moments/SquaredDifference:z:0@BatchNormGeneEncode1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
$BatchNormGeneEncode1/moments/SqueezeSqueeze*BatchNormGeneEncode1/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
&BatchNormGeneEncode1/moments/Squeeze_1Squeeze.BatchNormGeneEncode1/moments/variance:output:0*
T0*
_output_shapes	
:�*
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
:�*
dtype0�
(BatchNormGeneEncode1/AssignMovingAvg/subSub;BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOp:value:0-BatchNormGeneEncode1/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
(BatchNormGeneEncode1/AssignMovingAvg/mulMul,BatchNormGeneEncode1/AssignMovingAvg/sub:z:03BatchNormGeneEncode1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
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
:�*
dtype0�
*BatchNormGeneEncode1/AssignMovingAvg_1/subSub=BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOp:value:0/BatchNormGeneEncode1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
*BatchNormGeneEncode1/AssignMovingAvg_1/mulMul.BatchNormGeneEncode1/AssignMovingAvg_1/sub:z:05BatchNormGeneEncode1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
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
:�{
$BatchNormGeneEncode1/batchnorm/RsqrtRsqrt&BatchNormGeneEncode1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgeneencode1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"BatchNormGeneEncode1/batchnorm/mulMul(BatchNormGeneEncode1/batchnorm/Rsqrt:y:09BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
$BatchNormGeneEncode1/batchnorm/mul_1Mulgene_encoder_1/Sigmoid:y:0&BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
$BatchNormGeneEncode1/batchnorm/mul_2Mul-BatchNormGeneEncode1/moments/Squeeze:output:0&BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
-BatchNormGeneEncode1/batchnorm/ReadVariableOpReadVariableOp6batchnormgeneencode1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"BatchNormGeneEncode1/batchnorm/subSub5BatchNormGeneEncode1/batchnorm/ReadVariableOp:value:0(BatchNormGeneEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
$BatchNormGeneEncode1/batchnorm/add_1AddV2(BatchNormGeneEncode1/batchnorm/mul_1:z:0&BatchNormGeneEncode1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
'protein_encoder_2/MatMul/ReadVariableOpReadVariableOp0protein_encoder_2_matmul_readvariableop_resource*
_output_shapes

:/*
dtype0�
protein_encoder_2/MatMulMatMul+BatchNormProteinEncode1/batchnorm/add_1:z:0/protein_encoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(protein_encoder_2/BiasAdd/ReadVariableOpReadVariableOp1protein_encoder_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
protein_encoder_2/BiasAddBiasAdd"protein_encoder_2/MatMul:product:00protein_encoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������z
protein_encoder_2/SigmoidSigmoid"protein_encoder_2/BiasAdd:output:0*
T0*'
_output_shapes
:����������
$gene_encoder_2/MatMul/ReadVariableOpReadVariableOp-gene_encoder_2_matmul_readvariableop_resource*
_output_shapes
:	�P*
dtype0�
gene_encoder_2/MatMulMatMul(BatchNormGeneEncode1/batchnorm/add_1:z:0,gene_encoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
%gene_encoder_2/BiasAdd/ReadVariableOpReadVariableOp.gene_encoder_2_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
gene_encoder_2/BiasAddBiasAddgene_encoder_2/MatMul:product:0-gene_encoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pt
gene_encoder_2/SigmoidSigmoidgene_encoder_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������P}
3BatchNormGeneEncode2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
!BatchNormGeneEncode2/moments/meanMeangene_encoder_2/Sigmoid:y:0<BatchNormGeneEncode2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(�
)BatchNormGeneEncode2/moments/StopGradientStopGradient*BatchNormGeneEncode2/moments/mean:output:0*
T0*
_output_shapes

:P�
.BatchNormGeneEncode2/moments/SquaredDifferenceSquaredDifferencegene_encoder_2/Sigmoid:y:02BatchNormGeneEncode2/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������P�
7BatchNormGeneEncode2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
%BatchNormGeneEncode2/moments/varianceMean2BatchNormGeneEncode2/moments/SquaredDifference:z:0@BatchNormGeneEncode2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(�
$BatchNormGeneEncode2/moments/SqueezeSqueeze*BatchNormGeneEncode2/moments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 �
&BatchNormGeneEncode2/moments/Squeeze_1Squeeze.BatchNormGeneEncode2/moments/variance:output:0*
T0*
_output_shapes
:P*
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
:P*
dtype0�
(BatchNormGeneEncode2/AssignMovingAvg/subSub;BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOp:value:0-BatchNormGeneEncode2/moments/Squeeze:output:0*
T0*
_output_shapes
:P�
(BatchNormGeneEncode2/AssignMovingAvg/mulMul,BatchNormGeneEncode2/AssignMovingAvg/sub:z:03BatchNormGeneEncode2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P�
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
:P*
dtype0�
*BatchNormGeneEncode2/AssignMovingAvg_1/subSub=BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOp:value:0/BatchNormGeneEncode2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:P�
*BatchNormGeneEncode2/AssignMovingAvg_1/mulMul.BatchNormGeneEncode2/AssignMovingAvg_1/sub:z:05BatchNormGeneEncode2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P�
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
:Pz
$BatchNormGeneEncode2/batchnorm/RsqrtRsqrt&BatchNormGeneEncode2/batchnorm/add:z:0*
T0*
_output_shapes
:P�
1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgeneencode2_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0�
"BatchNormGeneEncode2/batchnorm/mulMul(BatchNormGeneEncode2/batchnorm/Rsqrt:y:09BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P�
$BatchNormGeneEncode2/batchnorm/mul_1Mulgene_encoder_2/Sigmoid:y:0&BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������P�
$BatchNormGeneEncode2/batchnorm/mul_2Mul-BatchNormGeneEncode2/moments/Squeeze:output:0&BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*
_output_shapes
:P�
-BatchNormGeneEncode2/batchnorm/ReadVariableOpReadVariableOp6batchnormgeneencode2_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0�
"BatchNormGeneEncode2/batchnorm/subSub5BatchNormGeneEncode2/batchnorm/ReadVariableOp:value:0(BatchNormGeneEncode2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:P�
$BatchNormGeneEncode2/batchnorm/add_1AddV2(BatchNormGeneEncode2/batchnorm/mul_1:z:0&BatchNormGeneEncode2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������P�
6BatchNormProteinEncode2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
$BatchNormProteinEncode2/moments/meanMeanprotein_encoder_2/Sigmoid:y:0?BatchNormProteinEncode2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
,BatchNormProteinEncode2/moments/StopGradientStopGradient-BatchNormProteinEncode2/moments/mean:output:0*
T0*
_output_shapes

:�
1BatchNormProteinEncode2/moments/SquaredDifferenceSquaredDifferenceprotein_encoder_2/Sigmoid:y:05BatchNormProteinEncode2/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
:BatchNormProteinEncode2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
(BatchNormProteinEncode2/moments/varianceMean5BatchNormProteinEncode2/moments/SquaredDifference:z:0CBatchNormProteinEncode2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
'BatchNormProteinEncode2/moments/SqueezeSqueeze-BatchNormProteinEncode2/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
)BatchNormProteinEncode2/moments/Squeeze_1Squeeze1BatchNormProteinEncode2/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 r
-BatchNormProteinEncode2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6BatchNormProteinEncode2/AssignMovingAvg/ReadVariableOpReadVariableOp?batchnormproteinencode2_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0�
+BatchNormProteinEncode2/AssignMovingAvg/subSub>BatchNormProteinEncode2/AssignMovingAvg/ReadVariableOp:value:00BatchNormProteinEncode2/moments/Squeeze:output:0*
T0*
_output_shapes
:�
+BatchNormProteinEncode2/AssignMovingAvg/mulMul/BatchNormProteinEncode2/AssignMovingAvg/sub:z:06BatchNormProteinEncode2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
'BatchNormProteinEncode2/AssignMovingAvgAssignSubVariableOp?batchnormproteinencode2_assignmovingavg_readvariableop_resource/BatchNormProteinEncode2/AssignMovingAvg/mul:z:07^BatchNormProteinEncode2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/BatchNormProteinEncode2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
8BatchNormProteinEncode2/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatchnormproteinencode2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0�
-BatchNormProteinEncode2/AssignMovingAvg_1/subSub@BatchNormProteinEncode2/AssignMovingAvg_1/ReadVariableOp:value:02BatchNormProteinEncode2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
-BatchNormProteinEncode2/AssignMovingAvg_1/mulMul1BatchNormProteinEncode2/AssignMovingAvg_1/sub:z:08BatchNormProteinEncode2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
)BatchNormProteinEncode2/AssignMovingAvg_1AssignSubVariableOpAbatchnormproteinencode2_assignmovingavg_1_readvariableop_resource1BatchNormProteinEncode2/AssignMovingAvg_1/mul:z:09^BatchNormProteinEncode2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'BatchNormProteinEncode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
%BatchNormProteinEncode2/batchnorm/addAddV22BatchNormProteinEncode2/moments/Squeeze_1:output:00BatchNormProteinEncode2/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
'BatchNormProteinEncode2/batchnorm/RsqrtRsqrt)BatchNormProteinEncode2/batchnorm/add:z:0*
T0*
_output_shapes
:�
4BatchNormProteinEncode2/batchnorm/mul/ReadVariableOpReadVariableOp=batchnormproteinencode2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
%BatchNormProteinEncode2/batchnorm/mulMul+BatchNormProteinEncode2/batchnorm/Rsqrt:y:0<BatchNormProteinEncode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
'BatchNormProteinEncode2/batchnorm/mul_1Mulprotein_encoder_2/Sigmoid:y:0)BatchNormProteinEncode2/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
'BatchNormProteinEncode2/batchnorm/mul_2Mul0BatchNormProteinEncode2/moments/Squeeze:output:0)BatchNormProteinEncode2/batchnorm/mul:z:0*
T0*
_output_shapes
:�
0BatchNormProteinEncode2/batchnorm/ReadVariableOpReadVariableOp9batchnormproteinencode2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
%BatchNormProteinEncode2/batchnorm/subSub8BatchNormProteinEncode2/batchnorm/ReadVariableOp:value:0+BatchNormProteinEncode2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
'BatchNormProteinEncode2/batchnorm/add_1AddV2+BatchNormProteinEncode2/batchnorm/mul_1:z:0)BatchNormProteinEncode2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������^
ConcatenateLayer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
ConcatenateLayer/concatConcatV2(BatchNormGeneEncode2/batchnorm/add_1:z:0+BatchNormProteinEncode2/batchnorm/add_1:z:0%ConcatenateLayer/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������[�
'EmbeddingDimDense/MatMul/ReadVariableOpReadVariableOp0embeddingdimdense_matmul_readvariableop_resource*
_output_shapes

:[@*
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
'protein_decoder_1/MatMul/ReadVariableOpReadVariableOp0protein_decoder_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
protein_decoder_1/MatMulMatMulEmbeddingDimDense/Sigmoid:y:0/protein_decoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(protein_decoder_1/BiasAdd/ReadVariableOpReadVariableOp1protein_decoder_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
protein_decoder_1/BiasAddBiasAdd"protein_decoder_1/MatMul:product:00protein_decoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������z
protein_decoder_1/SigmoidSigmoid"protein_decoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
$gene_decoder_1/MatMul/ReadVariableOpReadVariableOp-gene_decoder_1_matmul_readvariableop_resource*
_output_shapes

:@P*
dtype0�
gene_decoder_1/MatMulMatMulEmbeddingDimDense/Sigmoid:y:0,gene_decoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
%gene_decoder_1/BiasAdd/ReadVariableOpReadVariableOp.gene_decoder_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
gene_decoder_1/BiasAddBiasAddgene_decoder_1/MatMul:product:0-gene_decoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pt
gene_decoder_1/SigmoidSigmoidgene_decoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������P�
6BatchNormProteinDecode1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
$BatchNormProteinDecode1/moments/meanMeanprotein_decoder_1/Sigmoid:y:0?BatchNormProteinDecode1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
,BatchNormProteinDecode1/moments/StopGradientStopGradient-BatchNormProteinDecode1/moments/mean:output:0*
T0*
_output_shapes

:�
1BatchNormProteinDecode1/moments/SquaredDifferenceSquaredDifferenceprotein_decoder_1/Sigmoid:y:05BatchNormProteinDecode1/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
:BatchNormProteinDecode1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
(BatchNormProteinDecode1/moments/varianceMean5BatchNormProteinDecode1/moments/SquaredDifference:z:0CBatchNormProteinDecode1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
'BatchNormProteinDecode1/moments/SqueezeSqueeze-BatchNormProteinDecode1/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
)BatchNormProteinDecode1/moments/Squeeze_1Squeeze1BatchNormProteinDecode1/moments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0�
+BatchNormProteinDecode1/AssignMovingAvg/subSub>BatchNormProteinDecode1/AssignMovingAvg/ReadVariableOp:value:00BatchNormProteinDecode1/moments/Squeeze:output:0*
T0*
_output_shapes
:�
+BatchNormProteinDecode1/AssignMovingAvg/mulMul/BatchNormProteinDecode1/AssignMovingAvg/sub:z:06BatchNormProteinDecode1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
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
:*
dtype0�
-BatchNormProteinDecode1/AssignMovingAvg_1/subSub@BatchNormProteinDecode1/AssignMovingAvg_1/ReadVariableOp:value:02BatchNormProteinDecode1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
-BatchNormProteinDecode1/AssignMovingAvg_1/mulMul1BatchNormProteinDecode1/AssignMovingAvg_1/sub:z:08BatchNormProteinDecode1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
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
:�
'BatchNormProteinDecode1/batchnorm/RsqrtRsqrt)BatchNormProteinDecode1/batchnorm/add:z:0*
T0*
_output_shapes
:�
4BatchNormProteinDecode1/batchnorm/mul/ReadVariableOpReadVariableOp=batchnormproteindecode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
%BatchNormProteinDecode1/batchnorm/mulMul+BatchNormProteinDecode1/batchnorm/Rsqrt:y:0<BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
'BatchNormProteinDecode1/batchnorm/mul_1Mulprotein_decoder_1/Sigmoid:y:0)BatchNormProteinDecode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
'BatchNormProteinDecode1/batchnorm/mul_2Mul0BatchNormProteinDecode1/moments/Squeeze:output:0)BatchNormProteinDecode1/batchnorm/mul:z:0*
T0*
_output_shapes
:�
0BatchNormProteinDecode1/batchnorm/ReadVariableOpReadVariableOp9batchnormproteindecode1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
%BatchNormProteinDecode1/batchnorm/subSub8BatchNormProteinDecode1/batchnorm/ReadVariableOp:value:0+BatchNormProteinDecode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
'BatchNormProteinDecode1/batchnorm/add_1AddV2+BatchNormProteinDecode1/batchnorm/mul_1:z:0)BatchNormProteinDecode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}
3BatchNormGeneDecode1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
!BatchNormGeneDecode1/moments/meanMeangene_decoder_1/Sigmoid:y:0<BatchNormGeneDecode1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(�
)BatchNormGeneDecode1/moments/StopGradientStopGradient*BatchNormGeneDecode1/moments/mean:output:0*
T0*
_output_shapes

:P�
.BatchNormGeneDecode1/moments/SquaredDifferenceSquaredDifferencegene_decoder_1/Sigmoid:y:02BatchNormGeneDecode1/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������P�
7BatchNormGeneDecode1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
%BatchNormGeneDecode1/moments/varianceMean2BatchNormGeneDecode1/moments/SquaredDifference:z:0@BatchNormGeneDecode1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(�
$BatchNormGeneDecode1/moments/SqueezeSqueeze*BatchNormGeneDecode1/moments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 �
&BatchNormGeneDecode1/moments/Squeeze_1Squeeze.BatchNormGeneDecode1/moments/variance:output:0*
T0*
_output_shapes
:P*
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
:P*
dtype0�
(BatchNormGeneDecode1/AssignMovingAvg/subSub;BatchNormGeneDecode1/AssignMovingAvg/ReadVariableOp:value:0-BatchNormGeneDecode1/moments/Squeeze:output:0*
T0*
_output_shapes
:P�
(BatchNormGeneDecode1/AssignMovingAvg/mulMul,BatchNormGeneDecode1/AssignMovingAvg/sub:z:03BatchNormGeneDecode1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P�
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
:P*
dtype0�
*BatchNormGeneDecode1/AssignMovingAvg_1/subSub=BatchNormGeneDecode1/AssignMovingAvg_1/ReadVariableOp:value:0/BatchNormGeneDecode1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:P�
*BatchNormGeneDecode1/AssignMovingAvg_1/mulMul.BatchNormGeneDecode1/AssignMovingAvg_1/sub:z:05BatchNormGeneDecode1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P�
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
:Pz
$BatchNormGeneDecode1/batchnorm/RsqrtRsqrt&BatchNormGeneDecode1/batchnorm/add:z:0*
T0*
_output_shapes
:P�
1BatchNormGeneDecode1/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgenedecode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0�
"BatchNormGeneDecode1/batchnorm/mulMul(BatchNormGeneDecode1/batchnorm/Rsqrt:y:09BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P�
$BatchNormGeneDecode1/batchnorm/mul_1Mulgene_decoder_1/Sigmoid:y:0&BatchNormGeneDecode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������P�
$BatchNormGeneDecode1/batchnorm/mul_2Mul-BatchNormGeneDecode1/moments/Squeeze:output:0&BatchNormGeneDecode1/batchnorm/mul:z:0*
T0*
_output_shapes
:P�
-BatchNormGeneDecode1/batchnorm/ReadVariableOpReadVariableOp6batchnormgenedecode1_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0�
"BatchNormGeneDecode1/batchnorm/subSub5BatchNormGeneDecode1/batchnorm/ReadVariableOp:value:0(BatchNormGeneDecode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:P�
$BatchNormGeneDecode1/batchnorm/add_1AddV2(BatchNormGeneDecode1/batchnorm/mul_1:z:0&BatchNormGeneDecode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������P�
'protein_decoder_2/MatMul/ReadVariableOpReadVariableOp0protein_decoder_2_matmul_readvariableop_resource*
_output_shapes

:/*
dtype0�
protein_decoder_2/MatMulMatMul+BatchNormProteinDecode1/batchnorm/add_1:z:0/protein_decoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������/�
(protein_decoder_2/BiasAdd/ReadVariableOpReadVariableOp1protein_decoder_2_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0�
protein_decoder_2/BiasAddBiasAdd"protein_decoder_2/MatMul:product:00protein_decoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������/z
protein_decoder_2/SigmoidSigmoid"protein_decoder_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������/�
$gene_decoder_2/MatMul/ReadVariableOpReadVariableOp-gene_decoder_2_matmul_readvariableop_resource*
_output_shapes
:	P�*
dtype0�
gene_decoder_2/MatMulMatMul(BatchNormGeneDecode1/batchnorm/add_1:z:0,gene_decoder_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%gene_decoder_2/BiasAdd/ReadVariableOpReadVariableOp.gene_decoder_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
gene_decoder_2/BiasAddBiasAddgene_decoder_2/MatMul:product:0-gene_decoder_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
gene_decoder_2/SigmoidSigmoidgene_decoder_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
6BatchNormProteinDecode2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
$BatchNormProteinDecode2/moments/meanMeanprotein_decoder_2/Sigmoid:y:0?BatchNormProteinDecode2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(�
,BatchNormProteinDecode2/moments/StopGradientStopGradient-BatchNormProteinDecode2/moments/mean:output:0*
T0*
_output_shapes

:/�
1BatchNormProteinDecode2/moments/SquaredDifferenceSquaredDifferenceprotein_decoder_2/Sigmoid:y:05BatchNormProteinDecode2/moments/StopGradient:output:0*
T0*'
_output_shapes
:���������/�
:BatchNormProteinDecode2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
(BatchNormProteinDecode2/moments/varianceMean5BatchNormProteinDecode2/moments/SquaredDifference:z:0CBatchNormProteinDecode2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(�
'BatchNormProteinDecode2/moments/SqueezeSqueeze-BatchNormProteinDecode2/moments/mean:output:0*
T0*
_output_shapes
:/*
squeeze_dims
 �
)BatchNormProteinDecode2/moments/Squeeze_1Squeeze1BatchNormProteinDecode2/moments/variance:output:0*
T0*
_output_shapes
:/*
squeeze_dims
 r
-BatchNormProteinDecode2/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
6BatchNormProteinDecode2/AssignMovingAvg/ReadVariableOpReadVariableOp?batchnormproteindecode2_assignmovingavg_readvariableop_resource*
_output_shapes
:/*
dtype0�
+BatchNormProteinDecode2/AssignMovingAvg/subSub>BatchNormProteinDecode2/AssignMovingAvg/ReadVariableOp:value:00BatchNormProteinDecode2/moments/Squeeze:output:0*
T0*
_output_shapes
:/�
+BatchNormProteinDecode2/AssignMovingAvg/mulMul/BatchNormProteinDecode2/AssignMovingAvg/sub:z:06BatchNormProteinDecode2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:/�
'BatchNormProteinDecode2/AssignMovingAvgAssignSubVariableOp?batchnormproteindecode2_assignmovingavg_readvariableop_resource/BatchNormProteinDecode2/AssignMovingAvg/mul:z:07^BatchNormProteinDecode2/AssignMovingAvg/ReadVariableOp*
_output_shapes
 *
dtype0t
/BatchNormProteinDecode2/AssignMovingAvg_1/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<�
8BatchNormProteinDecode2/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatchnormproteindecode2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:/*
dtype0�
-BatchNormProteinDecode2/AssignMovingAvg_1/subSub@BatchNormProteinDecode2/AssignMovingAvg_1/ReadVariableOp:value:02BatchNormProteinDecode2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:/�
-BatchNormProteinDecode2/AssignMovingAvg_1/mulMul1BatchNormProteinDecode2/AssignMovingAvg_1/sub:z:08BatchNormProteinDecode2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:/�
)BatchNormProteinDecode2/AssignMovingAvg_1AssignSubVariableOpAbatchnormproteindecode2_assignmovingavg_1_readvariableop_resource1BatchNormProteinDecode2/AssignMovingAvg_1/mul:z:09^BatchNormProteinDecode2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'BatchNormProteinDecode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
%BatchNormProteinDecode2/batchnorm/addAddV22BatchNormProteinDecode2/moments/Squeeze_1:output:00BatchNormProteinDecode2/batchnorm/add/y:output:0*
T0*
_output_shapes
:/�
'BatchNormProteinDecode2/batchnorm/RsqrtRsqrt)BatchNormProteinDecode2/batchnorm/add:z:0*
T0*
_output_shapes
:/�
4BatchNormProteinDecode2/batchnorm/mul/ReadVariableOpReadVariableOp=batchnormproteindecode2_batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0�
%BatchNormProteinDecode2/batchnorm/mulMul+BatchNormProteinDecode2/batchnorm/Rsqrt:y:0<BatchNormProteinDecode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/�
'BatchNormProteinDecode2/batchnorm/mul_1Mulprotein_decoder_2/Sigmoid:y:0)BatchNormProteinDecode2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������/�
'BatchNormProteinDecode2/batchnorm/mul_2Mul0BatchNormProteinDecode2/moments/Squeeze:output:0)BatchNormProteinDecode2/batchnorm/mul:z:0*
T0*
_output_shapes
:/�
0BatchNormProteinDecode2/batchnorm/ReadVariableOpReadVariableOp9batchnormproteindecode2_batchnorm_readvariableop_resource*
_output_shapes
:/*
dtype0�
%BatchNormProteinDecode2/batchnorm/subSub8BatchNormProteinDecode2/batchnorm/ReadVariableOp:value:0+BatchNormProteinDecode2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:/�
'BatchNormProteinDecode2/batchnorm/add_1AddV2+BatchNormProteinDecode2/batchnorm/mul_1:z:0)BatchNormProteinDecode2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������/}
3BatchNormGeneDecode2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
!BatchNormGeneDecode2/moments/meanMeangene_decoder_2/Sigmoid:y:0<BatchNormGeneDecode2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
)BatchNormGeneDecode2/moments/StopGradientStopGradient*BatchNormGeneDecode2/moments/mean:output:0*
T0*
_output_shapes
:	��
.BatchNormGeneDecode2/moments/SquaredDifferenceSquaredDifferencegene_decoder_2/Sigmoid:y:02BatchNormGeneDecode2/moments/StopGradient:output:0*
T0*(
_output_shapes
:�����������
7BatchNormGeneDecode2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
%BatchNormGeneDecode2/moments/varianceMean2BatchNormGeneDecode2/moments/SquaredDifference:z:0@BatchNormGeneDecode2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(�
$BatchNormGeneDecode2/moments/SqueezeSqueeze*BatchNormGeneDecode2/moments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 �
&BatchNormGeneDecode2/moments/Squeeze_1Squeeze.BatchNormGeneDecode2/moments/variance:output:0*
T0*
_output_shapes	
:�*
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
:�*
dtype0�
(BatchNormGeneDecode2/AssignMovingAvg/subSub;BatchNormGeneDecode2/AssignMovingAvg/ReadVariableOp:value:0-BatchNormGeneDecode2/moments/Squeeze:output:0*
T0*
_output_shapes	
:��
(BatchNormGeneDecode2/AssignMovingAvg/mulMul,BatchNormGeneDecode2/AssignMovingAvg/sub:z:03BatchNormGeneDecode2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
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
:�*
dtype0�
*BatchNormGeneDecode2/AssignMovingAvg_1/subSub=BatchNormGeneDecode2/AssignMovingAvg_1/ReadVariableOp:value:0/BatchNormGeneDecode2/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:��
*BatchNormGeneDecode2/AssignMovingAvg_1/mulMul.BatchNormGeneDecode2/AssignMovingAvg_1/sub:z:05BatchNormGeneDecode2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
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
:�{
$BatchNormGeneDecode2/batchnorm/RsqrtRsqrt&BatchNormGeneDecode2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
1BatchNormGeneDecode2/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgenedecode2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"BatchNormGeneDecode2/batchnorm/mulMul(BatchNormGeneDecode2/batchnorm/Rsqrt:y:09BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
$BatchNormGeneDecode2/batchnorm/mul_1Mulgene_decoder_2/Sigmoid:y:0&BatchNormGeneDecode2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
$BatchNormGeneDecode2/batchnorm/mul_2Mul-BatchNormGeneDecode2/moments/Squeeze:output:0&BatchNormGeneDecode2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
-BatchNormGeneDecode2/batchnorm/ReadVariableOpReadVariableOp6batchnormgenedecode2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"BatchNormGeneDecode2/batchnorm/subSub5BatchNormGeneDecode2/batchnorm/ReadVariableOp:value:0(BatchNormGeneDecode2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
$BatchNormGeneDecode2/batchnorm/add_1AddV2(BatchNormGeneDecode2/batchnorm/mul_1:z:0&BatchNormGeneDecode2/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
*protein_decoder_last/MatMul/ReadVariableOpReadVariableOp3protein_decoder_last_matmul_readvariableop_resource*
_output_shapes
:	/�*
dtype0�
protein_decoder_last/MatMulMatMul+BatchNormProteinDecode2/batchnorm/add_1:z:02protein_decoder_last/MatMul/ReadVariableOp:value:0*
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
��
*
dtype0�
gene_decoder_last/MatMulMatMul(BatchNormGeneDecode2/batchnorm/add_1:z:0/gene_decoder_last/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
�
(gene_decoder_last/BiasAdd/ReadVariableOpReadVariableOp1gene_decoder_last_biasadd_readvariableop_resource*
_output_shapes	
:�
*
dtype0�
gene_decoder_last/BiasAddBiasAdd"gene_decoder_last/MatMul:product:00gene_decoder_last/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
{
gene_decoder_last/SigmoidSigmoid"gene_decoder_last/BiasAdd:output:0*
T0*(
_output_shapes
:����������
m
IdentityIdentitygene_decoder_last/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������
r

Identity_1Identity protein_decoder_last/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp%^BatchNormGeneDecode1/AssignMovingAvg4^BatchNormGeneDecode1/AssignMovingAvg/ReadVariableOp'^BatchNormGeneDecode1/AssignMovingAvg_16^BatchNormGeneDecode1/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneDecode1/batchnorm/ReadVariableOp2^BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp%^BatchNormGeneDecode2/AssignMovingAvg4^BatchNormGeneDecode2/AssignMovingAvg/ReadVariableOp'^BatchNormGeneDecode2/AssignMovingAvg_16^BatchNormGeneDecode2/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneDecode2/batchnorm/ReadVariableOp2^BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp%^BatchNormGeneEncode1/AssignMovingAvg4^BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOp'^BatchNormGeneEncode1/AssignMovingAvg_16^BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneEncode1/batchnorm/ReadVariableOp2^BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp%^BatchNormGeneEncode2/AssignMovingAvg4^BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOp'^BatchNormGeneEncode2/AssignMovingAvg_16^BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneEncode2/batchnorm/ReadVariableOp2^BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp(^BatchNormProteinDecode1/AssignMovingAvg7^BatchNormProteinDecode1/AssignMovingAvg/ReadVariableOp*^BatchNormProteinDecode1/AssignMovingAvg_19^BatchNormProteinDecode1/AssignMovingAvg_1/ReadVariableOp1^BatchNormProteinDecode1/batchnorm/ReadVariableOp5^BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp(^BatchNormProteinDecode2/AssignMovingAvg7^BatchNormProteinDecode2/AssignMovingAvg/ReadVariableOp*^BatchNormProteinDecode2/AssignMovingAvg_19^BatchNormProteinDecode2/AssignMovingAvg_1/ReadVariableOp1^BatchNormProteinDecode2/batchnorm/ReadVariableOp5^BatchNormProteinDecode2/batchnorm/mul/ReadVariableOp(^BatchNormProteinEncode1/AssignMovingAvg7^BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOp*^BatchNormProteinEncode1/AssignMovingAvg_19^BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp1^BatchNormProteinEncode1/batchnorm/ReadVariableOp5^BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp(^BatchNormProteinEncode2/AssignMovingAvg7^BatchNormProteinEncode2/AssignMovingAvg/ReadVariableOp*^BatchNormProteinEncode2/AssignMovingAvg_19^BatchNormProteinEncode2/AssignMovingAvg_1/ReadVariableOp1^BatchNormProteinEncode2/batchnorm/ReadVariableOp5^BatchNormProteinEncode2/batchnorm/mul/ReadVariableOp)^EmbeddingDimDense/BiasAdd/ReadVariableOp(^EmbeddingDimDense/MatMul/ReadVariableOp&^gene_decoder_1/BiasAdd/ReadVariableOp%^gene_decoder_1/MatMul/ReadVariableOp&^gene_decoder_2/BiasAdd/ReadVariableOp%^gene_decoder_2/MatMul/ReadVariableOp)^gene_decoder_last/BiasAdd/ReadVariableOp(^gene_decoder_last/MatMul/ReadVariableOp&^gene_encoder_1/BiasAdd/ReadVariableOp%^gene_encoder_1/MatMul/ReadVariableOp&^gene_encoder_2/BiasAdd/ReadVariableOp%^gene_encoder_2/MatMul/ReadVariableOp)^protein_decoder_1/BiasAdd/ReadVariableOp(^protein_decoder_1/MatMul/ReadVariableOp)^protein_decoder_2/BiasAdd/ReadVariableOp(^protein_decoder_2/MatMul/ReadVariableOp,^protein_decoder_last/BiasAdd/ReadVariableOp+^protein_decoder_last/MatMul/ReadVariableOp)^protein_encoder_1/BiasAdd/ReadVariableOp(^protein_encoder_1/MatMul/ReadVariableOp)^protein_encoder_2/BiasAdd/ReadVariableOp(^protein_encoder_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������
:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
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
'BatchNormProteinDecode2/AssignMovingAvg'BatchNormProteinDecode2/AssignMovingAvg2p
6BatchNormProteinDecode2/AssignMovingAvg/ReadVariableOp6BatchNormProteinDecode2/AssignMovingAvg/ReadVariableOp2V
)BatchNormProteinDecode2/AssignMovingAvg_1)BatchNormProteinDecode2/AssignMovingAvg_12t
8BatchNormProteinDecode2/AssignMovingAvg_1/ReadVariableOp8BatchNormProteinDecode2/AssignMovingAvg_1/ReadVariableOp2d
0BatchNormProteinDecode2/batchnorm/ReadVariableOp0BatchNormProteinDecode2/batchnorm/ReadVariableOp2l
4BatchNormProteinDecode2/batchnorm/mul/ReadVariableOp4BatchNormProteinDecode2/batchnorm/mul/ReadVariableOp2R
'BatchNormProteinEncode1/AssignMovingAvg'BatchNormProteinEncode1/AssignMovingAvg2p
6BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOp6BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOp2V
)BatchNormProteinEncode1/AssignMovingAvg_1)BatchNormProteinEncode1/AssignMovingAvg_12t
8BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp8BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp2d
0BatchNormProteinEncode1/batchnorm/ReadVariableOp0BatchNormProteinEncode1/batchnorm/ReadVariableOp2l
4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp2R
'BatchNormProteinEncode2/AssignMovingAvg'BatchNormProteinEncode2/AssignMovingAvg2p
6BatchNormProteinEncode2/AssignMovingAvg/ReadVariableOp6BatchNormProteinEncode2/AssignMovingAvg/ReadVariableOp2V
)BatchNormProteinEncode2/AssignMovingAvg_1)BatchNormProteinEncode2/AssignMovingAvg_12t
8BatchNormProteinEncode2/AssignMovingAvg_1/ReadVariableOp8BatchNormProteinEncode2/AssignMovingAvg_1/ReadVariableOp2d
0BatchNormProteinEncode2/batchnorm/ReadVariableOp0BatchNormProteinEncode2/batchnorm/ReadVariableOp2l
4BatchNormProteinEncode2/batchnorm/mul/ReadVariableOp4BatchNormProteinEncode2/batchnorm/mul/ReadVariableOp2T
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
'protein_decoder_1/MatMul/ReadVariableOp'protein_decoder_1/MatMul/ReadVariableOp2T
(protein_decoder_2/BiasAdd/ReadVariableOp(protein_decoder_2/BiasAdd/ReadVariableOp2R
'protein_decoder_2/MatMul/ReadVariableOp'protein_decoder_2/MatMul/ReadVariableOp2Z
+protein_decoder_last/BiasAdd/ReadVariableOp+protein_decoder_last/BiasAdd/ReadVariableOp2X
*protein_decoder_last/MatMul/ReadVariableOp*protein_decoder_last/MatMul/ReadVariableOp2T
(protein_encoder_1/BiasAdd/ReadVariableOp(protein_encoder_1/BiasAdd/ReadVariableOp2R
'protein_encoder_1/MatMul/ReadVariableOp'protein_encoder_1/MatMul/ReadVariableOp2T
(protein_encoder_2/BiasAdd/ReadVariableOp(protein_encoder_2/BiasAdd/ReadVariableOp2R
'protein_encoder_2/MatMul/ReadVariableOp'protein_encoder_2/MatMul/ReadVariableOp:R N
(
_output_shapes
:����������

"
_user_specified_name
inputs/0:RN
(
_output_shapes
:����������
"
_user_specified_name
inputs/1
�
�
(__inference_model_8_layer_call_fn_329350
gene_input_layer
protein_input_layer
unknown:	�/
	unknown_0:/
	unknown_1:
�
�
	unknown_2:	�
	unknown_3:/
	unknown_4:/
	unknown_5:/
	unknown_6:/
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:/

unknown_12:

unknown_13:	�P

unknown_14:P

unknown_15:P

unknown_16:P

unknown_17:P

unknown_18:P

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:[@

unknown_24:@

unknown_25:@

unknown_26:

unknown_27:@P

unknown_28:P

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:P

unknown_34:P

unknown_35:P

unknown_36:P

unknown_37:/

unknown_38:/

unknown_39:	P�

unknown_40:	�

unknown_41:/

unknown_42:/

unknown_43:/

unknown_44:/

unknown_45:	�

unknown_46:	�

unknown_47:	�

unknown_48:	�

unknown_49:	/�

unknown_50:	�

unknown_51:
��


unknown_52:	�

identity

identity_1��StatefulPartitionedCall�
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
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:����������
:����������*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./01234567*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_8_layer_call_and_return_conditional_losses_329237p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������
r

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
_input_shapes�
�:����������
:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:����������

*
_user_specified_nameGene_Input_Layer:]Y
(
_output_shapes
:����������
-
_user_specified_nameProtein_Input_Layer
�
�
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_328654

inputs/
!batchnorm_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P1
#batchnorm_readvariableop_1_resource:P1
#batchnorm_readvariableop_2_resource:P
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
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
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Pz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_331277

inputs/
!batchnorm_readvariableop_resource:/3
%batchnorm_mul_readvariableop_resource:/1
#batchnorm_readvariableop_1_resource:/1
#batchnorm_readvariableop_2_resource:/
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:/*
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
:/P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:/~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������/z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:/*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:/z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:/*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:/r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������/b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������/�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������/: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������/
 
_user_specified_nameinputs
�
�
5__inference_BatchNormGeneEncode1_layer_call_fn_331177

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_328373p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
P__inference_protein_decoder_last_layer_call_and_return_conditional_losses_329212

inputs1
matmul_readvariableop_resource:	/�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	/�*
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
:���������/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������/
 
_user_specified_nameinputs
�%
�
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_331311

inputs5
'assignmovingavg_readvariableop_resource:/7
)assignmovingavg_1_readvariableop_resource:/3
%batchnorm_mul_readvariableop_resource://
!batchnorm_readvariableop_resource:/
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:/�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������/l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:/*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:/*
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
:/*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:/x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:/�
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
:/*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:/~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:/�
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
:/P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:/~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������/h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:/v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:/*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:/r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������/b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������/�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������/: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������/
 
_user_specified_nameinputs
�%
�
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_328865

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
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
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
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
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
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
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
J__inference_gene_decoder_1_layer_call_and_return_conditional_losses_329125

inputs0
matmul_readvariableop_resource:@P-
biasadd_readvariableop_resource:P
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������PV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������PZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������Pw
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

�
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_329091

inputs0
matmul_readvariableop_resource:[@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:[@*
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
:���������[: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������[
 
_user_specified_nameinputs
�
�
2__inference_protein_encoder_2_layer_call_fn_331340

inputs
unknown:/
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_encoder_2_layer_call_and_return_conditional_losses_329030o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������/: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������/
 
_user_specified_nameinputs
�
�
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_328326

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
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
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_331544

inputs0
matmul_readvariableop_resource:[@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:[@*
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
:���������[: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������[
 
_user_specified_nameinputs
��
�9
!__inference__wrapped_model_328302
gene_input_layer
protein_input_layerK
8model_8_protein_encoder_1_matmul_readvariableop_resource:	�/G
9model_8_protein_encoder_1_biasadd_readvariableop_resource:/I
5model_8_gene_encoder_1_matmul_readvariableop_resource:
�
�E
6model_8_gene_encoder_1_biasadd_readvariableop_resource:	�O
Amodel_8_batchnormproteinencode1_batchnorm_readvariableop_resource:/S
Emodel_8_batchnormproteinencode1_batchnorm_mul_readvariableop_resource:/Q
Cmodel_8_batchnormproteinencode1_batchnorm_readvariableop_1_resource:/Q
Cmodel_8_batchnormproteinencode1_batchnorm_readvariableop_2_resource:/M
>model_8_batchnormgeneencode1_batchnorm_readvariableop_resource:	�Q
Bmodel_8_batchnormgeneencode1_batchnorm_mul_readvariableop_resource:	�O
@model_8_batchnormgeneencode1_batchnorm_readvariableop_1_resource:	�O
@model_8_batchnormgeneencode1_batchnorm_readvariableop_2_resource:	�J
8model_8_protein_encoder_2_matmul_readvariableop_resource:/G
9model_8_protein_encoder_2_biasadd_readvariableop_resource:H
5model_8_gene_encoder_2_matmul_readvariableop_resource:	�PD
6model_8_gene_encoder_2_biasadd_readvariableop_resource:PL
>model_8_batchnormgeneencode2_batchnorm_readvariableop_resource:PP
Bmodel_8_batchnormgeneencode2_batchnorm_mul_readvariableop_resource:PN
@model_8_batchnormgeneencode2_batchnorm_readvariableop_1_resource:PN
@model_8_batchnormgeneencode2_batchnorm_readvariableop_2_resource:PO
Amodel_8_batchnormproteinencode2_batchnorm_readvariableop_resource:S
Emodel_8_batchnormproteinencode2_batchnorm_mul_readvariableop_resource:Q
Cmodel_8_batchnormproteinencode2_batchnorm_readvariableop_1_resource:Q
Cmodel_8_batchnormproteinencode2_batchnorm_readvariableop_2_resource:J
8model_8_embeddingdimdense_matmul_readvariableop_resource:[@G
9model_8_embeddingdimdense_biasadd_readvariableop_resource:@J
8model_8_protein_decoder_1_matmul_readvariableop_resource:@G
9model_8_protein_decoder_1_biasadd_readvariableop_resource:G
5model_8_gene_decoder_1_matmul_readvariableop_resource:@PD
6model_8_gene_decoder_1_biasadd_readvariableop_resource:PO
Amodel_8_batchnormproteindecode1_batchnorm_readvariableop_resource:S
Emodel_8_batchnormproteindecode1_batchnorm_mul_readvariableop_resource:Q
Cmodel_8_batchnormproteindecode1_batchnorm_readvariableop_1_resource:Q
Cmodel_8_batchnormproteindecode1_batchnorm_readvariableop_2_resource:L
>model_8_batchnormgenedecode1_batchnorm_readvariableop_resource:PP
Bmodel_8_batchnormgenedecode1_batchnorm_mul_readvariableop_resource:PN
@model_8_batchnormgenedecode1_batchnorm_readvariableop_1_resource:PN
@model_8_batchnormgenedecode1_batchnorm_readvariableop_2_resource:PJ
8model_8_protein_decoder_2_matmul_readvariableop_resource:/G
9model_8_protein_decoder_2_biasadd_readvariableop_resource:/H
5model_8_gene_decoder_2_matmul_readvariableop_resource:	P�E
6model_8_gene_decoder_2_biasadd_readvariableop_resource:	�O
Amodel_8_batchnormproteindecode2_batchnorm_readvariableop_resource:/S
Emodel_8_batchnormproteindecode2_batchnorm_mul_readvariableop_resource:/Q
Cmodel_8_batchnormproteindecode2_batchnorm_readvariableop_1_resource:/Q
Cmodel_8_batchnormproteindecode2_batchnorm_readvariableop_2_resource:/M
>model_8_batchnormgenedecode2_batchnorm_readvariableop_resource:	�Q
Bmodel_8_batchnormgenedecode2_batchnorm_mul_readvariableop_resource:	�O
@model_8_batchnormgenedecode2_batchnorm_readvariableop_1_resource:	�O
@model_8_batchnormgenedecode2_batchnorm_readvariableop_2_resource:	�N
;model_8_protein_decoder_last_matmul_readvariableop_resource:	/�K
<model_8_protein_decoder_last_biasadd_readvariableop_resource:	�L
8model_8_gene_decoder_last_matmul_readvariableop_resource:
��
H
9model_8_gene_decoder_last_biasadd_readvariableop_resource:	�

identity

identity_1��5model_8/BatchNormGeneDecode1/batchnorm/ReadVariableOp�7model_8/BatchNormGeneDecode1/batchnorm/ReadVariableOp_1�7model_8/BatchNormGeneDecode1/batchnorm/ReadVariableOp_2�9model_8/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp�5model_8/BatchNormGeneDecode2/batchnorm/ReadVariableOp�7model_8/BatchNormGeneDecode2/batchnorm/ReadVariableOp_1�7model_8/BatchNormGeneDecode2/batchnorm/ReadVariableOp_2�9model_8/BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp�5model_8/BatchNormGeneEncode1/batchnorm/ReadVariableOp�7model_8/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1�7model_8/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2�9model_8/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp�5model_8/BatchNormGeneEncode2/batchnorm/ReadVariableOp�7model_8/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1�7model_8/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2�9model_8/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp�8model_8/BatchNormProteinDecode1/batchnorm/ReadVariableOp�:model_8/BatchNormProteinDecode1/batchnorm/ReadVariableOp_1�:model_8/BatchNormProteinDecode1/batchnorm/ReadVariableOp_2�<model_8/BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp�8model_8/BatchNormProteinDecode2/batchnorm/ReadVariableOp�:model_8/BatchNormProteinDecode2/batchnorm/ReadVariableOp_1�:model_8/BatchNormProteinDecode2/batchnorm/ReadVariableOp_2�<model_8/BatchNormProteinDecode2/batchnorm/mul/ReadVariableOp�8model_8/BatchNormProteinEncode1/batchnorm/ReadVariableOp�:model_8/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1�:model_8/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2�<model_8/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp�8model_8/BatchNormProteinEncode2/batchnorm/ReadVariableOp�:model_8/BatchNormProteinEncode2/batchnorm/ReadVariableOp_1�:model_8/BatchNormProteinEncode2/batchnorm/ReadVariableOp_2�<model_8/BatchNormProteinEncode2/batchnorm/mul/ReadVariableOp�0model_8/EmbeddingDimDense/BiasAdd/ReadVariableOp�/model_8/EmbeddingDimDense/MatMul/ReadVariableOp�-model_8/gene_decoder_1/BiasAdd/ReadVariableOp�,model_8/gene_decoder_1/MatMul/ReadVariableOp�-model_8/gene_decoder_2/BiasAdd/ReadVariableOp�,model_8/gene_decoder_2/MatMul/ReadVariableOp�0model_8/gene_decoder_last/BiasAdd/ReadVariableOp�/model_8/gene_decoder_last/MatMul/ReadVariableOp�-model_8/gene_encoder_1/BiasAdd/ReadVariableOp�,model_8/gene_encoder_1/MatMul/ReadVariableOp�-model_8/gene_encoder_2/BiasAdd/ReadVariableOp�,model_8/gene_encoder_2/MatMul/ReadVariableOp�0model_8/protein_decoder_1/BiasAdd/ReadVariableOp�/model_8/protein_decoder_1/MatMul/ReadVariableOp�0model_8/protein_decoder_2/BiasAdd/ReadVariableOp�/model_8/protein_decoder_2/MatMul/ReadVariableOp�3model_8/protein_decoder_last/BiasAdd/ReadVariableOp�2model_8/protein_decoder_last/MatMul/ReadVariableOp�0model_8/protein_encoder_1/BiasAdd/ReadVariableOp�/model_8/protein_encoder_1/MatMul/ReadVariableOp�0model_8/protein_encoder_2/BiasAdd/ReadVariableOp�/model_8/protein_encoder_2/MatMul/ReadVariableOp�
/model_8/protein_encoder_1/MatMul/ReadVariableOpReadVariableOp8model_8_protein_encoder_1_matmul_readvariableop_resource*
_output_shapes
:	�/*
dtype0�
 model_8/protein_encoder_1/MatMulMatMulprotein_input_layer7model_8/protein_encoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������/�
0model_8/protein_encoder_1/BiasAdd/ReadVariableOpReadVariableOp9model_8_protein_encoder_1_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0�
!model_8/protein_encoder_1/BiasAddBiasAdd*model_8/protein_encoder_1/MatMul:product:08model_8/protein_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������/�
!model_8/protein_encoder_1/SigmoidSigmoid*model_8/protein_encoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������/�
,model_8/gene_encoder_1/MatMul/ReadVariableOpReadVariableOp5model_8_gene_encoder_1_matmul_readvariableop_resource* 
_output_shapes
:
�
�*
dtype0�
model_8/gene_encoder_1/MatMulMatMulgene_input_layer4model_8/gene_encoder_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-model_8/gene_encoder_1/BiasAdd/ReadVariableOpReadVariableOp6model_8_gene_encoder_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_8/gene_encoder_1/BiasAddBiasAdd'model_8/gene_encoder_1/MatMul:product:05model_8/gene_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
model_8/gene_encoder_1/SigmoidSigmoid'model_8/gene_encoder_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
8model_8/BatchNormProteinEncode1/batchnorm/ReadVariableOpReadVariableOpAmodel_8_batchnormproteinencode1_batchnorm_readvariableop_resource*
_output_shapes
:/*
dtype0t
/model_8/BatchNormProteinEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-model_8/BatchNormProteinEncode1/batchnorm/addAddV2@model_8/BatchNormProteinEncode1/batchnorm/ReadVariableOp:value:08model_8/BatchNormProteinEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:/�
/model_8/BatchNormProteinEncode1/batchnorm/RsqrtRsqrt1model_8/BatchNormProteinEncode1/batchnorm/add:z:0*
T0*
_output_shapes
:/�
<model_8/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOpReadVariableOpEmodel_8_batchnormproteinencode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0�
-model_8/BatchNormProteinEncode1/batchnorm/mulMul3model_8/BatchNormProteinEncode1/batchnorm/Rsqrt:y:0Dmodel_8/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/�
/model_8/BatchNormProteinEncode1/batchnorm/mul_1Mul%model_8/protein_encoder_1/Sigmoid:y:01model_8/BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������/�
:model_8/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1ReadVariableOpCmodel_8_batchnormproteinencode1_batchnorm_readvariableop_1_resource*
_output_shapes
:/*
dtype0�
/model_8/BatchNormProteinEncode1/batchnorm/mul_2MulBmodel_8/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1:value:01model_8/BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*
_output_shapes
:/�
:model_8/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2ReadVariableOpCmodel_8_batchnormproteinencode1_batchnorm_readvariableop_2_resource*
_output_shapes
:/*
dtype0�
-model_8/BatchNormProteinEncode1/batchnorm/subSubBmodel_8/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2:value:03model_8/BatchNormProteinEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:/�
/model_8/BatchNormProteinEncode1/batchnorm/add_1AddV23model_8/BatchNormProteinEncode1/batchnorm/mul_1:z:01model_8/BatchNormProteinEncode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������/�
5model_8/BatchNormGeneEncode1/batchnorm/ReadVariableOpReadVariableOp>model_8_batchnormgeneencode1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
,model_8/BatchNormGeneEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
*model_8/BatchNormGeneEncode1/batchnorm/addAddV2=model_8/BatchNormGeneEncode1/batchnorm/ReadVariableOp:value:05model_8/BatchNormGeneEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
,model_8/BatchNormGeneEncode1/batchnorm/RsqrtRsqrt.model_8/BatchNormGeneEncode1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
9model_8/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOpReadVariableOpBmodel_8_batchnormgeneencode1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*model_8/BatchNormGeneEncode1/batchnorm/mulMul0model_8/BatchNormGeneEncode1/batchnorm/Rsqrt:y:0Amodel_8/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
,model_8/BatchNormGeneEncode1/batchnorm/mul_1Mul"model_8/gene_encoder_1/Sigmoid:y:0.model_8/BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
7model_8/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1ReadVariableOp@model_8_batchnormgeneencode1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
,model_8/BatchNormGeneEncode1/batchnorm/mul_2Mul?model_8/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1:value:0.model_8/BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
7model_8/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2ReadVariableOp@model_8_batchnormgeneencode1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
*model_8/BatchNormGeneEncode1/batchnorm/subSub?model_8/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2:value:00model_8/BatchNormGeneEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
,model_8/BatchNormGeneEncode1/batchnorm/add_1AddV20model_8/BatchNormGeneEncode1/batchnorm/mul_1:z:0.model_8/BatchNormGeneEncode1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
/model_8/protein_encoder_2/MatMul/ReadVariableOpReadVariableOp8model_8_protein_encoder_2_matmul_readvariableop_resource*
_output_shapes

:/*
dtype0�
 model_8/protein_encoder_2/MatMulMatMul3model_8/BatchNormProteinEncode1/batchnorm/add_1:z:07model_8/protein_encoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
0model_8/protein_encoder_2/BiasAdd/ReadVariableOpReadVariableOp9model_8_protein_encoder_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!model_8/protein_encoder_2/BiasAddBiasAdd*model_8/protein_encoder_2/MatMul:product:08model_8/protein_encoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!model_8/protein_encoder_2/SigmoidSigmoid*model_8/protein_encoder_2/BiasAdd:output:0*
T0*'
_output_shapes
:����������
,model_8/gene_encoder_2/MatMul/ReadVariableOpReadVariableOp5model_8_gene_encoder_2_matmul_readvariableop_resource*
_output_shapes
:	�P*
dtype0�
model_8/gene_encoder_2/MatMulMatMul0model_8/BatchNormGeneEncode1/batchnorm/add_1:z:04model_8/gene_encoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
-model_8/gene_encoder_2/BiasAdd/ReadVariableOpReadVariableOp6model_8_gene_encoder_2_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
model_8/gene_encoder_2/BiasAddBiasAdd'model_8/gene_encoder_2/MatMul:product:05model_8/gene_encoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
model_8/gene_encoder_2/SigmoidSigmoid'model_8/gene_encoder_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������P�
5model_8/BatchNormGeneEncode2/batchnorm/ReadVariableOpReadVariableOp>model_8_batchnormgeneencode2_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0q
,model_8/BatchNormGeneEncode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
*model_8/BatchNormGeneEncode2/batchnorm/addAddV2=model_8/BatchNormGeneEncode2/batchnorm/ReadVariableOp:value:05model_8/BatchNormGeneEncode2/batchnorm/add/y:output:0*
T0*
_output_shapes
:P�
,model_8/BatchNormGeneEncode2/batchnorm/RsqrtRsqrt.model_8/BatchNormGeneEncode2/batchnorm/add:z:0*
T0*
_output_shapes
:P�
9model_8/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOpReadVariableOpBmodel_8_batchnormgeneencode2_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0�
*model_8/BatchNormGeneEncode2/batchnorm/mulMul0model_8/BatchNormGeneEncode2/batchnorm/Rsqrt:y:0Amodel_8/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P�
,model_8/BatchNormGeneEncode2/batchnorm/mul_1Mul"model_8/gene_encoder_2/Sigmoid:y:0.model_8/BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������P�
7model_8/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1ReadVariableOp@model_8_batchnormgeneencode2_batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0�
,model_8/BatchNormGeneEncode2/batchnorm/mul_2Mul?model_8/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1:value:0.model_8/BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*
_output_shapes
:P�
7model_8/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2ReadVariableOp@model_8_batchnormgeneencode2_batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0�
*model_8/BatchNormGeneEncode2/batchnorm/subSub?model_8/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2:value:00model_8/BatchNormGeneEncode2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:P�
,model_8/BatchNormGeneEncode2/batchnorm/add_1AddV20model_8/BatchNormGeneEncode2/batchnorm/mul_1:z:0.model_8/BatchNormGeneEncode2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������P�
8model_8/BatchNormProteinEncode2/batchnorm/ReadVariableOpReadVariableOpAmodel_8_batchnormproteinencode2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0t
/model_8/BatchNormProteinEncode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-model_8/BatchNormProteinEncode2/batchnorm/addAddV2@model_8/BatchNormProteinEncode2/batchnorm/ReadVariableOp:value:08model_8/BatchNormProteinEncode2/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
/model_8/BatchNormProteinEncode2/batchnorm/RsqrtRsqrt1model_8/BatchNormProteinEncode2/batchnorm/add:z:0*
T0*
_output_shapes
:�
<model_8/BatchNormProteinEncode2/batchnorm/mul/ReadVariableOpReadVariableOpEmodel_8_batchnormproteinencode2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
-model_8/BatchNormProteinEncode2/batchnorm/mulMul3model_8/BatchNormProteinEncode2/batchnorm/Rsqrt:y:0Dmodel_8/BatchNormProteinEncode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
/model_8/BatchNormProteinEncode2/batchnorm/mul_1Mul%model_8/protein_encoder_2/Sigmoid:y:01model_8/BatchNormProteinEncode2/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
:model_8/BatchNormProteinEncode2/batchnorm/ReadVariableOp_1ReadVariableOpCmodel_8_batchnormproteinencode2_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
/model_8/BatchNormProteinEncode2/batchnorm/mul_2MulBmodel_8/BatchNormProteinEncode2/batchnorm/ReadVariableOp_1:value:01model_8/BatchNormProteinEncode2/batchnorm/mul:z:0*
T0*
_output_shapes
:�
:model_8/BatchNormProteinEncode2/batchnorm/ReadVariableOp_2ReadVariableOpCmodel_8_batchnormproteinencode2_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
-model_8/BatchNormProteinEncode2/batchnorm/subSubBmodel_8/BatchNormProteinEncode2/batchnorm/ReadVariableOp_2:value:03model_8/BatchNormProteinEncode2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
/model_8/BatchNormProteinEncode2/batchnorm/add_1AddV23model_8/BatchNormProteinEncode2/batchnorm/mul_1:z:01model_8/BatchNormProteinEncode2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������f
$model_8/ConcatenateLayer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model_8/ConcatenateLayer/concatConcatV20model_8/BatchNormGeneEncode2/batchnorm/add_1:z:03model_8/BatchNormProteinEncode2/batchnorm/add_1:z:0-model_8/ConcatenateLayer/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������[�
/model_8/EmbeddingDimDense/MatMul/ReadVariableOpReadVariableOp8model_8_embeddingdimdense_matmul_readvariableop_resource*
_output_shapes

:[@*
dtype0�
 model_8/EmbeddingDimDense/MatMulMatMul(model_8/ConcatenateLayer/concat:output:07model_8/EmbeddingDimDense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
0model_8/EmbeddingDimDense/BiasAdd/ReadVariableOpReadVariableOp9model_8_embeddingdimdense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
!model_8/EmbeddingDimDense/BiasAddBiasAdd*model_8/EmbeddingDimDense/MatMul:product:08model_8/EmbeddingDimDense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
!model_8/EmbeddingDimDense/SigmoidSigmoid*model_8/EmbeddingDimDense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
/model_8/protein_decoder_1/MatMul/ReadVariableOpReadVariableOp8model_8_protein_decoder_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
 model_8/protein_decoder_1/MatMulMatMul%model_8/EmbeddingDimDense/Sigmoid:y:07model_8/protein_decoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
0model_8/protein_decoder_1/BiasAdd/ReadVariableOpReadVariableOp9model_8_protein_decoder_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
!model_8/protein_decoder_1/BiasAddBiasAdd*model_8/protein_decoder_1/MatMul:product:08model_8/protein_decoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
!model_8/protein_decoder_1/SigmoidSigmoid*model_8/protein_decoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
,model_8/gene_decoder_1/MatMul/ReadVariableOpReadVariableOp5model_8_gene_decoder_1_matmul_readvariableop_resource*
_output_shapes

:@P*
dtype0�
model_8/gene_decoder_1/MatMulMatMul%model_8/EmbeddingDimDense/Sigmoid:y:04model_8/gene_decoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
-model_8/gene_decoder_1/BiasAdd/ReadVariableOpReadVariableOp6model_8_gene_decoder_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
model_8/gene_decoder_1/BiasAddBiasAdd'model_8/gene_decoder_1/MatMul:product:05model_8/gene_decoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
model_8/gene_decoder_1/SigmoidSigmoid'model_8/gene_decoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������P�
8model_8/BatchNormProteinDecode1/batchnorm/ReadVariableOpReadVariableOpAmodel_8_batchnormproteindecode1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0t
/model_8/BatchNormProteinDecode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-model_8/BatchNormProteinDecode1/batchnorm/addAddV2@model_8/BatchNormProteinDecode1/batchnorm/ReadVariableOp:value:08model_8/BatchNormProteinDecode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
/model_8/BatchNormProteinDecode1/batchnorm/RsqrtRsqrt1model_8/BatchNormProteinDecode1/batchnorm/add:z:0*
T0*
_output_shapes
:�
<model_8/BatchNormProteinDecode1/batchnorm/mul/ReadVariableOpReadVariableOpEmodel_8_batchnormproteindecode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
-model_8/BatchNormProteinDecode1/batchnorm/mulMul3model_8/BatchNormProteinDecode1/batchnorm/Rsqrt:y:0Dmodel_8/BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
/model_8/BatchNormProteinDecode1/batchnorm/mul_1Mul%model_8/protein_decoder_1/Sigmoid:y:01model_8/BatchNormProteinDecode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
:model_8/BatchNormProteinDecode1/batchnorm/ReadVariableOp_1ReadVariableOpCmodel_8_batchnormproteindecode1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
/model_8/BatchNormProteinDecode1/batchnorm/mul_2MulBmodel_8/BatchNormProteinDecode1/batchnorm/ReadVariableOp_1:value:01model_8/BatchNormProteinDecode1/batchnorm/mul:z:0*
T0*
_output_shapes
:�
:model_8/BatchNormProteinDecode1/batchnorm/ReadVariableOp_2ReadVariableOpCmodel_8_batchnormproteindecode1_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
-model_8/BatchNormProteinDecode1/batchnorm/subSubBmodel_8/BatchNormProteinDecode1/batchnorm/ReadVariableOp_2:value:03model_8/BatchNormProteinDecode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
/model_8/BatchNormProteinDecode1/batchnorm/add_1AddV23model_8/BatchNormProteinDecode1/batchnorm/mul_1:z:01model_8/BatchNormProteinDecode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
5model_8/BatchNormGeneDecode1/batchnorm/ReadVariableOpReadVariableOp>model_8_batchnormgenedecode1_batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0q
,model_8/BatchNormGeneDecode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
*model_8/BatchNormGeneDecode1/batchnorm/addAddV2=model_8/BatchNormGeneDecode1/batchnorm/ReadVariableOp:value:05model_8/BatchNormGeneDecode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:P�
,model_8/BatchNormGeneDecode1/batchnorm/RsqrtRsqrt.model_8/BatchNormGeneDecode1/batchnorm/add:z:0*
T0*
_output_shapes
:P�
9model_8/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOpReadVariableOpBmodel_8_batchnormgenedecode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0�
*model_8/BatchNormGeneDecode1/batchnorm/mulMul0model_8/BatchNormGeneDecode1/batchnorm/Rsqrt:y:0Amodel_8/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P�
,model_8/BatchNormGeneDecode1/batchnorm/mul_1Mul"model_8/gene_decoder_1/Sigmoid:y:0.model_8/BatchNormGeneDecode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������P�
7model_8/BatchNormGeneDecode1/batchnorm/ReadVariableOp_1ReadVariableOp@model_8_batchnormgenedecode1_batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0�
,model_8/BatchNormGeneDecode1/batchnorm/mul_2Mul?model_8/BatchNormGeneDecode1/batchnorm/ReadVariableOp_1:value:0.model_8/BatchNormGeneDecode1/batchnorm/mul:z:0*
T0*
_output_shapes
:P�
7model_8/BatchNormGeneDecode1/batchnorm/ReadVariableOp_2ReadVariableOp@model_8_batchnormgenedecode1_batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0�
*model_8/BatchNormGeneDecode1/batchnorm/subSub?model_8/BatchNormGeneDecode1/batchnorm/ReadVariableOp_2:value:00model_8/BatchNormGeneDecode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:P�
,model_8/BatchNormGeneDecode1/batchnorm/add_1AddV20model_8/BatchNormGeneDecode1/batchnorm/mul_1:z:0.model_8/BatchNormGeneDecode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������P�
/model_8/protein_decoder_2/MatMul/ReadVariableOpReadVariableOp8model_8_protein_decoder_2_matmul_readvariableop_resource*
_output_shapes

:/*
dtype0�
 model_8/protein_decoder_2/MatMulMatMul3model_8/BatchNormProteinDecode1/batchnorm/add_1:z:07model_8/protein_decoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������/�
0model_8/protein_decoder_2/BiasAdd/ReadVariableOpReadVariableOp9model_8_protein_decoder_2_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0�
!model_8/protein_decoder_2/BiasAddBiasAdd*model_8/protein_decoder_2/MatMul:product:08model_8/protein_decoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������/�
!model_8/protein_decoder_2/SigmoidSigmoid*model_8/protein_decoder_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������/�
,model_8/gene_decoder_2/MatMul/ReadVariableOpReadVariableOp5model_8_gene_decoder_2_matmul_readvariableop_resource*
_output_shapes
:	P�*
dtype0�
model_8/gene_decoder_2/MatMulMatMul0model_8/BatchNormGeneDecode1/batchnorm/add_1:z:04model_8/gene_decoder_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-model_8/gene_decoder_2/BiasAdd/ReadVariableOpReadVariableOp6model_8_gene_decoder_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_8/gene_decoder_2/BiasAddBiasAdd'model_8/gene_decoder_2/MatMul:product:05model_8/gene_decoder_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
model_8/gene_decoder_2/SigmoidSigmoid'model_8/gene_decoder_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
8model_8/BatchNormProteinDecode2/batchnorm/ReadVariableOpReadVariableOpAmodel_8_batchnormproteindecode2_batchnorm_readvariableop_resource*
_output_shapes
:/*
dtype0t
/model_8/BatchNormProteinDecode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
-model_8/BatchNormProteinDecode2/batchnorm/addAddV2@model_8/BatchNormProteinDecode2/batchnorm/ReadVariableOp:value:08model_8/BatchNormProteinDecode2/batchnorm/add/y:output:0*
T0*
_output_shapes
:/�
/model_8/BatchNormProteinDecode2/batchnorm/RsqrtRsqrt1model_8/BatchNormProteinDecode2/batchnorm/add:z:0*
T0*
_output_shapes
:/�
<model_8/BatchNormProteinDecode2/batchnorm/mul/ReadVariableOpReadVariableOpEmodel_8_batchnormproteindecode2_batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0�
-model_8/BatchNormProteinDecode2/batchnorm/mulMul3model_8/BatchNormProteinDecode2/batchnorm/Rsqrt:y:0Dmodel_8/BatchNormProteinDecode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/�
/model_8/BatchNormProteinDecode2/batchnorm/mul_1Mul%model_8/protein_decoder_2/Sigmoid:y:01model_8/BatchNormProteinDecode2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������/�
:model_8/BatchNormProteinDecode2/batchnorm/ReadVariableOp_1ReadVariableOpCmodel_8_batchnormproteindecode2_batchnorm_readvariableop_1_resource*
_output_shapes
:/*
dtype0�
/model_8/BatchNormProteinDecode2/batchnorm/mul_2MulBmodel_8/BatchNormProteinDecode2/batchnorm/ReadVariableOp_1:value:01model_8/BatchNormProteinDecode2/batchnorm/mul:z:0*
T0*
_output_shapes
:/�
:model_8/BatchNormProteinDecode2/batchnorm/ReadVariableOp_2ReadVariableOpCmodel_8_batchnormproteindecode2_batchnorm_readvariableop_2_resource*
_output_shapes
:/*
dtype0�
-model_8/BatchNormProteinDecode2/batchnorm/subSubBmodel_8/BatchNormProteinDecode2/batchnorm/ReadVariableOp_2:value:03model_8/BatchNormProteinDecode2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:/�
/model_8/BatchNormProteinDecode2/batchnorm/add_1AddV23model_8/BatchNormProteinDecode2/batchnorm/mul_1:z:01model_8/BatchNormProteinDecode2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������/�
5model_8/BatchNormGeneDecode2/batchnorm/ReadVariableOpReadVariableOp>model_8_batchnormgenedecode2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
,model_8/BatchNormGeneDecode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
*model_8/BatchNormGeneDecode2/batchnorm/addAddV2=model_8/BatchNormGeneDecode2/batchnorm/ReadVariableOp:value:05model_8/BatchNormGeneDecode2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
,model_8/BatchNormGeneDecode2/batchnorm/RsqrtRsqrt.model_8/BatchNormGeneDecode2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
9model_8/BatchNormGeneDecode2/batchnorm/mul/ReadVariableOpReadVariableOpBmodel_8_batchnormgenedecode2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*model_8/BatchNormGeneDecode2/batchnorm/mulMul0model_8/BatchNormGeneDecode2/batchnorm/Rsqrt:y:0Amodel_8/BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
,model_8/BatchNormGeneDecode2/batchnorm/mul_1Mul"model_8/gene_decoder_2/Sigmoid:y:0.model_8/BatchNormGeneDecode2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
7model_8/BatchNormGeneDecode2/batchnorm/ReadVariableOp_1ReadVariableOp@model_8_batchnormgenedecode2_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
,model_8/BatchNormGeneDecode2/batchnorm/mul_2Mul?model_8/BatchNormGeneDecode2/batchnorm/ReadVariableOp_1:value:0.model_8/BatchNormGeneDecode2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
7model_8/BatchNormGeneDecode2/batchnorm/ReadVariableOp_2ReadVariableOp@model_8_batchnormgenedecode2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
*model_8/BatchNormGeneDecode2/batchnorm/subSub?model_8/BatchNormGeneDecode2/batchnorm/ReadVariableOp_2:value:00model_8/BatchNormGeneDecode2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
,model_8/BatchNormGeneDecode2/batchnorm/add_1AddV20model_8/BatchNormGeneDecode2/batchnorm/mul_1:z:0.model_8/BatchNormGeneDecode2/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
2model_8/protein_decoder_last/MatMul/ReadVariableOpReadVariableOp;model_8_protein_decoder_last_matmul_readvariableop_resource*
_output_shapes
:	/�*
dtype0�
#model_8/protein_decoder_last/MatMulMatMul3model_8/BatchNormProteinDecode2/batchnorm/add_1:z:0:model_8/protein_decoder_last/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
3model_8/protein_decoder_last/BiasAdd/ReadVariableOpReadVariableOp<model_8_protein_decoder_last_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
$model_8/protein_decoder_last/BiasAddBiasAdd-model_8/protein_decoder_last/MatMul:product:0;model_8/protein_decoder_last/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$model_8/protein_decoder_last/SigmoidSigmoid-model_8/protein_decoder_last/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
/model_8/gene_decoder_last/MatMul/ReadVariableOpReadVariableOp8model_8_gene_decoder_last_matmul_readvariableop_resource* 
_output_shapes
:
��
*
dtype0�
 model_8/gene_decoder_last/MatMulMatMul0model_8/BatchNormGeneDecode2/batchnorm/add_1:z:07model_8/gene_decoder_last/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
�
0model_8/gene_decoder_last/BiasAdd/ReadVariableOpReadVariableOp9model_8_gene_decoder_last_biasadd_readvariableop_resource*
_output_shapes	
:�
*
dtype0�
!model_8/gene_decoder_last/BiasAddBiasAdd*model_8/gene_decoder_last/MatMul:product:08model_8/gene_decoder_last/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
�
!model_8/gene_decoder_last/SigmoidSigmoid*model_8/gene_decoder_last/BiasAdd:output:0*
T0*(
_output_shapes
:����������
u
IdentityIdentity%model_8/gene_decoder_last/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������
z

Identity_1Identity(model_8/protein_decoder_last/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp6^model_8/BatchNormGeneDecode1/batchnorm/ReadVariableOp8^model_8/BatchNormGeneDecode1/batchnorm/ReadVariableOp_18^model_8/BatchNormGeneDecode1/batchnorm/ReadVariableOp_2:^model_8/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp6^model_8/BatchNormGeneDecode2/batchnorm/ReadVariableOp8^model_8/BatchNormGeneDecode2/batchnorm/ReadVariableOp_18^model_8/BatchNormGeneDecode2/batchnorm/ReadVariableOp_2:^model_8/BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp6^model_8/BatchNormGeneEncode1/batchnorm/ReadVariableOp8^model_8/BatchNormGeneEncode1/batchnorm/ReadVariableOp_18^model_8/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2:^model_8/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp6^model_8/BatchNormGeneEncode2/batchnorm/ReadVariableOp8^model_8/BatchNormGeneEncode2/batchnorm/ReadVariableOp_18^model_8/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2:^model_8/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp9^model_8/BatchNormProteinDecode1/batchnorm/ReadVariableOp;^model_8/BatchNormProteinDecode1/batchnorm/ReadVariableOp_1;^model_8/BatchNormProteinDecode1/batchnorm/ReadVariableOp_2=^model_8/BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp9^model_8/BatchNormProteinDecode2/batchnorm/ReadVariableOp;^model_8/BatchNormProteinDecode2/batchnorm/ReadVariableOp_1;^model_8/BatchNormProteinDecode2/batchnorm/ReadVariableOp_2=^model_8/BatchNormProteinDecode2/batchnorm/mul/ReadVariableOp9^model_8/BatchNormProteinEncode1/batchnorm/ReadVariableOp;^model_8/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1;^model_8/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2=^model_8/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp9^model_8/BatchNormProteinEncode2/batchnorm/ReadVariableOp;^model_8/BatchNormProteinEncode2/batchnorm/ReadVariableOp_1;^model_8/BatchNormProteinEncode2/batchnorm/ReadVariableOp_2=^model_8/BatchNormProteinEncode2/batchnorm/mul/ReadVariableOp1^model_8/EmbeddingDimDense/BiasAdd/ReadVariableOp0^model_8/EmbeddingDimDense/MatMul/ReadVariableOp.^model_8/gene_decoder_1/BiasAdd/ReadVariableOp-^model_8/gene_decoder_1/MatMul/ReadVariableOp.^model_8/gene_decoder_2/BiasAdd/ReadVariableOp-^model_8/gene_decoder_2/MatMul/ReadVariableOp1^model_8/gene_decoder_last/BiasAdd/ReadVariableOp0^model_8/gene_decoder_last/MatMul/ReadVariableOp.^model_8/gene_encoder_1/BiasAdd/ReadVariableOp-^model_8/gene_encoder_1/MatMul/ReadVariableOp.^model_8/gene_encoder_2/BiasAdd/ReadVariableOp-^model_8/gene_encoder_2/MatMul/ReadVariableOp1^model_8/protein_decoder_1/BiasAdd/ReadVariableOp0^model_8/protein_decoder_1/MatMul/ReadVariableOp1^model_8/protein_decoder_2/BiasAdd/ReadVariableOp0^model_8/protein_decoder_2/MatMul/ReadVariableOp4^model_8/protein_decoder_last/BiasAdd/ReadVariableOp3^model_8/protein_decoder_last/MatMul/ReadVariableOp1^model_8/protein_encoder_1/BiasAdd/ReadVariableOp0^model_8/protein_encoder_1/MatMul/ReadVariableOp1^model_8/protein_encoder_2/BiasAdd/ReadVariableOp0^model_8/protein_encoder_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������
:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2n
5model_8/BatchNormGeneDecode1/batchnorm/ReadVariableOp5model_8/BatchNormGeneDecode1/batchnorm/ReadVariableOp2r
7model_8/BatchNormGeneDecode1/batchnorm/ReadVariableOp_17model_8/BatchNormGeneDecode1/batchnorm/ReadVariableOp_12r
7model_8/BatchNormGeneDecode1/batchnorm/ReadVariableOp_27model_8/BatchNormGeneDecode1/batchnorm/ReadVariableOp_22v
9model_8/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp9model_8/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp2n
5model_8/BatchNormGeneDecode2/batchnorm/ReadVariableOp5model_8/BatchNormGeneDecode2/batchnorm/ReadVariableOp2r
7model_8/BatchNormGeneDecode2/batchnorm/ReadVariableOp_17model_8/BatchNormGeneDecode2/batchnorm/ReadVariableOp_12r
7model_8/BatchNormGeneDecode2/batchnorm/ReadVariableOp_27model_8/BatchNormGeneDecode2/batchnorm/ReadVariableOp_22v
9model_8/BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp9model_8/BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp2n
5model_8/BatchNormGeneEncode1/batchnorm/ReadVariableOp5model_8/BatchNormGeneEncode1/batchnorm/ReadVariableOp2r
7model_8/BatchNormGeneEncode1/batchnorm/ReadVariableOp_17model_8/BatchNormGeneEncode1/batchnorm/ReadVariableOp_12r
7model_8/BatchNormGeneEncode1/batchnorm/ReadVariableOp_27model_8/BatchNormGeneEncode1/batchnorm/ReadVariableOp_22v
9model_8/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp9model_8/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp2n
5model_8/BatchNormGeneEncode2/batchnorm/ReadVariableOp5model_8/BatchNormGeneEncode2/batchnorm/ReadVariableOp2r
7model_8/BatchNormGeneEncode2/batchnorm/ReadVariableOp_17model_8/BatchNormGeneEncode2/batchnorm/ReadVariableOp_12r
7model_8/BatchNormGeneEncode2/batchnorm/ReadVariableOp_27model_8/BatchNormGeneEncode2/batchnorm/ReadVariableOp_22v
9model_8/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp9model_8/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp2t
8model_8/BatchNormProteinDecode1/batchnorm/ReadVariableOp8model_8/BatchNormProteinDecode1/batchnorm/ReadVariableOp2x
:model_8/BatchNormProteinDecode1/batchnorm/ReadVariableOp_1:model_8/BatchNormProteinDecode1/batchnorm/ReadVariableOp_12x
:model_8/BatchNormProteinDecode1/batchnorm/ReadVariableOp_2:model_8/BatchNormProteinDecode1/batchnorm/ReadVariableOp_22|
<model_8/BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp<model_8/BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp2t
8model_8/BatchNormProteinDecode2/batchnorm/ReadVariableOp8model_8/BatchNormProteinDecode2/batchnorm/ReadVariableOp2x
:model_8/BatchNormProteinDecode2/batchnorm/ReadVariableOp_1:model_8/BatchNormProteinDecode2/batchnorm/ReadVariableOp_12x
:model_8/BatchNormProteinDecode2/batchnorm/ReadVariableOp_2:model_8/BatchNormProteinDecode2/batchnorm/ReadVariableOp_22|
<model_8/BatchNormProteinDecode2/batchnorm/mul/ReadVariableOp<model_8/BatchNormProteinDecode2/batchnorm/mul/ReadVariableOp2t
8model_8/BatchNormProteinEncode1/batchnorm/ReadVariableOp8model_8/BatchNormProteinEncode1/batchnorm/ReadVariableOp2x
:model_8/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1:model_8/BatchNormProteinEncode1/batchnorm/ReadVariableOp_12x
:model_8/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2:model_8/BatchNormProteinEncode1/batchnorm/ReadVariableOp_22|
<model_8/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp<model_8/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp2t
8model_8/BatchNormProteinEncode2/batchnorm/ReadVariableOp8model_8/BatchNormProteinEncode2/batchnorm/ReadVariableOp2x
:model_8/BatchNormProteinEncode2/batchnorm/ReadVariableOp_1:model_8/BatchNormProteinEncode2/batchnorm/ReadVariableOp_12x
:model_8/BatchNormProteinEncode2/batchnorm/ReadVariableOp_2:model_8/BatchNormProteinEncode2/batchnorm/ReadVariableOp_22|
<model_8/BatchNormProteinEncode2/batchnorm/mul/ReadVariableOp<model_8/BatchNormProteinEncode2/batchnorm/mul/ReadVariableOp2d
0model_8/EmbeddingDimDense/BiasAdd/ReadVariableOp0model_8/EmbeddingDimDense/BiasAdd/ReadVariableOp2b
/model_8/EmbeddingDimDense/MatMul/ReadVariableOp/model_8/EmbeddingDimDense/MatMul/ReadVariableOp2^
-model_8/gene_decoder_1/BiasAdd/ReadVariableOp-model_8/gene_decoder_1/BiasAdd/ReadVariableOp2\
,model_8/gene_decoder_1/MatMul/ReadVariableOp,model_8/gene_decoder_1/MatMul/ReadVariableOp2^
-model_8/gene_decoder_2/BiasAdd/ReadVariableOp-model_8/gene_decoder_2/BiasAdd/ReadVariableOp2\
,model_8/gene_decoder_2/MatMul/ReadVariableOp,model_8/gene_decoder_2/MatMul/ReadVariableOp2d
0model_8/gene_decoder_last/BiasAdd/ReadVariableOp0model_8/gene_decoder_last/BiasAdd/ReadVariableOp2b
/model_8/gene_decoder_last/MatMul/ReadVariableOp/model_8/gene_decoder_last/MatMul/ReadVariableOp2^
-model_8/gene_encoder_1/BiasAdd/ReadVariableOp-model_8/gene_encoder_1/BiasAdd/ReadVariableOp2\
,model_8/gene_encoder_1/MatMul/ReadVariableOp,model_8/gene_encoder_1/MatMul/ReadVariableOp2^
-model_8/gene_encoder_2/BiasAdd/ReadVariableOp-model_8/gene_encoder_2/BiasAdd/ReadVariableOp2\
,model_8/gene_encoder_2/MatMul/ReadVariableOp,model_8/gene_encoder_2/MatMul/ReadVariableOp2d
0model_8/protein_decoder_1/BiasAdd/ReadVariableOp0model_8/protein_decoder_1/BiasAdd/ReadVariableOp2b
/model_8/protein_decoder_1/MatMul/ReadVariableOp/model_8/protein_decoder_1/MatMul/ReadVariableOp2d
0model_8/protein_decoder_2/BiasAdd/ReadVariableOp0model_8/protein_decoder_2/BiasAdd/ReadVariableOp2b
/model_8/protein_decoder_2/MatMul/ReadVariableOp/model_8/protein_decoder_2/MatMul/ReadVariableOp2j
3model_8/protein_decoder_last/BiasAdd/ReadVariableOp3model_8/protein_decoder_last/BiasAdd/ReadVariableOp2h
2model_8/protein_decoder_last/MatMul/ReadVariableOp2model_8/protein_decoder_last/MatMul/ReadVariableOp2d
0model_8/protein_encoder_1/BiasAdd/ReadVariableOp0model_8/protein_encoder_1/BiasAdd/ReadVariableOp2b
/model_8/protein_encoder_1/MatMul/ReadVariableOp/model_8/protein_encoder_1/MatMul/ReadVariableOp2d
0model_8/protein_encoder_2/BiasAdd/ReadVariableOp0model_8/protein_encoder_2/BiasAdd/ReadVariableOp2b
/model_8/protein_encoder_2/MatMul/ReadVariableOp/model_8/protein_encoder_2/MatMul/ReadVariableOp:Z V
(
_output_shapes
:����������

*
_user_specified_nameGene_Input_Layer:]Y
(
_output_shapes
:����������
-
_user_specified_nameProtein_Input_Layer
�%
�
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_328783

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
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
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
2__inference_protein_decoder_2_layer_call_fn_331773

inputs
unknown:/
	unknown_0:/
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_decoder_2_layer_call_and_return_conditional_losses_329160o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������/`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_329047

inputs1
matmul_readvariableop_resource:	�P-
biasadd_readvariableop_resource:P
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������PV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������PZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������Pw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�%
�
S__inference_BatchNormProteinEncode2_layer_call_and_return_conditional_losses_331511

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
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
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
x
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_331524
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������[W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������["
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������P:���������:Q M
'
_output_shapes
:���������P
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
S__inference_BatchNormProteinEncode2_layer_call_and_return_conditional_losses_331477

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
J__inference_gene_decoder_2_layer_call_and_return_conditional_losses_329177

inputs1
matmul_readvariableop_resource:	P�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	P�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
(__inference_model_8_layer_call_fn_329949
gene_input_layer
protein_input_layer
unknown:	�/
	unknown_0:/
	unknown_1:
�
�
	unknown_2:	�
	unknown_3:/
	unknown_4:/
	unknown_5:/
	unknown_6:/
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:/

unknown_12:

unknown_13:	�P

unknown_14:P

unknown_15:P

unknown_16:P

unknown_17:P

unknown_18:P

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:[@

unknown_24:@

unknown_25:@

unknown_26:

unknown_27:@P

unknown_28:P

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:P

unknown_34:P

unknown_35:P

unknown_36:P

unknown_37:/

unknown_38:/

unknown_39:	P�

unknown_40:	�

unknown_41:/

unknown_42:/

unknown_43:/

unknown_44:/

unknown_45:	�

unknown_46:	�

unknown_47:	�

unknown_48:	�

unknown_49:	/�

unknown_50:	�

unknown_51:
��


unknown_52:	�

identity

identity_1��StatefulPartitionedCall�
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
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:����������
:����������*H
_read_only_resource_inputs*
(&	"#&'()*+./234567*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_8_layer_call_and_return_conditional_losses_329720p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������
r

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
_input_shapes�
�:����������
:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:����������

*
_user_specified_nameGene_Input_Layer:]Y
(
_output_shapes
:����������
-
_user_specified_nameProtein_Input_Layer
�

�
M__inference_protein_encoder_2_layer_call_and_return_conditional_losses_331351

inputs0
matmul_readvariableop_resource:/-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������/
 
_user_specified_nameinputs
�%
�
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_328537

inputs5
'assignmovingavg_readvariableop_resource:P7
)assignmovingavg_1_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P/
!batchnorm_readvariableop_resource:P
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:P�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������Pl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:P*
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
:P*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Px
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P�
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
:P*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:P~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P�
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
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Ph
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
��
�
C__inference_model_8_layer_call_and_return_conditional_losses_330083
gene_input_layer
protein_input_layer+
protein_encoder_1_329953:	�/&
protein_encoder_1_329955:/)
gene_encoder_1_329958:
�
�$
gene_encoder_1_329960:	�,
batchnormproteinencode1_329963:/,
batchnormproteinencode1_329965:/,
batchnormproteinencode1_329967:/,
batchnormproteinencode1_329969:/*
batchnormgeneencode1_329972:	�*
batchnormgeneencode1_329974:	�*
batchnormgeneencode1_329976:	�*
batchnormgeneencode1_329978:	�*
protein_encoder_2_329981:/&
protein_encoder_2_329983:(
gene_encoder_2_329986:	�P#
gene_encoder_2_329988:P)
batchnormgeneencode2_329991:P)
batchnormgeneencode2_329993:P)
batchnormgeneencode2_329995:P)
batchnormgeneencode2_329997:P,
batchnormproteinencode2_330000:,
batchnormproteinencode2_330002:,
batchnormproteinencode2_330004:,
batchnormproteinencode2_330006:*
embeddingdimdense_330010:[@&
embeddingdimdense_330012:@*
protein_decoder_1_330015:@&
protein_decoder_1_330017:'
gene_decoder_1_330020:@P#
gene_decoder_1_330022:P,
batchnormproteindecode1_330025:,
batchnormproteindecode1_330027:,
batchnormproteindecode1_330029:,
batchnormproteindecode1_330031:)
batchnormgenedecode1_330034:P)
batchnormgenedecode1_330036:P)
batchnormgenedecode1_330038:P)
batchnormgenedecode1_330040:P*
protein_decoder_2_330043:/&
protein_decoder_2_330045:/(
gene_decoder_2_330048:	P�$
gene_decoder_2_330050:	�,
batchnormproteindecode2_330053:/,
batchnormproteindecode2_330055:/,
batchnormproteindecode2_330057:/,
batchnormproteindecode2_330059:/*
batchnormgenedecode2_330062:	�*
batchnormgenedecode2_330064:	�*
batchnormgenedecode2_330066:	�*
batchnormgenedecode2_330068:	�.
protein_decoder_last_330071:	/�*
protein_decoder_last_330073:	�,
gene_decoder_last_330076:
��
'
gene_decoder_last_330078:	�

identity

identity_1��,BatchNormGeneDecode1/StatefulPartitionedCall�,BatchNormGeneDecode2/StatefulPartitionedCall�,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�/BatchNormProteinDecode1/StatefulPartitionedCall�/BatchNormProteinDecode2/StatefulPartitionedCall�/BatchNormProteinEncode1/StatefulPartitionedCall�/BatchNormProteinEncode2/StatefulPartitionedCall�)EmbeddingDimDense/StatefulPartitionedCall�&gene_decoder_1/StatefulPartitionedCall�&gene_decoder_2/StatefulPartitionedCall�)gene_decoder_last/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�)protein_decoder_1/StatefulPartitionedCall�)protein_decoder_2/StatefulPartitionedCall�,protein_decoder_last/StatefulPartitionedCall�)protein_encoder_1/StatefulPartitionedCall�)protein_encoder_2/StatefulPartitionedCall�
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallprotein_input_layerprotein_encoder_1_329953protein_encoder_1_329955*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_328978�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallgene_input_layergene_encoder_1_329958gene_encoder_1_329960*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_328995�
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_329963batchnormproteinencode1_329965batchnormproteinencode1_329967batchnormproteinencode1_329969*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_328408�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_329972batchnormgeneencode1_329974batchnormgeneencode1_329976batchnormgeneencode1_329978*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_328326�
)protein_encoder_2/StatefulPartitionedCallStatefulPartitionedCall8BatchNormProteinEncode1/StatefulPartitionedCall:output:0protein_encoder_2_329981protein_encoder_2_329983*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_encoder_2_layer_call_and_return_conditional_losses_329030�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_329986gene_encoder_2_329988*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_329047�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_329991batchnormgeneencode2_329993batchnormgeneencode2_329995batchnormgeneencode2_329997*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_328490�
/BatchNormProteinEncode2/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_2/StatefulPartitionedCall:output:0batchnormproteinencode2_330000batchnormproteinencode2_330002batchnormproteinencode2_330004batchnormproteinencode2_330006*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinEncode2_layer_call_and_return_conditional_losses_328572�
 ConcatenateLayer/PartitionedCallPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:08BatchNormProteinEncode2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������[* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_329078�
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_330010embeddingdimdense_330012*
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
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_329091�
)protein_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0protein_decoder_1_330015protein_decoder_1_330017*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_decoder_1_layer_call_and_return_conditional_losses_329108�
&gene_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0gene_decoder_1_330020gene_decoder_1_330022*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_decoder_1_layer_call_and_return_conditional_losses_329125�
/BatchNormProteinDecode1/StatefulPartitionedCallStatefulPartitionedCall2protein_decoder_1/StatefulPartitionedCall:output:0batchnormproteindecode1_330025batchnormproteindecode1_330027batchnormproteindecode1_330029batchnormproteindecode1_330031*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_328736�
,BatchNormGeneDecode1/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_1/StatefulPartitionedCall:output:0batchnormgenedecode1_330034batchnormgenedecode1_330036batchnormgenedecode1_330038batchnormgenedecode1_330040*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_328654�
)protein_decoder_2/StatefulPartitionedCallStatefulPartitionedCall8BatchNormProteinDecode1/StatefulPartitionedCall:output:0protein_decoder_2_330043protein_decoder_2_330045*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_decoder_2_layer_call_and_return_conditional_losses_329160�
&gene_decoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode1/StatefulPartitionedCall:output:0gene_decoder_2_330048gene_decoder_2_330050*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_decoder_2_layer_call_and_return_conditional_losses_329177�
/BatchNormProteinDecode2/StatefulPartitionedCallStatefulPartitionedCall2protein_decoder_2/StatefulPartitionedCall:output:0batchnormproteindecode2_330053batchnormproteindecode2_330055batchnormproteindecode2_330057batchnormproteindecode2_330059*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinDecode2_layer_call_and_return_conditional_losses_328900�
,BatchNormGeneDecode2/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_2/StatefulPartitionedCall:output:0batchnormgenedecode2_330062batchnormgenedecode2_330064batchnormgenedecode2_330066batchnormgenedecode2_330068*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_328818�
,protein_decoder_last/StatefulPartitionedCallStatefulPartitionedCall8BatchNormProteinDecode2/StatefulPartitionedCall:output:0protein_decoder_last_330071protein_decoder_last_330073*
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
P__inference_protein_decoder_last_layer_call_and_return_conditional_losses_329212�
)gene_decoder_last/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode2/StatefulPartitionedCall:output:0gene_decoder_last_330076gene_decoder_last_330078*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_gene_decoder_last_layer_call_and_return_conditional_losses_329229�
IdentityIdentity2gene_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������
�

Identity_1Identity5protein_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp-^BatchNormGeneDecode1/StatefulPartitionedCall-^BatchNormGeneDecode2/StatefulPartitionedCall-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinDecode1/StatefulPartitionedCall0^BatchNormProteinDecode2/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall0^BatchNormProteinEncode2/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_decoder_1/StatefulPartitionedCall'^gene_decoder_2/StatefulPartitionedCall*^gene_decoder_last/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_decoder_1/StatefulPartitionedCall*^protein_decoder_2/StatefulPartitionedCall-^protein_decoder_last/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*^protein_encoder_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������
:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneDecode1/StatefulPartitionedCall,BatchNormGeneDecode1/StatefulPartitionedCall2\
,BatchNormGeneDecode2/StatefulPartitionedCall,BatchNormGeneDecode2/StatefulPartitionedCall2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2b
/BatchNormProteinDecode1/StatefulPartitionedCall/BatchNormProteinDecode1/StatefulPartitionedCall2b
/BatchNormProteinDecode2/StatefulPartitionedCall/BatchNormProteinDecode2/StatefulPartitionedCall2b
/BatchNormProteinEncode1/StatefulPartitionedCall/BatchNormProteinEncode1/StatefulPartitionedCall2b
/BatchNormProteinEncode2/StatefulPartitionedCall/BatchNormProteinEncode2/StatefulPartitionedCall2V
)EmbeddingDimDense/StatefulPartitionedCall)EmbeddingDimDense/StatefulPartitionedCall2P
&gene_decoder_1/StatefulPartitionedCall&gene_decoder_1/StatefulPartitionedCall2P
&gene_decoder_2/StatefulPartitionedCall&gene_decoder_2/StatefulPartitionedCall2V
)gene_decoder_last/StatefulPartitionedCall)gene_decoder_last/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall2V
)protein_decoder_1/StatefulPartitionedCall)protein_decoder_1/StatefulPartitionedCall2V
)protein_decoder_2/StatefulPartitionedCall)protein_decoder_2/StatefulPartitionedCall2\
,protein_decoder_last/StatefulPartitionedCall,protein_decoder_last/StatefulPartitionedCall2V
)protein_encoder_1/StatefulPartitionedCall)protein_encoder_1/StatefulPartitionedCall2V
)protein_encoder_2/StatefulPartitionedCall)protein_encoder_2/StatefulPartitionedCall:Z V
(
_output_shapes
:����������

*
_user_specified_nameGene_Input_Layer:]Y
(
_output_shapes
:����������
-
_user_specified_nameProtein_Input_Layer
�
�
5__inference_BatchNormGeneDecode1_layer_call_fn_331610

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_328701o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�

�
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_331131

inputs2
matmul_readvariableop_resource:
�
�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�
�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������

 
_user_specified_nameinputs
�

�
M__inference_gene_decoder_last_layer_call_and_return_conditional_losses_331964

inputs2
matmul_readvariableop_resource:
��
.
biasadd_readvariableop_resource:	�

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�
*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������
[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_331197

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
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
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
(__inference_model_8_layer_call_fn_330455
inputs_0
inputs_1
unknown:	�/
	unknown_0:/
	unknown_1:
�
�
	unknown_2:	�
	unknown_3:/
	unknown_4:/
	unknown_5:/
	unknown_6:/
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:/

unknown_12:

unknown_13:	�P

unknown_14:P

unknown_15:P

unknown_16:P

unknown_17:P

unknown_18:P

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:[@

unknown_24:@

unknown_25:@

unknown_26:

unknown_27:@P

unknown_28:P

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:P

unknown_34:P

unknown_35:P

unknown_36:P

unknown_37:/

unknown_38:/

unknown_39:	P�

unknown_40:	�

unknown_41:/

unknown_42:/

unknown_43:/

unknown_44:/

unknown_45:	�

unknown_46:	�

unknown_47:	�

unknown_48:	�

unknown_49:	/�

unknown_50:	�

unknown_51:
��


unknown_52:	�

identity

identity_1��StatefulPartitionedCall�
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
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:����������
:����������*H
_read_only_resource_inputs*
(&	"#&'()*+./234567*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_8_layer_call_and_return_conditional_losses_329720p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������
r

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
_input_shapes�
�:����������
:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:����������

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
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_331231

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
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
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
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
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
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
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
8__inference_BatchNormProteinEncode1_layer_call_fn_331257

inputs
unknown:/
	unknown_0:/
	unknown_1:/
	unknown_2:/
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_328455o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������/`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������/: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������/
 
_user_specified_nameinputs
�
�
S__inference_BatchNormProteinDecode2_layer_call_and_return_conditional_losses_328900

inputs/
!batchnorm_readvariableop_resource:/3
%batchnorm_mul_readvariableop_resource:/1
#batchnorm_readvariableop_1_resource:/1
#batchnorm_readvariableop_2_resource:/
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:/*
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
:/P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:/~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������/z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:/*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:/z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:/*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:/r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������/b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������/�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������/: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������/
 
_user_specified_nameinputs
�
�
8__inference_BatchNormProteinDecode2_layer_call_fn_331877

inputs
unknown:/
	unknown_0:/
	unknown_1:/
	unknown_2:/
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinDecode2_layer_call_and_return_conditional_losses_328900o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������/`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������/: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������/
 
_user_specified_nameinputs
�
�
/__inference_gene_decoder_1_layer_call_fn_331553

inputs
unknown:@P
	unknown_0:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_decoder_1_layer_call_and_return_conditional_losses_329125o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P`
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
2__inference_protein_decoder_1_layer_call_fn_331573

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_decoder_1_layer_call_and_return_conditional_losses_329108o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
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
�
v
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_329078

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������[W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������["
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������P:���������:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
S__inference_BatchNormProteinDecode2_layer_call_and_return_conditional_losses_331910

inputs/
!batchnorm_readvariableop_resource:/3
%batchnorm_mul_readvariableop_resource:/1
#batchnorm_readvariableop_1_resource:/1
#batchnorm_readvariableop_2_resource:/
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:/*
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
:/P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:/~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������/z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:/*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:/z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:/*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:/r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������/b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������/�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������/: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������/
 
_user_specified_nameinputs
�
�
C__inference_model_8_layer_call_and_return_conditional_losses_329720

inputs
inputs_1+
protein_encoder_1_329590:	�/&
protein_encoder_1_329592:/)
gene_encoder_1_329595:
�
�$
gene_encoder_1_329597:	�,
batchnormproteinencode1_329600:/,
batchnormproteinencode1_329602:/,
batchnormproteinencode1_329604:/,
batchnormproteinencode1_329606:/*
batchnormgeneencode1_329609:	�*
batchnormgeneencode1_329611:	�*
batchnormgeneencode1_329613:	�*
batchnormgeneencode1_329615:	�*
protein_encoder_2_329618:/&
protein_encoder_2_329620:(
gene_encoder_2_329623:	�P#
gene_encoder_2_329625:P)
batchnormgeneencode2_329628:P)
batchnormgeneencode2_329630:P)
batchnormgeneencode2_329632:P)
batchnormgeneencode2_329634:P,
batchnormproteinencode2_329637:,
batchnormproteinencode2_329639:,
batchnormproteinencode2_329641:,
batchnormproteinencode2_329643:*
embeddingdimdense_329647:[@&
embeddingdimdense_329649:@*
protein_decoder_1_329652:@&
protein_decoder_1_329654:'
gene_decoder_1_329657:@P#
gene_decoder_1_329659:P,
batchnormproteindecode1_329662:,
batchnormproteindecode1_329664:,
batchnormproteindecode1_329666:,
batchnormproteindecode1_329668:)
batchnormgenedecode1_329671:P)
batchnormgenedecode1_329673:P)
batchnormgenedecode1_329675:P)
batchnormgenedecode1_329677:P*
protein_decoder_2_329680:/&
protein_decoder_2_329682:/(
gene_decoder_2_329685:	P�$
gene_decoder_2_329687:	�,
batchnormproteindecode2_329690:/,
batchnormproteindecode2_329692:/,
batchnormproteindecode2_329694:/,
batchnormproteindecode2_329696:/*
batchnormgenedecode2_329699:	�*
batchnormgenedecode2_329701:	�*
batchnormgenedecode2_329703:	�*
batchnormgenedecode2_329705:	�.
protein_decoder_last_329708:	/�*
protein_decoder_last_329710:	�,
gene_decoder_last_329713:
��
'
gene_decoder_last_329715:	�

identity

identity_1��,BatchNormGeneDecode1/StatefulPartitionedCall�,BatchNormGeneDecode2/StatefulPartitionedCall�,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�/BatchNormProteinDecode1/StatefulPartitionedCall�/BatchNormProteinDecode2/StatefulPartitionedCall�/BatchNormProteinEncode1/StatefulPartitionedCall�/BatchNormProteinEncode2/StatefulPartitionedCall�)EmbeddingDimDense/StatefulPartitionedCall�&gene_decoder_1/StatefulPartitionedCall�&gene_decoder_2/StatefulPartitionedCall�)gene_decoder_last/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�)protein_decoder_1/StatefulPartitionedCall�)protein_decoder_2/StatefulPartitionedCall�,protein_decoder_last/StatefulPartitionedCall�)protein_encoder_1/StatefulPartitionedCall�)protein_encoder_2/StatefulPartitionedCall�
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1protein_encoder_1_329590protein_encoder_1_329592*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_328978�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputsgene_encoder_1_329595gene_encoder_1_329597*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_328995�
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_329600batchnormproteinencode1_329602batchnormproteinencode1_329604batchnormproteinencode1_329606*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_328455�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_329609batchnormgeneencode1_329611batchnormgeneencode1_329613batchnormgeneencode1_329615*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_328373�
)protein_encoder_2/StatefulPartitionedCallStatefulPartitionedCall8BatchNormProteinEncode1/StatefulPartitionedCall:output:0protein_encoder_2_329618protein_encoder_2_329620*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_encoder_2_layer_call_and_return_conditional_losses_329030�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_329623gene_encoder_2_329625*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_329047�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_329628batchnormgeneencode2_329630batchnormgeneencode2_329632batchnormgeneencode2_329634*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_328537�
/BatchNormProteinEncode2/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_2/StatefulPartitionedCall:output:0batchnormproteinencode2_329637batchnormproteinencode2_329639batchnormproteinencode2_329641batchnormproteinencode2_329643*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinEncode2_layer_call_and_return_conditional_losses_328619�
 ConcatenateLayer/PartitionedCallPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:08BatchNormProteinEncode2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������[* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_329078�
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_329647embeddingdimdense_329649*
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
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_329091�
)protein_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0protein_decoder_1_329652protein_decoder_1_329654*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_decoder_1_layer_call_and_return_conditional_losses_329108�
&gene_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0gene_decoder_1_329657gene_decoder_1_329659*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_decoder_1_layer_call_and_return_conditional_losses_329125�
/BatchNormProteinDecode1/StatefulPartitionedCallStatefulPartitionedCall2protein_decoder_1/StatefulPartitionedCall:output:0batchnormproteindecode1_329662batchnormproteindecode1_329664batchnormproteindecode1_329666batchnormproteindecode1_329668*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_328783�
,BatchNormGeneDecode1/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_1/StatefulPartitionedCall:output:0batchnormgenedecode1_329671batchnormgenedecode1_329673batchnormgenedecode1_329675batchnormgenedecode1_329677*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_328701�
)protein_decoder_2/StatefulPartitionedCallStatefulPartitionedCall8BatchNormProteinDecode1/StatefulPartitionedCall:output:0protein_decoder_2_329680protein_decoder_2_329682*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_decoder_2_layer_call_and_return_conditional_losses_329160�
&gene_decoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode1/StatefulPartitionedCall:output:0gene_decoder_2_329685gene_decoder_2_329687*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_decoder_2_layer_call_and_return_conditional_losses_329177�
/BatchNormProteinDecode2/StatefulPartitionedCallStatefulPartitionedCall2protein_decoder_2/StatefulPartitionedCall:output:0batchnormproteindecode2_329690batchnormproteindecode2_329692batchnormproteindecode2_329694batchnormproteindecode2_329696*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinDecode2_layer_call_and_return_conditional_losses_328947�
,BatchNormGeneDecode2/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_2/StatefulPartitionedCall:output:0batchnormgenedecode2_329699batchnormgenedecode2_329701batchnormgenedecode2_329703batchnormgenedecode2_329705*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_328865�
,protein_decoder_last/StatefulPartitionedCallStatefulPartitionedCall8BatchNormProteinDecode2/StatefulPartitionedCall:output:0protein_decoder_last_329708protein_decoder_last_329710*
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
P__inference_protein_decoder_last_layer_call_and_return_conditional_losses_329212�
)gene_decoder_last/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode2/StatefulPartitionedCall:output:0gene_decoder_last_329713gene_decoder_last_329715*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_gene_decoder_last_layer_call_and_return_conditional_losses_329229�
IdentityIdentity2gene_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������
�

Identity_1Identity5protein_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp-^BatchNormGeneDecode1/StatefulPartitionedCall-^BatchNormGeneDecode2/StatefulPartitionedCall-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinDecode1/StatefulPartitionedCall0^BatchNormProteinDecode2/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall0^BatchNormProteinEncode2/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_decoder_1/StatefulPartitionedCall'^gene_decoder_2/StatefulPartitionedCall*^gene_decoder_last/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_decoder_1/StatefulPartitionedCall*^protein_decoder_2/StatefulPartitionedCall-^protein_decoder_last/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*^protein_encoder_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������
:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneDecode1/StatefulPartitionedCall,BatchNormGeneDecode1/StatefulPartitionedCall2\
,BatchNormGeneDecode2/StatefulPartitionedCall,BatchNormGeneDecode2/StatefulPartitionedCall2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2b
/BatchNormProteinDecode1/StatefulPartitionedCall/BatchNormProteinDecode1/StatefulPartitionedCall2b
/BatchNormProteinDecode2/StatefulPartitionedCall/BatchNormProteinDecode2/StatefulPartitionedCall2b
/BatchNormProteinEncode1/StatefulPartitionedCall/BatchNormProteinEncode1/StatefulPartitionedCall2b
/BatchNormProteinEncode2/StatefulPartitionedCall/BatchNormProteinEncode2/StatefulPartitionedCall2V
)EmbeddingDimDense/StatefulPartitionedCall)EmbeddingDimDense/StatefulPartitionedCall2P
&gene_decoder_1/StatefulPartitionedCall&gene_decoder_1/StatefulPartitionedCall2P
&gene_decoder_2/StatefulPartitionedCall&gene_decoder_2/StatefulPartitionedCall2V
)gene_decoder_last/StatefulPartitionedCall)gene_decoder_last/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall2V
)protein_decoder_1/StatefulPartitionedCall)protein_decoder_1/StatefulPartitionedCall2V
)protein_decoder_2/StatefulPartitionedCall)protein_decoder_2/StatefulPartitionedCall2\
,protein_decoder_last/StatefulPartitionedCall,protein_decoder_last/StatefulPartitionedCall2V
)protein_encoder_1/StatefulPartitionedCall)protein_encoder_1/StatefulPartitionedCall2V
)protein_encoder_2/StatefulPartitionedCall)protein_encoder_2/StatefulPartitionedCall:P L
(
_output_shapes
:����������

 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs
Ϸ
�2
C__inference_model_8_layer_call_and_return_conditional_losses_330668
inputs_0
inputs_1C
0protein_encoder_1_matmul_readvariableop_resource:	�/?
1protein_encoder_1_biasadd_readvariableop_resource:/A
-gene_encoder_1_matmul_readvariableop_resource:
�
�=
.gene_encoder_1_biasadd_readvariableop_resource:	�G
9batchnormproteinencode1_batchnorm_readvariableop_resource:/K
=batchnormproteinencode1_batchnorm_mul_readvariableop_resource:/I
;batchnormproteinencode1_batchnorm_readvariableop_1_resource:/I
;batchnormproteinencode1_batchnorm_readvariableop_2_resource:/E
6batchnormgeneencode1_batchnorm_readvariableop_resource:	�I
:batchnormgeneencode1_batchnorm_mul_readvariableop_resource:	�G
8batchnormgeneencode1_batchnorm_readvariableop_1_resource:	�G
8batchnormgeneencode1_batchnorm_readvariableop_2_resource:	�B
0protein_encoder_2_matmul_readvariableop_resource:/?
1protein_encoder_2_biasadd_readvariableop_resource:@
-gene_encoder_2_matmul_readvariableop_resource:	�P<
.gene_encoder_2_biasadd_readvariableop_resource:PD
6batchnormgeneencode2_batchnorm_readvariableop_resource:PH
:batchnormgeneencode2_batchnorm_mul_readvariableop_resource:PF
8batchnormgeneencode2_batchnorm_readvariableop_1_resource:PF
8batchnormgeneencode2_batchnorm_readvariableop_2_resource:PG
9batchnormproteinencode2_batchnorm_readvariableop_resource:K
=batchnormproteinencode2_batchnorm_mul_readvariableop_resource:I
;batchnormproteinencode2_batchnorm_readvariableop_1_resource:I
;batchnormproteinencode2_batchnorm_readvariableop_2_resource:B
0embeddingdimdense_matmul_readvariableop_resource:[@?
1embeddingdimdense_biasadd_readvariableop_resource:@B
0protein_decoder_1_matmul_readvariableop_resource:@?
1protein_decoder_1_biasadd_readvariableop_resource:?
-gene_decoder_1_matmul_readvariableop_resource:@P<
.gene_decoder_1_biasadd_readvariableop_resource:PG
9batchnormproteindecode1_batchnorm_readvariableop_resource:K
=batchnormproteindecode1_batchnorm_mul_readvariableop_resource:I
;batchnormproteindecode1_batchnorm_readvariableop_1_resource:I
;batchnormproteindecode1_batchnorm_readvariableop_2_resource:D
6batchnormgenedecode1_batchnorm_readvariableop_resource:PH
:batchnormgenedecode1_batchnorm_mul_readvariableop_resource:PF
8batchnormgenedecode1_batchnorm_readvariableop_1_resource:PF
8batchnormgenedecode1_batchnorm_readvariableop_2_resource:PB
0protein_decoder_2_matmul_readvariableop_resource:/?
1protein_decoder_2_biasadd_readvariableop_resource:/@
-gene_decoder_2_matmul_readvariableop_resource:	P�=
.gene_decoder_2_biasadd_readvariableop_resource:	�G
9batchnormproteindecode2_batchnorm_readvariableop_resource:/K
=batchnormproteindecode2_batchnorm_mul_readvariableop_resource:/I
;batchnormproteindecode2_batchnorm_readvariableop_1_resource:/I
;batchnormproteindecode2_batchnorm_readvariableop_2_resource:/E
6batchnormgenedecode2_batchnorm_readvariableop_resource:	�I
:batchnormgenedecode2_batchnorm_mul_readvariableop_resource:	�G
8batchnormgenedecode2_batchnorm_readvariableop_1_resource:	�G
8batchnormgenedecode2_batchnorm_readvariableop_2_resource:	�F
3protein_decoder_last_matmul_readvariableop_resource:	/�C
4protein_decoder_last_biasadd_readvariableop_resource:	�D
0gene_decoder_last_matmul_readvariableop_resource:
��
@
1gene_decoder_last_biasadd_readvariableop_resource:	�

identity

identity_1��-BatchNormGeneDecode1/batchnorm/ReadVariableOp�/BatchNormGeneDecode1/batchnorm/ReadVariableOp_1�/BatchNormGeneDecode1/batchnorm/ReadVariableOp_2�1BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp�-BatchNormGeneDecode2/batchnorm/ReadVariableOp�/BatchNormGeneDecode2/batchnorm/ReadVariableOp_1�/BatchNormGeneDecode2/batchnorm/ReadVariableOp_2�1BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp�-BatchNormGeneEncode1/batchnorm/ReadVariableOp�/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1�/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2�1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp�-BatchNormGeneEncode2/batchnorm/ReadVariableOp�/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1�/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2�1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp�0BatchNormProteinDecode1/batchnorm/ReadVariableOp�2BatchNormProteinDecode1/batchnorm/ReadVariableOp_1�2BatchNormProteinDecode1/batchnorm/ReadVariableOp_2�4BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp�0BatchNormProteinDecode2/batchnorm/ReadVariableOp�2BatchNormProteinDecode2/batchnorm/ReadVariableOp_1�2BatchNormProteinDecode2/batchnorm/ReadVariableOp_2�4BatchNormProteinDecode2/batchnorm/mul/ReadVariableOp�0BatchNormProteinEncode1/batchnorm/ReadVariableOp�2BatchNormProteinEncode1/batchnorm/ReadVariableOp_1�2BatchNormProteinEncode1/batchnorm/ReadVariableOp_2�4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp�0BatchNormProteinEncode2/batchnorm/ReadVariableOp�2BatchNormProteinEncode2/batchnorm/ReadVariableOp_1�2BatchNormProteinEncode2/batchnorm/ReadVariableOp_2�4BatchNormProteinEncode2/batchnorm/mul/ReadVariableOp�(EmbeddingDimDense/BiasAdd/ReadVariableOp�'EmbeddingDimDense/MatMul/ReadVariableOp�%gene_decoder_1/BiasAdd/ReadVariableOp�$gene_decoder_1/MatMul/ReadVariableOp�%gene_decoder_2/BiasAdd/ReadVariableOp�$gene_decoder_2/MatMul/ReadVariableOp�(gene_decoder_last/BiasAdd/ReadVariableOp�'gene_decoder_last/MatMul/ReadVariableOp�%gene_encoder_1/BiasAdd/ReadVariableOp�$gene_encoder_1/MatMul/ReadVariableOp�%gene_encoder_2/BiasAdd/ReadVariableOp�$gene_encoder_2/MatMul/ReadVariableOp�(protein_decoder_1/BiasAdd/ReadVariableOp�'protein_decoder_1/MatMul/ReadVariableOp�(protein_decoder_2/BiasAdd/ReadVariableOp�'protein_decoder_2/MatMul/ReadVariableOp�+protein_decoder_last/BiasAdd/ReadVariableOp�*protein_decoder_last/MatMul/ReadVariableOp�(protein_encoder_1/BiasAdd/ReadVariableOp�'protein_encoder_1/MatMul/ReadVariableOp�(protein_encoder_2/BiasAdd/ReadVariableOp�'protein_encoder_2/MatMul/ReadVariableOp�
'protein_encoder_1/MatMul/ReadVariableOpReadVariableOp0protein_encoder_1_matmul_readvariableop_resource*
_output_shapes
:	�/*
dtype0�
protein_encoder_1/MatMulMatMulinputs_1/protein_encoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������/�
(protein_encoder_1/BiasAdd/ReadVariableOpReadVariableOp1protein_encoder_1_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0�
protein_encoder_1/BiasAddBiasAdd"protein_encoder_1/MatMul:product:00protein_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������/z
protein_encoder_1/SigmoidSigmoid"protein_encoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������/�
$gene_encoder_1/MatMul/ReadVariableOpReadVariableOp-gene_encoder_1_matmul_readvariableop_resource* 
_output_shapes
:
�
�*
dtype0�
gene_encoder_1/MatMulMatMulinputs_0,gene_encoder_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%gene_encoder_1/BiasAdd/ReadVariableOpReadVariableOp.gene_encoder_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
gene_encoder_1/BiasAddBiasAddgene_encoder_1/MatMul:product:0-gene_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
gene_encoder_1/SigmoidSigmoidgene_encoder_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
0BatchNormProteinEncode1/batchnorm/ReadVariableOpReadVariableOp9batchnormproteinencode1_batchnorm_readvariableop_resource*
_output_shapes
:/*
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
:/�
'BatchNormProteinEncode1/batchnorm/RsqrtRsqrt)BatchNormProteinEncode1/batchnorm/add:z:0*
T0*
_output_shapes
:/�
4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOpReadVariableOp=batchnormproteinencode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0�
%BatchNormProteinEncode1/batchnorm/mulMul+BatchNormProteinEncode1/batchnorm/Rsqrt:y:0<BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/�
'BatchNormProteinEncode1/batchnorm/mul_1Mulprotein_encoder_1/Sigmoid:y:0)BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������/�
2BatchNormProteinEncode1/batchnorm/ReadVariableOp_1ReadVariableOp;batchnormproteinencode1_batchnorm_readvariableop_1_resource*
_output_shapes
:/*
dtype0�
'BatchNormProteinEncode1/batchnorm/mul_2Mul:BatchNormProteinEncode1/batchnorm/ReadVariableOp_1:value:0)BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*
_output_shapes
:/�
2BatchNormProteinEncode1/batchnorm/ReadVariableOp_2ReadVariableOp;batchnormproteinencode1_batchnorm_readvariableop_2_resource*
_output_shapes
:/*
dtype0�
%BatchNormProteinEncode1/batchnorm/subSub:BatchNormProteinEncode1/batchnorm/ReadVariableOp_2:value:0+BatchNormProteinEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:/�
'BatchNormProteinEncode1/batchnorm/add_1AddV2+BatchNormProteinEncode1/batchnorm/mul_1:z:0)BatchNormProteinEncode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������/�
-BatchNormGeneEncode1/batchnorm/ReadVariableOpReadVariableOp6batchnormgeneencode1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
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
:�{
$BatchNormGeneEncode1/batchnorm/RsqrtRsqrt&BatchNormGeneEncode1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgeneencode1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"BatchNormGeneEncode1/batchnorm/mulMul(BatchNormGeneEncode1/batchnorm/Rsqrt:y:09BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
$BatchNormGeneEncode1/batchnorm/mul_1Mulgene_encoder_1/Sigmoid:y:0&BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1ReadVariableOp8batchnormgeneencode1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
$BatchNormGeneEncode1/batchnorm/mul_2Mul7BatchNormGeneEncode1/batchnorm/ReadVariableOp_1:value:0&BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2ReadVariableOp8batchnormgeneencode1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
"BatchNormGeneEncode1/batchnorm/subSub7BatchNormGeneEncode1/batchnorm/ReadVariableOp_2:value:0(BatchNormGeneEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
$BatchNormGeneEncode1/batchnorm/add_1AddV2(BatchNormGeneEncode1/batchnorm/mul_1:z:0&BatchNormGeneEncode1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
'protein_encoder_2/MatMul/ReadVariableOpReadVariableOp0protein_encoder_2_matmul_readvariableop_resource*
_output_shapes

:/*
dtype0�
protein_encoder_2/MatMulMatMul+BatchNormProteinEncode1/batchnorm/add_1:z:0/protein_encoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(protein_encoder_2/BiasAdd/ReadVariableOpReadVariableOp1protein_encoder_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
protein_encoder_2/BiasAddBiasAdd"protein_encoder_2/MatMul:product:00protein_encoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������z
protein_encoder_2/SigmoidSigmoid"protein_encoder_2/BiasAdd:output:0*
T0*'
_output_shapes
:����������
$gene_encoder_2/MatMul/ReadVariableOpReadVariableOp-gene_encoder_2_matmul_readvariableop_resource*
_output_shapes
:	�P*
dtype0�
gene_encoder_2/MatMulMatMul(BatchNormGeneEncode1/batchnorm/add_1:z:0,gene_encoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
%gene_encoder_2/BiasAdd/ReadVariableOpReadVariableOp.gene_encoder_2_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
gene_encoder_2/BiasAddBiasAddgene_encoder_2/MatMul:product:0-gene_encoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pt
gene_encoder_2/SigmoidSigmoidgene_encoder_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������P�
-BatchNormGeneEncode2/batchnorm/ReadVariableOpReadVariableOp6batchnormgeneencode2_batchnorm_readvariableop_resource*
_output_shapes
:P*
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
:Pz
$BatchNormGeneEncode2/batchnorm/RsqrtRsqrt&BatchNormGeneEncode2/batchnorm/add:z:0*
T0*
_output_shapes
:P�
1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgeneencode2_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0�
"BatchNormGeneEncode2/batchnorm/mulMul(BatchNormGeneEncode2/batchnorm/Rsqrt:y:09BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P�
$BatchNormGeneEncode2/batchnorm/mul_1Mulgene_encoder_2/Sigmoid:y:0&BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������P�
/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1ReadVariableOp8batchnormgeneencode2_batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0�
$BatchNormGeneEncode2/batchnorm/mul_2Mul7BatchNormGeneEncode2/batchnorm/ReadVariableOp_1:value:0&BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*
_output_shapes
:P�
/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2ReadVariableOp8batchnormgeneencode2_batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0�
"BatchNormGeneEncode2/batchnorm/subSub7BatchNormGeneEncode2/batchnorm/ReadVariableOp_2:value:0(BatchNormGeneEncode2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:P�
$BatchNormGeneEncode2/batchnorm/add_1AddV2(BatchNormGeneEncode2/batchnorm/mul_1:z:0&BatchNormGeneEncode2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������P�
0BatchNormProteinEncode2/batchnorm/ReadVariableOpReadVariableOp9batchnormproteinencode2_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'BatchNormProteinEncode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
%BatchNormProteinEncode2/batchnorm/addAddV28BatchNormProteinEncode2/batchnorm/ReadVariableOp:value:00BatchNormProteinEncode2/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
'BatchNormProteinEncode2/batchnorm/RsqrtRsqrt)BatchNormProteinEncode2/batchnorm/add:z:0*
T0*
_output_shapes
:�
4BatchNormProteinEncode2/batchnorm/mul/ReadVariableOpReadVariableOp=batchnormproteinencode2_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
%BatchNormProteinEncode2/batchnorm/mulMul+BatchNormProteinEncode2/batchnorm/Rsqrt:y:0<BatchNormProteinEncode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
'BatchNormProteinEncode2/batchnorm/mul_1Mulprotein_encoder_2/Sigmoid:y:0)BatchNormProteinEncode2/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
2BatchNormProteinEncode2/batchnorm/ReadVariableOp_1ReadVariableOp;batchnormproteinencode2_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
'BatchNormProteinEncode2/batchnorm/mul_2Mul:BatchNormProteinEncode2/batchnorm/ReadVariableOp_1:value:0)BatchNormProteinEncode2/batchnorm/mul:z:0*
T0*
_output_shapes
:�
2BatchNormProteinEncode2/batchnorm/ReadVariableOp_2ReadVariableOp;batchnormproteinencode2_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
%BatchNormProteinEncode2/batchnorm/subSub:BatchNormProteinEncode2/batchnorm/ReadVariableOp_2:value:0+BatchNormProteinEncode2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
'BatchNormProteinEncode2/batchnorm/add_1AddV2+BatchNormProteinEncode2/batchnorm/mul_1:z:0)BatchNormProteinEncode2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������^
ConcatenateLayer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
ConcatenateLayer/concatConcatV2(BatchNormGeneEncode2/batchnorm/add_1:z:0+BatchNormProteinEncode2/batchnorm/add_1:z:0%ConcatenateLayer/concat/axis:output:0*
N*
T0*'
_output_shapes
:���������[�
'EmbeddingDimDense/MatMul/ReadVariableOpReadVariableOp0embeddingdimdense_matmul_readvariableop_resource*
_output_shapes

:[@*
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
'protein_decoder_1/MatMul/ReadVariableOpReadVariableOp0protein_decoder_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
protein_decoder_1/MatMulMatMulEmbeddingDimDense/Sigmoid:y:0/protein_decoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(protein_decoder_1/BiasAdd/ReadVariableOpReadVariableOp1protein_decoder_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
protein_decoder_1/BiasAddBiasAdd"protein_decoder_1/MatMul:product:00protein_decoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������z
protein_decoder_1/SigmoidSigmoid"protein_decoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
$gene_decoder_1/MatMul/ReadVariableOpReadVariableOp-gene_decoder_1_matmul_readvariableop_resource*
_output_shapes

:@P*
dtype0�
gene_decoder_1/MatMulMatMulEmbeddingDimDense/Sigmoid:y:0,gene_decoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P�
%gene_decoder_1/BiasAdd/ReadVariableOpReadVariableOp.gene_decoder_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype0�
gene_decoder_1/BiasAddBiasAddgene_decoder_1/MatMul:product:0-gene_decoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pt
gene_decoder_1/SigmoidSigmoidgene_decoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������P�
0BatchNormProteinDecode1/batchnorm/ReadVariableOpReadVariableOp9batchnormproteindecode1_batchnorm_readvariableop_resource*
_output_shapes
:*
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
:�
'BatchNormProteinDecode1/batchnorm/RsqrtRsqrt)BatchNormProteinDecode1/batchnorm/add:z:0*
T0*
_output_shapes
:�
4BatchNormProteinDecode1/batchnorm/mul/ReadVariableOpReadVariableOp=batchnormproteindecode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
%BatchNormProteinDecode1/batchnorm/mulMul+BatchNormProteinDecode1/batchnorm/Rsqrt:y:0<BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
'BatchNormProteinDecode1/batchnorm/mul_1Mulprotein_decoder_1/Sigmoid:y:0)BatchNormProteinDecode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
2BatchNormProteinDecode1/batchnorm/ReadVariableOp_1ReadVariableOp;batchnormproteindecode1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
'BatchNormProteinDecode1/batchnorm/mul_2Mul:BatchNormProteinDecode1/batchnorm/ReadVariableOp_1:value:0)BatchNormProteinDecode1/batchnorm/mul:z:0*
T0*
_output_shapes
:�
2BatchNormProteinDecode1/batchnorm/ReadVariableOp_2ReadVariableOp;batchnormproteindecode1_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
%BatchNormProteinDecode1/batchnorm/subSub:BatchNormProteinDecode1/batchnorm/ReadVariableOp_2:value:0+BatchNormProteinDecode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
'BatchNormProteinDecode1/batchnorm/add_1AddV2+BatchNormProteinDecode1/batchnorm/mul_1:z:0)BatchNormProteinDecode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
-BatchNormGeneDecode1/batchnorm/ReadVariableOpReadVariableOp6batchnormgenedecode1_batchnorm_readvariableop_resource*
_output_shapes
:P*
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
:Pz
$BatchNormGeneDecode1/batchnorm/RsqrtRsqrt&BatchNormGeneDecode1/batchnorm/add:z:0*
T0*
_output_shapes
:P�
1BatchNormGeneDecode1/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgenedecode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0�
"BatchNormGeneDecode1/batchnorm/mulMul(BatchNormGeneDecode1/batchnorm/Rsqrt:y:09BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:P�
$BatchNormGeneDecode1/batchnorm/mul_1Mulgene_decoder_1/Sigmoid:y:0&BatchNormGeneDecode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������P�
/BatchNormGeneDecode1/batchnorm/ReadVariableOp_1ReadVariableOp8batchnormgenedecode1_batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0�
$BatchNormGeneDecode1/batchnorm/mul_2Mul7BatchNormGeneDecode1/batchnorm/ReadVariableOp_1:value:0&BatchNormGeneDecode1/batchnorm/mul:z:0*
T0*
_output_shapes
:P�
/BatchNormGeneDecode1/batchnorm/ReadVariableOp_2ReadVariableOp8batchnormgenedecode1_batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0�
"BatchNormGeneDecode1/batchnorm/subSub7BatchNormGeneDecode1/batchnorm/ReadVariableOp_2:value:0(BatchNormGeneDecode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:P�
$BatchNormGeneDecode1/batchnorm/add_1AddV2(BatchNormGeneDecode1/batchnorm/mul_1:z:0&BatchNormGeneDecode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������P�
'protein_decoder_2/MatMul/ReadVariableOpReadVariableOp0protein_decoder_2_matmul_readvariableop_resource*
_output_shapes

:/*
dtype0�
protein_decoder_2/MatMulMatMul+BatchNormProteinDecode1/batchnorm/add_1:z:0/protein_decoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������/�
(protein_decoder_2/BiasAdd/ReadVariableOpReadVariableOp1protein_decoder_2_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0�
protein_decoder_2/BiasAddBiasAdd"protein_decoder_2/MatMul:product:00protein_decoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������/z
protein_decoder_2/SigmoidSigmoid"protein_decoder_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������/�
$gene_decoder_2/MatMul/ReadVariableOpReadVariableOp-gene_decoder_2_matmul_readvariableop_resource*
_output_shapes
:	P�*
dtype0�
gene_decoder_2/MatMulMatMul(BatchNormGeneDecode1/batchnorm/add_1:z:0,gene_decoder_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
%gene_decoder_2/BiasAdd/ReadVariableOpReadVariableOp.gene_decoder_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
gene_decoder_2/BiasAddBiasAddgene_decoder_2/MatMul:product:0-gene_decoder_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
gene_decoder_2/SigmoidSigmoidgene_decoder_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
0BatchNormProteinDecode2/batchnorm/ReadVariableOpReadVariableOp9batchnormproteindecode2_batchnorm_readvariableop_resource*
_output_shapes
:/*
dtype0l
'BatchNormProteinDecode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
%BatchNormProteinDecode2/batchnorm/addAddV28BatchNormProteinDecode2/batchnorm/ReadVariableOp:value:00BatchNormProteinDecode2/batchnorm/add/y:output:0*
T0*
_output_shapes
:/�
'BatchNormProteinDecode2/batchnorm/RsqrtRsqrt)BatchNormProteinDecode2/batchnorm/add:z:0*
T0*
_output_shapes
:/�
4BatchNormProteinDecode2/batchnorm/mul/ReadVariableOpReadVariableOp=batchnormproteindecode2_batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0�
%BatchNormProteinDecode2/batchnorm/mulMul+BatchNormProteinDecode2/batchnorm/Rsqrt:y:0<BatchNormProteinDecode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/�
'BatchNormProteinDecode2/batchnorm/mul_1Mulprotein_decoder_2/Sigmoid:y:0)BatchNormProteinDecode2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������/�
2BatchNormProteinDecode2/batchnorm/ReadVariableOp_1ReadVariableOp;batchnormproteindecode2_batchnorm_readvariableop_1_resource*
_output_shapes
:/*
dtype0�
'BatchNormProteinDecode2/batchnorm/mul_2Mul:BatchNormProteinDecode2/batchnorm/ReadVariableOp_1:value:0)BatchNormProteinDecode2/batchnorm/mul:z:0*
T0*
_output_shapes
:/�
2BatchNormProteinDecode2/batchnorm/ReadVariableOp_2ReadVariableOp;batchnormproteindecode2_batchnorm_readvariableop_2_resource*
_output_shapes
:/*
dtype0�
%BatchNormProteinDecode2/batchnorm/subSub:BatchNormProteinDecode2/batchnorm/ReadVariableOp_2:value:0+BatchNormProteinDecode2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:/�
'BatchNormProteinDecode2/batchnorm/add_1AddV2+BatchNormProteinDecode2/batchnorm/mul_1:z:0)BatchNormProteinDecode2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������/�
-BatchNormGeneDecode2/batchnorm/ReadVariableOpReadVariableOp6batchnormgenedecode2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
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
:�{
$BatchNormGeneDecode2/batchnorm/RsqrtRsqrt&BatchNormGeneDecode2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
1BatchNormGeneDecode2/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgenedecode2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"BatchNormGeneDecode2/batchnorm/mulMul(BatchNormGeneDecode2/batchnorm/Rsqrt:y:09BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
$BatchNormGeneDecode2/batchnorm/mul_1Mulgene_decoder_2/Sigmoid:y:0&BatchNormGeneDecode2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
/BatchNormGeneDecode2/batchnorm/ReadVariableOp_1ReadVariableOp8batchnormgenedecode2_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
$BatchNormGeneDecode2/batchnorm/mul_2Mul7BatchNormGeneDecode2/batchnorm/ReadVariableOp_1:value:0&BatchNormGeneDecode2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
/BatchNormGeneDecode2/batchnorm/ReadVariableOp_2ReadVariableOp8batchnormgenedecode2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
"BatchNormGeneDecode2/batchnorm/subSub7BatchNormGeneDecode2/batchnorm/ReadVariableOp_2:value:0(BatchNormGeneDecode2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
$BatchNormGeneDecode2/batchnorm/add_1AddV2(BatchNormGeneDecode2/batchnorm/mul_1:z:0&BatchNormGeneDecode2/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
*protein_decoder_last/MatMul/ReadVariableOpReadVariableOp3protein_decoder_last_matmul_readvariableop_resource*
_output_shapes
:	/�*
dtype0�
protein_decoder_last/MatMulMatMul+BatchNormProteinDecode2/batchnorm/add_1:z:02protein_decoder_last/MatMul/ReadVariableOp:value:0*
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
��
*
dtype0�
gene_decoder_last/MatMulMatMul(BatchNormGeneDecode2/batchnorm/add_1:z:0/gene_decoder_last/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
�
(gene_decoder_last/BiasAdd/ReadVariableOpReadVariableOp1gene_decoder_last_biasadd_readvariableop_resource*
_output_shapes	
:�
*
dtype0�
gene_decoder_last/BiasAddBiasAdd"gene_decoder_last/MatMul:product:00gene_decoder_last/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
{
gene_decoder_last/SigmoidSigmoid"gene_decoder_last/BiasAdd:output:0*
T0*(
_output_shapes
:����������
m
IdentityIdentitygene_decoder_last/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������
r

Identity_1Identity protein_decoder_last/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp.^BatchNormGeneDecode1/batchnorm/ReadVariableOp0^BatchNormGeneDecode1/batchnorm/ReadVariableOp_10^BatchNormGeneDecode1/batchnorm/ReadVariableOp_22^BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp.^BatchNormGeneDecode2/batchnorm/ReadVariableOp0^BatchNormGeneDecode2/batchnorm/ReadVariableOp_10^BatchNormGeneDecode2/batchnorm/ReadVariableOp_22^BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp.^BatchNormGeneEncode1/batchnorm/ReadVariableOp0^BatchNormGeneEncode1/batchnorm/ReadVariableOp_10^BatchNormGeneEncode1/batchnorm/ReadVariableOp_22^BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp.^BatchNormGeneEncode2/batchnorm/ReadVariableOp0^BatchNormGeneEncode2/batchnorm/ReadVariableOp_10^BatchNormGeneEncode2/batchnorm/ReadVariableOp_22^BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp1^BatchNormProteinDecode1/batchnorm/ReadVariableOp3^BatchNormProteinDecode1/batchnorm/ReadVariableOp_13^BatchNormProteinDecode1/batchnorm/ReadVariableOp_25^BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp1^BatchNormProteinDecode2/batchnorm/ReadVariableOp3^BatchNormProteinDecode2/batchnorm/ReadVariableOp_13^BatchNormProteinDecode2/batchnorm/ReadVariableOp_25^BatchNormProteinDecode2/batchnorm/mul/ReadVariableOp1^BatchNormProteinEncode1/batchnorm/ReadVariableOp3^BatchNormProteinEncode1/batchnorm/ReadVariableOp_13^BatchNormProteinEncode1/batchnorm/ReadVariableOp_25^BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp1^BatchNormProteinEncode2/batchnorm/ReadVariableOp3^BatchNormProteinEncode2/batchnorm/ReadVariableOp_13^BatchNormProteinEncode2/batchnorm/ReadVariableOp_25^BatchNormProteinEncode2/batchnorm/mul/ReadVariableOp)^EmbeddingDimDense/BiasAdd/ReadVariableOp(^EmbeddingDimDense/MatMul/ReadVariableOp&^gene_decoder_1/BiasAdd/ReadVariableOp%^gene_decoder_1/MatMul/ReadVariableOp&^gene_decoder_2/BiasAdd/ReadVariableOp%^gene_decoder_2/MatMul/ReadVariableOp)^gene_decoder_last/BiasAdd/ReadVariableOp(^gene_decoder_last/MatMul/ReadVariableOp&^gene_encoder_1/BiasAdd/ReadVariableOp%^gene_encoder_1/MatMul/ReadVariableOp&^gene_encoder_2/BiasAdd/ReadVariableOp%^gene_encoder_2/MatMul/ReadVariableOp)^protein_decoder_1/BiasAdd/ReadVariableOp(^protein_decoder_1/MatMul/ReadVariableOp)^protein_decoder_2/BiasAdd/ReadVariableOp(^protein_decoder_2/MatMul/ReadVariableOp,^protein_decoder_last/BiasAdd/ReadVariableOp+^protein_decoder_last/MatMul/ReadVariableOp)^protein_encoder_1/BiasAdd/ReadVariableOp(^protein_encoder_1/MatMul/ReadVariableOp)^protein_encoder_2/BiasAdd/ReadVariableOp(^protein_encoder_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������
:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
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
0BatchNormProteinDecode2/batchnorm/ReadVariableOp0BatchNormProteinDecode2/batchnorm/ReadVariableOp2h
2BatchNormProteinDecode2/batchnorm/ReadVariableOp_12BatchNormProteinDecode2/batchnorm/ReadVariableOp_12h
2BatchNormProteinDecode2/batchnorm/ReadVariableOp_22BatchNormProteinDecode2/batchnorm/ReadVariableOp_22l
4BatchNormProteinDecode2/batchnorm/mul/ReadVariableOp4BatchNormProteinDecode2/batchnorm/mul/ReadVariableOp2d
0BatchNormProteinEncode1/batchnorm/ReadVariableOp0BatchNormProteinEncode1/batchnorm/ReadVariableOp2h
2BatchNormProteinEncode1/batchnorm/ReadVariableOp_12BatchNormProteinEncode1/batchnorm/ReadVariableOp_12h
2BatchNormProteinEncode1/batchnorm/ReadVariableOp_22BatchNormProteinEncode1/batchnorm/ReadVariableOp_22l
4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp2d
0BatchNormProteinEncode2/batchnorm/ReadVariableOp0BatchNormProteinEncode2/batchnorm/ReadVariableOp2h
2BatchNormProteinEncode2/batchnorm/ReadVariableOp_12BatchNormProteinEncode2/batchnorm/ReadVariableOp_12h
2BatchNormProteinEncode2/batchnorm/ReadVariableOp_22BatchNormProteinEncode2/batchnorm/ReadVariableOp_22l
4BatchNormProteinEncode2/batchnorm/mul/ReadVariableOp4BatchNormProteinEncode2/batchnorm/mul/ReadVariableOp2T
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
'protein_decoder_1/MatMul/ReadVariableOp'protein_decoder_1/MatMul/ReadVariableOp2T
(protein_decoder_2/BiasAdd/ReadVariableOp(protein_decoder_2/BiasAdd/ReadVariableOp2R
'protein_decoder_2/MatMul/ReadVariableOp'protein_decoder_2/MatMul/ReadVariableOp2Z
+protein_decoder_last/BiasAdd/ReadVariableOp+protein_decoder_last/BiasAdd/ReadVariableOp2X
*protein_decoder_last/MatMul/ReadVariableOp*protein_decoder_last/MatMul/ReadVariableOp2T
(protein_encoder_1/BiasAdd/ReadVariableOp(protein_encoder_1/BiasAdd/ReadVariableOp2R
'protein_encoder_1/MatMul/ReadVariableOp'protein_encoder_1/MatMul/ReadVariableOp2T
(protein_encoder_2/BiasAdd/ReadVariableOp(protein_encoder_2/BiasAdd/ReadVariableOp2R
'protein_encoder_2/MatMul/ReadVariableOp'protein_encoder_2/MatMul/ReadVariableOp:R N
(
_output_shapes
:����������

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
S__inference_BatchNormProteinDecode2_layer_call_and_return_conditional_losses_328947

inputs5
'assignmovingavg_readvariableop_resource:/7
)assignmovingavg_1_readvariableop_resource:/3
%batchnorm_mul_readvariableop_resource://
!batchnorm_readvariableop_resource:/
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:/�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������/l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:/*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:/*
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
:/*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:/x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:/�
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
:/*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:/~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:/�
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
:/P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:/~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������/h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:/v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:/*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:/r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������/b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������/�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������/: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������/
 
_user_specified_nameinputs
��
�
C__inference_model_8_layer_call_and_return_conditional_losses_330217
gene_input_layer
protein_input_layer+
protein_encoder_1_330087:	�/&
protein_encoder_1_330089:/)
gene_encoder_1_330092:
�
�$
gene_encoder_1_330094:	�,
batchnormproteinencode1_330097:/,
batchnormproteinencode1_330099:/,
batchnormproteinencode1_330101:/,
batchnormproteinencode1_330103:/*
batchnormgeneencode1_330106:	�*
batchnormgeneencode1_330108:	�*
batchnormgeneencode1_330110:	�*
batchnormgeneencode1_330112:	�*
protein_encoder_2_330115:/&
protein_encoder_2_330117:(
gene_encoder_2_330120:	�P#
gene_encoder_2_330122:P)
batchnormgeneencode2_330125:P)
batchnormgeneencode2_330127:P)
batchnormgeneencode2_330129:P)
batchnormgeneencode2_330131:P,
batchnormproteinencode2_330134:,
batchnormproteinencode2_330136:,
batchnormproteinencode2_330138:,
batchnormproteinencode2_330140:*
embeddingdimdense_330144:[@&
embeddingdimdense_330146:@*
protein_decoder_1_330149:@&
protein_decoder_1_330151:'
gene_decoder_1_330154:@P#
gene_decoder_1_330156:P,
batchnormproteindecode1_330159:,
batchnormproteindecode1_330161:,
batchnormproteindecode1_330163:,
batchnormproteindecode1_330165:)
batchnormgenedecode1_330168:P)
batchnormgenedecode1_330170:P)
batchnormgenedecode1_330172:P)
batchnormgenedecode1_330174:P*
protein_decoder_2_330177:/&
protein_decoder_2_330179:/(
gene_decoder_2_330182:	P�$
gene_decoder_2_330184:	�,
batchnormproteindecode2_330187:/,
batchnormproteindecode2_330189:/,
batchnormproteindecode2_330191:/,
batchnormproteindecode2_330193:/*
batchnormgenedecode2_330196:	�*
batchnormgenedecode2_330198:	�*
batchnormgenedecode2_330200:	�*
batchnormgenedecode2_330202:	�.
protein_decoder_last_330205:	/�*
protein_decoder_last_330207:	�,
gene_decoder_last_330210:
��
'
gene_decoder_last_330212:	�

identity

identity_1��,BatchNormGeneDecode1/StatefulPartitionedCall�,BatchNormGeneDecode2/StatefulPartitionedCall�,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�/BatchNormProteinDecode1/StatefulPartitionedCall�/BatchNormProteinDecode2/StatefulPartitionedCall�/BatchNormProteinEncode1/StatefulPartitionedCall�/BatchNormProteinEncode2/StatefulPartitionedCall�)EmbeddingDimDense/StatefulPartitionedCall�&gene_decoder_1/StatefulPartitionedCall�&gene_decoder_2/StatefulPartitionedCall�)gene_decoder_last/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�)protein_decoder_1/StatefulPartitionedCall�)protein_decoder_2/StatefulPartitionedCall�,protein_decoder_last/StatefulPartitionedCall�)protein_encoder_1/StatefulPartitionedCall�)protein_encoder_2/StatefulPartitionedCall�
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallprotein_input_layerprotein_encoder_1_330087protein_encoder_1_330089*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_328978�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallgene_input_layergene_encoder_1_330092gene_encoder_1_330094*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_328995�
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_330097batchnormproteinencode1_330099batchnormproteinencode1_330101batchnormproteinencode1_330103*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_328455�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_330106batchnormgeneencode1_330108batchnormgeneencode1_330110batchnormgeneencode1_330112*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_328373�
)protein_encoder_2/StatefulPartitionedCallStatefulPartitionedCall8BatchNormProteinEncode1/StatefulPartitionedCall:output:0protein_encoder_2_330115protein_encoder_2_330117*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_encoder_2_layer_call_and_return_conditional_losses_329030�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_330120gene_encoder_2_330122*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_329047�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_330125batchnormgeneencode2_330127batchnormgeneencode2_330129batchnormgeneencode2_330131*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_328537�
/BatchNormProteinEncode2/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_2/StatefulPartitionedCall:output:0batchnormproteinencode2_330134batchnormproteinencode2_330136batchnormproteinencode2_330138batchnormproteinencode2_330140*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinEncode2_layer_call_and_return_conditional_losses_328619�
 ConcatenateLayer/PartitionedCallPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:08BatchNormProteinEncode2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������[* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_329078�
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_330144embeddingdimdense_330146*
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
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_329091�
)protein_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0protein_decoder_1_330149protein_decoder_1_330151*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_decoder_1_layer_call_and_return_conditional_losses_329108�
&gene_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0gene_decoder_1_330154gene_decoder_1_330156*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_decoder_1_layer_call_and_return_conditional_losses_329125�
/BatchNormProteinDecode1/StatefulPartitionedCallStatefulPartitionedCall2protein_decoder_1/StatefulPartitionedCall:output:0batchnormproteindecode1_330159batchnormproteindecode1_330161batchnormproteindecode1_330163batchnormproteindecode1_330165*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_328783�
,BatchNormGeneDecode1/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_1/StatefulPartitionedCall:output:0batchnormgenedecode1_330168batchnormgenedecode1_330170batchnormgenedecode1_330172batchnormgenedecode1_330174*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_328701�
)protein_decoder_2/StatefulPartitionedCallStatefulPartitionedCall8BatchNormProteinDecode1/StatefulPartitionedCall:output:0protein_decoder_2_330177protein_decoder_2_330179*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_decoder_2_layer_call_and_return_conditional_losses_329160�
&gene_decoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode1/StatefulPartitionedCall:output:0gene_decoder_2_330182gene_decoder_2_330184*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_decoder_2_layer_call_and_return_conditional_losses_329177�
/BatchNormProteinDecode2/StatefulPartitionedCallStatefulPartitionedCall2protein_decoder_2/StatefulPartitionedCall:output:0batchnormproteindecode2_330187batchnormproteindecode2_330189batchnormproteindecode2_330191batchnormproteindecode2_330193*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinDecode2_layer_call_and_return_conditional_losses_328947�
,BatchNormGeneDecode2/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_2/StatefulPartitionedCall:output:0batchnormgenedecode2_330196batchnormgenedecode2_330198batchnormgenedecode2_330200batchnormgenedecode2_330202*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_328865�
,protein_decoder_last/StatefulPartitionedCallStatefulPartitionedCall8BatchNormProteinDecode2/StatefulPartitionedCall:output:0protein_decoder_last_330205protein_decoder_last_330207*
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
P__inference_protein_decoder_last_layer_call_and_return_conditional_losses_329212�
)gene_decoder_last/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode2/StatefulPartitionedCall:output:0gene_decoder_last_330210gene_decoder_last_330212*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_gene_decoder_last_layer_call_and_return_conditional_losses_329229�
IdentityIdentity2gene_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������
�

Identity_1Identity5protein_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp-^BatchNormGeneDecode1/StatefulPartitionedCall-^BatchNormGeneDecode2/StatefulPartitionedCall-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinDecode1/StatefulPartitionedCall0^BatchNormProteinDecode2/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall0^BatchNormProteinEncode2/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_decoder_1/StatefulPartitionedCall'^gene_decoder_2/StatefulPartitionedCall*^gene_decoder_last/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_decoder_1/StatefulPartitionedCall*^protein_decoder_2/StatefulPartitionedCall-^protein_decoder_last/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*^protein_encoder_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������
:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneDecode1/StatefulPartitionedCall,BatchNormGeneDecode1/StatefulPartitionedCall2\
,BatchNormGeneDecode2/StatefulPartitionedCall,BatchNormGeneDecode2/StatefulPartitionedCall2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2b
/BatchNormProteinDecode1/StatefulPartitionedCall/BatchNormProteinDecode1/StatefulPartitionedCall2b
/BatchNormProteinDecode2/StatefulPartitionedCall/BatchNormProteinDecode2/StatefulPartitionedCall2b
/BatchNormProteinEncode1/StatefulPartitionedCall/BatchNormProteinEncode1/StatefulPartitionedCall2b
/BatchNormProteinEncode2/StatefulPartitionedCall/BatchNormProteinEncode2/StatefulPartitionedCall2V
)EmbeddingDimDense/StatefulPartitionedCall)EmbeddingDimDense/StatefulPartitionedCall2P
&gene_decoder_1/StatefulPartitionedCall&gene_decoder_1/StatefulPartitionedCall2P
&gene_decoder_2/StatefulPartitionedCall&gene_decoder_2/StatefulPartitionedCall2V
)gene_decoder_last/StatefulPartitionedCall)gene_decoder_last/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall2V
)protein_decoder_1/StatefulPartitionedCall)protein_decoder_1/StatefulPartitionedCall2V
)protein_decoder_2/StatefulPartitionedCall)protein_decoder_2/StatefulPartitionedCall2\
,protein_decoder_last/StatefulPartitionedCall,protein_decoder_last/StatefulPartitionedCall2V
)protein_encoder_1/StatefulPartitionedCall)protein_encoder_1/StatefulPartitionedCall2V
)protein_encoder_2/StatefulPartitionedCall)protein_encoder_2/StatefulPartitionedCall:Z V
(
_output_shapes
:����������

*
_user_specified_nameGene_Input_Layer:]Y
(
_output_shapes
:����������
-
_user_specified_nameProtein_Input_Layer
�
�
$__inference_signature_wrapper_331111
gene_input_layer
protein_input_layer
unknown:	�/
	unknown_0:/
	unknown_1:
�
�
	unknown_2:	�
	unknown_3:/
	unknown_4:/
	unknown_5:/
	unknown_6:/
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:/

unknown_12:

unknown_13:	�P

unknown_14:P

unknown_15:P

unknown_16:P

unknown_17:P

unknown_18:P

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:[@

unknown_24:@

unknown_25:@

unknown_26:

unknown_27:@P

unknown_28:P

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:P

unknown_34:P

unknown_35:P

unknown_36:P

unknown_37:/

unknown_38:/

unknown_39:	P�

unknown_40:	�

unknown_41:/

unknown_42:/

unknown_43:/

unknown_44:/

unknown_45:	�

unknown_46:	�

unknown_47:	�

unknown_48:	�

unknown_49:	/�

unknown_50:	�

unknown_51:
��


unknown_52:	�

identity

identity_1��StatefulPartitionedCall�
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
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:����������
:����������*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./01234567*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_328302p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������
r

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
_input_shapes�
�:����������
:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:����������

*
_user_specified_nameGene_Input_Layer:]Y
(
_output_shapes
:����������
-
_user_specified_nameProtein_Input_Layer
�

�
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_328978

inputs1
matmul_readvariableop_resource:	�/-
biasadd_readvariableop_resource:/
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�/*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������/r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������/V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������/Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������/w
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
�
�
C__inference_model_8_layer_call_and_return_conditional_losses_329237

inputs
inputs_1+
protein_encoder_1_328979:	�/&
protein_encoder_1_328981:/)
gene_encoder_1_328996:
�
�$
gene_encoder_1_328998:	�,
batchnormproteinencode1_329001:/,
batchnormproteinencode1_329003:/,
batchnormproteinencode1_329005:/,
batchnormproteinencode1_329007:/*
batchnormgeneencode1_329010:	�*
batchnormgeneencode1_329012:	�*
batchnormgeneencode1_329014:	�*
batchnormgeneencode1_329016:	�*
protein_encoder_2_329031:/&
protein_encoder_2_329033:(
gene_encoder_2_329048:	�P#
gene_encoder_2_329050:P)
batchnormgeneencode2_329053:P)
batchnormgeneencode2_329055:P)
batchnormgeneencode2_329057:P)
batchnormgeneencode2_329059:P,
batchnormproteinencode2_329062:,
batchnormproteinencode2_329064:,
batchnormproteinencode2_329066:,
batchnormproteinencode2_329068:*
embeddingdimdense_329092:[@&
embeddingdimdense_329094:@*
protein_decoder_1_329109:@&
protein_decoder_1_329111:'
gene_decoder_1_329126:@P#
gene_decoder_1_329128:P,
batchnormproteindecode1_329131:,
batchnormproteindecode1_329133:,
batchnormproteindecode1_329135:,
batchnormproteindecode1_329137:)
batchnormgenedecode1_329140:P)
batchnormgenedecode1_329142:P)
batchnormgenedecode1_329144:P)
batchnormgenedecode1_329146:P*
protein_decoder_2_329161:/&
protein_decoder_2_329163:/(
gene_decoder_2_329178:	P�$
gene_decoder_2_329180:	�,
batchnormproteindecode2_329183:/,
batchnormproteindecode2_329185:/,
batchnormproteindecode2_329187:/,
batchnormproteindecode2_329189:/*
batchnormgenedecode2_329192:	�*
batchnormgenedecode2_329194:	�*
batchnormgenedecode2_329196:	�*
batchnormgenedecode2_329198:	�.
protein_decoder_last_329213:	/�*
protein_decoder_last_329215:	�,
gene_decoder_last_329230:
��
'
gene_decoder_last_329232:	�

identity

identity_1��,BatchNormGeneDecode1/StatefulPartitionedCall�,BatchNormGeneDecode2/StatefulPartitionedCall�,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�/BatchNormProteinDecode1/StatefulPartitionedCall�/BatchNormProteinDecode2/StatefulPartitionedCall�/BatchNormProteinEncode1/StatefulPartitionedCall�/BatchNormProteinEncode2/StatefulPartitionedCall�)EmbeddingDimDense/StatefulPartitionedCall�&gene_decoder_1/StatefulPartitionedCall�&gene_decoder_2/StatefulPartitionedCall�)gene_decoder_last/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�)protein_decoder_1/StatefulPartitionedCall�)protein_decoder_2/StatefulPartitionedCall�,protein_decoder_last/StatefulPartitionedCall�)protein_encoder_1/StatefulPartitionedCall�)protein_encoder_2/StatefulPartitionedCall�
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1protein_encoder_1_328979protein_encoder_1_328981*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_328978�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputsgene_encoder_1_328996gene_encoder_1_328998*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_328995�
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_329001batchnormproteinencode1_329003batchnormproteinencode1_329005batchnormproteinencode1_329007*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_328408�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_329010batchnormgeneencode1_329012batchnormgeneencode1_329014batchnormgeneencode1_329016*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_328326�
)protein_encoder_2/StatefulPartitionedCallStatefulPartitionedCall8BatchNormProteinEncode1/StatefulPartitionedCall:output:0protein_encoder_2_329031protein_encoder_2_329033*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_encoder_2_layer_call_and_return_conditional_losses_329030�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_329048gene_encoder_2_329050*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_329047�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_329053batchnormgeneencode2_329055batchnormgeneencode2_329057batchnormgeneencode2_329059*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_328490�
/BatchNormProteinEncode2/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_2/StatefulPartitionedCall:output:0batchnormproteinencode2_329062batchnormproteinencode2_329064batchnormproteinencode2_329066batchnormproteinencode2_329068*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinEncode2_layer_call_and_return_conditional_losses_328572�
 ConcatenateLayer/PartitionedCallPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:08BatchNormProteinEncode2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������[* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_329078�
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_329092embeddingdimdense_329094*
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
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_329091�
)protein_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0protein_decoder_1_329109protein_decoder_1_329111*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_decoder_1_layer_call_and_return_conditional_losses_329108�
&gene_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0gene_decoder_1_329126gene_decoder_1_329128*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_decoder_1_layer_call_and_return_conditional_losses_329125�
/BatchNormProteinDecode1/StatefulPartitionedCallStatefulPartitionedCall2protein_decoder_1/StatefulPartitionedCall:output:0batchnormproteindecode1_329131batchnormproteindecode1_329133batchnormproteindecode1_329135batchnormproteindecode1_329137*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_328736�
,BatchNormGeneDecode1/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_1/StatefulPartitionedCall:output:0batchnormgenedecode1_329140batchnormgenedecode1_329142batchnormgenedecode1_329144batchnormgenedecode1_329146*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_328654�
)protein_decoder_2/StatefulPartitionedCallStatefulPartitionedCall8BatchNormProteinDecode1/StatefulPartitionedCall:output:0protein_decoder_2_329161protein_decoder_2_329163*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_decoder_2_layer_call_and_return_conditional_losses_329160�
&gene_decoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode1/StatefulPartitionedCall:output:0gene_decoder_2_329178gene_decoder_2_329180*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_decoder_2_layer_call_and_return_conditional_losses_329177�
/BatchNormProteinDecode2/StatefulPartitionedCallStatefulPartitionedCall2protein_decoder_2/StatefulPartitionedCall:output:0batchnormproteindecode2_329183batchnormproteindecode2_329185batchnormproteindecode2_329187batchnormproteindecode2_329189*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinDecode2_layer_call_and_return_conditional_losses_328900�
,BatchNormGeneDecode2/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_2/StatefulPartitionedCall:output:0batchnormgenedecode2_329192batchnormgenedecode2_329194batchnormgenedecode2_329196batchnormgenedecode2_329198*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_328818�
,protein_decoder_last/StatefulPartitionedCallStatefulPartitionedCall8BatchNormProteinDecode2/StatefulPartitionedCall:output:0protein_decoder_last_329213protein_decoder_last_329215*
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
P__inference_protein_decoder_last_layer_call_and_return_conditional_losses_329212�
)gene_decoder_last/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode2/StatefulPartitionedCall:output:0gene_decoder_last_329230gene_decoder_last_329232*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_gene_decoder_last_layer_call_and_return_conditional_losses_329229�
IdentityIdentity2gene_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������
�

Identity_1Identity5protein_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp-^BatchNormGeneDecode1/StatefulPartitionedCall-^BatchNormGeneDecode2/StatefulPartitionedCall-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinDecode1/StatefulPartitionedCall0^BatchNormProteinDecode2/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall0^BatchNormProteinEncode2/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_decoder_1/StatefulPartitionedCall'^gene_decoder_2/StatefulPartitionedCall*^gene_decoder_last/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_decoder_1/StatefulPartitionedCall*^protein_decoder_2/StatefulPartitionedCall-^protein_decoder_last/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*^protein_encoder_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:����������
:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneDecode1/StatefulPartitionedCall,BatchNormGeneDecode1/StatefulPartitionedCall2\
,BatchNormGeneDecode2/StatefulPartitionedCall,BatchNormGeneDecode2/StatefulPartitionedCall2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2b
/BatchNormProteinDecode1/StatefulPartitionedCall/BatchNormProteinDecode1/StatefulPartitionedCall2b
/BatchNormProteinDecode2/StatefulPartitionedCall/BatchNormProteinDecode2/StatefulPartitionedCall2b
/BatchNormProteinEncode1/StatefulPartitionedCall/BatchNormProteinEncode1/StatefulPartitionedCall2b
/BatchNormProteinEncode2/StatefulPartitionedCall/BatchNormProteinEncode2/StatefulPartitionedCall2V
)EmbeddingDimDense/StatefulPartitionedCall)EmbeddingDimDense/StatefulPartitionedCall2P
&gene_decoder_1/StatefulPartitionedCall&gene_decoder_1/StatefulPartitionedCall2P
&gene_decoder_2/StatefulPartitionedCall&gene_decoder_2/StatefulPartitionedCall2V
)gene_decoder_last/StatefulPartitionedCall)gene_decoder_last/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall2V
)protein_decoder_1/StatefulPartitionedCall)protein_decoder_1/StatefulPartitionedCall2V
)protein_decoder_2/StatefulPartitionedCall)protein_decoder_2/StatefulPartitionedCall2\
,protein_decoder_last/StatefulPartitionedCall,protein_decoder_last/StatefulPartitionedCall2V
)protein_encoder_1/StatefulPartitionedCall)protein_encoder_1/StatefulPartitionedCall2V
)protein_encoder_2/StatefulPartitionedCall)protein_encoder_2/StatefulPartitionedCall:P L
(
_output_shapes
:����������

 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_328490

inputs/
!batchnorm_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P1
#batchnorm_readvariableop_1_resource:P1
#batchnorm_readvariableop_2_resource:P
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
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
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Pz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�%
�
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_328455

inputs5
'assignmovingavg_readvariableop_resource:/7
)assignmovingavg_1_readvariableop_resource:/3
%batchnorm_mul_readvariableop_resource://
!batchnorm_readvariableop_resource:/
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:/�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������/l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:/*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:/*
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
:/*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:/x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:/�
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
:/*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:/~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:/�
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
:/P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:/~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������/h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:/v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:/*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:/r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������/b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������/�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������/: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������/
 
_user_specified_nameinputs
�

�
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_328995

inputs2
matmul_readvariableop_resource:
�
�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�
�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������

 
_user_specified_nameinputs
�

�
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_331331

inputs1
matmul_readvariableop_resource:	�P-
biasadd_readvariableop_resource:P
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������PV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������PZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������Pw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
5__inference_BatchNormGeneDecode1_layer_call_fn_331597

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_328654o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
(__inference_model_8_layer_call_fn_330339
inputs_0
inputs_1
unknown:	�/
	unknown_0:/
	unknown_1:
�
�
	unknown_2:	�
	unknown_3:/
	unknown_4:/
	unknown_5:/
	unknown_6:/
	unknown_7:	�
	unknown_8:	�
	unknown_9:	�

unknown_10:	�

unknown_11:/

unknown_12:

unknown_13:	�P

unknown_14:P

unknown_15:P

unknown_16:P

unknown_17:P

unknown_18:P

unknown_19:

unknown_20:

unknown_21:

unknown_22:

unknown_23:[@

unknown_24:@

unknown_25:@

unknown_26:

unknown_27:@P

unknown_28:P

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:P

unknown_34:P

unknown_35:P

unknown_36:P

unknown_37:/

unknown_38:/

unknown_39:	P�

unknown_40:	�

unknown_41:/

unknown_42:/

unknown_43:/

unknown_44:/

unknown_45:	�

unknown_46:	�

unknown_47:	�

unknown_48:	�

unknown_49:	/�

unknown_50:	�

unknown_51:
��


unknown_52:	�

identity

identity_1��StatefulPartitionedCall�
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
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50
unknown_51
unknown_52*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *<
_output_shapes*
(:����������
:����������*X
_read_only_resource_inputs:
86	
 !"#$%&'()*+,-./01234567*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_8_layer_call_and_return_conditional_losses_329237p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������
r

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
_input_shapes�
�:����������
:����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:����������

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
8__inference_BatchNormProteinEncode2_layer_call_fn_331444

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinEncode2_layer_call_and_return_conditional_losses_328572o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
8__inference_BatchNormProteinDecode1_layer_call_fn_331690

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_328783o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_331431

inputs5
'assignmovingavg_readvariableop_resource:P7
)assignmovingavg_1_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P/
!batchnorm_readvariableop_resource:P
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:P�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������Pl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:P*
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
:P*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Px
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P�
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
:P*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:P~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P�
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
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Ph
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
/__inference_gene_encoder_1_layer_call_fn_331120

inputs
unknown:
�
�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_328995p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������
: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������

 
_user_specified_nameinputs
�

�
M__inference_protein_decoder_1_layer_call_and_return_conditional_losses_329108

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
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
5__inference_BatchNormGeneDecode2_layer_call_fn_331797

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_328818p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�%
�
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_328701

inputs5
'assignmovingavg_readvariableop_resource:P7
)assignmovingavg_1_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P/
!batchnorm_readvariableop_resource:P
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:P�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������Pl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:P*
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
:P*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Px
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P�
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
:P*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:P~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P�
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
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Ph
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
]
1__inference_ConcatenateLayer_layer_call_fn_331517
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������[* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_329078`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������["
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������P:���������:Q M
'
_output_shapes
:���������P
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�
�
S__inference_BatchNormProteinEncode2_layer_call_and_return_conditional_losses_328572

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_331151

inputs1
matmul_readvariableop_resource:	�/-
biasadd_readvariableop_resource:/
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�/*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������/r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������/V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������/Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������/w
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
�
�
8__inference_BatchNormProteinDecode2_layer_call_fn_331890

inputs
unknown:/
	unknown_0:/
	unknown_1:/
	unknown_2:/
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinDecode2_layer_call_and_return_conditional_losses_328947o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������/`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������/: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������/
 
_user_specified_nameinputs
�
�
8__inference_BatchNormProteinEncode2_layer_call_fn_331457

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinEncode2_layer_call_and_return_conditional_losses_328619o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_331744

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
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
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
5__inference_BatchNormGeneEncode2_layer_call_fn_331377

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_328537o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
5__inference_BatchNormGeneDecode2_layer_call_fn_331810

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_328865p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�%
�
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_331864

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
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
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
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
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
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
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
J__inference_gene_decoder_2_layer_call_and_return_conditional_losses_331764

inputs1
matmul_readvariableop_resource:	P�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	P�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�%
�
S__inference_BatchNormProteinDecode2_layer_call_and_return_conditional_losses_331944

inputs5
'assignmovingavg_readvariableop_resource:/7
)assignmovingavg_1_readvariableop_resource:/3
%batchnorm_mul_readvariableop_resource://
!batchnorm_readvariableop_resource:/
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:/�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������/l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:/*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:/*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:/*
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
:/*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:/x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:/�
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
:/*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:/~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:/�
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
:/P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:/~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������/h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:/v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:/*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:/r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������/b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������/�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������/: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������/
 
_user_specified_nameinputs
�
�
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_331830

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
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
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_328736

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
M__inference_protein_decoder_2_layer_call_and_return_conditional_losses_329160

inputs0
matmul_readvariableop_resource:/-
biasadd_readvariableop_resource:/
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������/r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������/V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������/Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������/w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_331397

inputs/
!batchnorm_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P1
#batchnorm_readvariableop_1_resource:P1
#batchnorm_readvariableop_2_resource:P
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
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
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Pz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_331630

inputs/
!batchnorm_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P1
#batchnorm_readvariableop_1_resource:P1
#batchnorm_readvariableop_2_resource:P
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
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
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Pz
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:P*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pz
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:P*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
2__inference_protein_encoder_1_layer_call_fn_331140

inputs
unknown:	�/
	unknown_0:/
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_328978o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������/`
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
�
�
5__inference_BatchNormGeneEncode1_layer_call_fn_331164

inputs
unknown:	�
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_328326p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
5__inference_protein_decoder_last_layer_call_fn_331973

inputs
unknown:	/�
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
P__inference_protein_decoder_last_layer_call_and_return_conditional_losses_329212p
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
:���������/: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������/
 
_user_specified_nameinputs
�%
�
S__inference_BatchNormProteinEncode2_layer_call_and_return_conditional_losses_328619

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
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
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
M__inference_protein_encoder_2_layer_call_and_return_conditional_losses_329030

inputs0
matmul_readvariableop_resource:/-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������/
 
_user_specified_nameinputs
�
�
2__inference_gene_decoder_last_layer_call_fn_331953

inputs
unknown:
��

	unknown_0:	�

identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_gene_decoder_last_layer_call_and_return_conditional_losses_329229p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
J__inference_gene_decoder_1_layer_call_and_return_conditional_losses_331564

inputs0
matmul_readvariableop_resource:@P-
biasadd_readvariableop_resource:P
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@P*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Pr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������PV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������PZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������Pw
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
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_331664

inputs5
'assignmovingavg_readvariableop_resource:P7
)assignmovingavg_1_readvariableop_resource:P3
%batchnorm_mul_readvariableop_resource:P/
!batchnorm_readvariableop_resource:P
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:P�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������Pl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:P*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:P*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:P*
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
:P*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:Px
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:P�
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
:P*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:P~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:P�
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
:PP
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:P~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:P*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:Pc
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������Ph
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:Pv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:P*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:Pr
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������Pb
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������P�
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
/__inference_gene_encoder_2_layer_call_fn_331320

inputs
unknown:	�P
	unknown_0:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_329047o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
M__inference_gene_decoder_last_layer_call_and_return_conditional_losses_329229

inputs2
matmul_readvariableop_resource:
��
.
biasadd_readvariableop_resource:	�

identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�
*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
W
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:����������
[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�%
�
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_328373

inputs6
'assignmovingavg_readvariableop_resource:	�8
)assignmovingavg_1_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�0
!batchnorm_readvariableop_resource:	�
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	��
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:����������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	�*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:�*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:�*
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
:�*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:�y
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:��
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
:�*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:��
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
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������i
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�w
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
5__inference_BatchNormGeneEncode2_layer_call_fn_331364

inputs
unknown:P
	unknown_0:P
	unknown_1:P
	unknown_2:P
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_328490o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������P`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������P: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_328818

inputs0
!batchnorm_readvariableop_resource:	�4
%batchnorm_mul_readvariableop_resource:	�2
#batchnorm_readvariableop_1_resource:	�2
#batchnorm_readvariableop_2_resource:	�
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:�*
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
:�Q
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:�
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:�d
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:����������{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:�{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:�s
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:����������c
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:�����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
8__inference_BatchNormProteinEncode1_layer_call_fn_331244

inputs
unknown:/
	unknown_0:/
	unknown_1:/
	unknown_2:/
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_328408o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������/`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������/: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������/
 
_user_specified_nameinputs
�
�
/__inference_gene_decoder_2_layer_call_fn_331753

inputs
unknown:	P�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_gene_decoder_2_layer_call_and_return_conditional_losses_329177p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������P: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_331710

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
M__inference_protein_decoder_2_layer_call_and_return_conditional_losses_331784

inputs0
matmul_readvariableop_resource:/-
biasadd_readvariableop_resource:/
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:/*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������/r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:/*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������/V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������/Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������/w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�D
__inference__traced_save_332432
file_prefix4
0savev2_gene_encoder_1_kernel_read_readvariableop2
.savev2_gene_encoder_1_bias_read_readvariableop7
3savev2_protein_encoder_1_kernel_read_readvariableop5
1savev2_protein_encoder_1_bias_read_readvariableop9
5savev2_batchnormgeneencode1_gamma_read_readvariableop8
4savev2_batchnormgeneencode1_beta_read_readvariableop?
;savev2_batchnormgeneencode1_moving_mean_read_readvariableopC
?savev2_batchnormgeneencode1_moving_variance_read_readvariableop<
8savev2_batchnormproteinencode1_gamma_read_readvariableop;
7savev2_batchnormproteinencode1_beta_read_readvariableopB
>savev2_batchnormproteinencode1_moving_mean_read_readvariableopF
Bsavev2_batchnormproteinencode1_moving_variance_read_readvariableop4
0savev2_gene_encoder_2_kernel_read_readvariableop2
.savev2_gene_encoder_2_bias_read_readvariableop7
3savev2_protein_encoder_2_kernel_read_readvariableop5
1savev2_protein_encoder_2_bias_read_readvariableop9
5savev2_batchnormgeneencode2_gamma_read_readvariableop8
4savev2_batchnormgeneencode2_beta_read_readvariableop?
;savev2_batchnormgeneencode2_moving_mean_read_readvariableopC
?savev2_batchnormgeneencode2_moving_variance_read_readvariableop<
8savev2_batchnormproteinencode2_gamma_read_readvariableop;
7savev2_batchnormproteinencode2_beta_read_readvariableopB
>savev2_batchnormproteinencode2_moving_mean_read_readvariableopF
Bsavev2_batchnormproteinencode2_moving_variance_read_readvariableop7
3savev2_embeddingdimdense_kernel_read_readvariableop5
1savev2_embeddingdimdense_bias_read_readvariableop4
0savev2_gene_decoder_1_kernel_read_readvariableop2
.savev2_gene_decoder_1_bias_read_readvariableop7
3savev2_protein_decoder_1_kernel_read_readvariableop5
1savev2_protein_decoder_1_bias_read_readvariableop9
5savev2_batchnormgenedecode1_gamma_read_readvariableop8
4savev2_batchnormgenedecode1_beta_read_readvariableop?
;savev2_batchnormgenedecode1_moving_mean_read_readvariableopC
?savev2_batchnormgenedecode1_moving_variance_read_readvariableop<
8savev2_batchnormproteindecode1_gamma_read_readvariableop;
7savev2_batchnormproteindecode1_beta_read_readvariableopB
>savev2_batchnormproteindecode1_moving_mean_read_readvariableopF
Bsavev2_batchnormproteindecode1_moving_variance_read_readvariableop4
0savev2_gene_decoder_2_kernel_read_readvariableop2
.savev2_gene_decoder_2_bias_read_readvariableop7
3savev2_protein_decoder_2_kernel_read_readvariableop5
1savev2_protein_decoder_2_bias_read_readvariableop9
5savev2_batchnormgenedecode2_gamma_read_readvariableop8
4savev2_batchnormgenedecode2_beta_read_readvariableop?
;savev2_batchnormgenedecode2_moving_mean_read_readvariableopC
?savev2_batchnormgenedecode2_moving_variance_read_readvariableop<
8savev2_batchnormproteindecode2_gamma_read_readvariableop;
7savev2_batchnormproteindecode2_beta_read_readvariableopB
>savev2_batchnormproteindecode2_moving_mean_read_readvariableopF
Bsavev2_batchnormproteindecode2_moving_variance_read_readvariableop7
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
5savev2_adam_gene_encoder_1_bias_m_read_readvariableop>
:savev2_adam_protein_encoder_1_kernel_m_read_readvariableop<
8savev2_adam_protein_encoder_1_bias_m_read_readvariableop@
<savev2_adam_batchnormgeneencode1_gamma_m_read_readvariableop?
;savev2_adam_batchnormgeneencode1_beta_m_read_readvariableopC
?savev2_adam_batchnormproteinencode1_gamma_m_read_readvariableopB
>savev2_adam_batchnormproteinencode1_beta_m_read_readvariableop;
7savev2_adam_gene_encoder_2_kernel_m_read_readvariableop9
5savev2_adam_gene_encoder_2_bias_m_read_readvariableop>
:savev2_adam_protein_encoder_2_kernel_m_read_readvariableop<
8savev2_adam_protein_encoder_2_bias_m_read_readvariableop@
<savev2_adam_batchnormgeneencode2_gamma_m_read_readvariableop?
;savev2_adam_batchnormgeneencode2_beta_m_read_readvariableopC
?savev2_adam_batchnormproteinencode2_gamma_m_read_readvariableopB
>savev2_adam_batchnormproteinencode2_beta_m_read_readvariableop>
:savev2_adam_embeddingdimdense_kernel_m_read_readvariableop<
8savev2_adam_embeddingdimdense_bias_m_read_readvariableop;
7savev2_adam_gene_decoder_1_kernel_m_read_readvariableop9
5savev2_adam_gene_decoder_1_bias_m_read_readvariableop>
:savev2_adam_protein_decoder_1_kernel_m_read_readvariableop<
8savev2_adam_protein_decoder_1_bias_m_read_readvariableop@
<savev2_adam_batchnormgenedecode1_gamma_m_read_readvariableop?
;savev2_adam_batchnormgenedecode1_beta_m_read_readvariableopC
?savev2_adam_batchnormproteindecode1_gamma_m_read_readvariableopB
>savev2_adam_batchnormproteindecode1_beta_m_read_readvariableop;
7savev2_adam_gene_decoder_2_kernel_m_read_readvariableop9
5savev2_adam_gene_decoder_2_bias_m_read_readvariableop>
:savev2_adam_protein_decoder_2_kernel_m_read_readvariableop<
8savev2_adam_protein_decoder_2_bias_m_read_readvariableop@
<savev2_adam_batchnormgenedecode2_gamma_m_read_readvariableop?
;savev2_adam_batchnormgenedecode2_beta_m_read_readvariableopC
?savev2_adam_batchnormproteindecode2_gamma_m_read_readvariableopB
>savev2_adam_batchnormproteindecode2_beta_m_read_readvariableop>
:savev2_adam_gene_decoder_last_kernel_m_read_readvariableop<
8savev2_adam_gene_decoder_last_bias_m_read_readvariableopA
=savev2_adam_protein_decoder_last_kernel_m_read_readvariableop?
;savev2_adam_protein_decoder_last_bias_m_read_readvariableop;
7savev2_adam_gene_encoder_1_kernel_v_read_readvariableop9
5savev2_adam_gene_encoder_1_bias_v_read_readvariableop>
:savev2_adam_protein_encoder_1_kernel_v_read_readvariableop<
8savev2_adam_protein_encoder_1_bias_v_read_readvariableop@
<savev2_adam_batchnormgeneencode1_gamma_v_read_readvariableop?
;savev2_adam_batchnormgeneencode1_beta_v_read_readvariableopC
?savev2_adam_batchnormproteinencode1_gamma_v_read_readvariableopB
>savev2_adam_batchnormproteinencode1_beta_v_read_readvariableop;
7savev2_adam_gene_encoder_2_kernel_v_read_readvariableop9
5savev2_adam_gene_encoder_2_bias_v_read_readvariableop>
:savev2_adam_protein_encoder_2_kernel_v_read_readvariableop<
8savev2_adam_protein_encoder_2_bias_v_read_readvariableop@
<savev2_adam_batchnormgeneencode2_gamma_v_read_readvariableop?
;savev2_adam_batchnormgeneencode2_beta_v_read_readvariableopC
?savev2_adam_batchnormproteinencode2_gamma_v_read_readvariableopB
>savev2_adam_batchnormproteinencode2_beta_v_read_readvariableop>
:savev2_adam_embeddingdimdense_kernel_v_read_readvariableop<
8savev2_adam_embeddingdimdense_bias_v_read_readvariableop;
7savev2_adam_gene_decoder_1_kernel_v_read_readvariableop9
5savev2_adam_gene_decoder_1_bias_v_read_readvariableop>
:savev2_adam_protein_decoder_1_kernel_v_read_readvariableop<
8savev2_adam_protein_decoder_1_bias_v_read_readvariableop@
<savev2_adam_batchnormgenedecode1_gamma_v_read_readvariableop?
;savev2_adam_batchnormgenedecode1_beta_v_read_readvariableopC
?savev2_adam_batchnormproteindecode1_gamma_v_read_readvariableopB
>savev2_adam_batchnormproteindecode1_beta_v_read_readvariableop;
7savev2_adam_gene_decoder_2_kernel_v_read_readvariableop9
5savev2_adam_gene_decoder_2_bias_v_read_readvariableop>
:savev2_adam_protein_decoder_2_kernel_v_read_readvariableop<
8savev2_adam_protein_decoder_2_bias_v_read_readvariableop@
<savev2_adam_batchnormgenedecode2_gamma_v_read_readvariableop?
;savev2_adam_batchnormgenedecode2_beta_v_read_readvariableopC
?savev2_adam_batchnormproteindecode2_gamma_v_read_readvariableopB
>savev2_adam_batchnormproteindecode2_beta_v_read_readvariableop>
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
: �O
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�N
value�NB�N�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-16/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-16/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-16/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-17/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-3/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-6/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-11/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-12/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-15/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-16/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-16/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-17/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-17/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-18/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-18/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes	
:�*
dtype0*�
value�B��B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �A
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_gene_encoder_1_kernel_read_readvariableop.savev2_gene_encoder_1_bias_read_readvariableop3savev2_protein_encoder_1_kernel_read_readvariableop1savev2_protein_encoder_1_bias_read_readvariableop5savev2_batchnormgeneencode1_gamma_read_readvariableop4savev2_batchnormgeneencode1_beta_read_readvariableop;savev2_batchnormgeneencode1_moving_mean_read_readvariableop?savev2_batchnormgeneencode1_moving_variance_read_readvariableop8savev2_batchnormproteinencode1_gamma_read_readvariableop7savev2_batchnormproteinencode1_beta_read_readvariableop>savev2_batchnormproteinencode1_moving_mean_read_readvariableopBsavev2_batchnormproteinencode1_moving_variance_read_readvariableop0savev2_gene_encoder_2_kernel_read_readvariableop.savev2_gene_encoder_2_bias_read_readvariableop3savev2_protein_encoder_2_kernel_read_readvariableop1savev2_protein_encoder_2_bias_read_readvariableop5savev2_batchnormgeneencode2_gamma_read_readvariableop4savev2_batchnormgeneencode2_beta_read_readvariableop;savev2_batchnormgeneencode2_moving_mean_read_readvariableop?savev2_batchnormgeneencode2_moving_variance_read_readvariableop8savev2_batchnormproteinencode2_gamma_read_readvariableop7savev2_batchnormproteinencode2_beta_read_readvariableop>savev2_batchnormproteinencode2_moving_mean_read_readvariableopBsavev2_batchnormproteinencode2_moving_variance_read_readvariableop3savev2_embeddingdimdense_kernel_read_readvariableop1savev2_embeddingdimdense_bias_read_readvariableop0savev2_gene_decoder_1_kernel_read_readvariableop.savev2_gene_decoder_1_bias_read_readvariableop3savev2_protein_decoder_1_kernel_read_readvariableop1savev2_protein_decoder_1_bias_read_readvariableop5savev2_batchnormgenedecode1_gamma_read_readvariableop4savev2_batchnormgenedecode1_beta_read_readvariableop;savev2_batchnormgenedecode1_moving_mean_read_readvariableop?savev2_batchnormgenedecode1_moving_variance_read_readvariableop8savev2_batchnormproteindecode1_gamma_read_readvariableop7savev2_batchnormproteindecode1_beta_read_readvariableop>savev2_batchnormproteindecode1_moving_mean_read_readvariableopBsavev2_batchnormproteindecode1_moving_variance_read_readvariableop0savev2_gene_decoder_2_kernel_read_readvariableop.savev2_gene_decoder_2_bias_read_readvariableop3savev2_protein_decoder_2_kernel_read_readvariableop1savev2_protein_decoder_2_bias_read_readvariableop5savev2_batchnormgenedecode2_gamma_read_readvariableop4savev2_batchnormgenedecode2_beta_read_readvariableop;savev2_batchnormgenedecode2_moving_mean_read_readvariableop?savev2_batchnormgenedecode2_moving_variance_read_readvariableop8savev2_batchnormproteindecode2_gamma_read_readvariableop7savev2_batchnormproteindecode2_beta_read_readvariableop>savev2_batchnormproteindecode2_moving_mean_read_readvariableopBsavev2_batchnormproteindecode2_moving_variance_read_readvariableop3savev2_gene_decoder_last_kernel_read_readvariableop1savev2_gene_decoder_last_bias_read_readvariableop6savev2_protein_decoder_last_kernel_read_readvariableop4savev2_protein_decoder_last_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop7savev2_adam_gene_encoder_1_kernel_m_read_readvariableop5savev2_adam_gene_encoder_1_bias_m_read_readvariableop:savev2_adam_protein_encoder_1_kernel_m_read_readvariableop8savev2_adam_protein_encoder_1_bias_m_read_readvariableop<savev2_adam_batchnormgeneencode1_gamma_m_read_readvariableop;savev2_adam_batchnormgeneencode1_beta_m_read_readvariableop?savev2_adam_batchnormproteinencode1_gamma_m_read_readvariableop>savev2_adam_batchnormproteinencode1_beta_m_read_readvariableop7savev2_adam_gene_encoder_2_kernel_m_read_readvariableop5savev2_adam_gene_encoder_2_bias_m_read_readvariableop:savev2_adam_protein_encoder_2_kernel_m_read_readvariableop8savev2_adam_protein_encoder_2_bias_m_read_readvariableop<savev2_adam_batchnormgeneencode2_gamma_m_read_readvariableop;savev2_adam_batchnormgeneencode2_beta_m_read_readvariableop?savev2_adam_batchnormproteinencode2_gamma_m_read_readvariableop>savev2_adam_batchnormproteinencode2_beta_m_read_readvariableop:savev2_adam_embeddingdimdense_kernel_m_read_readvariableop8savev2_adam_embeddingdimdense_bias_m_read_readvariableop7savev2_adam_gene_decoder_1_kernel_m_read_readvariableop5savev2_adam_gene_decoder_1_bias_m_read_readvariableop:savev2_adam_protein_decoder_1_kernel_m_read_readvariableop8savev2_adam_protein_decoder_1_bias_m_read_readvariableop<savev2_adam_batchnormgenedecode1_gamma_m_read_readvariableop;savev2_adam_batchnormgenedecode1_beta_m_read_readvariableop?savev2_adam_batchnormproteindecode1_gamma_m_read_readvariableop>savev2_adam_batchnormproteindecode1_beta_m_read_readvariableop7savev2_adam_gene_decoder_2_kernel_m_read_readvariableop5savev2_adam_gene_decoder_2_bias_m_read_readvariableop:savev2_adam_protein_decoder_2_kernel_m_read_readvariableop8savev2_adam_protein_decoder_2_bias_m_read_readvariableop<savev2_adam_batchnormgenedecode2_gamma_m_read_readvariableop;savev2_adam_batchnormgenedecode2_beta_m_read_readvariableop?savev2_adam_batchnormproteindecode2_gamma_m_read_readvariableop>savev2_adam_batchnormproteindecode2_beta_m_read_readvariableop:savev2_adam_gene_decoder_last_kernel_m_read_readvariableop8savev2_adam_gene_decoder_last_bias_m_read_readvariableop=savev2_adam_protein_decoder_last_kernel_m_read_readvariableop;savev2_adam_protein_decoder_last_bias_m_read_readvariableop7savev2_adam_gene_encoder_1_kernel_v_read_readvariableop5savev2_adam_gene_encoder_1_bias_v_read_readvariableop:savev2_adam_protein_encoder_1_kernel_v_read_readvariableop8savev2_adam_protein_encoder_1_bias_v_read_readvariableop<savev2_adam_batchnormgeneencode1_gamma_v_read_readvariableop;savev2_adam_batchnormgeneencode1_beta_v_read_readvariableop?savev2_adam_batchnormproteinencode1_gamma_v_read_readvariableop>savev2_adam_batchnormproteinencode1_beta_v_read_readvariableop7savev2_adam_gene_encoder_2_kernel_v_read_readvariableop5savev2_adam_gene_encoder_2_bias_v_read_readvariableop:savev2_adam_protein_encoder_2_kernel_v_read_readvariableop8savev2_adam_protein_encoder_2_bias_v_read_readvariableop<savev2_adam_batchnormgeneencode2_gamma_v_read_readvariableop;savev2_adam_batchnormgeneencode2_beta_v_read_readvariableop?savev2_adam_batchnormproteinencode2_gamma_v_read_readvariableop>savev2_adam_batchnormproteinencode2_beta_v_read_readvariableop:savev2_adam_embeddingdimdense_kernel_v_read_readvariableop8savev2_adam_embeddingdimdense_bias_v_read_readvariableop7savev2_adam_gene_decoder_1_kernel_v_read_readvariableop5savev2_adam_gene_decoder_1_bias_v_read_readvariableop:savev2_adam_protein_decoder_1_kernel_v_read_readvariableop8savev2_adam_protein_decoder_1_bias_v_read_readvariableop<savev2_adam_batchnormgenedecode1_gamma_v_read_readvariableop;savev2_adam_batchnormgenedecode1_beta_v_read_readvariableop?savev2_adam_batchnormproteindecode1_gamma_v_read_readvariableop>savev2_adam_batchnormproteindecode1_beta_v_read_readvariableop7savev2_adam_gene_decoder_2_kernel_v_read_readvariableop5savev2_adam_gene_decoder_2_bias_v_read_readvariableop:savev2_adam_protein_decoder_2_kernel_v_read_readvariableop8savev2_adam_protein_decoder_2_bias_v_read_readvariableop<savev2_adam_batchnormgenedecode2_gamma_v_read_readvariableop;savev2_adam_batchnormgenedecode2_beta_v_read_readvariableop?savev2_adam_batchnormproteindecode2_gamma_v_read_readvariableop>savev2_adam_batchnormproteindecode2_beta_v_read_readvariableop:savev2_adam_gene_decoder_last_kernel_v_read_readvariableop8savev2_adam_gene_decoder_last_bias_v_read_readvariableop=savev2_adam_protein_decoder_last_kernel_v_read_readvariableop;savev2_adam_protein_decoder_last_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *�
dtypes�
�2�	�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :
�
�:�:	�/:/:�:�:�:�:/:/:/:/:	�P:P:/::P:P:P:P:::::[@:@:@P:P:@::P:P:P:P:::::	P�:�:/:/:�:�:�:�:/:/:/:/:
��
:�
:	/�:�: : : : : : : : : : : :
�
�:�:	�/:/:�:�:/:/:	�P:P:/::P:P:::[@:@:@P:P:@::P:P:::	P�:�:/:/:�:�:/:/:
��
:�
:	/�:�:
�
�:�:	�/:/:�:�:/:/:	�P:P:/::P:P:::[@:@:@P:P:@::P:P:::	P�:�:/:/:�:�:/:/:
��
:�
:	/�:�: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
�
�:!

_output_shapes	
:�:%!

_output_shapes
:	�/: 

_output_shapes
:/:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�:!

_output_shapes	
:�: 	

_output_shapes
:/: 


_output_shapes
:/: 

_output_shapes
:/: 

_output_shapes
:/:%!

_output_shapes
:	�P: 

_output_shapes
:P:$ 

_output_shapes

:/: 

_output_shapes
:: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:P: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::$ 

_output_shapes

:[@: 

_output_shapes
:@:$ 

_output_shapes

:@P: 

_output_shapes
:P:$ 

_output_shapes

:@: 

_output_shapes
:: 

_output_shapes
:P:  

_output_shapes
:P: !

_output_shapes
:P: "

_output_shapes
:P: #

_output_shapes
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
::%'!

_output_shapes
:	P�:!(

_output_shapes	
:�:$) 

_output_shapes

:/: *

_output_shapes
:/:!+

_output_shapes	
:�:!,

_output_shapes	
:�:!-

_output_shapes	
:�:!.

_output_shapes	
:�: /

_output_shapes
:/: 0

_output_shapes
:/: 1

_output_shapes
:/: 2

_output_shapes
:/:&3"
 
_output_shapes
:
��
:!4

_output_shapes	
:�
:%5!

_output_shapes
:	/�:!6

_output_shapes	
:�:7

_output_shapes
: :8

_output_shapes
: :9

_output_shapes
: ::

_output_shapes
: :;

_output_shapes
: :<

_output_shapes
: :=

_output_shapes
: :>

_output_shapes
: :?

_output_shapes
: :@

_output_shapes
: :A

_output_shapes
: :&B"
 
_output_shapes
:
�
�:!C

_output_shapes	
:�:%D!

_output_shapes
:	�/: E

_output_shapes
:/:!F

_output_shapes	
:�:!G

_output_shapes	
:�: H

_output_shapes
:/: I

_output_shapes
:/:%J!

_output_shapes
:	�P: K

_output_shapes
:P:$L 

_output_shapes

:/: M

_output_shapes
:: N

_output_shapes
:P: O

_output_shapes
:P: P

_output_shapes
:: Q

_output_shapes
::$R 

_output_shapes

:[@: S

_output_shapes
:@:$T 

_output_shapes

:@P: U

_output_shapes
:P:$V 

_output_shapes

:@: W

_output_shapes
:: X

_output_shapes
:P: Y

_output_shapes
:P: Z

_output_shapes
:: [

_output_shapes
::%\!

_output_shapes
:	P�:!]

_output_shapes	
:�:$^ 

_output_shapes

:/: _

_output_shapes
:/:!`

_output_shapes	
:�:!a

_output_shapes	
:�: b

_output_shapes
:/: c

_output_shapes
:/:&d"
 
_output_shapes
:
��
:!e

_output_shapes	
:�
:%f!

_output_shapes
:	/�:!g

_output_shapes	
:�:&h"
 
_output_shapes
:
�
�:!i

_output_shapes	
:�:%j!

_output_shapes
:	�/: k

_output_shapes
:/:!l

_output_shapes	
:�:!m

_output_shapes	
:�: n

_output_shapes
:/: o

_output_shapes
:/:%p!

_output_shapes
:	�P: q

_output_shapes
:P:$r 

_output_shapes

:/: s

_output_shapes
:: t

_output_shapes
:P: u

_output_shapes
:P: v

_output_shapes
:: w

_output_shapes
::$x 

_output_shapes

:[@: y

_output_shapes
:@:$z 

_output_shapes

:@P: {

_output_shapes
:P:$| 

_output_shapes

:@: }

_output_shapes
:: ~

_output_shapes
:P: 

_output_shapes
:P:!�

_output_shapes
::!�

_output_shapes
::&�!

_output_shapes
:	P�:"�

_output_shapes	
:�:%� 

_output_shapes

:/:!�

_output_shapes
:/:"�

_output_shapes	
:�:"�

_output_shapes	
:�:!�

_output_shapes
:/:!�

_output_shapes
:/:'�"
 
_output_shapes
:
��
:"�

_output_shapes	
:�
:&�!

_output_shapes
:	/�:"�

_output_shapes	
:�:�

_output_shapes
: 
�

�
P__inference_protein_decoder_last_layer_call_and_return_conditional_losses_331984

inputs1
matmul_readvariableop_resource:	/�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	/�*
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
:���������/: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������/
 
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
"serving_default_Gene_Input_Layer:0����������

T
Protein_Input_Layer=
%serving_default_Protein_Input_Layer:0����������F
gene_decoder_last1
StatefulPartitionedCall:0����������
I
protein_decoder_last1
StatefulPartitionedCall:1����������tensorflow/serving/predict:��
�
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer_with_weights-4
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer_with_weights-7

layer-9
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
layer_with_weights-15
layer-18
layer_with_weights-16
layer-19
layer_with_weights-17
layer-20
layer_with_weights-18
layer-21
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
�

 kernel
!bias
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_layer
�

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
�
0axis
	1gamma
2beta
3moving_mean
4moving_variance
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses"
_tf_keras_layer
�
;axis
	<gamma
=beta
>moving_mean
?moving_variance
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Fkernel
Gbias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
L__call__
*M&call_and_return_all_conditional_losses"
_tf_keras_layer
�

Nkernel
Obias
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
�
Vaxis
	Wgamma
Xbeta
Ymoving_mean
Zmoving_variance
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses"
_tf_keras_layer
�
aaxis
	bgamma
cbeta
dmoving_mean
emoving_variance
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
j__call__
*k&call_and_return_all_conditional_losses"
_tf_keras_layer
�
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
p__call__
*q&call_and_return_all_conditional_losses"
_tf_keras_layer
�

rkernel
sbias
t	variables
utrainable_variables
vregularization_losses
w	keras_api
x__call__
*y&call_and_return_all_conditional_losses"
_tf_keras_layer
�

zkernel
{bias
|	variables
}trainable_variables
~regularization_losses
	keras_api
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
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate m�!m�(m�)m�1m�2m�<m�=m�Fm�Gm�Nm�Om�Wm�Xm�bm�cm�rm�sm�zm�{m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m� v�!v�(v�)v�1v�2v�<v�=v�Fv�Gv�Nv�Ov�Wv�Xv�bv�cv�rv�sv�zv�{v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�"
	optimizer
�
 0
!1
(2
)3
14
25
36
47
<8
=9
>10
?11
F12
G13
N14
O15
W16
X17
Y18
Z19
b20
c21
d22
e23
r24
s25
z26
{27
�28
�29
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
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53"
trackable_list_wrapper
�
 0
!1
(2
)3
14
25
<6
=7
F8
G9
N10
O11
W12
X13
b14
c15
r16
s17
z18
{19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_model_8_layer_call_fn_329350
(__inference_model_8_layer_call_fn_330339
(__inference_model_8_layer_call_fn_330455
(__inference_model_8_layer_call_fn_329949�
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
C__inference_model_8_layer_call_and_return_conditional_losses_330668
C__inference_model_8_layer_call_and_return_conditional_losses_330993
C__inference_model_8_layer_call_and_return_conditional_losses_330083
C__inference_model_8_layer_call_and_return_conditional_losses_330217�
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
!__inference__wrapped_model_328302Gene_Input_LayerProtein_Input_Layer"�
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
�
�2gene_encoder_1/kernel
": �2gene_encoder_1/bias
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
�2�
/__inference_gene_encoder_1_layer_call_fn_331120�
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
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_331131�
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
+:)	�/2protein_encoder_1/kernel
$:"/2protein_encoder_1/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
�2�
2__inference_protein_encoder_1_layer_call_fn_331140�
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
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_331151�
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
):'�2BatchNormGeneEncode1/gamma
(:&�2BatchNormGeneEncode1/beta
1:/� (2 BatchNormGeneEncode1/moving_mean
5:3� (2$BatchNormGeneEncode1/moving_variance
<
10
21
32
43"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
�2�
5__inference_BatchNormGeneEncode1_layer_call_fn_331164
5__inference_BatchNormGeneEncode1_layer_call_fn_331177�
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
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_331197
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_331231�
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
+:)/2BatchNormProteinEncode1/gamma
*:(/2BatchNormProteinEncode1/beta
3:1/ (2#BatchNormProteinEncode1/moving_mean
7:5/ (2'BatchNormProteinEncode1/moving_variance
<
<0
=1
>2
?3"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
�2�
8__inference_BatchNormProteinEncode1_layer_call_fn_331244
8__inference_BatchNormProteinEncode1_layer_call_fn_331257�
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
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_331277
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_331311�
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
(:&	�P2gene_encoder_2/kernel
!:P2gene_encoder_2/bias
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
H	variables
Itrainable_variables
Jregularization_losses
L__call__
*M&call_and_return_all_conditional_losses
&M"call_and_return_conditional_losses"
_generic_user_object
�2�
/__inference_gene_encoder_2_layer_call_fn_331320�
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
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_331331�
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
*:(/2protein_encoder_2/kernel
$:"2protein_encoder_2/bias
.
N0
O1"
trackable_list_wrapper
.
N0
O1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
�2�
2__inference_protein_encoder_2_layer_call_fn_331340�
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
M__inference_protein_encoder_2_layer_call_and_return_conditional_losses_331351�
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
(:&P2BatchNormGeneEncode2/gamma
':%P2BatchNormGeneEncode2/beta
0:.P (2 BatchNormGeneEncode2/moving_mean
4:2P (2$BatchNormGeneEncode2/moving_variance
<
W0
X1
Y2
Z3"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
�2�
5__inference_BatchNormGeneEncode2_layer_call_fn_331364
5__inference_BatchNormGeneEncode2_layer_call_fn_331377�
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
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_331397
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_331431�
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
+:)2BatchNormProteinEncode2/gamma
*:(2BatchNormProteinEncode2/beta
3:1 (2#BatchNormProteinEncode2/moving_mean
7:5 (2'BatchNormProteinEncode2/moving_variance
<
b0
c1
d2
e3"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
f	variables
gtrainable_variables
hregularization_losses
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
�2�
8__inference_BatchNormProteinEncode2_layer_call_fn_331444
8__inference_BatchNormProteinEncode2_layer_call_fn_331457�
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
S__inference_BatchNormProteinEncode2_layer_call_and_return_conditional_losses_331477
S__inference_BatchNormProteinEncode2_layer_call_and_return_conditional_losses_331511�
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
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
l	variables
mtrainable_variables
nregularization_losses
p__call__
*q&call_and_return_all_conditional_losses
&q"call_and_return_conditional_losses"
_generic_user_object
�2�
1__inference_ConcatenateLayer_layer_call_fn_331517�
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
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_331524�
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
*:([@2EmbeddingDimDense/kernel
$:"@2EmbeddingDimDense/bias
.
r0
s1"
trackable_list_wrapper
.
r0
s1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
t	variables
utrainable_variables
vregularization_losses
x__call__
*y&call_and_return_all_conditional_losses
&y"call_and_return_conditional_losses"
_generic_user_object
�2�
2__inference_EmbeddingDimDense_layer_call_fn_331533�
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
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_331544�
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
':%@P2gene_decoder_1/kernel
!:P2gene_decoder_1/bias
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
|	variables
}trainable_variables
~regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2�
/__inference_gene_decoder_1_layer_call_fn_331553�
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
J__inference_gene_decoder_1_layer_call_and_return_conditional_losses_331564�
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
*:(@2protein_decoder_1/kernel
$:"2protein_decoder_1/bias
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
2__inference_protein_decoder_1_layer_call_fn_331573�
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
M__inference_protein_decoder_1_layer_call_and_return_conditional_losses_331584�
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
(:&P2BatchNormGeneDecode1/gamma
':%P2BatchNormGeneDecode1/beta
0:.P (2 BatchNormGeneDecode1/moving_mean
4:2P (2$BatchNormGeneDecode1/moving_variance
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
�2�
5__inference_BatchNormGeneDecode1_layer_call_fn_331597
5__inference_BatchNormGeneDecode1_layer_call_fn_331610�
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
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_331630
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_331664�
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
+:)2BatchNormProteinDecode1/gamma
*:(2BatchNormProteinDecode1/beta
3:1 (2#BatchNormProteinDecode1/moving_mean
7:5 (2'BatchNormProteinDecode1/moving_variance
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
�2�
8__inference_BatchNormProteinDecode1_layer_call_fn_331677
8__inference_BatchNormProteinDecode1_layer_call_fn_331690�
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
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_331710
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_331744�
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
(:&	P�2gene_decoder_2/kernel
": �2gene_decoder_2/bias
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
/__inference_gene_decoder_2_layer_call_fn_331753�
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
J__inference_gene_decoder_2_layer_call_and_return_conditional_losses_331764�
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
*:(/2protein_decoder_2/kernel
$:"/2protein_decoder_2/bias
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
2__inference_protein_decoder_2_layer_call_fn_331773�
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
M__inference_protein_decoder_2_layer_call_and_return_conditional_losses_331784�
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
):'�2BatchNormGeneDecode2/gamma
(:&�2BatchNormGeneDecode2/beta
1:/� (2 BatchNormGeneDecode2/moving_mean
5:3� (2$BatchNormGeneDecode2/moving_variance
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
�2�
5__inference_BatchNormGeneDecode2_layer_call_fn_331797
5__inference_BatchNormGeneDecode2_layer_call_fn_331810�
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
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_331830
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_331864�
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
+:)/2BatchNormProteinDecode2/gamma
*:(/2BatchNormProteinDecode2/beta
3:1/ (2#BatchNormProteinDecode2/moving_mean
7:5/ (2'BatchNormProteinDecode2/moving_variance
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
�2�
8__inference_BatchNormProteinDecode2_layer_call_fn_331877
8__inference_BatchNormProteinDecode2_layer_call_fn_331890�
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
S__inference_BatchNormProteinDecode2_layer_call_and_return_conditional_losses_331910
S__inference_BatchNormProteinDecode2_layer_call_and_return_conditional_losses_331944�
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
��
2gene_decoder_last/kernel
%:#�
2gene_decoder_last/bias
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
2__inference_gene_decoder_last_layer_call_fn_331953�
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
M__inference_gene_decoder_last_layer_call_and_return_conditional_losses_331964�
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
.:,	/�2protein_decoder_last/kernel
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
5__inference_protein_decoder_last_layer_call_fn_331973�
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
P__inference_protein_decoder_last_layer_call_and_return_conditional_losses_331984�
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
�
30
41
>2
?3
Y4
Z5
d6
e7
�8
�9
�10
�11
�12
�13
�14
�15"
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
17
18
19
20
21"
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
$__inference_signature_wrapper_331111Gene_Input_LayerProtein_Input_Layer"�
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
30
41"
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
>0
?1"
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
Y0
Z1"
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
d0
e1"
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
�
�2Adam/gene_encoder_1/kernel/m
':%�2Adam/gene_encoder_1/bias/m
0:.	�/2Adam/protein_encoder_1/kernel/m
):'/2Adam/protein_encoder_1/bias/m
.:,�2!Adam/BatchNormGeneEncode1/gamma/m
-:+�2 Adam/BatchNormGeneEncode1/beta/m
0:./2$Adam/BatchNormProteinEncode1/gamma/m
/:-/2#Adam/BatchNormProteinEncode1/beta/m
-:+	�P2Adam/gene_encoder_2/kernel/m
&:$P2Adam/gene_encoder_2/bias/m
/:-/2Adam/protein_encoder_2/kernel/m
):'2Adam/protein_encoder_2/bias/m
-:+P2!Adam/BatchNormGeneEncode2/gamma/m
,:*P2 Adam/BatchNormGeneEncode2/beta/m
0:.2$Adam/BatchNormProteinEncode2/gamma/m
/:-2#Adam/BatchNormProteinEncode2/beta/m
/:-[@2Adam/EmbeddingDimDense/kernel/m
):'@2Adam/EmbeddingDimDense/bias/m
,:*@P2Adam/gene_decoder_1/kernel/m
&:$P2Adam/gene_decoder_1/bias/m
/:-@2Adam/protein_decoder_1/kernel/m
):'2Adam/protein_decoder_1/bias/m
-:+P2!Adam/BatchNormGeneDecode1/gamma/m
,:*P2 Adam/BatchNormGeneDecode1/beta/m
0:.2$Adam/BatchNormProteinDecode1/gamma/m
/:-2#Adam/BatchNormProteinDecode1/beta/m
-:+	P�2Adam/gene_decoder_2/kernel/m
':%�2Adam/gene_decoder_2/bias/m
/:-/2Adam/protein_decoder_2/kernel/m
):'/2Adam/protein_decoder_2/bias/m
.:,�2!Adam/BatchNormGeneDecode2/gamma/m
-:+�2 Adam/BatchNormGeneDecode2/beta/m
0:./2$Adam/BatchNormProteinDecode2/gamma/m
/:-/2#Adam/BatchNormProteinDecode2/beta/m
1:/
��
2Adam/gene_decoder_last/kernel/m
*:(�
2Adam/gene_decoder_last/bias/m
3:1	/�2"Adam/protein_decoder_last/kernel/m
-:+�2 Adam/protein_decoder_last/bias/m
.:,
�
�2Adam/gene_encoder_1/kernel/v
':%�2Adam/gene_encoder_1/bias/v
0:.	�/2Adam/protein_encoder_1/kernel/v
):'/2Adam/protein_encoder_1/bias/v
.:,�2!Adam/BatchNormGeneEncode1/gamma/v
-:+�2 Adam/BatchNormGeneEncode1/beta/v
0:./2$Adam/BatchNormProteinEncode1/gamma/v
/:-/2#Adam/BatchNormProteinEncode1/beta/v
-:+	�P2Adam/gene_encoder_2/kernel/v
&:$P2Adam/gene_encoder_2/bias/v
/:-/2Adam/protein_encoder_2/kernel/v
):'2Adam/protein_encoder_2/bias/v
-:+P2!Adam/BatchNormGeneEncode2/gamma/v
,:*P2 Adam/BatchNormGeneEncode2/beta/v
0:.2$Adam/BatchNormProteinEncode2/gamma/v
/:-2#Adam/BatchNormProteinEncode2/beta/v
/:-[@2Adam/EmbeddingDimDense/kernel/v
):'@2Adam/EmbeddingDimDense/bias/v
,:*@P2Adam/gene_decoder_1/kernel/v
&:$P2Adam/gene_decoder_1/bias/v
/:-@2Adam/protein_decoder_1/kernel/v
):'2Adam/protein_decoder_1/bias/v
-:+P2!Adam/BatchNormGeneDecode1/gamma/v
,:*P2 Adam/BatchNormGeneDecode1/beta/v
0:.2$Adam/BatchNormProteinDecode1/gamma/v
/:-2#Adam/BatchNormProteinDecode1/beta/v
-:+	P�2Adam/gene_decoder_2/kernel/v
':%�2Adam/gene_decoder_2/bias/v
/:-/2Adam/protein_decoder_2/kernel/v
):'/2Adam/protein_decoder_2/bias/v
.:,�2!Adam/BatchNormGeneDecode2/gamma/v
-:+�2 Adam/BatchNormGeneDecode2/beta/v
0:./2$Adam/BatchNormProteinDecode2/gamma/v
/:-/2#Adam/BatchNormProteinDecode2/beta/v
1:/
��
2Adam/gene_decoder_last/kernel/v
*:(�
2Adam/gene_decoder_last/bias/v
3:1	/�2"Adam/protein_decoder_last/kernel/v
-:+�2 Adam/protein_decoder_last/bias/v�
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_331630f����3�0
)�&
 �
inputs���������P
p 
� "%�"
�
0���������P
� �
P__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_331664f����3�0
)�&
 �
inputs���������P
p
� "%�"
�
0���������P
� �
5__inference_BatchNormGeneDecode1_layer_call_fn_331597Y����3�0
)�&
 �
inputs���������P
p 
� "����������P�
5__inference_BatchNormGeneDecode1_layer_call_fn_331610Y����3�0
)�&
 �
inputs���������P
p
� "����������P�
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_331830h����4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
P__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_331864h����4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
5__inference_BatchNormGeneDecode2_layer_call_fn_331797[����4�1
*�'
!�
inputs����������
p 
� "������������
5__inference_BatchNormGeneDecode2_layer_call_fn_331810[����4�1
*�'
!�
inputs����������
p
� "������������
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_331197d41324�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_331231d34124�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
5__inference_BatchNormGeneEncode1_layer_call_fn_331164W41324�1
*�'
!�
inputs����������
p 
� "������������
5__inference_BatchNormGeneEncode1_layer_call_fn_331177W34124�1
*�'
!�
inputs����������
p
� "������������
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_331397bZWYX3�0
)�&
 �
inputs���������P
p 
� "%�"
�
0���������P
� �
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_331431bYZWX3�0
)�&
 �
inputs���������P
p
� "%�"
�
0���������P
� �
5__inference_BatchNormGeneEncode2_layer_call_fn_331364UZWYX3�0
)�&
 �
inputs���������P
p 
� "����������P�
5__inference_BatchNormGeneEncode2_layer_call_fn_331377UYZWX3�0
)�&
 �
inputs���������P
p
� "����������P�
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_331710f����3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
S__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_331744f����3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
8__inference_BatchNormProteinDecode1_layer_call_fn_331677Y����3�0
)�&
 �
inputs���������
p 
� "�����������
8__inference_BatchNormProteinDecode1_layer_call_fn_331690Y����3�0
)�&
 �
inputs���������
p
� "�����������
S__inference_BatchNormProteinDecode2_layer_call_and_return_conditional_losses_331910f����3�0
)�&
 �
inputs���������/
p 
� "%�"
�
0���������/
� �
S__inference_BatchNormProteinDecode2_layer_call_and_return_conditional_losses_331944f����3�0
)�&
 �
inputs���������/
p
� "%�"
�
0���������/
� �
8__inference_BatchNormProteinDecode2_layer_call_fn_331877Y����3�0
)�&
 �
inputs���������/
p 
� "����������/�
8__inference_BatchNormProteinDecode2_layer_call_fn_331890Y����3�0
)�&
 �
inputs���������/
p
� "����������/�
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_331277b?<>=3�0
)�&
 �
inputs���������/
p 
� "%�"
�
0���������/
� �
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_331311b>?<=3�0
)�&
 �
inputs���������/
p
� "%�"
�
0���������/
� �
8__inference_BatchNormProteinEncode1_layer_call_fn_331244U?<>=3�0
)�&
 �
inputs���������/
p 
� "����������/�
8__inference_BatchNormProteinEncode1_layer_call_fn_331257U>?<=3�0
)�&
 �
inputs���������/
p
� "����������/�
S__inference_BatchNormProteinEncode2_layer_call_and_return_conditional_losses_331477bebdc3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
S__inference_BatchNormProteinEncode2_layer_call_and_return_conditional_losses_331511bdebc3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
8__inference_BatchNormProteinEncode2_layer_call_fn_331444Uebdc3�0
)�&
 �
inputs���������
p 
� "�����������
8__inference_BatchNormProteinEncode2_layer_call_fn_331457Udebc3�0
)�&
 �
inputs���������
p
� "�����������
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_331524�Z�W
P�M
K�H
"�
inputs/0���������P
"�
inputs/1���������
� "%�"
�
0���������[
� �
1__inference_ConcatenateLayer_layer_call_fn_331517vZ�W
P�M
K�H
"�
inputs/0���������P
"�
inputs/1���������
� "����������[�
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_331544\rs/�,
%�"
 �
inputs���������[
� "%�"
�
0���������@
� �
2__inference_EmbeddingDimDense_layer_call_fn_331533Ors/�,
%�"
 �
inputs���������[
� "����������@�
!__inference__wrapped_model_328302�P() !?<>=4132NOFGZWYXebdcrs��z{������������������������o�l
e�b
`�]
+�(
Gene_Input_Layer����������

.�+
Protein_Input_Layer����������
� "���
A
gene_decoder_last,�)
gene_decoder_last����������

G
protein_decoder_last/�,
protein_decoder_last�����������
J__inference_gene_decoder_1_layer_call_and_return_conditional_losses_331564\z{/�,
%�"
 �
inputs���������@
� "%�"
�
0���������P
� �
/__inference_gene_decoder_1_layer_call_fn_331553Oz{/�,
%�"
 �
inputs���������@
� "����������P�
J__inference_gene_decoder_2_layer_call_and_return_conditional_losses_331764_��/�,
%�"
 �
inputs���������P
� "&�#
�
0����������
� �
/__inference_gene_decoder_2_layer_call_fn_331753R��/�,
%�"
 �
inputs���������P
� "������������
M__inference_gene_decoder_last_layer_call_and_return_conditional_losses_331964`��0�-
&�#
!�
inputs����������
� "&�#
�
0����������

� �
2__inference_gene_decoder_last_layer_call_fn_331953S��0�-
&�#
!�
inputs����������
� "�����������
�
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_331131^ !0�-
&�#
!�
inputs����������

� "&�#
�
0����������
� �
/__inference_gene_encoder_1_layer_call_fn_331120Q !0�-
&�#
!�
inputs����������

� "������������
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_331331]FG0�-
&�#
!�
inputs����������
� "%�"
�
0���������P
� �
/__inference_gene_encoder_2_layer_call_fn_331320PFG0�-
&�#
!�
inputs����������
� "����������P�
C__inference_model_8_layer_call_and_return_conditional_losses_330083�P() !?<>=4132NOFGZWYXebdcrs��z{������������������������w�t
m�j
`�]
+�(
Gene_Input_Layer����������

.�+
Protein_Input_Layer����������
p 

 
� "M�J
C�@
�
0/0����������

�
0/1����������
� �
C__inference_model_8_layer_call_and_return_conditional_losses_330217�P() !>?<=3412NOFGYZWXdebcrs��z{������������������������w�t
m�j
`�]
+�(
Gene_Input_Layer����������

.�+
Protein_Input_Layer����������
p

 
� "M�J
C�@
�
0/0����������

�
0/1����������
� �
C__inference_model_8_layer_call_and_return_conditional_losses_330668�P() !?<>=4132NOFGZWYXebdcrs��z{������������������������d�a
Z�W
M�J
#� 
inputs/0����������

#� 
inputs/1����������
p 

 
� "M�J
C�@
�
0/0����������

�
0/1����������
� �
C__inference_model_8_layer_call_and_return_conditional_losses_330993�P() !>?<=3412NOFGYZWXdebcrs��z{������������������������d�a
Z�W
M�J
#� 
inputs/0����������

#� 
inputs/1����������
p

 
� "M�J
C�@
�
0/0����������

�
0/1����������
� �
(__inference_model_8_layer_call_fn_329350�P() !?<>=4132NOFGZWYXebdcrs��z{������������������������w�t
m�j
`�]
+�(
Gene_Input_Layer����������

.�+
Protein_Input_Layer����������
p 

 
� "?�<
�
0����������

�
1�����������
(__inference_model_8_layer_call_fn_329949�P() !>?<=3412NOFGYZWXdebcrs��z{������������������������w�t
m�j
`�]
+�(
Gene_Input_Layer����������

.�+
Protein_Input_Layer����������
p

 
� "?�<
�
0����������

�
1�����������
(__inference_model_8_layer_call_fn_330339�P() !?<>=4132NOFGZWYXebdcrs��z{������������������������d�a
Z�W
M�J
#� 
inputs/0����������

#� 
inputs/1����������
p 

 
� "?�<
�
0����������

�
1�����������
(__inference_model_8_layer_call_fn_330455�P() !>?<=3412NOFGYZWXdebcrs��z{������������������������d�a
Z�W
M�J
#� 
inputs/0����������

#� 
inputs/1����������
p

 
� "?�<
�
0����������

�
1�����������
M__inference_protein_decoder_1_layer_call_and_return_conditional_losses_331584^��/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� �
2__inference_protein_decoder_1_layer_call_fn_331573Q��/�,
%�"
 �
inputs���������@
� "�����������
M__inference_protein_decoder_2_layer_call_and_return_conditional_losses_331784^��/�,
%�"
 �
inputs���������
� "%�"
�
0���������/
� �
2__inference_protein_decoder_2_layer_call_fn_331773Q��/�,
%�"
 �
inputs���������
� "����������/�
P__inference_protein_decoder_last_layer_call_and_return_conditional_losses_331984_��/�,
%�"
 �
inputs���������/
� "&�#
�
0����������
� �
5__inference_protein_decoder_last_layer_call_fn_331973R��/�,
%�"
 �
inputs���������/
� "������������
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_331151]()0�-
&�#
!�
inputs����������
� "%�"
�
0���������/
� �
2__inference_protein_encoder_1_layer_call_fn_331140P()0�-
&�#
!�
inputs����������
� "����������/�
M__inference_protein_encoder_2_layer_call_and_return_conditional_losses_331351\NO/�,
%�"
 �
inputs���������/
� "%�"
�
0���������
� �
2__inference_protein_encoder_2_layer_call_fn_331340ONO/�,
%�"
 �
inputs���������/
� "�����������
$__inference_signature_wrapper_331111�P() !?<>=4132NOFGZWYXebdcrs��z{���������������������������
� 
���
?
Gene_Input_Layer+�(
Gene_Input_Layer����������

E
Protein_Input_Layer.�+
Protein_Input_Layer����������"���
A
gene_decoder_last,�)
gene_decoder_last����������

G
protein_decoder_last/�,
protein_decoder_last����������