�� 
��
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
 �"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8Ę
�
 Adam/protein_decoder_last/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:W*1
shared_name" Adam/protein_decoder_last/bias/v
�
4Adam/protein_decoder_last/bias/v/Read/ReadVariableOpReadVariableOp Adam/protein_decoder_last/bias/v*
_output_shapes
:W*
dtype0
�
"Adam/protein_decoder_last/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:W*3
shared_name$"Adam/protein_decoder_last/kernel/v
�
6Adam/protein_decoder_last/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/protein_decoder_last/kernel/v*
_output_shapes

:W*
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
#Adam/BatchNormProteinDecode1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/BatchNormProteinDecode1/beta/v
�
7Adam/BatchNormProteinDecode1/beta/v/Read/ReadVariableOpReadVariableOp#Adam/BatchNormProteinDecode1/beta/v*
_output_shapes
:*
dtype0
�
$Adam/BatchNormProteinDecode1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/BatchNormProteinDecode1/gamma/v
�
8Adam/BatchNormProteinDecode1/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/BatchNormProteinDecode1/gamma/v*
_output_shapes
:*
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
Adam/protein_decoder_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/protein_decoder_1/bias/v
�
1Adam/protein_decoder_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/protein_decoder_1/bias/v*
_output_shapes
:*
dtype0
�
Adam/protein_decoder_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!Adam/protein_decoder_1/kernel/v
�
3Adam/protein_decoder_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/protein_decoder_1/kernel/v*
_output_shapes

:@*
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
#Adam/BatchNormProteinEncode1/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/BatchNormProteinEncode1/beta/v
�
7Adam/BatchNormProteinEncode1/beta/v/Read/ReadVariableOpReadVariableOp#Adam/BatchNormProteinEncode1/beta/v*
_output_shapes
:*
dtype0
�
$Adam/BatchNormProteinEncode1/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/BatchNormProteinEncode1/gamma/v
�
8Adam/BatchNormProteinEncode1/gamma/v/Read/ReadVariableOpReadVariableOp$Adam/BatchNormProteinEncode1/gamma/v*
_output_shapes
:*
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
Adam/protein_encoder_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/protein_encoder_1/bias/v
�
1Adam/protein_encoder_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/protein_encoder_1/bias/v*
_output_shapes
:*
dtype0
�
Adam/protein_encoder_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:W*0
shared_name!Adam/protein_encoder_1/kernel/v
�
3Adam/protein_encoder_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/protein_encoder_1/kernel/v*
_output_shapes

:W*
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
 Adam/protein_decoder_last/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:W*1
shared_name" Adam/protein_decoder_last/bias/m
�
4Adam/protein_decoder_last/bias/m/Read/ReadVariableOpReadVariableOp Adam/protein_decoder_last/bias/m*
_output_shapes
:W*
dtype0
�
"Adam/protein_decoder_last/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:W*3
shared_name$"Adam/protein_decoder_last/kernel/m
�
6Adam/protein_decoder_last/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/protein_decoder_last/kernel/m*
_output_shapes

:W*
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
#Adam/BatchNormProteinDecode1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/BatchNormProteinDecode1/beta/m
�
7Adam/BatchNormProteinDecode1/beta/m/Read/ReadVariableOpReadVariableOp#Adam/BatchNormProteinDecode1/beta/m*
_output_shapes
:*
dtype0
�
$Adam/BatchNormProteinDecode1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/BatchNormProteinDecode1/gamma/m
�
8Adam/BatchNormProteinDecode1/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/BatchNormProteinDecode1/gamma/m*
_output_shapes
:*
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
Adam/protein_decoder_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/protein_decoder_1/bias/m
�
1Adam/protein_decoder_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/protein_decoder_1/bias/m*
_output_shapes
:*
dtype0
�
Adam/protein_decoder_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*0
shared_name!Adam/protein_decoder_1/kernel/m
�
3Adam/protein_decoder_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/protein_decoder_1/kernel/m*
_output_shapes

:@*
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
#Adam/BatchNormProteinEncode1/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/BatchNormProteinEncode1/beta/m
�
7Adam/BatchNormProteinEncode1/beta/m/Read/ReadVariableOpReadVariableOp#Adam/BatchNormProteinEncode1/beta/m*
_output_shapes
:*
dtype0
�
$Adam/BatchNormProteinEncode1/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$Adam/BatchNormProteinEncode1/gamma/m
�
8Adam/BatchNormProteinEncode1/gamma/m/Read/ReadVariableOpReadVariableOp$Adam/BatchNormProteinEncode1/gamma/m*
_output_shapes
:*
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
Adam/protein_encoder_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameAdam/protein_encoder_1/bias/m
�
1Adam/protein_encoder_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/protein_encoder_1/bias/m*
_output_shapes
:*
dtype0
�
Adam/protein_encoder_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:W*0
shared_name!Adam/protein_encoder_1/kernel/m
�
3Adam/protein_encoder_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/protein_encoder_1/kernel/m*
_output_shapes

:W*
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
�
protein_decoder_last/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:W**
shared_nameprotein_decoder_last/bias
�
-protein_decoder_last/bias/Read/ReadVariableOpReadVariableOpprotein_decoder_last/bias*
_output_shapes
:W*
dtype0
�
protein_decoder_last/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:W*,
shared_nameprotein_decoder_last/kernel
�
/protein_decoder_last/kernel/Read/ReadVariableOpReadVariableOpprotein_decoder_last/kernel*
_output_shapes

:W*
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
'BatchNormProteinDecode1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'BatchNormProteinDecode1/moving_variance
�
;BatchNormProteinDecode1/moving_variance/Read/ReadVariableOpReadVariableOp'BatchNormProteinDecode1/moving_variance*
_output_shapes
:*
dtype0
�
#BatchNormProteinDecode1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#BatchNormProteinDecode1/moving_mean
�
7BatchNormProteinDecode1/moving_mean/Read/ReadVariableOpReadVariableOp#BatchNormProteinDecode1/moving_mean*
_output_shapes
:*
dtype0
�
BatchNormProteinDecode1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameBatchNormProteinDecode1/beta
�
0BatchNormProteinDecode1/beta/Read/ReadVariableOpReadVariableOpBatchNormProteinDecode1/beta*
_output_shapes
:*
dtype0
�
BatchNormProteinDecode1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameBatchNormProteinDecode1/gamma
�
1BatchNormProteinDecode1/gamma/Read/ReadVariableOpReadVariableOpBatchNormProteinDecode1/gamma*
_output_shapes
:*
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
protein_decoder_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameprotein_decoder_1/bias
}
*protein_decoder_1/bias/Read/ReadVariableOpReadVariableOpprotein_decoder_1/bias*
_output_shapes
:*
dtype0
�
protein_decoder_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*)
shared_nameprotein_decoder_1/kernel
�
,protein_decoder_1/kernel/Read/ReadVariableOpReadVariableOpprotein_decoder_1/kernel*
_output_shapes

:@*
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
'BatchNormProteinEncode1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'BatchNormProteinEncode1/moving_variance
�
;BatchNormProteinEncode1/moving_variance/Read/ReadVariableOpReadVariableOp'BatchNormProteinEncode1/moving_variance*
_output_shapes
:*
dtype0
�
#BatchNormProteinEncode1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#BatchNormProteinEncode1/moving_mean
�
7BatchNormProteinEncode1/moving_mean/Read/ReadVariableOpReadVariableOp#BatchNormProteinEncode1/moving_mean*
_output_shapes
:*
dtype0
�
BatchNormProteinEncode1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameBatchNormProteinEncode1/beta
�
0BatchNormProteinEncode1/beta/Read/ReadVariableOpReadVariableOpBatchNormProteinEncode1/beta*
_output_shapes
:*
dtype0
�
BatchNormProteinEncode1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameBatchNormProteinEncode1/gamma
�
1BatchNormProteinEncode1/gamma/Read/ReadVariableOpReadVariableOpBatchNormProteinEncode1/gamma*
_output_shapes
:*
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
�
protein_encoder_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameprotein_encoder_1/bias
}
*protein_encoder_1/bias/Read/ReadVariableOpReadVariableOpprotein_encoder_1/bias*
_output_shapes
:*
dtype0
�
protein_encoder_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:W*)
shared_nameprotein_encoder_1/kernel
�
,protein_encoder_1/kernel/Read/ReadVariableOpReadVariableOpprotein_encoder_1/kernel*
_output_shapes

:W*
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
 serving_default_Gene_Input_LayerPlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
#serving_default_Protein_Input_LayerPlaceholder*'
_output_shapes
:���������W*
dtype0*
shape:���������W
�
StatefulPartitionedCallStatefulPartitionedCall serving_default_Gene_Input_Layer#serving_default_Protein_Input_Layergene_encoder_1/kernelgene_encoder_1/bias$BatchNormGeneEncode1/moving_varianceBatchNormGeneEncode1/gamma BatchNormGeneEncode1/moving_meanBatchNormGeneEncode1/betaprotein_encoder_1/kernelprotein_encoder_1/biasgene_encoder_2/kernelgene_encoder_2/bias$BatchNormGeneEncode2/moving_varianceBatchNormGeneEncode2/gamma BatchNormGeneEncode2/moving_meanBatchNormGeneEncode2/beta'BatchNormProteinEncode1/moving_varianceBatchNormProteinEncode1/gamma#BatchNormProteinEncode1/moving_meanBatchNormProteinEncode1/betaEmbeddingDimDense/kernelEmbeddingDimDense/biasgene_decoder_1/kernelgene_decoder_1/bias$BatchNormGeneDecode1/moving_varianceBatchNormGeneDecode1/gamma BatchNormGeneDecode1/moving_meanBatchNormGeneDecode1/betaprotein_decoder_1/kernelprotein_decoder_1/biasgene_decoder_2/kernelgene_decoder_2/bias'BatchNormProteinDecode1/moving_varianceBatchNormProteinDecode1/gamma#BatchNormProteinDecode1/moving_meanBatchNormProteinDecode1/beta$BatchNormGeneDecode2/moving_varianceBatchNormGeneDecode2/gamma BatchNormGeneDecode2/moving_meanBatchNormGeneDecode2/betaprotein_decoder_last/kernelprotein_decoder_last/biasgene_decoder_last/kernelgene_decoder_last/bias*7
Tin0
.2,*
Tout
2*
_collective_manager_ids
 *;
_output_shapes)
':����������:���������W*L
_read_only_resource_inputs.
,*	
 !"#$%&'()*+*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_1898546

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
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias*
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
*axis
	+gamma
,beta
-moving_mean
.moving_variance*
* 
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias*
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias*
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses
Eaxis
	Fgamma
Gbeta
Hmoving_mean
Imoving_variance*
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance*
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses* 
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses

akernel
bbias*
�
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

ikernel
jbias*
�
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses
qaxis
	rgamma
sbeta
tmoving_mean
umoving_variance*
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses

|kernel
}bias*
�
~	variables
trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
"0
#1
+2
,3
-4
.5
56
67
=8
>9
F10
G11
H12
I13
Q14
R15
S16
T17
a18
b19
i20
j21
r22
s23
t24
u25
|26
}27
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
�41*
�
"0
#1
+2
,3
54
65
=6
>7
F8
G9
Q10
R11
a12
b13
i14
j15
r16
s17
|18
}19
�20
�21
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
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
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
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate"m�#m�+m�,m�5m�6m�=m�>m�Fm�Gm�Qm�Rm�am�bm�im�jm�rm�sm�|m�}m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�"v�#v�+v�,v�5v�6v�=v�>v�Fv�Gv�Qv�Rv�av�bv�iv�jv�rv�sv�|v�}v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�*

�serving_default* 

"0
#1*

"0
#1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
e_
VARIABLE_VALUEgene_encoder_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEgene_encoder_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
+0
,1
-2
.3*

+0
,1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

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
50
61*

50
61*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
e_
VARIABLE_VALUEgene_encoder_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEgene_encoder_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

=0
>1*

=0
>1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
hb
VARIABLE_VALUEprotein_encoder_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEprotein_encoder_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
F0
G1
H2
I3*
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
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ic
VARIABLE_VALUEBatchNormGeneEncode2/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEBatchNormGeneEncode2/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE BatchNormGeneEncode2/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE$BatchNormGeneEncode2/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
Q0
R1
S2
T3*

Q0
R1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
lf
VARIABLE_VALUEBatchNormProteinEncode1/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEBatchNormProteinEncode1/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#BatchNormProteinEncode1/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
�z
VARIABLE_VALUE'BatchNormProteinEncode1/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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

�trace_0* 

�trace_0* 

a0
b1*

a0
b1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
hb
VARIABLE_VALUEEmbeddingDimDense/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEEmbeddingDimDense/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

i0
j1*

i0
j1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
e_
VARIABLE_VALUEgene_decoder_1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEgene_decoder_1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
r0
s1
t2
u3*
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
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ic
VARIABLE_VALUEBatchNormGeneDecode1/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEBatchNormGeneDecode1/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE BatchNormGeneDecode1/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE$BatchNormGeneDecode1/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

|0
}1*

|0
}1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
e_
VARIABLE_VALUEgene_decoder_2/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEgene_decoder_2/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
~	variables
trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
ic
VARIABLE_VALUEprotein_decoder_1/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEprotein_decoder_1/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
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

�trace_0
�trace_1* 

�trace_0
�trace_1* 
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

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
mg
VARIABLE_VALUEBatchNormProteinDecode1/gamma6layer_with_weights-12/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUEBatchNormProteinDecode1/beta5layer_with_weights-12/beta/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE#BatchNormProteinDecode1/moving_mean<layer_with_weights-12/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
�{
VARIABLE_VALUE'BatchNormProteinDecode1/moving_variance@layer_with_weights-12/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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

�trace_0* 

�trace_0* 
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

�trace_0* 

�trace_0* 
lf
VARIABLE_VALUEprotein_decoder_last/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEprotein_decoder_last/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*
^
-0
.1
H2
I3
S4
T5
t6
u7
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
-0
.1*
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

H0
I1*
* 
* 
* 
* 
* 
* 
* 
* 

S0
T1*
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
* 
* 
* 
* 
* 

t0
u1*
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
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
<
�	variables
�	keras_api

�total

�count*
<
�	variables
�	keras_api

�total

�count*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
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
SM
VARIABLE_VALUEtotal4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*
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
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�.
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)gene_encoder_1/kernel/Read/ReadVariableOp'gene_encoder_1/bias/Read/ReadVariableOp.BatchNormGeneEncode1/gamma/Read/ReadVariableOp-BatchNormGeneEncode1/beta/Read/ReadVariableOp4BatchNormGeneEncode1/moving_mean/Read/ReadVariableOp8BatchNormGeneEncode1/moving_variance/Read/ReadVariableOp)gene_encoder_2/kernel/Read/ReadVariableOp'gene_encoder_2/bias/Read/ReadVariableOp,protein_encoder_1/kernel/Read/ReadVariableOp*protein_encoder_1/bias/Read/ReadVariableOp.BatchNormGeneEncode2/gamma/Read/ReadVariableOp-BatchNormGeneEncode2/beta/Read/ReadVariableOp4BatchNormGeneEncode2/moving_mean/Read/ReadVariableOp8BatchNormGeneEncode2/moving_variance/Read/ReadVariableOp1BatchNormProteinEncode1/gamma/Read/ReadVariableOp0BatchNormProteinEncode1/beta/Read/ReadVariableOp7BatchNormProteinEncode1/moving_mean/Read/ReadVariableOp;BatchNormProteinEncode1/moving_variance/Read/ReadVariableOp,EmbeddingDimDense/kernel/Read/ReadVariableOp*EmbeddingDimDense/bias/Read/ReadVariableOp)gene_decoder_1/kernel/Read/ReadVariableOp'gene_decoder_1/bias/Read/ReadVariableOp.BatchNormGeneDecode1/gamma/Read/ReadVariableOp-BatchNormGeneDecode1/beta/Read/ReadVariableOp4BatchNormGeneDecode1/moving_mean/Read/ReadVariableOp8BatchNormGeneDecode1/moving_variance/Read/ReadVariableOp)gene_decoder_2/kernel/Read/ReadVariableOp'gene_decoder_2/bias/Read/ReadVariableOp,protein_decoder_1/kernel/Read/ReadVariableOp*protein_decoder_1/bias/Read/ReadVariableOp.BatchNormGeneDecode2/gamma/Read/ReadVariableOp-BatchNormGeneDecode2/beta/Read/ReadVariableOp4BatchNormGeneDecode2/moving_mean/Read/ReadVariableOp8BatchNormGeneDecode2/moving_variance/Read/ReadVariableOp1BatchNormProteinDecode1/gamma/Read/ReadVariableOp0BatchNormProteinDecode1/beta/Read/ReadVariableOp7BatchNormProteinDecode1/moving_mean/Read/ReadVariableOp;BatchNormProteinDecode1/moving_variance/Read/ReadVariableOp,gene_decoder_last/kernel/Read/ReadVariableOp*gene_decoder_last/bias/Read/ReadVariableOp/protein_decoder_last/kernel/Read/ReadVariableOp-protein_decoder_last/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp0Adam/gene_encoder_1/kernel/m/Read/ReadVariableOp.Adam/gene_encoder_1/bias/m/Read/ReadVariableOp5Adam/BatchNormGeneEncode1/gamma/m/Read/ReadVariableOp4Adam/BatchNormGeneEncode1/beta/m/Read/ReadVariableOp0Adam/gene_encoder_2/kernel/m/Read/ReadVariableOp.Adam/gene_encoder_2/bias/m/Read/ReadVariableOp3Adam/protein_encoder_1/kernel/m/Read/ReadVariableOp1Adam/protein_encoder_1/bias/m/Read/ReadVariableOp5Adam/BatchNormGeneEncode2/gamma/m/Read/ReadVariableOp4Adam/BatchNormGeneEncode2/beta/m/Read/ReadVariableOp8Adam/BatchNormProteinEncode1/gamma/m/Read/ReadVariableOp7Adam/BatchNormProteinEncode1/beta/m/Read/ReadVariableOp3Adam/EmbeddingDimDense/kernel/m/Read/ReadVariableOp1Adam/EmbeddingDimDense/bias/m/Read/ReadVariableOp0Adam/gene_decoder_1/kernel/m/Read/ReadVariableOp.Adam/gene_decoder_1/bias/m/Read/ReadVariableOp5Adam/BatchNormGeneDecode1/gamma/m/Read/ReadVariableOp4Adam/BatchNormGeneDecode1/beta/m/Read/ReadVariableOp0Adam/gene_decoder_2/kernel/m/Read/ReadVariableOp.Adam/gene_decoder_2/bias/m/Read/ReadVariableOp3Adam/protein_decoder_1/kernel/m/Read/ReadVariableOp1Adam/protein_decoder_1/bias/m/Read/ReadVariableOp5Adam/BatchNormGeneDecode2/gamma/m/Read/ReadVariableOp4Adam/BatchNormGeneDecode2/beta/m/Read/ReadVariableOp8Adam/BatchNormProteinDecode1/gamma/m/Read/ReadVariableOp7Adam/BatchNormProteinDecode1/beta/m/Read/ReadVariableOp3Adam/gene_decoder_last/kernel/m/Read/ReadVariableOp1Adam/gene_decoder_last/bias/m/Read/ReadVariableOp6Adam/protein_decoder_last/kernel/m/Read/ReadVariableOp4Adam/protein_decoder_last/bias/m/Read/ReadVariableOp0Adam/gene_encoder_1/kernel/v/Read/ReadVariableOp.Adam/gene_encoder_1/bias/v/Read/ReadVariableOp5Adam/BatchNormGeneEncode1/gamma/v/Read/ReadVariableOp4Adam/BatchNormGeneEncode1/beta/v/Read/ReadVariableOp0Adam/gene_encoder_2/kernel/v/Read/ReadVariableOp.Adam/gene_encoder_2/bias/v/Read/ReadVariableOp3Adam/protein_encoder_1/kernel/v/Read/ReadVariableOp1Adam/protein_encoder_1/bias/v/Read/ReadVariableOp5Adam/BatchNormGeneEncode2/gamma/v/Read/ReadVariableOp4Adam/BatchNormGeneEncode2/beta/v/Read/ReadVariableOp8Adam/BatchNormProteinEncode1/gamma/v/Read/ReadVariableOp7Adam/BatchNormProteinEncode1/beta/v/Read/ReadVariableOp3Adam/EmbeddingDimDense/kernel/v/Read/ReadVariableOp1Adam/EmbeddingDimDense/bias/v/Read/ReadVariableOp0Adam/gene_decoder_1/kernel/v/Read/ReadVariableOp.Adam/gene_decoder_1/bias/v/Read/ReadVariableOp5Adam/BatchNormGeneDecode1/gamma/v/Read/ReadVariableOp4Adam/BatchNormGeneDecode1/beta/v/Read/ReadVariableOp0Adam/gene_decoder_2/kernel/v/Read/ReadVariableOp.Adam/gene_decoder_2/bias/v/Read/ReadVariableOp3Adam/protein_decoder_1/kernel/v/Read/ReadVariableOp1Adam/protein_decoder_1/bias/v/Read/ReadVariableOp5Adam/BatchNormGeneDecode2/gamma/v/Read/ReadVariableOp4Adam/BatchNormGeneDecode2/beta/v/Read/ReadVariableOp8Adam/BatchNormProteinDecode1/gamma/v/Read/ReadVariableOp7Adam/BatchNormProteinDecode1/beta/v/Read/ReadVariableOp3Adam/gene_decoder_last/kernel/v/Read/ReadVariableOp1Adam/gene_decoder_last/bias/v/Read/ReadVariableOp6Adam/protein_decoder_last/kernel/v/Read/ReadVariableOp4Adam/protein_decoder_last/bias/v/Read/ReadVariableOpConst*~
Tinw
u2s	*
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
 __inference__traced_save_1900185
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegene_encoder_1/kernelgene_encoder_1/biasBatchNormGeneEncode1/gammaBatchNormGeneEncode1/beta BatchNormGeneEncode1/moving_mean$BatchNormGeneEncode1/moving_variancegene_encoder_2/kernelgene_encoder_2/biasprotein_encoder_1/kernelprotein_encoder_1/biasBatchNormGeneEncode2/gammaBatchNormGeneEncode2/beta BatchNormGeneEncode2/moving_mean$BatchNormGeneEncode2/moving_varianceBatchNormProteinEncode1/gammaBatchNormProteinEncode1/beta#BatchNormProteinEncode1/moving_mean'BatchNormProteinEncode1/moving_varianceEmbeddingDimDense/kernelEmbeddingDimDense/biasgene_decoder_1/kernelgene_decoder_1/biasBatchNormGeneDecode1/gammaBatchNormGeneDecode1/beta BatchNormGeneDecode1/moving_mean$BatchNormGeneDecode1/moving_variancegene_decoder_2/kernelgene_decoder_2/biasprotein_decoder_1/kernelprotein_decoder_1/biasBatchNormGeneDecode2/gammaBatchNormGeneDecode2/beta BatchNormGeneDecode2/moving_mean$BatchNormGeneDecode2/moving_varianceBatchNormProteinDecode1/gammaBatchNormProteinDecode1/beta#BatchNormProteinDecode1/moving_mean'BatchNormProteinDecode1/moving_variancegene_decoder_last/kernelgene_decoder_last/biasprotein_decoder_last/kernelprotein_decoder_last/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_2count_2total_1count_1totalcountAdam/gene_encoder_1/kernel/mAdam/gene_encoder_1/bias/m!Adam/BatchNormGeneEncode1/gamma/m Adam/BatchNormGeneEncode1/beta/mAdam/gene_encoder_2/kernel/mAdam/gene_encoder_2/bias/mAdam/protein_encoder_1/kernel/mAdam/protein_encoder_1/bias/m!Adam/BatchNormGeneEncode2/gamma/m Adam/BatchNormGeneEncode2/beta/m$Adam/BatchNormProteinEncode1/gamma/m#Adam/BatchNormProteinEncode1/beta/mAdam/EmbeddingDimDense/kernel/mAdam/EmbeddingDimDense/bias/mAdam/gene_decoder_1/kernel/mAdam/gene_decoder_1/bias/m!Adam/BatchNormGeneDecode1/gamma/m Adam/BatchNormGeneDecode1/beta/mAdam/gene_decoder_2/kernel/mAdam/gene_decoder_2/bias/mAdam/protein_decoder_1/kernel/mAdam/protein_decoder_1/bias/m!Adam/BatchNormGeneDecode2/gamma/m Adam/BatchNormGeneDecode2/beta/m$Adam/BatchNormProteinDecode1/gamma/m#Adam/BatchNormProteinDecode1/beta/mAdam/gene_decoder_last/kernel/mAdam/gene_decoder_last/bias/m"Adam/protein_decoder_last/kernel/m Adam/protein_decoder_last/bias/mAdam/gene_encoder_1/kernel/vAdam/gene_encoder_1/bias/v!Adam/BatchNormGeneEncode1/gamma/v Adam/BatchNormGeneEncode1/beta/vAdam/gene_encoder_2/kernel/vAdam/gene_encoder_2/bias/vAdam/protein_encoder_1/kernel/vAdam/protein_encoder_1/bias/v!Adam/BatchNormGeneEncode2/gamma/v Adam/BatchNormGeneEncode2/beta/v$Adam/BatchNormProteinEncode1/gamma/v#Adam/BatchNormProteinEncode1/beta/vAdam/EmbeddingDimDense/kernel/vAdam/EmbeddingDimDense/bias/vAdam/gene_decoder_1/kernel/vAdam/gene_decoder_1/bias/v!Adam/BatchNormGeneDecode1/gamma/v Adam/BatchNormGeneDecode1/beta/vAdam/gene_decoder_2/kernel/vAdam/gene_decoder_2/bias/vAdam/protein_decoder_1/kernel/vAdam/protein_decoder_1/bias/v!Adam/BatchNormGeneDecode2/gamma/v Adam/BatchNormGeneDecode2/beta/v$Adam/BatchNormProteinDecode1/gamma/v#Adam/BatchNormProteinDecode1/beta/vAdam/gene_decoder_last/kernel/vAdam/gene_decoder_last/bias/v"Adam/protein_decoder_last/kernel/v Adam/protein_decoder_last/bias/v*}
Tinv
t2r*
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
#__inference__traced_restore_1900534��
�
�
T__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_1897381

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
Q__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_1897346

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
9__inference_BatchNormProteinDecode1_layer_call_fn_1899714

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_1897381o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1899448

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
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
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
3__inference_protein_decoder_1_layer_call_fn_1899610

inputs
unknown:@
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_protein_decoder_1_layer_call_and_return_conditional_losses_1897589o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
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
�
�
6__inference_BatchNormGeneEncode2_layer_call_fn_1899301

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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1897053o
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
6__inference_BatchNormGeneDecode1_layer_call_fn_1899514

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
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_1897217o
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
�
y
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1899461
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
&:���������}:���������:Q M
'
_output_shapes
:���������}
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�%
�
T__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_1897428

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
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
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
3__inference_gene_decoder_last_layer_call_fn_1899790

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
GPU 2J 8� *W
fRRP
N__inference_gene_decoder_last_layer_call_and_return_conditional_losses_1897658p
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
�
�
6__inference_BatchNormGeneEncode2_layer_call_fn_1899314

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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1897100o
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
Q__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_1899701

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
�e
�
E__inference_model_40_layer_call_and_return_conditional_losses_1898053

inputs
inputs_1*
gene_encoder_1_1897951:
��%
gene_encoder_1_1897953:	�+
batchnormgeneencode1_1897956:	�+
batchnormgeneencode1_1897958:	�+
batchnormgeneencode1_1897960:	�+
batchnormgeneencode1_1897962:	�+
protein_encoder_1_1897965:W'
protein_encoder_1_1897967:)
gene_encoder_2_1897970:	�}$
gene_encoder_2_1897972:}*
batchnormgeneencode2_1897975:}*
batchnormgeneencode2_1897977:}*
batchnormgeneencode2_1897979:}*
batchnormgeneencode2_1897981:}-
batchnormproteinencode1_1897984:-
batchnormproteinencode1_1897986:-
batchnormproteinencode1_1897988:-
batchnormproteinencode1_1897990:,
embeddingdimdense_1897994:	�@'
embeddingdimdense_1897996:@(
gene_decoder_1_1897999:@}$
gene_decoder_1_1898001:}*
batchnormgenedecode1_1898004:}*
batchnormgenedecode1_1898006:}*
batchnormgenedecode1_1898008:}*
batchnormgenedecode1_1898010:}+
protein_decoder_1_1898013:@'
protein_decoder_1_1898015:)
gene_decoder_2_1898018:	}�%
gene_decoder_2_1898020:	�-
batchnormproteindecode1_1898023:-
batchnormproteindecode1_1898025:-
batchnormproteindecode1_1898027:-
batchnormproteindecode1_1898029:+
batchnormgenedecode2_1898032:	�+
batchnormgenedecode2_1898034:	�+
batchnormgenedecode2_1898036:	�+
batchnormgenedecode2_1898038:	�.
protein_decoder_last_1898041:W*
protein_decoder_last_1898043:W-
gene_decoder_last_1898046:
��(
gene_decoder_last_1898048:	�
identity

identity_1��,BatchNormGeneDecode1/StatefulPartitionedCall�,BatchNormGeneDecode2/StatefulPartitionedCall�,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�/BatchNormProteinDecode1/StatefulPartitionedCall�/BatchNormProteinEncode1/StatefulPartitionedCall�)EmbeddingDimDense/StatefulPartitionedCall�&gene_decoder_1/StatefulPartitionedCall�&gene_decoder_2/StatefulPartitionedCall�)gene_decoder_last/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�)protein_decoder_1/StatefulPartitionedCall�,protein_decoder_last/StatefulPartitionedCall�)protein_encoder_1/StatefulPartitionedCall�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputsgene_encoder_1_1897951gene_encoder_1_1897953*
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
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_1897459�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_1897956batchnormgeneencode1_1897958batchnormgeneencode1_1897960batchnormgeneencode1_1897962*
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
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1897018�
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1protein_encoder_1_1897965protein_encoder_1_1897967*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_protein_encoder_1_layer_call_and_return_conditional_losses_1897485�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_1897970gene_encoder_2_1897972*
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
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_1897502�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_1897975batchnormgeneencode2_1897977batchnormgeneencode2_1897979batchnormgeneencode2_1897981*
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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1897100�
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_1897984batchnormproteinencode1_1897986batchnormproteinencode1_1897988batchnormproteinencode1_1897990*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1897182�
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
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1897533�
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_1897994embeddingdimdense_1897996*
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
GPU 2J 8� *W
fRRP
N__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_1897546�
&gene_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0gene_decoder_1_1897999gene_decoder_1_1898001*
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
K__inference_gene_decoder_1_layer_call_and_return_conditional_losses_1897563�
,BatchNormGeneDecode1/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_1/StatefulPartitionedCall:output:0batchnormgenedecode1_1898004batchnormgenedecode1_1898006batchnormgenedecode1_1898008batchnormgenedecode1_1898010*
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
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_1897264�
)protein_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0protein_decoder_1_1898013protein_decoder_1_1898015*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_protein_decoder_1_layer_call_and_return_conditional_losses_1897589�
&gene_decoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode1/StatefulPartitionedCall:output:0gene_decoder_2_1898018gene_decoder_2_1898020*
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
K__inference_gene_decoder_2_layer_call_and_return_conditional_losses_1897606�
/BatchNormProteinDecode1/StatefulPartitionedCallStatefulPartitionedCall2protein_decoder_1/StatefulPartitionedCall:output:0batchnormproteindecode1_1898023batchnormproteindecode1_1898025batchnormproteindecode1_1898027batchnormproteindecode1_1898029*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_1897428�
,BatchNormGeneDecode2/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_2/StatefulPartitionedCall:output:0batchnormgenedecode2_1898032batchnormgenedecode2_1898034batchnormgenedecode2_1898036batchnormgenedecode2_1898038*
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
Q__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_1897346�
,protein_decoder_last/StatefulPartitionedCallStatefulPartitionedCall8BatchNormProteinDecode1/StatefulPartitionedCall:output:0protein_decoder_last_1898041protein_decoder_last_1898043*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������W*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_protein_decoder_last_layer_call_and_return_conditional_losses_1897641�
)gene_decoder_last/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode2/StatefulPartitionedCall:output:0gene_decoder_last_1898046gene_decoder_last_1898048*
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
GPU 2J 8� *W
fRRP
N__inference_gene_decoder_last_layer_call_and_return_conditional_losses_1897658�
IdentityIdentity2gene_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������

Identity_1Identity5protein_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������W�
NoOpNoOp-^BatchNormGeneDecode1/StatefulPartitionedCall-^BatchNormGeneDecode2/StatefulPartitionedCall-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinDecode1/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_decoder_1/StatefulPartitionedCall'^gene_decoder_2/StatefulPartitionedCall*^gene_decoder_last/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_decoder_1/StatefulPartitionedCall-^protein_decoder_last/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes}
{:����������:���������W: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
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
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������W
 
_user_specified_nameinputs
�f
�
E__inference_model_40_layer_call_and_return_conditional_losses_1897666

inputs
inputs_1*
gene_encoder_1_1897460:
��%
gene_encoder_1_1897462:	�+
batchnormgeneencode1_1897465:	�+
batchnormgeneencode1_1897467:	�+
batchnormgeneencode1_1897469:	�+
batchnormgeneencode1_1897471:	�+
protein_encoder_1_1897486:W'
protein_encoder_1_1897488:)
gene_encoder_2_1897503:	�}$
gene_encoder_2_1897505:}*
batchnormgeneencode2_1897508:}*
batchnormgeneencode2_1897510:}*
batchnormgeneencode2_1897512:}*
batchnormgeneencode2_1897514:}-
batchnormproteinencode1_1897517:-
batchnormproteinencode1_1897519:-
batchnormproteinencode1_1897521:-
batchnormproteinencode1_1897523:,
embeddingdimdense_1897547:	�@'
embeddingdimdense_1897549:@(
gene_decoder_1_1897564:@}$
gene_decoder_1_1897566:}*
batchnormgenedecode1_1897569:}*
batchnormgenedecode1_1897571:}*
batchnormgenedecode1_1897573:}*
batchnormgenedecode1_1897575:}+
protein_decoder_1_1897590:@'
protein_decoder_1_1897592:)
gene_decoder_2_1897607:	}�%
gene_decoder_2_1897609:	�-
batchnormproteindecode1_1897612:-
batchnormproteindecode1_1897614:-
batchnormproteindecode1_1897616:-
batchnormproteindecode1_1897618:+
batchnormgenedecode2_1897621:	�+
batchnormgenedecode2_1897623:	�+
batchnormgenedecode2_1897625:	�+
batchnormgenedecode2_1897627:	�.
protein_decoder_last_1897642:W*
protein_decoder_last_1897644:W-
gene_decoder_last_1897659:
��(
gene_decoder_last_1897661:	�
identity

identity_1��,BatchNormGeneDecode1/StatefulPartitionedCall�,BatchNormGeneDecode2/StatefulPartitionedCall�,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�/BatchNormProteinDecode1/StatefulPartitionedCall�/BatchNormProteinEncode1/StatefulPartitionedCall�)EmbeddingDimDense/StatefulPartitionedCall�&gene_decoder_1/StatefulPartitionedCall�&gene_decoder_2/StatefulPartitionedCall�)gene_decoder_last/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�)protein_decoder_1/StatefulPartitionedCall�,protein_decoder_last/StatefulPartitionedCall�)protein_encoder_1/StatefulPartitionedCall�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputsgene_encoder_1_1897460gene_encoder_1_1897462*
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
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_1897459�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_1897465batchnormgeneencode1_1897467batchnormgeneencode1_1897469batchnormgeneencode1_1897471*
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
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1896971�
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1protein_encoder_1_1897486protein_encoder_1_1897488*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_protein_encoder_1_layer_call_and_return_conditional_losses_1897485�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_1897503gene_encoder_2_1897505*
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
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_1897502�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_1897508batchnormgeneencode2_1897510batchnormgeneencode2_1897512batchnormgeneencode2_1897514*
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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1897053�
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_1897517batchnormproteinencode1_1897519batchnormproteinencode1_1897521batchnormproteinencode1_1897523*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1897135�
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
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1897533�
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_1897547embeddingdimdense_1897549*
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
GPU 2J 8� *W
fRRP
N__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_1897546�
&gene_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0gene_decoder_1_1897564gene_decoder_1_1897566*
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
K__inference_gene_decoder_1_layer_call_and_return_conditional_losses_1897563�
,BatchNormGeneDecode1/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_1/StatefulPartitionedCall:output:0batchnormgenedecode1_1897569batchnormgenedecode1_1897571batchnormgenedecode1_1897573batchnormgenedecode1_1897575*
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
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_1897217�
)protein_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0protein_decoder_1_1897590protein_decoder_1_1897592*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_protein_decoder_1_layer_call_and_return_conditional_losses_1897589�
&gene_decoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode1/StatefulPartitionedCall:output:0gene_decoder_2_1897607gene_decoder_2_1897609*
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
K__inference_gene_decoder_2_layer_call_and_return_conditional_losses_1897606�
/BatchNormProteinDecode1/StatefulPartitionedCallStatefulPartitionedCall2protein_decoder_1/StatefulPartitionedCall:output:0batchnormproteindecode1_1897612batchnormproteindecode1_1897614batchnormproteindecode1_1897616batchnormproteindecode1_1897618*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_1897381�
,BatchNormGeneDecode2/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_2/StatefulPartitionedCall:output:0batchnormgenedecode2_1897621batchnormgenedecode2_1897623batchnormgenedecode2_1897625batchnormgenedecode2_1897627*
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
Q__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_1897299�
,protein_decoder_last/StatefulPartitionedCallStatefulPartitionedCall8BatchNormProteinDecode1/StatefulPartitionedCall:output:0protein_decoder_last_1897642protein_decoder_last_1897644*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������W*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_protein_decoder_last_layer_call_and_return_conditional_losses_1897641�
)gene_decoder_last/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode2/StatefulPartitionedCall:output:0gene_decoder_last_1897659gene_decoder_last_1897661*
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
GPU 2J 8� *W
fRRP
N__inference_gene_decoder_last_layer_call_and_return_conditional_losses_1897658�
IdentityIdentity2gene_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������

Identity_1Identity5protein_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������W�
NoOpNoOp-^BatchNormGeneDecode1/StatefulPartitionedCall-^BatchNormGeneDecode2/StatefulPartitionedCall-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinDecode1/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_decoder_1/StatefulPartitionedCall'^gene_decoder_2/StatefulPartitionedCall*^gene_decoder_last/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_decoder_1/StatefulPartitionedCall-^protein_decoder_last/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes}
{:����������:���������W: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
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
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������W
 
_user_specified_nameinputs
��
�6
 __inference__traced_save_1900185
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
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop;
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_gene_encoder_1_kernel_read_readvariableop.savev2_gene_encoder_1_bias_read_readvariableop5savev2_batchnormgeneencode1_gamma_read_readvariableop4savev2_batchnormgeneencode1_beta_read_readvariableop;savev2_batchnormgeneencode1_moving_mean_read_readvariableop?savev2_batchnormgeneencode1_moving_variance_read_readvariableop0savev2_gene_encoder_2_kernel_read_readvariableop.savev2_gene_encoder_2_bias_read_readvariableop3savev2_protein_encoder_1_kernel_read_readvariableop1savev2_protein_encoder_1_bias_read_readvariableop5savev2_batchnormgeneencode2_gamma_read_readvariableop4savev2_batchnormgeneencode2_beta_read_readvariableop;savev2_batchnormgeneencode2_moving_mean_read_readvariableop?savev2_batchnormgeneencode2_moving_variance_read_readvariableop8savev2_batchnormproteinencode1_gamma_read_readvariableop7savev2_batchnormproteinencode1_beta_read_readvariableop>savev2_batchnormproteinencode1_moving_mean_read_readvariableopBsavev2_batchnormproteinencode1_moving_variance_read_readvariableop3savev2_embeddingdimdense_kernel_read_readvariableop1savev2_embeddingdimdense_bias_read_readvariableop0savev2_gene_decoder_1_kernel_read_readvariableop.savev2_gene_decoder_1_bias_read_readvariableop5savev2_batchnormgenedecode1_gamma_read_readvariableop4savev2_batchnormgenedecode1_beta_read_readvariableop;savev2_batchnormgenedecode1_moving_mean_read_readvariableop?savev2_batchnormgenedecode1_moving_variance_read_readvariableop0savev2_gene_decoder_2_kernel_read_readvariableop.savev2_gene_decoder_2_bias_read_readvariableop3savev2_protein_decoder_1_kernel_read_readvariableop1savev2_protein_decoder_1_bias_read_readvariableop5savev2_batchnormgenedecode2_gamma_read_readvariableop4savev2_batchnormgenedecode2_beta_read_readvariableop;savev2_batchnormgenedecode2_moving_mean_read_readvariableop?savev2_batchnormgenedecode2_moving_variance_read_readvariableop8savev2_batchnormproteindecode1_gamma_read_readvariableop7savev2_batchnormproteindecode1_beta_read_readvariableop>savev2_batchnormproteindecode1_moving_mean_read_readvariableopBsavev2_batchnormproteindecode1_moving_variance_read_readvariableop3savev2_gene_decoder_last_kernel_read_readvariableop1savev2_gene_decoder_last_bias_read_readvariableop6savev2_protein_decoder_last_kernel_read_readvariableop4savev2_protein_decoder_last_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop7savev2_adam_gene_encoder_1_kernel_m_read_readvariableop5savev2_adam_gene_encoder_1_bias_m_read_readvariableop<savev2_adam_batchnormgeneencode1_gamma_m_read_readvariableop;savev2_adam_batchnormgeneencode1_beta_m_read_readvariableop7savev2_adam_gene_encoder_2_kernel_m_read_readvariableop5savev2_adam_gene_encoder_2_bias_m_read_readvariableop:savev2_adam_protein_encoder_1_kernel_m_read_readvariableop8savev2_adam_protein_encoder_1_bias_m_read_readvariableop<savev2_adam_batchnormgeneencode2_gamma_m_read_readvariableop;savev2_adam_batchnormgeneencode2_beta_m_read_readvariableop?savev2_adam_batchnormproteinencode1_gamma_m_read_readvariableop>savev2_adam_batchnormproteinencode1_beta_m_read_readvariableop:savev2_adam_embeddingdimdense_kernel_m_read_readvariableop8savev2_adam_embeddingdimdense_bias_m_read_readvariableop7savev2_adam_gene_decoder_1_kernel_m_read_readvariableop5savev2_adam_gene_decoder_1_bias_m_read_readvariableop<savev2_adam_batchnormgenedecode1_gamma_m_read_readvariableop;savev2_adam_batchnormgenedecode1_beta_m_read_readvariableop7savev2_adam_gene_decoder_2_kernel_m_read_readvariableop5savev2_adam_gene_decoder_2_bias_m_read_readvariableop:savev2_adam_protein_decoder_1_kernel_m_read_readvariableop8savev2_adam_protein_decoder_1_bias_m_read_readvariableop<savev2_adam_batchnormgenedecode2_gamma_m_read_readvariableop;savev2_adam_batchnormgenedecode2_beta_m_read_readvariableop?savev2_adam_batchnormproteindecode1_gamma_m_read_readvariableop>savev2_adam_batchnormproteindecode1_beta_m_read_readvariableop:savev2_adam_gene_decoder_last_kernel_m_read_readvariableop8savev2_adam_gene_decoder_last_bias_m_read_readvariableop=savev2_adam_protein_decoder_last_kernel_m_read_readvariableop;savev2_adam_protein_decoder_last_bias_m_read_readvariableop7savev2_adam_gene_encoder_1_kernel_v_read_readvariableop5savev2_adam_gene_encoder_1_bias_v_read_readvariableop<savev2_adam_batchnormgeneencode1_gamma_v_read_readvariableop;savev2_adam_batchnormgeneencode1_beta_v_read_readvariableop7savev2_adam_gene_encoder_2_kernel_v_read_readvariableop5savev2_adam_gene_encoder_2_bias_v_read_readvariableop:savev2_adam_protein_encoder_1_kernel_v_read_readvariableop8savev2_adam_protein_encoder_1_bias_v_read_readvariableop<savev2_adam_batchnormgeneencode2_gamma_v_read_readvariableop;savev2_adam_batchnormgeneencode2_beta_v_read_readvariableop?savev2_adam_batchnormproteinencode1_gamma_v_read_readvariableop>savev2_adam_batchnormproteinencode1_beta_v_read_readvariableop:savev2_adam_embeddingdimdense_kernel_v_read_readvariableop8savev2_adam_embeddingdimdense_bias_v_read_readvariableop7savev2_adam_gene_decoder_1_kernel_v_read_readvariableop5savev2_adam_gene_decoder_1_bias_v_read_readvariableop<savev2_adam_batchnormgenedecode1_gamma_v_read_readvariableop;savev2_adam_batchnormgenedecode1_beta_v_read_readvariableop7savev2_adam_gene_decoder_2_kernel_v_read_readvariableop5savev2_adam_gene_decoder_2_bias_v_read_readvariableop:savev2_adam_protein_decoder_1_kernel_v_read_readvariableop8savev2_adam_protein_decoder_1_bias_v_read_readvariableop<savev2_adam_batchnormgenedecode2_gamma_v_read_readvariableop;savev2_adam_batchnormgenedecode2_beta_v_read_readvariableop?savev2_adam_batchnormproteindecode1_gamma_v_read_readvariableop>savev2_adam_batchnormproteindecode1_beta_v_read_readvariableop:savev2_adam_gene_decoder_last_kernel_v_read_readvariableop8savev2_adam_gene_decoder_last_bias_v_read_readvariableop=savev2_adam_protein_decoder_last_kernel_v_read_readvariableop;savev2_adam_protein_decoder_last_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
��:�:�:�:�:�:	�}:}:W::}:}:}:}:::::	�@:@:@}:}:}:}:}:}:	}�:�:@::�:�:�:�:::::
��:�:W:W: : : : : : : : : : : :
��:�:�:�:	�}:}:W::}:}:::	�@:@:@}:}:}:}:	}�:�:@::�:�:::
��:�:W:W:
��:�:�:�:	�}:}:W::}:}:::	�@:@:@}:}:}:}:	}�:�:@::�:�:::
��:�:W:W: 2(
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
:}:$	 

_output_shapes

:W: 


_output_shapes
:: 
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
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::%!

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

:@: 

_output_shapes
::!
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
:: $

_output_shapes
:: %

_output_shapes
:: &

_output_shapes
::&'"
 
_output_shapes
:
��:!(

_output_shapes	
:�:$) 

_output_shapes

:W: *

_output_shapes
:W:+
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
:}:$< 

_output_shapes

:W: =

_output_shapes
:: >

_output_shapes
:}: ?

_output_shapes
:}: @

_output_shapes
:: A

_output_shapes
::%B!

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

:@: K

_output_shapes
::!L

_output_shapes	
:�:!M

_output_shapes	
:�: N

_output_shapes
:: O

_output_shapes
::&P"
 
_output_shapes
:
��:!Q

_output_shapes	
:�:$R 

_output_shapes

:W: S

_output_shapes
:W:&T"
 
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
:}:$Z 

_output_shapes

:W: [

_output_shapes
:: \

_output_shapes
:}: ]

_output_shapes
:}: ^

_output_shapes
:: _

_output_shapes
::%`!

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

:@: i

_output_shapes
::!j

_output_shapes	
:�:!k

_output_shapes	
:�: l

_output_shapes
:: m

_output_shapes
::&n"
 
_output_shapes
:
��:!o

_output_shapes	
:�:$p 

_output_shapes

:W: q

_output_shapes
:W:r

_output_shapes
: 
�%
�
T__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_1899781

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
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
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1897135

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
��
�L
#__inference__traced_restore_1900534
file_prefix:
&assignvariableop_gene_encoder_1_kernel:
��5
&assignvariableop_1_gene_encoder_1_bias:	�<
-assignvariableop_2_batchnormgeneencode1_gamma:	�;
,assignvariableop_3_batchnormgeneencode1_beta:	�B
3assignvariableop_4_batchnormgeneencode1_moving_mean:	�F
7assignvariableop_5_batchnormgeneencode1_moving_variance:	�;
(assignvariableop_6_gene_encoder_2_kernel:	�}4
&assignvariableop_7_gene_encoder_2_bias:}=
+assignvariableop_8_protein_encoder_1_kernel:W7
)assignvariableop_9_protein_encoder_1_bias:<
.assignvariableop_10_batchnormgeneencode2_gamma:};
-assignvariableop_11_batchnormgeneencode2_beta:}B
4assignvariableop_12_batchnormgeneencode2_moving_mean:}F
8assignvariableop_13_batchnormgeneencode2_moving_variance:}?
1assignvariableop_14_batchnormproteinencode1_gamma:>
0assignvariableop_15_batchnormproteinencode1_beta:E
7assignvariableop_16_batchnormproteinencode1_moving_mean:I
;assignvariableop_17_batchnormproteinencode1_moving_variance:?
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
,assignvariableop_28_protein_decoder_1_kernel:@8
*assignvariableop_29_protein_decoder_1_bias:=
.assignvariableop_30_batchnormgenedecode2_gamma:	�<
-assignvariableop_31_batchnormgenedecode2_beta:	�C
4assignvariableop_32_batchnormgenedecode2_moving_mean:	�G
8assignvariableop_33_batchnormgenedecode2_moving_variance:	�?
1assignvariableop_34_batchnormproteindecode1_gamma:>
0assignvariableop_35_batchnormproteindecode1_beta:E
7assignvariableop_36_batchnormproteindecode1_moving_mean:I
;assignvariableop_37_batchnormproteindecode1_moving_variance:@
,assignvariableop_38_gene_decoder_last_kernel:
��9
*assignvariableop_39_gene_decoder_last_bias:	�A
/assignvariableop_40_protein_decoder_last_kernel:W;
-assignvariableop_41_protein_decoder_last_bias:W'
assignvariableop_42_adam_iter:	 )
assignvariableop_43_adam_beta_1: )
assignvariableop_44_adam_beta_2: (
assignvariableop_45_adam_decay: 0
&assignvariableop_46_adam_learning_rate: %
assignvariableop_47_total_2: %
assignvariableop_48_count_2: %
assignvariableop_49_total_1: %
assignvariableop_50_count_1: #
assignvariableop_51_total: #
assignvariableop_52_count: D
0assignvariableop_53_adam_gene_encoder_1_kernel_m:
��=
.assignvariableop_54_adam_gene_encoder_1_bias_m:	�D
5assignvariableop_55_adam_batchnormgeneencode1_gamma_m:	�C
4assignvariableop_56_adam_batchnormgeneencode1_beta_m:	�C
0assignvariableop_57_adam_gene_encoder_2_kernel_m:	�}<
.assignvariableop_58_adam_gene_encoder_2_bias_m:}E
3assignvariableop_59_adam_protein_encoder_1_kernel_m:W?
1assignvariableop_60_adam_protein_encoder_1_bias_m:C
5assignvariableop_61_adam_batchnormgeneencode2_gamma_m:}B
4assignvariableop_62_adam_batchnormgeneencode2_beta_m:}F
8assignvariableop_63_adam_batchnormproteinencode1_gamma_m:E
7assignvariableop_64_adam_batchnormproteinencode1_beta_m:F
3assignvariableop_65_adam_embeddingdimdense_kernel_m:	�@?
1assignvariableop_66_adam_embeddingdimdense_bias_m:@B
0assignvariableop_67_adam_gene_decoder_1_kernel_m:@}<
.assignvariableop_68_adam_gene_decoder_1_bias_m:}C
5assignvariableop_69_adam_batchnormgenedecode1_gamma_m:}B
4assignvariableop_70_adam_batchnormgenedecode1_beta_m:}C
0assignvariableop_71_adam_gene_decoder_2_kernel_m:	}�=
.assignvariableop_72_adam_gene_decoder_2_bias_m:	�E
3assignvariableop_73_adam_protein_decoder_1_kernel_m:@?
1assignvariableop_74_adam_protein_decoder_1_bias_m:D
5assignvariableop_75_adam_batchnormgenedecode2_gamma_m:	�C
4assignvariableop_76_adam_batchnormgenedecode2_beta_m:	�F
8assignvariableop_77_adam_batchnormproteindecode1_gamma_m:E
7assignvariableop_78_adam_batchnormproteindecode1_beta_m:G
3assignvariableop_79_adam_gene_decoder_last_kernel_m:
��@
1assignvariableop_80_adam_gene_decoder_last_bias_m:	�H
6assignvariableop_81_adam_protein_decoder_last_kernel_m:WB
4assignvariableop_82_adam_protein_decoder_last_bias_m:WD
0assignvariableop_83_adam_gene_encoder_1_kernel_v:
��=
.assignvariableop_84_adam_gene_encoder_1_bias_v:	�D
5assignvariableop_85_adam_batchnormgeneencode1_gamma_v:	�C
4assignvariableop_86_adam_batchnormgeneencode1_beta_v:	�C
0assignvariableop_87_adam_gene_encoder_2_kernel_v:	�}<
.assignvariableop_88_adam_gene_encoder_2_bias_v:}E
3assignvariableop_89_adam_protein_encoder_1_kernel_v:W?
1assignvariableop_90_adam_protein_encoder_1_bias_v:C
5assignvariableop_91_adam_batchnormgeneencode2_gamma_v:}B
4assignvariableop_92_adam_batchnormgeneencode2_beta_v:}F
8assignvariableop_93_adam_batchnormproteinencode1_gamma_v:E
7assignvariableop_94_adam_batchnormproteinencode1_beta_v:F
3assignvariableop_95_adam_embeddingdimdense_kernel_v:	�@?
1assignvariableop_96_adam_embeddingdimdense_bias_v:@B
0assignvariableop_97_adam_gene_decoder_1_kernel_v:@}<
.assignvariableop_98_adam_gene_decoder_1_bias_v:}C
5assignvariableop_99_adam_batchnormgenedecode1_gamma_v:}C
5assignvariableop_100_adam_batchnormgenedecode1_beta_v:}D
1assignvariableop_101_adam_gene_decoder_2_kernel_v:	}�>
/assignvariableop_102_adam_gene_decoder_2_bias_v:	�F
4assignvariableop_103_adam_protein_decoder_1_kernel_v:@@
2assignvariableop_104_adam_protein_decoder_1_bias_v:E
6assignvariableop_105_adam_batchnormgenedecode2_gamma_v:	�D
5assignvariableop_106_adam_batchnormgenedecode2_beta_v:	�G
9assignvariableop_107_adam_batchnormproteindecode1_gamma_v:F
8assignvariableop_108_adam_batchnormproteindecode1_beta_v:H
4assignvariableop_109_adam_gene_decoder_last_kernel_v:
��A
2assignvariableop_110_adam_gene_decoder_last_bias_v:	�I
7assignvariableop_111_adam_protein_decoder_last_kernel_v:WC
5assignvariableop_112_adam_protein_decoder_last_bias_v:W
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
AssignVariableOp_47AssignVariableOpassignvariableop_47_total_2Identity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOpassignvariableop_48_count_2Identity_48:output:0"/device:CPU:0*
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
AssignVariableOp_51AssignVariableOpassignvariableop_51_totalIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOpassignvariableop_52_countIdentity_52:output:0"/device:CPU:0*
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
�

�
N__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_1899481

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
�
�
Q__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_1897299

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
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_1899581

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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1897053

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
��
�'
E__inference_model_40_layer_call_and_return_conditional_losses_1898897
inputs_0
inputs_1A
-gene_encoder_1_matmul_readvariableop_resource:
��=
.gene_encoder_1_biasadd_readvariableop_resource:	�E
6batchnormgeneencode1_batchnorm_readvariableop_resource:	�I
:batchnormgeneencode1_batchnorm_mul_readvariableop_resource:	�G
8batchnormgeneencode1_batchnorm_readvariableop_1_resource:	�G
8batchnormgeneencode1_batchnorm_readvariableop_2_resource:	�B
0protein_encoder_1_matmul_readvariableop_resource:W?
1protein_encoder_1_biasadd_readvariableop_resource:@
-gene_encoder_2_matmul_readvariableop_resource:	�}<
.gene_encoder_2_biasadd_readvariableop_resource:}D
6batchnormgeneencode2_batchnorm_readvariableop_resource:}H
:batchnormgeneencode2_batchnorm_mul_readvariableop_resource:}F
8batchnormgeneencode2_batchnorm_readvariableop_1_resource:}F
8batchnormgeneencode2_batchnorm_readvariableop_2_resource:}G
9batchnormproteinencode1_batchnorm_readvariableop_resource:K
=batchnormproteinencode1_batchnorm_mul_readvariableop_resource:I
;batchnormproteinencode1_batchnorm_readvariableop_1_resource:I
;batchnormproteinencode1_batchnorm_readvariableop_2_resource:C
0embeddingdimdense_matmul_readvariableop_resource:	�@?
1embeddingdimdense_biasadd_readvariableop_resource:@?
-gene_decoder_1_matmul_readvariableop_resource:@}<
.gene_decoder_1_biasadd_readvariableop_resource:}D
6batchnormgenedecode1_batchnorm_readvariableop_resource:}H
:batchnormgenedecode1_batchnorm_mul_readvariableop_resource:}F
8batchnormgenedecode1_batchnorm_readvariableop_1_resource:}F
8batchnormgenedecode1_batchnorm_readvariableop_2_resource:}B
0protein_decoder_1_matmul_readvariableop_resource:@?
1protein_decoder_1_biasadd_readvariableop_resource:@
-gene_decoder_2_matmul_readvariableop_resource:	}�=
.gene_decoder_2_biasadd_readvariableop_resource:	�G
9batchnormproteindecode1_batchnorm_readvariableop_resource:K
=batchnormproteindecode1_batchnorm_mul_readvariableop_resource:I
;batchnormproteindecode1_batchnorm_readvariableop_1_resource:I
;batchnormproteindecode1_batchnorm_readvariableop_2_resource:E
6batchnormgenedecode2_batchnorm_readvariableop_resource:	�I
:batchnormgenedecode2_batchnorm_mul_readvariableop_resource:	�G
8batchnormgenedecode2_batchnorm_readvariableop_1_resource:	�G
8batchnormgenedecode2_batchnorm_readvariableop_2_resource:	�E
3protein_decoder_last_matmul_readvariableop_resource:WB
4protein_decoder_last_biasadd_readvariableop_resource:WD
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

:W*
dtype0�
protein_encoder_1/MatMulMatMulinputs_1/protein_encoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(protein_encoder_1/BiasAdd/ReadVariableOpReadVariableOp1protein_encoder_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
protein_encoder_1/BiasAddBiasAdd"protein_encoder_1/MatMul:product:00protein_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������z
protein_encoder_1/SigmoidSigmoid"protein_encoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
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
:*
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
:�
'BatchNormProteinEncode1/batchnorm/RsqrtRsqrt)BatchNormProteinEncode1/batchnorm/add:z:0*
T0*
_output_shapes
:�
4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOpReadVariableOp=batchnormproteinencode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
%BatchNormProteinEncode1/batchnorm/mulMul+BatchNormProteinEncode1/batchnorm/Rsqrt:y:0<BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
'BatchNormProteinEncode1/batchnorm/mul_1Mulprotein_encoder_1/Sigmoid:y:0)BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
2BatchNormProteinEncode1/batchnorm/ReadVariableOp_1ReadVariableOp;batchnormproteinencode1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
'BatchNormProteinEncode1/batchnorm/mul_2Mul:BatchNormProteinEncode1/batchnorm/ReadVariableOp_1:value:0)BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*
_output_shapes
:�
2BatchNormProteinEncode1/batchnorm/ReadVariableOp_2ReadVariableOp;batchnormproteinencode1_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
%BatchNormProteinEncode1/batchnorm/subSub:BatchNormProteinEncode1/batchnorm/ReadVariableOp_2:value:0+BatchNormProteinEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
'BatchNormProteinEncode1/batchnorm/add_1AddV2+BatchNormProteinEncode1/batchnorm/mul_1:z:0)BatchNormProteinEncode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������^
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

:@*
dtype0�
protein_decoder_1/MatMulMatMulEmbeddingDimDense/Sigmoid:y:0/protein_decoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(protein_decoder_1/BiasAdd/ReadVariableOpReadVariableOp1protein_decoder_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
protein_decoder_1/BiasAddBiasAdd"protein_decoder_1/MatMul:product:00protein_decoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������z
protein_decoder_1/SigmoidSigmoid"protein_decoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
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
:*
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
:�
'BatchNormProteinDecode1/batchnorm/RsqrtRsqrt)BatchNormProteinDecode1/batchnorm/add:z:0*
T0*
_output_shapes
:�
4BatchNormProteinDecode1/batchnorm/mul/ReadVariableOpReadVariableOp=batchnormproteindecode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
%BatchNormProteinDecode1/batchnorm/mulMul+BatchNormProteinDecode1/batchnorm/Rsqrt:y:0<BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
'BatchNormProteinDecode1/batchnorm/mul_1Mulprotein_decoder_1/Sigmoid:y:0)BatchNormProteinDecode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
2BatchNormProteinDecode1/batchnorm/ReadVariableOp_1ReadVariableOp;batchnormproteindecode1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
'BatchNormProteinDecode1/batchnorm/mul_2Mul:BatchNormProteinDecode1/batchnorm/ReadVariableOp_1:value:0)BatchNormProteinDecode1/batchnorm/mul:z:0*
T0*
_output_shapes
:�
2BatchNormProteinDecode1/batchnorm/ReadVariableOp_2ReadVariableOp;batchnormproteindecode1_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
%BatchNormProteinDecode1/batchnorm/subSub:BatchNormProteinDecode1/batchnorm/ReadVariableOp_2:value:0+BatchNormProteinDecode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
'BatchNormProteinDecode1/batchnorm/add_1AddV2+BatchNormProteinDecode1/batchnorm/mul_1:z:0)BatchNormProteinDecode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
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

:W*
dtype0�
protein_decoder_last/MatMulMatMul+BatchNormProteinDecode1/batchnorm/add_1:z:02protein_decoder_last/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������W�
+protein_decoder_last/BiasAdd/ReadVariableOpReadVariableOp4protein_decoder_last_biasadd_readvariableop_resource*
_output_shapes
:W*
dtype0�
protein_decoder_last/BiasAddBiasAdd%protein_decoder_last/MatMul:product:03protein_decoder_last/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������W�
protein_decoder_last/SigmoidSigmoid%protein_decoder_last/BiasAdd:output:0*
T0*'
_output_shapes
:���������W�
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
:����������q

Identity_1Identity protein_decoder_last/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������W�
NoOpNoOp.^BatchNormGeneDecode1/batchnorm/ReadVariableOp0^BatchNormGeneDecode1/batchnorm/ReadVariableOp_10^BatchNormGeneDecode1/batchnorm/ReadVariableOp_22^BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp.^BatchNormGeneDecode2/batchnorm/ReadVariableOp0^BatchNormGeneDecode2/batchnorm/ReadVariableOp_10^BatchNormGeneDecode2/batchnorm/ReadVariableOp_22^BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp.^BatchNormGeneEncode1/batchnorm/ReadVariableOp0^BatchNormGeneEncode1/batchnorm/ReadVariableOp_10^BatchNormGeneEncode1/batchnorm/ReadVariableOp_22^BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp.^BatchNormGeneEncode2/batchnorm/ReadVariableOp0^BatchNormGeneEncode2/batchnorm/ReadVariableOp_10^BatchNormGeneEncode2/batchnorm/ReadVariableOp_22^BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp1^BatchNormProteinDecode1/batchnorm/ReadVariableOp3^BatchNormProteinDecode1/batchnorm/ReadVariableOp_13^BatchNormProteinDecode1/batchnorm/ReadVariableOp_25^BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp1^BatchNormProteinEncode1/batchnorm/ReadVariableOp3^BatchNormProteinEncode1/batchnorm/ReadVariableOp_13^BatchNormProteinEncode1/batchnorm/ReadVariableOp_25^BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp)^EmbeddingDimDense/BiasAdd/ReadVariableOp(^EmbeddingDimDense/MatMul/ReadVariableOp&^gene_decoder_1/BiasAdd/ReadVariableOp%^gene_decoder_1/MatMul/ReadVariableOp&^gene_decoder_2/BiasAdd/ReadVariableOp%^gene_decoder_2/MatMul/ReadVariableOp)^gene_decoder_last/BiasAdd/ReadVariableOp(^gene_decoder_last/MatMul/ReadVariableOp&^gene_encoder_1/BiasAdd/ReadVariableOp%^gene_encoder_1/MatMul/ReadVariableOp&^gene_encoder_2/BiasAdd/ReadVariableOp%^gene_encoder_2/MatMul/ReadVariableOp)^protein_decoder_1/BiasAdd/ReadVariableOp(^protein_decoder_1/MatMul/ReadVariableOp,^protein_decoder_last/BiasAdd/ReadVariableOp+^protein_decoder_last/MatMul/ReadVariableOp)^protein_encoder_1/BiasAdd/ReadVariableOp(^protein_encoder_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes}
{:����������:���������W: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2^
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
inputs/0:QM
'
_output_shapes
:���������W
"
_user_specified_name
inputs/1
�
�-
"__inference__wrapped_model_1896947
gene_input_layer
protein_input_layerJ
6model_40_gene_encoder_1_matmul_readvariableop_resource:
��F
7model_40_gene_encoder_1_biasadd_readvariableop_resource:	�N
?model_40_batchnormgeneencode1_batchnorm_readvariableop_resource:	�R
Cmodel_40_batchnormgeneencode1_batchnorm_mul_readvariableop_resource:	�P
Amodel_40_batchnormgeneencode1_batchnorm_readvariableop_1_resource:	�P
Amodel_40_batchnormgeneencode1_batchnorm_readvariableop_2_resource:	�K
9model_40_protein_encoder_1_matmul_readvariableop_resource:WH
:model_40_protein_encoder_1_biasadd_readvariableop_resource:I
6model_40_gene_encoder_2_matmul_readvariableop_resource:	�}E
7model_40_gene_encoder_2_biasadd_readvariableop_resource:}M
?model_40_batchnormgeneencode2_batchnorm_readvariableop_resource:}Q
Cmodel_40_batchnormgeneencode2_batchnorm_mul_readvariableop_resource:}O
Amodel_40_batchnormgeneencode2_batchnorm_readvariableop_1_resource:}O
Amodel_40_batchnormgeneencode2_batchnorm_readvariableop_2_resource:}P
Bmodel_40_batchnormproteinencode1_batchnorm_readvariableop_resource:T
Fmodel_40_batchnormproteinencode1_batchnorm_mul_readvariableop_resource:R
Dmodel_40_batchnormproteinencode1_batchnorm_readvariableop_1_resource:R
Dmodel_40_batchnormproteinencode1_batchnorm_readvariableop_2_resource:L
9model_40_embeddingdimdense_matmul_readvariableop_resource:	�@H
:model_40_embeddingdimdense_biasadd_readvariableop_resource:@H
6model_40_gene_decoder_1_matmul_readvariableop_resource:@}E
7model_40_gene_decoder_1_biasadd_readvariableop_resource:}M
?model_40_batchnormgenedecode1_batchnorm_readvariableop_resource:}Q
Cmodel_40_batchnormgenedecode1_batchnorm_mul_readvariableop_resource:}O
Amodel_40_batchnormgenedecode1_batchnorm_readvariableop_1_resource:}O
Amodel_40_batchnormgenedecode1_batchnorm_readvariableop_2_resource:}K
9model_40_protein_decoder_1_matmul_readvariableop_resource:@H
:model_40_protein_decoder_1_biasadd_readvariableop_resource:I
6model_40_gene_decoder_2_matmul_readvariableop_resource:	}�F
7model_40_gene_decoder_2_biasadd_readvariableop_resource:	�P
Bmodel_40_batchnormproteindecode1_batchnorm_readvariableop_resource:T
Fmodel_40_batchnormproteindecode1_batchnorm_mul_readvariableop_resource:R
Dmodel_40_batchnormproteindecode1_batchnorm_readvariableop_1_resource:R
Dmodel_40_batchnormproteindecode1_batchnorm_readvariableop_2_resource:N
?model_40_batchnormgenedecode2_batchnorm_readvariableop_resource:	�R
Cmodel_40_batchnormgenedecode2_batchnorm_mul_readvariableop_resource:	�P
Amodel_40_batchnormgenedecode2_batchnorm_readvariableop_1_resource:	�P
Amodel_40_batchnormgenedecode2_batchnorm_readvariableop_2_resource:	�N
<model_40_protein_decoder_last_matmul_readvariableop_resource:WK
=model_40_protein_decoder_last_biasadd_readvariableop_resource:WM
9model_40_gene_decoder_last_matmul_readvariableop_resource:
��I
:model_40_gene_decoder_last_biasadd_readvariableop_resource:	�
identity

identity_1��6model_40/BatchNormGeneDecode1/batchnorm/ReadVariableOp�8model_40/BatchNormGeneDecode1/batchnorm/ReadVariableOp_1�8model_40/BatchNormGeneDecode1/batchnorm/ReadVariableOp_2�:model_40/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp�6model_40/BatchNormGeneDecode2/batchnorm/ReadVariableOp�8model_40/BatchNormGeneDecode2/batchnorm/ReadVariableOp_1�8model_40/BatchNormGeneDecode2/batchnorm/ReadVariableOp_2�:model_40/BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp�6model_40/BatchNormGeneEncode1/batchnorm/ReadVariableOp�8model_40/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1�8model_40/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2�:model_40/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp�6model_40/BatchNormGeneEncode2/batchnorm/ReadVariableOp�8model_40/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1�8model_40/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2�:model_40/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp�9model_40/BatchNormProteinDecode1/batchnorm/ReadVariableOp�;model_40/BatchNormProteinDecode1/batchnorm/ReadVariableOp_1�;model_40/BatchNormProteinDecode1/batchnorm/ReadVariableOp_2�=model_40/BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp�9model_40/BatchNormProteinEncode1/batchnorm/ReadVariableOp�;model_40/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1�;model_40/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2�=model_40/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp�1model_40/EmbeddingDimDense/BiasAdd/ReadVariableOp�0model_40/EmbeddingDimDense/MatMul/ReadVariableOp�.model_40/gene_decoder_1/BiasAdd/ReadVariableOp�-model_40/gene_decoder_1/MatMul/ReadVariableOp�.model_40/gene_decoder_2/BiasAdd/ReadVariableOp�-model_40/gene_decoder_2/MatMul/ReadVariableOp�1model_40/gene_decoder_last/BiasAdd/ReadVariableOp�0model_40/gene_decoder_last/MatMul/ReadVariableOp�.model_40/gene_encoder_1/BiasAdd/ReadVariableOp�-model_40/gene_encoder_1/MatMul/ReadVariableOp�.model_40/gene_encoder_2/BiasAdd/ReadVariableOp�-model_40/gene_encoder_2/MatMul/ReadVariableOp�1model_40/protein_decoder_1/BiasAdd/ReadVariableOp�0model_40/protein_decoder_1/MatMul/ReadVariableOp�4model_40/protein_decoder_last/BiasAdd/ReadVariableOp�3model_40/protein_decoder_last/MatMul/ReadVariableOp�1model_40/protein_encoder_1/BiasAdd/ReadVariableOp�0model_40/protein_encoder_1/MatMul/ReadVariableOp�
-model_40/gene_encoder_1/MatMul/ReadVariableOpReadVariableOp6model_40_gene_encoder_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model_40/gene_encoder_1/MatMulMatMulgene_input_layer5model_40/gene_encoder_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.model_40/gene_encoder_1/BiasAdd/ReadVariableOpReadVariableOp7model_40_gene_encoder_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_40/gene_encoder_1/BiasAddBiasAdd(model_40/gene_encoder_1/MatMul:product:06model_40/gene_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
model_40/gene_encoder_1/SigmoidSigmoid(model_40/gene_encoder_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
6model_40/BatchNormGeneEncode1/batchnorm/ReadVariableOpReadVariableOp?model_40_batchnormgeneencode1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0r
-model_40/BatchNormGeneEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
+model_40/BatchNormGeneEncode1/batchnorm/addAddV2>model_40/BatchNormGeneEncode1/batchnorm/ReadVariableOp:value:06model_40/BatchNormGeneEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
-model_40/BatchNormGeneEncode1/batchnorm/RsqrtRsqrt/model_40/BatchNormGeneEncode1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
:model_40/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_40_batchnormgeneencode1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+model_40/BatchNormGeneEncode1/batchnorm/mulMul1model_40/BatchNormGeneEncode1/batchnorm/Rsqrt:y:0Bmodel_40/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
-model_40/BatchNormGeneEncode1/batchnorm/mul_1Mul#model_40/gene_encoder_1/Sigmoid:y:0/model_40/BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
8model_40/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_40_batchnormgeneencode1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
-model_40/BatchNormGeneEncode1/batchnorm/mul_2Mul@model_40/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1:value:0/model_40/BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
8model_40/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_40_batchnormgeneencode1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
+model_40/BatchNormGeneEncode1/batchnorm/subSub@model_40/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2:value:01model_40/BatchNormGeneEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
-model_40/BatchNormGeneEncode1/batchnorm/add_1AddV21model_40/BatchNormGeneEncode1/batchnorm/mul_1:z:0/model_40/BatchNormGeneEncode1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
0model_40/protein_encoder_1/MatMul/ReadVariableOpReadVariableOp9model_40_protein_encoder_1_matmul_readvariableop_resource*
_output_shapes

:W*
dtype0�
!model_40/protein_encoder_1/MatMulMatMulprotein_input_layer8model_40/protein_encoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1model_40/protein_encoder_1/BiasAdd/ReadVariableOpReadVariableOp:model_40_protein_encoder_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
"model_40/protein_encoder_1/BiasAddBiasAdd+model_40/protein_encoder_1/MatMul:product:09model_40/protein_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"model_40/protein_encoder_1/SigmoidSigmoid+model_40/protein_encoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
-model_40/gene_encoder_2/MatMul/ReadVariableOpReadVariableOp6model_40_gene_encoder_2_matmul_readvariableop_resource*
_output_shapes
:	�}*
dtype0�
model_40/gene_encoder_2/MatMulMatMul1model_40/BatchNormGeneEncode1/batchnorm/add_1:z:05model_40/gene_encoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}�
.model_40/gene_encoder_2/BiasAdd/ReadVariableOpReadVariableOp7model_40_gene_encoder_2_biasadd_readvariableop_resource*
_output_shapes
:}*
dtype0�
model_40/gene_encoder_2/BiasAddBiasAdd(model_40/gene_encoder_2/MatMul:product:06model_40/gene_encoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}�
model_40/gene_encoder_2/SigmoidSigmoid(model_40/gene_encoder_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������}�
6model_40/BatchNormGeneEncode2/batchnorm/ReadVariableOpReadVariableOp?model_40_batchnormgeneencode2_batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0r
-model_40/BatchNormGeneEncode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
+model_40/BatchNormGeneEncode2/batchnorm/addAddV2>model_40/BatchNormGeneEncode2/batchnorm/ReadVariableOp:value:06model_40/BatchNormGeneEncode2/batchnorm/add/y:output:0*
T0*
_output_shapes
:}�
-model_40/BatchNormGeneEncode2/batchnorm/RsqrtRsqrt/model_40/BatchNormGeneEncode2/batchnorm/add:z:0*
T0*
_output_shapes
:}�
:model_40/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_40_batchnormgeneencode2_batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0�
+model_40/BatchNormGeneEncode2/batchnorm/mulMul1model_40/BatchNormGeneEncode2/batchnorm/Rsqrt:y:0Bmodel_40/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}�
-model_40/BatchNormGeneEncode2/batchnorm/mul_1Mul#model_40/gene_encoder_2/Sigmoid:y:0/model_40/BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������}�
8model_40/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_40_batchnormgeneencode2_batchnorm_readvariableop_1_resource*
_output_shapes
:}*
dtype0�
-model_40/BatchNormGeneEncode2/batchnorm/mul_2Mul@model_40/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1:value:0/model_40/BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*
_output_shapes
:}�
8model_40/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_40_batchnormgeneencode2_batchnorm_readvariableop_2_resource*
_output_shapes
:}*
dtype0�
+model_40/BatchNormGeneEncode2/batchnorm/subSub@model_40/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2:value:01model_40/BatchNormGeneEncode2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:}�
-model_40/BatchNormGeneEncode2/batchnorm/add_1AddV21model_40/BatchNormGeneEncode2/batchnorm/mul_1:z:0/model_40/BatchNormGeneEncode2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}�
9model_40/BatchNormProteinEncode1/batchnorm/ReadVariableOpReadVariableOpBmodel_40_batchnormproteinencode1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0u
0model_40/BatchNormProteinEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
.model_40/BatchNormProteinEncode1/batchnorm/addAddV2Amodel_40/BatchNormProteinEncode1/batchnorm/ReadVariableOp:value:09model_40/BatchNormProteinEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
0model_40/BatchNormProteinEncode1/batchnorm/RsqrtRsqrt2model_40/BatchNormProteinEncode1/batchnorm/add:z:0*
T0*
_output_shapes
:�
=model_40/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOpReadVariableOpFmodel_40_batchnormproteinencode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
.model_40/BatchNormProteinEncode1/batchnorm/mulMul4model_40/BatchNormProteinEncode1/batchnorm/Rsqrt:y:0Emodel_40/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
0model_40/BatchNormProteinEncode1/batchnorm/mul_1Mul&model_40/protein_encoder_1/Sigmoid:y:02model_40/BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
;model_40/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1ReadVariableOpDmodel_40_batchnormproteinencode1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
0model_40/BatchNormProteinEncode1/batchnorm/mul_2MulCmodel_40/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1:value:02model_40/BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*
_output_shapes
:�
;model_40/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2ReadVariableOpDmodel_40_batchnormproteinencode1_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
.model_40/BatchNormProteinEncode1/batchnorm/subSubCmodel_40/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2:value:04model_40/BatchNormProteinEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
0model_40/BatchNormProteinEncode1/batchnorm/add_1AddV24model_40/BatchNormProteinEncode1/batchnorm/mul_1:z:02model_40/BatchNormProteinEncode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������g
%model_40/ConcatenateLayer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
 model_40/ConcatenateLayer/concatConcatV21model_40/BatchNormGeneEncode2/batchnorm/add_1:z:04model_40/BatchNormProteinEncode1/batchnorm/add_1:z:0.model_40/ConcatenateLayer/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
0model_40/EmbeddingDimDense/MatMul/ReadVariableOpReadVariableOp9model_40_embeddingdimdense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
!model_40/EmbeddingDimDense/MatMulMatMul)model_40/ConcatenateLayer/concat:output:08model_40/EmbeddingDimDense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
1model_40/EmbeddingDimDense/BiasAdd/ReadVariableOpReadVariableOp:model_40_embeddingdimdense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
"model_40/EmbeddingDimDense/BiasAddBiasAdd+model_40/EmbeddingDimDense/MatMul:product:09model_40/EmbeddingDimDense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
"model_40/EmbeddingDimDense/SigmoidSigmoid+model_40/EmbeddingDimDense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
-model_40/gene_decoder_1/MatMul/ReadVariableOpReadVariableOp6model_40_gene_decoder_1_matmul_readvariableop_resource*
_output_shapes

:@}*
dtype0�
model_40/gene_decoder_1/MatMulMatMul&model_40/EmbeddingDimDense/Sigmoid:y:05model_40/gene_decoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}�
.model_40/gene_decoder_1/BiasAdd/ReadVariableOpReadVariableOp7model_40_gene_decoder_1_biasadd_readvariableop_resource*
_output_shapes
:}*
dtype0�
model_40/gene_decoder_1/BiasAddBiasAdd(model_40/gene_decoder_1/MatMul:product:06model_40/gene_decoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}�
model_40/gene_decoder_1/SigmoidSigmoid(model_40/gene_decoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������}�
6model_40/BatchNormGeneDecode1/batchnorm/ReadVariableOpReadVariableOp?model_40_batchnormgenedecode1_batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0r
-model_40/BatchNormGeneDecode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
+model_40/BatchNormGeneDecode1/batchnorm/addAddV2>model_40/BatchNormGeneDecode1/batchnorm/ReadVariableOp:value:06model_40/BatchNormGeneDecode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:}�
-model_40/BatchNormGeneDecode1/batchnorm/RsqrtRsqrt/model_40/BatchNormGeneDecode1/batchnorm/add:z:0*
T0*
_output_shapes
:}�
:model_40/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_40_batchnormgenedecode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0�
+model_40/BatchNormGeneDecode1/batchnorm/mulMul1model_40/BatchNormGeneDecode1/batchnorm/Rsqrt:y:0Bmodel_40/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}�
-model_40/BatchNormGeneDecode1/batchnorm/mul_1Mul#model_40/gene_decoder_1/Sigmoid:y:0/model_40/BatchNormGeneDecode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������}�
8model_40/BatchNormGeneDecode1/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_40_batchnormgenedecode1_batchnorm_readvariableop_1_resource*
_output_shapes
:}*
dtype0�
-model_40/BatchNormGeneDecode1/batchnorm/mul_2Mul@model_40/BatchNormGeneDecode1/batchnorm/ReadVariableOp_1:value:0/model_40/BatchNormGeneDecode1/batchnorm/mul:z:0*
T0*
_output_shapes
:}�
8model_40/BatchNormGeneDecode1/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_40_batchnormgenedecode1_batchnorm_readvariableop_2_resource*
_output_shapes
:}*
dtype0�
+model_40/BatchNormGeneDecode1/batchnorm/subSub@model_40/BatchNormGeneDecode1/batchnorm/ReadVariableOp_2:value:01model_40/BatchNormGeneDecode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:}�
-model_40/BatchNormGeneDecode1/batchnorm/add_1AddV21model_40/BatchNormGeneDecode1/batchnorm/mul_1:z:0/model_40/BatchNormGeneDecode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}�
0model_40/protein_decoder_1/MatMul/ReadVariableOpReadVariableOp9model_40_protein_decoder_1_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
!model_40/protein_decoder_1/MatMulMatMul&model_40/EmbeddingDimDense/Sigmoid:y:08model_40/protein_decoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
1model_40/protein_decoder_1/BiasAdd/ReadVariableOpReadVariableOp:model_40_protein_decoder_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
"model_40/protein_decoder_1/BiasAddBiasAdd+model_40/protein_decoder_1/MatMul:product:09model_40/protein_decoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
"model_40/protein_decoder_1/SigmoidSigmoid+model_40/protein_decoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
-model_40/gene_decoder_2/MatMul/ReadVariableOpReadVariableOp6model_40_gene_decoder_2_matmul_readvariableop_resource*
_output_shapes
:	}�*
dtype0�
model_40/gene_decoder_2/MatMulMatMul1model_40/BatchNormGeneDecode1/batchnorm/add_1:z:05model_40/gene_decoder_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.model_40/gene_decoder_2/BiasAdd/ReadVariableOpReadVariableOp7model_40_gene_decoder_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_40/gene_decoder_2/BiasAddBiasAdd(model_40/gene_decoder_2/MatMul:product:06model_40/gene_decoder_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
model_40/gene_decoder_2/SigmoidSigmoid(model_40/gene_decoder_2/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
9model_40/BatchNormProteinDecode1/batchnorm/ReadVariableOpReadVariableOpBmodel_40_batchnormproteindecode1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0u
0model_40/BatchNormProteinDecode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
.model_40/BatchNormProteinDecode1/batchnorm/addAddV2Amodel_40/BatchNormProteinDecode1/batchnorm/ReadVariableOp:value:09model_40/BatchNormProteinDecode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:�
0model_40/BatchNormProteinDecode1/batchnorm/RsqrtRsqrt2model_40/BatchNormProteinDecode1/batchnorm/add:z:0*
T0*
_output_shapes
:�
=model_40/BatchNormProteinDecode1/batchnorm/mul/ReadVariableOpReadVariableOpFmodel_40_batchnormproteindecode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
.model_40/BatchNormProteinDecode1/batchnorm/mulMul4model_40/BatchNormProteinDecode1/batchnorm/Rsqrt:y:0Emodel_40/BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
0model_40/BatchNormProteinDecode1/batchnorm/mul_1Mul&model_40/protein_decoder_1/Sigmoid:y:02model_40/BatchNormProteinDecode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
;model_40/BatchNormProteinDecode1/batchnorm/ReadVariableOp_1ReadVariableOpDmodel_40_batchnormproteindecode1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0�
0model_40/BatchNormProteinDecode1/batchnorm/mul_2MulCmodel_40/BatchNormProteinDecode1/batchnorm/ReadVariableOp_1:value:02model_40/BatchNormProteinDecode1/batchnorm/mul:z:0*
T0*
_output_shapes
:�
;model_40/BatchNormProteinDecode1/batchnorm/ReadVariableOp_2ReadVariableOpDmodel_40_batchnormproteindecode1_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0�
.model_40/BatchNormProteinDecode1/batchnorm/subSubCmodel_40/BatchNormProteinDecode1/batchnorm/ReadVariableOp_2:value:04model_40/BatchNormProteinDecode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
0model_40/BatchNormProteinDecode1/batchnorm/add_1AddV24model_40/BatchNormProteinDecode1/batchnorm/mul_1:z:02model_40/BatchNormProteinDecode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:����������
6model_40/BatchNormGeneDecode2/batchnorm/ReadVariableOpReadVariableOp?model_40_batchnormgenedecode2_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0r
-model_40/BatchNormGeneDecode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
+model_40/BatchNormGeneDecode2/batchnorm/addAddV2>model_40/BatchNormGeneDecode2/batchnorm/ReadVariableOp:value:06model_40/BatchNormGeneDecode2/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
-model_40/BatchNormGeneDecode2/batchnorm/RsqrtRsqrt/model_40/BatchNormGeneDecode2/batchnorm/add:z:0*
T0*
_output_shapes	
:��
:model_40/BatchNormGeneDecode2/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_40_batchnormgenedecode2_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+model_40/BatchNormGeneDecode2/batchnorm/mulMul1model_40/BatchNormGeneDecode2/batchnorm/Rsqrt:y:0Bmodel_40/BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
-model_40/BatchNormGeneDecode2/batchnorm/mul_1Mul#model_40/gene_decoder_2/Sigmoid:y:0/model_40/BatchNormGeneDecode2/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
8model_40/BatchNormGeneDecode2/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_40_batchnormgenedecode2_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
-model_40/BatchNormGeneDecode2/batchnorm/mul_2Mul@model_40/BatchNormGeneDecode2/batchnorm/ReadVariableOp_1:value:0/model_40/BatchNormGeneDecode2/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
8model_40/BatchNormGeneDecode2/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_40_batchnormgenedecode2_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
+model_40/BatchNormGeneDecode2/batchnorm/subSub@model_40/BatchNormGeneDecode2/batchnorm/ReadVariableOp_2:value:01model_40/BatchNormGeneDecode2/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
-model_40/BatchNormGeneDecode2/batchnorm/add_1AddV21model_40/BatchNormGeneDecode2/batchnorm/mul_1:z:0/model_40/BatchNormGeneDecode2/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
3model_40/protein_decoder_last/MatMul/ReadVariableOpReadVariableOp<model_40_protein_decoder_last_matmul_readvariableop_resource*
_output_shapes

:W*
dtype0�
$model_40/protein_decoder_last/MatMulMatMul4model_40/BatchNormProteinDecode1/batchnorm/add_1:z:0;model_40/protein_decoder_last/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������W�
4model_40/protein_decoder_last/BiasAdd/ReadVariableOpReadVariableOp=model_40_protein_decoder_last_biasadd_readvariableop_resource*
_output_shapes
:W*
dtype0�
%model_40/protein_decoder_last/BiasAddBiasAdd.model_40/protein_decoder_last/MatMul:product:0<model_40/protein_decoder_last/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������W�
%model_40/protein_decoder_last/SigmoidSigmoid.model_40/protein_decoder_last/BiasAdd:output:0*
T0*'
_output_shapes
:���������W�
0model_40/gene_decoder_last/MatMul/ReadVariableOpReadVariableOp9model_40_gene_decoder_last_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
!model_40/gene_decoder_last/MatMulMatMul1model_40/BatchNormGeneDecode2/batchnorm/add_1:z:08model_40/gene_decoder_last/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
1model_40/gene_decoder_last/BiasAdd/ReadVariableOpReadVariableOp:model_40_gene_decoder_last_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
"model_40/gene_decoder_last/BiasAddBiasAdd+model_40/gene_decoder_last/MatMul:product:09model_40/gene_decoder_last/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
"model_40/gene_decoder_last/SigmoidSigmoid+model_40/gene_decoder_last/BiasAdd:output:0*
T0*(
_output_shapes
:����������v
IdentityIdentity&model_40/gene_decoder_last/Sigmoid:y:0^NoOp*
T0*(
_output_shapes
:����������z

Identity_1Identity)model_40/protein_decoder_last/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������W�
NoOpNoOp7^model_40/BatchNormGeneDecode1/batchnorm/ReadVariableOp9^model_40/BatchNormGeneDecode1/batchnorm/ReadVariableOp_19^model_40/BatchNormGeneDecode1/batchnorm/ReadVariableOp_2;^model_40/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp7^model_40/BatchNormGeneDecode2/batchnorm/ReadVariableOp9^model_40/BatchNormGeneDecode2/batchnorm/ReadVariableOp_19^model_40/BatchNormGeneDecode2/batchnorm/ReadVariableOp_2;^model_40/BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp7^model_40/BatchNormGeneEncode1/batchnorm/ReadVariableOp9^model_40/BatchNormGeneEncode1/batchnorm/ReadVariableOp_19^model_40/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2;^model_40/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp7^model_40/BatchNormGeneEncode2/batchnorm/ReadVariableOp9^model_40/BatchNormGeneEncode2/batchnorm/ReadVariableOp_19^model_40/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2;^model_40/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:^model_40/BatchNormProteinDecode1/batchnorm/ReadVariableOp<^model_40/BatchNormProteinDecode1/batchnorm/ReadVariableOp_1<^model_40/BatchNormProteinDecode1/batchnorm/ReadVariableOp_2>^model_40/BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp:^model_40/BatchNormProteinEncode1/batchnorm/ReadVariableOp<^model_40/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1<^model_40/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2>^model_40/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp2^model_40/EmbeddingDimDense/BiasAdd/ReadVariableOp1^model_40/EmbeddingDimDense/MatMul/ReadVariableOp/^model_40/gene_decoder_1/BiasAdd/ReadVariableOp.^model_40/gene_decoder_1/MatMul/ReadVariableOp/^model_40/gene_decoder_2/BiasAdd/ReadVariableOp.^model_40/gene_decoder_2/MatMul/ReadVariableOp2^model_40/gene_decoder_last/BiasAdd/ReadVariableOp1^model_40/gene_decoder_last/MatMul/ReadVariableOp/^model_40/gene_encoder_1/BiasAdd/ReadVariableOp.^model_40/gene_encoder_1/MatMul/ReadVariableOp/^model_40/gene_encoder_2/BiasAdd/ReadVariableOp.^model_40/gene_encoder_2/MatMul/ReadVariableOp2^model_40/protein_decoder_1/BiasAdd/ReadVariableOp1^model_40/protein_decoder_1/MatMul/ReadVariableOp5^model_40/protein_decoder_last/BiasAdd/ReadVariableOp4^model_40/protein_decoder_last/MatMul/ReadVariableOp2^model_40/protein_encoder_1/BiasAdd/ReadVariableOp1^model_40/protein_encoder_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes}
{:����������:���������W: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6model_40/BatchNormGeneDecode1/batchnorm/ReadVariableOp6model_40/BatchNormGeneDecode1/batchnorm/ReadVariableOp2t
8model_40/BatchNormGeneDecode1/batchnorm/ReadVariableOp_18model_40/BatchNormGeneDecode1/batchnorm/ReadVariableOp_12t
8model_40/BatchNormGeneDecode1/batchnorm/ReadVariableOp_28model_40/BatchNormGeneDecode1/batchnorm/ReadVariableOp_22x
:model_40/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp:model_40/BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp2p
6model_40/BatchNormGeneDecode2/batchnorm/ReadVariableOp6model_40/BatchNormGeneDecode2/batchnorm/ReadVariableOp2t
8model_40/BatchNormGeneDecode2/batchnorm/ReadVariableOp_18model_40/BatchNormGeneDecode2/batchnorm/ReadVariableOp_12t
8model_40/BatchNormGeneDecode2/batchnorm/ReadVariableOp_28model_40/BatchNormGeneDecode2/batchnorm/ReadVariableOp_22x
:model_40/BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp:model_40/BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp2p
6model_40/BatchNormGeneEncode1/batchnorm/ReadVariableOp6model_40/BatchNormGeneEncode1/batchnorm/ReadVariableOp2t
8model_40/BatchNormGeneEncode1/batchnorm/ReadVariableOp_18model_40/BatchNormGeneEncode1/batchnorm/ReadVariableOp_12t
8model_40/BatchNormGeneEncode1/batchnorm/ReadVariableOp_28model_40/BatchNormGeneEncode1/batchnorm/ReadVariableOp_22x
:model_40/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp:model_40/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp2p
6model_40/BatchNormGeneEncode2/batchnorm/ReadVariableOp6model_40/BatchNormGeneEncode2/batchnorm/ReadVariableOp2t
8model_40/BatchNormGeneEncode2/batchnorm/ReadVariableOp_18model_40/BatchNormGeneEncode2/batchnorm/ReadVariableOp_12t
8model_40/BatchNormGeneEncode2/batchnorm/ReadVariableOp_28model_40/BatchNormGeneEncode2/batchnorm/ReadVariableOp_22x
:model_40/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:model_40/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp2v
9model_40/BatchNormProteinDecode1/batchnorm/ReadVariableOp9model_40/BatchNormProteinDecode1/batchnorm/ReadVariableOp2z
;model_40/BatchNormProteinDecode1/batchnorm/ReadVariableOp_1;model_40/BatchNormProteinDecode1/batchnorm/ReadVariableOp_12z
;model_40/BatchNormProteinDecode1/batchnorm/ReadVariableOp_2;model_40/BatchNormProteinDecode1/batchnorm/ReadVariableOp_22~
=model_40/BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp=model_40/BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp2v
9model_40/BatchNormProteinEncode1/batchnorm/ReadVariableOp9model_40/BatchNormProteinEncode1/batchnorm/ReadVariableOp2z
;model_40/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1;model_40/BatchNormProteinEncode1/batchnorm/ReadVariableOp_12z
;model_40/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2;model_40/BatchNormProteinEncode1/batchnorm/ReadVariableOp_22~
=model_40/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp=model_40/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp2f
1model_40/EmbeddingDimDense/BiasAdd/ReadVariableOp1model_40/EmbeddingDimDense/BiasAdd/ReadVariableOp2d
0model_40/EmbeddingDimDense/MatMul/ReadVariableOp0model_40/EmbeddingDimDense/MatMul/ReadVariableOp2`
.model_40/gene_decoder_1/BiasAdd/ReadVariableOp.model_40/gene_decoder_1/BiasAdd/ReadVariableOp2^
-model_40/gene_decoder_1/MatMul/ReadVariableOp-model_40/gene_decoder_1/MatMul/ReadVariableOp2`
.model_40/gene_decoder_2/BiasAdd/ReadVariableOp.model_40/gene_decoder_2/BiasAdd/ReadVariableOp2^
-model_40/gene_decoder_2/MatMul/ReadVariableOp-model_40/gene_decoder_2/MatMul/ReadVariableOp2f
1model_40/gene_decoder_last/BiasAdd/ReadVariableOp1model_40/gene_decoder_last/BiasAdd/ReadVariableOp2d
0model_40/gene_decoder_last/MatMul/ReadVariableOp0model_40/gene_decoder_last/MatMul/ReadVariableOp2`
.model_40/gene_encoder_1/BiasAdd/ReadVariableOp.model_40/gene_encoder_1/BiasAdd/ReadVariableOp2^
-model_40/gene_encoder_1/MatMul/ReadVariableOp-model_40/gene_encoder_1/MatMul/ReadVariableOp2`
.model_40/gene_encoder_2/BiasAdd/ReadVariableOp.model_40/gene_encoder_2/BiasAdd/ReadVariableOp2^
-model_40/gene_encoder_2/MatMul/ReadVariableOp-model_40/gene_encoder_2/MatMul/ReadVariableOp2f
1model_40/protein_decoder_1/BiasAdd/ReadVariableOp1model_40/protein_decoder_1/BiasAdd/ReadVariableOp2d
0model_40/protein_decoder_1/MatMul/ReadVariableOp0model_40/protein_decoder_1/MatMul/ReadVariableOp2l
4model_40/protein_decoder_last/BiasAdd/ReadVariableOp4model_40/protein_decoder_last/BiasAdd/ReadVariableOp2j
3model_40/protein_decoder_last/MatMul/ReadVariableOp3model_40/protein_decoder_last/MatMul/ReadVariableOp2f
1model_40/protein_encoder_1/BiasAdd/ReadVariableOp1model_40/protein_encoder_1/BiasAdd/ReadVariableOp2d
0model_40/protein_encoder_1/MatMul/ReadVariableOp0model_40/protein_encoder_1/MatMul/ReadVariableOp:Z V
(
_output_shapes
:����������
*
_user_specified_nameGene_Input_Layer:\X
'
_output_shapes
:���������W
-
_user_specified_nameProtein_Input_Layer
�
�
3__inference_protein_encoder_1_layer_call_fn_1899277

inputs
unknown:W
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_protein_encoder_1_layer_call_and_return_conditional_losses_1897485o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������W: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������W
 
_user_specified_nameinputs
�

�
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_1899268

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
�
�	
*__inference_model_40_layer_call_fn_1898638
inputs_0
inputs_1
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:W
	unknown_6:
	unknown_7:	�}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:	�@

unknown_18:@

unknown_19:@}

unknown_20:}

unknown_21:}

unknown_22:}

unknown_23:}

unknown_24:}

unknown_25:@

unknown_26:

unknown_27:	}�

unknown_28:	�

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:	�

unknown_34:	�

unknown_35:	�

unknown_36:	�

unknown_37:W

unknown_38:W

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
 *;
_output_shapes)
':����������:���������W*L
_read_only_resource_inputs.
,*	
 !"#$%&'()*+*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_40_layer_call_and_return_conditional_losses_1897666p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������W`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes}
{:����������:���������W: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������W
"
_user_specified_name
inputs/1
�

�
Q__inference_protein_decoder_last_layer_call_and_return_conditional_losses_1899821

inputs0
matmul_readvariableop_resource:W-
biasadd_readvariableop_resource:W
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:W*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Wr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:W*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������WV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������WZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������Ww
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_1897217

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
�

�
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_1897502

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
�

�
K__inference_gene_decoder_1_layer_call_and_return_conditional_losses_1899501

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
3__inference_EmbeddingDimDense_layer_call_fn_1899470

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
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_1897546o
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
�
�
0__inference_gene_decoder_2_layer_call_fn_1899590

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
K__inference_gene_decoder_2_layer_call_and_return_conditional_losses_1897606p
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
�

�
K__inference_gene_decoder_2_layer_call_and_return_conditional_losses_1897606

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
�
�	
%__inference_signature_wrapper_1898546
gene_input_layer
protein_input_layer
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:W
	unknown_6:
	unknown_7:	�}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:	�@

unknown_18:@

unknown_19:@}

unknown_20:}

unknown_21:}

unknown_22:}

unknown_23:}

unknown_24:}

unknown_25:@

unknown_26:

unknown_27:	}�

unknown_28:	�

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:	�

unknown_34:	�

unknown_35:	�

unknown_36:	�

unknown_37:W

unknown_38:W

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
 *;
_output_shapes)
':����������:���������W*L
_read_only_resource_inputs.
,*	
 !"#$%&'()*+*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_1896947p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������W`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes}
{:����������:���������W: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:����������
*
_user_specified_nameGene_Input_Layer:\X
'
_output_shapes
:���������W
-
_user_specified_nameProtein_Input_Layer
�%
�
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1899368

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
�
�
6__inference_BatchNormGeneEncode1_layer_call_fn_1899181

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
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1896971p
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
9__inference_BatchNormProteinEncode1_layer_call_fn_1899394

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1897182o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1897182

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identity��AssignMovingAvg�AssignMovingAvg/ReadVariableOp�AssignMovingAvg_1� AssignMovingAvg_1/ReadVariableOp�batchnorm/ReadVariableOp�batchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:�
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:���������l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0�
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
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
:*
dtype0�
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1897100

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
0__inference_gene_encoder_1_layer_call_fn_1899157

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
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_1897459p
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
�
�

*__inference_model_40_layer_call_fn_1898234
gene_input_layer
protein_input_layer
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:W
	unknown_6:
	unknown_7:	�}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:	�@

unknown_18:@

unknown_19:@}

unknown_20:}

unknown_21:}

unknown_22:}

unknown_23:}

unknown_24:}

unknown_25:@

unknown_26:

unknown_27:	}�

unknown_28:	�

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:	�

unknown_34:	�

unknown_35:	�

unknown_36:	�

unknown_37:W

unknown_38:W

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
 *;
_output_shapes)
':����������:���������W*@
_read_only_resource_inputs"
 	
"#&'()*+*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_40_layer_call_and_return_conditional_losses_1898053p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������W`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes}
{:����������:���������W: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:����������
*
_user_specified_nameGene_Input_Layer:\X
'
_output_shapes
:���������W
-
_user_specified_nameProtein_Input_Layer
�%
�
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1897018

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
6__inference_BatchNormGeneDecode1_layer_call_fn_1899527

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
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_1897264o
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
N__inference_gene_decoder_last_layer_call_and_return_conditional_losses_1899801

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
�

�
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_1899168

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
�
�
6__inference_BatchNormGeneDecode2_layer_call_fn_1899634

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
Q__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_1897299p
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
�f
�
E__inference_model_40_layer_call_and_return_conditional_losses_1898446
gene_input_layer
protein_input_layer*
gene_encoder_1_1898344:
��%
gene_encoder_1_1898346:	�+
batchnormgeneencode1_1898349:	�+
batchnormgeneencode1_1898351:	�+
batchnormgeneencode1_1898353:	�+
batchnormgeneencode1_1898355:	�+
protein_encoder_1_1898358:W'
protein_encoder_1_1898360:)
gene_encoder_2_1898363:	�}$
gene_encoder_2_1898365:}*
batchnormgeneencode2_1898368:}*
batchnormgeneencode2_1898370:}*
batchnormgeneencode2_1898372:}*
batchnormgeneencode2_1898374:}-
batchnormproteinencode1_1898377:-
batchnormproteinencode1_1898379:-
batchnormproteinencode1_1898381:-
batchnormproteinencode1_1898383:,
embeddingdimdense_1898387:	�@'
embeddingdimdense_1898389:@(
gene_decoder_1_1898392:@}$
gene_decoder_1_1898394:}*
batchnormgenedecode1_1898397:}*
batchnormgenedecode1_1898399:}*
batchnormgenedecode1_1898401:}*
batchnormgenedecode1_1898403:}+
protein_decoder_1_1898406:@'
protein_decoder_1_1898408:)
gene_decoder_2_1898411:	}�%
gene_decoder_2_1898413:	�-
batchnormproteindecode1_1898416:-
batchnormproteindecode1_1898418:-
batchnormproteindecode1_1898420:-
batchnormproteindecode1_1898422:+
batchnormgenedecode2_1898425:	�+
batchnormgenedecode2_1898427:	�+
batchnormgenedecode2_1898429:	�+
batchnormgenedecode2_1898431:	�.
protein_decoder_last_1898434:W*
protein_decoder_last_1898436:W-
gene_decoder_last_1898439:
��(
gene_decoder_last_1898441:	�
identity

identity_1��,BatchNormGeneDecode1/StatefulPartitionedCall�,BatchNormGeneDecode2/StatefulPartitionedCall�,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�/BatchNormProteinDecode1/StatefulPartitionedCall�/BatchNormProteinEncode1/StatefulPartitionedCall�)EmbeddingDimDense/StatefulPartitionedCall�&gene_decoder_1/StatefulPartitionedCall�&gene_decoder_2/StatefulPartitionedCall�)gene_decoder_last/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�)protein_decoder_1/StatefulPartitionedCall�,protein_decoder_last/StatefulPartitionedCall�)protein_encoder_1/StatefulPartitionedCall�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallgene_input_layergene_encoder_1_1898344gene_encoder_1_1898346*
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
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_1897459�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_1898349batchnormgeneencode1_1898351batchnormgeneencode1_1898353batchnormgeneencode1_1898355*
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
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1897018�
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallprotein_input_layerprotein_encoder_1_1898358protein_encoder_1_1898360*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_protein_encoder_1_layer_call_and_return_conditional_losses_1897485�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_1898363gene_encoder_2_1898365*
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
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_1897502�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_1898368batchnormgeneencode2_1898370batchnormgeneencode2_1898372batchnormgeneencode2_1898374*
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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1897100�
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_1898377batchnormproteinencode1_1898379batchnormproteinencode1_1898381batchnormproteinencode1_1898383*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1897182�
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
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1897533�
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_1898387embeddingdimdense_1898389*
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
GPU 2J 8� *W
fRRP
N__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_1897546�
&gene_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0gene_decoder_1_1898392gene_decoder_1_1898394*
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
K__inference_gene_decoder_1_layer_call_and_return_conditional_losses_1897563�
,BatchNormGeneDecode1/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_1/StatefulPartitionedCall:output:0batchnormgenedecode1_1898397batchnormgenedecode1_1898399batchnormgenedecode1_1898401batchnormgenedecode1_1898403*
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
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_1897264�
)protein_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0protein_decoder_1_1898406protein_decoder_1_1898408*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_protein_decoder_1_layer_call_and_return_conditional_losses_1897589�
&gene_decoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode1/StatefulPartitionedCall:output:0gene_decoder_2_1898411gene_decoder_2_1898413*
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
K__inference_gene_decoder_2_layer_call_and_return_conditional_losses_1897606�
/BatchNormProteinDecode1/StatefulPartitionedCallStatefulPartitionedCall2protein_decoder_1/StatefulPartitionedCall:output:0batchnormproteindecode1_1898416batchnormproteindecode1_1898418batchnormproteindecode1_1898420batchnormproteindecode1_1898422*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_1897428�
,BatchNormGeneDecode2/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_2/StatefulPartitionedCall:output:0batchnormgenedecode2_1898425batchnormgenedecode2_1898427batchnormgenedecode2_1898429batchnormgenedecode2_1898431*
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
Q__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_1897346�
,protein_decoder_last/StatefulPartitionedCallStatefulPartitionedCall8BatchNormProteinDecode1/StatefulPartitionedCall:output:0protein_decoder_last_1898434protein_decoder_last_1898436*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������W*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_protein_decoder_last_layer_call_and_return_conditional_losses_1897641�
)gene_decoder_last/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode2/StatefulPartitionedCall:output:0gene_decoder_last_1898439gene_decoder_last_1898441*
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
GPU 2J 8� *W
fRRP
N__inference_gene_decoder_last_layer_call_and_return_conditional_losses_1897658�
IdentityIdentity2gene_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������

Identity_1Identity5protein_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������W�
NoOpNoOp-^BatchNormGeneDecode1/StatefulPartitionedCall-^BatchNormGeneDecode2/StatefulPartitionedCall-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinDecode1/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_decoder_1/StatefulPartitionedCall'^gene_decoder_2/StatefulPartitionedCall*^gene_decoder_last/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_decoder_1/StatefulPartitionedCall-^protein_decoder_last/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes}
{:����������:���������W: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
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
_user_specified_nameGene_Input_Layer:\X
'
_output_shapes
:���������W
-
_user_specified_nameProtein_Input_Layer
�
�

*__inference_model_40_layer_call_fn_1897755
gene_input_layer
protein_input_layer
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:W
	unknown_6:
	unknown_7:	�}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:	�@

unknown_18:@

unknown_19:@}

unknown_20:}

unknown_21:}

unknown_22:}

unknown_23:}

unknown_24:}

unknown_25:@

unknown_26:

unknown_27:	}�

unknown_28:	�

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:	�

unknown_34:	�

unknown_35:	�

unknown_36:	�

unknown_37:W

unknown_38:W

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
 *;
_output_shapes)
':����������:���������W*L
_read_only_resource_inputs.
,*	
 !"#$%&'()*+*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_40_layer_call_and_return_conditional_losses_1897666p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������W`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes}
{:����������:���������W: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:����������
*
_user_specified_nameGene_Input_Layer:\X
'
_output_shapes
:���������W
-
_user_specified_nameProtein_Input_Layer
�
�
T__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_1899747

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
w
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1897533

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
&:���������}:���������:O K
'
_output_shapes
:���������}
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
N__inference_protein_encoder_1_layer_call_and_return_conditional_losses_1897485

inputs0
matmul_readvariableop_resource:W-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:W*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������W: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������W
 
_user_specified_nameinputs
ċ
�,
E__inference_model_40_layer_call_and_return_conditional_losses_1899148
inputs_0
inputs_1A
-gene_encoder_1_matmul_readvariableop_resource:
��=
.gene_encoder_1_biasadd_readvariableop_resource:	�K
<batchnormgeneencode1_assignmovingavg_readvariableop_resource:	�M
>batchnormgeneencode1_assignmovingavg_1_readvariableop_resource:	�I
:batchnormgeneencode1_batchnorm_mul_readvariableop_resource:	�E
6batchnormgeneencode1_batchnorm_readvariableop_resource:	�B
0protein_encoder_1_matmul_readvariableop_resource:W?
1protein_encoder_1_biasadd_readvariableop_resource:@
-gene_encoder_2_matmul_readvariableop_resource:	�}<
.gene_encoder_2_biasadd_readvariableop_resource:}J
<batchnormgeneencode2_assignmovingavg_readvariableop_resource:}L
>batchnormgeneencode2_assignmovingavg_1_readvariableop_resource:}H
:batchnormgeneencode2_batchnorm_mul_readvariableop_resource:}D
6batchnormgeneencode2_batchnorm_readvariableop_resource:}M
?batchnormproteinencode1_assignmovingavg_readvariableop_resource:O
Abatchnormproteinencode1_assignmovingavg_1_readvariableop_resource:K
=batchnormproteinencode1_batchnorm_mul_readvariableop_resource:G
9batchnormproteinencode1_batchnorm_readvariableop_resource:C
0embeddingdimdense_matmul_readvariableop_resource:	�@?
1embeddingdimdense_biasadd_readvariableop_resource:@?
-gene_decoder_1_matmul_readvariableop_resource:@}<
.gene_decoder_1_biasadd_readvariableop_resource:}J
<batchnormgenedecode1_assignmovingavg_readvariableop_resource:}L
>batchnormgenedecode1_assignmovingavg_1_readvariableop_resource:}H
:batchnormgenedecode1_batchnorm_mul_readvariableop_resource:}D
6batchnormgenedecode1_batchnorm_readvariableop_resource:}B
0protein_decoder_1_matmul_readvariableop_resource:@?
1protein_decoder_1_biasadd_readvariableop_resource:@
-gene_decoder_2_matmul_readvariableop_resource:	}�=
.gene_decoder_2_biasadd_readvariableop_resource:	�M
?batchnormproteindecode1_assignmovingavg_readvariableop_resource:O
Abatchnormproteindecode1_assignmovingavg_1_readvariableop_resource:K
=batchnormproteindecode1_batchnorm_mul_readvariableop_resource:G
9batchnormproteindecode1_batchnorm_readvariableop_resource:K
<batchnormgenedecode2_assignmovingavg_readvariableop_resource:	�M
>batchnormgenedecode2_assignmovingavg_1_readvariableop_resource:	�I
:batchnormgenedecode2_batchnorm_mul_readvariableop_resource:	�E
6batchnormgenedecode2_batchnorm_readvariableop_resource:	�E
3protein_decoder_last_matmul_readvariableop_resource:WB
4protein_decoder_last_biasadd_readvariableop_resource:WD
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

:W*
dtype0�
protein_encoder_1/MatMulMatMulinputs_1/protein_encoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(protein_encoder_1/BiasAdd/ReadVariableOpReadVariableOp1protein_encoder_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
protein_encoder_1/BiasAddBiasAdd"protein_encoder_1/MatMul:product:00protein_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������z
protein_encoder_1/SigmoidSigmoid"protein_encoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
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

:*
	keep_dims(�
,BatchNormProteinEncode1/moments/StopGradientStopGradient-BatchNormProteinEncode1/moments/mean:output:0*
T0*
_output_shapes

:�
1BatchNormProteinEncode1/moments/SquaredDifferenceSquaredDifferenceprotein_encoder_1/Sigmoid:y:05BatchNormProteinEncode1/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
:BatchNormProteinEncode1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
(BatchNormProteinEncode1/moments/varianceMean5BatchNormProteinEncode1/moments/SquaredDifference:z:0CBatchNormProteinEncode1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
'BatchNormProteinEncode1/moments/SqueezeSqueeze-BatchNormProteinEncode1/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
)BatchNormProteinEncode1/moments/Squeeze_1Squeeze1BatchNormProteinEncode1/moments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0�
+BatchNormProteinEncode1/AssignMovingAvg/subSub>BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOp:value:00BatchNormProteinEncode1/moments/Squeeze:output:0*
T0*
_output_shapes
:�
+BatchNormProteinEncode1/AssignMovingAvg/mulMul/BatchNormProteinEncode1/AssignMovingAvg/sub:z:06BatchNormProteinEncode1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
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
:*
dtype0�
-BatchNormProteinEncode1/AssignMovingAvg_1/subSub@BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp:value:02BatchNormProteinEncode1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
-BatchNormProteinEncode1/AssignMovingAvg_1/mulMul1BatchNormProteinEncode1/AssignMovingAvg_1/sub:z:08BatchNormProteinEncode1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
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
:�
'BatchNormProteinEncode1/batchnorm/RsqrtRsqrt)BatchNormProteinEncode1/batchnorm/add:z:0*
T0*
_output_shapes
:�
4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOpReadVariableOp=batchnormproteinencode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
%BatchNormProteinEncode1/batchnorm/mulMul+BatchNormProteinEncode1/batchnorm/Rsqrt:y:0<BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
'BatchNormProteinEncode1/batchnorm/mul_1Mulprotein_encoder_1/Sigmoid:y:0)BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
'BatchNormProteinEncode1/batchnorm/mul_2Mul0BatchNormProteinEncode1/moments/Squeeze:output:0)BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*
_output_shapes
:�
0BatchNormProteinEncode1/batchnorm/ReadVariableOpReadVariableOp9batchnormproteinencode1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
%BatchNormProteinEncode1/batchnorm/subSub8BatchNormProteinEncode1/batchnorm/ReadVariableOp:value:0+BatchNormProteinEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
'BatchNormProteinEncode1/batchnorm/add_1AddV2+BatchNormProteinEncode1/batchnorm/mul_1:z:0)BatchNormProteinEncode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������^
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

:@*
dtype0�
protein_decoder_1/MatMulMatMulEmbeddingDimDense/Sigmoid:y:0/protein_decoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
(protein_decoder_1/BiasAdd/ReadVariableOpReadVariableOp1protein_decoder_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
protein_decoder_1/BiasAddBiasAdd"protein_decoder_1/MatMul:product:00protein_decoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������z
protein_decoder_1/SigmoidSigmoid"protein_decoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:����������
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

:*
	keep_dims(�
,BatchNormProteinDecode1/moments/StopGradientStopGradient-BatchNormProteinDecode1/moments/mean:output:0*
T0*
_output_shapes

:�
1BatchNormProteinDecode1/moments/SquaredDifferenceSquaredDifferenceprotein_decoder_1/Sigmoid:y:05BatchNormProteinDecode1/moments/StopGradient:output:0*
T0*'
_output_shapes
:����������
:BatchNormProteinDecode1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: �
(BatchNormProteinDecode1/moments/varianceMean5BatchNormProteinDecode1/moments/SquaredDifference:z:0CBatchNormProteinDecode1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(�
'BatchNormProteinDecode1/moments/SqueezeSqueeze-BatchNormProteinDecode1/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 �
)BatchNormProteinDecode1/moments/Squeeze_1Squeeze1BatchNormProteinDecode1/moments/variance:output:0*
T0*
_output_shapes
:*
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
:*
dtype0�
+BatchNormProteinDecode1/AssignMovingAvg/subSub>BatchNormProteinDecode1/AssignMovingAvg/ReadVariableOp:value:00BatchNormProteinDecode1/moments/Squeeze:output:0*
T0*
_output_shapes
:�
+BatchNormProteinDecode1/AssignMovingAvg/mulMul/BatchNormProteinDecode1/AssignMovingAvg/sub:z:06BatchNormProteinDecode1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:�
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
:*
dtype0�
-BatchNormProteinDecode1/AssignMovingAvg_1/subSub@BatchNormProteinDecode1/AssignMovingAvg_1/ReadVariableOp:value:02BatchNormProteinDecode1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:�
-BatchNormProteinDecode1/AssignMovingAvg_1/mulMul1BatchNormProteinDecode1/AssignMovingAvg_1/sub:z:08BatchNormProteinDecode1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:�
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
:�
'BatchNormProteinDecode1/batchnorm/RsqrtRsqrt)BatchNormProteinDecode1/batchnorm/add:z:0*
T0*
_output_shapes
:�
4BatchNormProteinDecode1/batchnorm/mul/ReadVariableOpReadVariableOp=batchnormproteindecode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0�
%BatchNormProteinDecode1/batchnorm/mulMul+BatchNormProteinDecode1/batchnorm/Rsqrt:y:0<BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:�
'BatchNormProteinDecode1/batchnorm/mul_1Mulprotein_decoder_1/Sigmoid:y:0)BatchNormProteinDecode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:����������
'BatchNormProteinDecode1/batchnorm/mul_2Mul0BatchNormProteinDecode1/moments/Squeeze:output:0)BatchNormProteinDecode1/batchnorm/mul:z:0*
T0*
_output_shapes
:�
0BatchNormProteinDecode1/batchnorm/ReadVariableOpReadVariableOp9batchnormproteindecode1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0�
%BatchNormProteinDecode1/batchnorm/subSub8BatchNormProteinDecode1/batchnorm/ReadVariableOp:value:0+BatchNormProteinDecode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:�
'BatchNormProteinDecode1/batchnorm/add_1AddV2+BatchNormProteinDecode1/batchnorm/mul_1:z:0)BatchNormProteinDecode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}
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

:W*
dtype0�
protein_decoder_last/MatMulMatMul+BatchNormProteinDecode1/batchnorm/add_1:z:02protein_decoder_last/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������W�
+protein_decoder_last/BiasAdd/ReadVariableOpReadVariableOp4protein_decoder_last_biasadd_readvariableop_resource*
_output_shapes
:W*
dtype0�
protein_decoder_last/BiasAddBiasAdd%protein_decoder_last/MatMul:product:03protein_decoder_last/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������W�
protein_decoder_last/SigmoidSigmoid%protein_decoder_last/BiasAdd:output:0*
T0*'
_output_shapes
:���������W�
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
:����������q

Identity_1Identity protein_decoder_last/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������W�
NoOpNoOp%^BatchNormGeneDecode1/AssignMovingAvg4^BatchNormGeneDecode1/AssignMovingAvg/ReadVariableOp'^BatchNormGeneDecode1/AssignMovingAvg_16^BatchNormGeneDecode1/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneDecode1/batchnorm/ReadVariableOp2^BatchNormGeneDecode1/batchnorm/mul/ReadVariableOp%^BatchNormGeneDecode2/AssignMovingAvg4^BatchNormGeneDecode2/AssignMovingAvg/ReadVariableOp'^BatchNormGeneDecode2/AssignMovingAvg_16^BatchNormGeneDecode2/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneDecode2/batchnorm/ReadVariableOp2^BatchNormGeneDecode2/batchnorm/mul/ReadVariableOp%^BatchNormGeneEncode1/AssignMovingAvg4^BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOp'^BatchNormGeneEncode1/AssignMovingAvg_16^BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneEncode1/batchnorm/ReadVariableOp2^BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp%^BatchNormGeneEncode2/AssignMovingAvg4^BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOp'^BatchNormGeneEncode2/AssignMovingAvg_16^BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneEncode2/batchnorm/ReadVariableOp2^BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp(^BatchNormProteinDecode1/AssignMovingAvg7^BatchNormProteinDecode1/AssignMovingAvg/ReadVariableOp*^BatchNormProteinDecode1/AssignMovingAvg_19^BatchNormProteinDecode1/AssignMovingAvg_1/ReadVariableOp1^BatchNormProteinDecode1/batchnorm/ReadVariableOp5^BatchNormProteinDecode1/batchnorm/mul/ReadVariableOp(^BatchNormProteinEncode1/AssignMovingAvg7^BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOp*^BatchNormProteinEncode1/AssignMovingAvg_19^BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp1^BatchNormProteinEncode1/batchnorm/ReadVariableOp5^BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp)^EmbeddingDimDense/BiasAdd/ReadVariableOp(^EmbeddingDimDense/MatMul/ReadVariableOp&^gene_decoder_1/BiasAdd/ReadVariableOp%^gene_decoder_1/MatMul/ReadVariableOp&^gene_decoder_2/BiasAdd/ReadVariableOp%^gene_decoder_2/MatMul/ReadVariableOp)^gene_decoder_last/BiasAdd/ReadVariableOp(^gene_decoder_last/MatMul/ReadVariableOp&^gene_encoder_1/BiasAdd/ReadVariableOp%^gene_encoder_1/MatMul/ReadVariableOp&^gene_encoder_2/BiasAdd/ReadVariableOp%^gene_encoder_2/MatMul/ReadVariableOp)^protein_decoder_1/BiasAdd/ReadVariableOp(^protein_decoder_1/MatMul/ReadVariableOp,^protein_decoder_last/BiasAdd/ReadVariableOp+^protein_decoder_last/MatMul/ReadVariableOp)^protein_encoder_1/BiasAdd/ReadVariableOp(^protein_encoder_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes}
{:����������:���������W: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
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
inputs/0:QM
'
_output_shapes
:���������W
"
_user_specified_name
inputs/1
�

�
Q__inference_protein_decoder_last_layer_call_and_return_conditional_losses_1897641

inputs0
matmul_readvariableop_resource:W-
biasadd_readvariableop_resource:W
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:W*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������Wr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:W*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������WV
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������WZ
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������Ww
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_1897459

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
�f
�
E__inference_model_40_layer_call_and_return_conditional_losses_1898340
gene_input_layer
protein_input_layer*
gene_encoder_1_1898238:
��%
gene_encoder_1_1898240:	�+
batchnormgeneencode1_1898243:	�+
batchnormgeneencode1_1898245:	�+
batchnormgeneencode1_1898247:	�+
batchnormgeneencode1_1898249:	�+
protein_encoder_1_1898252:W'
protein_encoder_1_1898254:)
gene_encoder_2_1898257:	�}$
gene_encoder_2_1898259:}*
batchnormgeneencode2_1898262:}*
batchnormgeneencode2_1898264:}*
batchnormgeneencode2_1898266:}*
batchnormgeneencode2_1898268:}-
batchnormproteinencode1_1898271:-
batchnormproteinencode1_1898273:-
batchnormproteinencode1_1898275:-
batchnormproteinencode1_1898277:,
embeddingdimdense_1898281:	�@'
embeddingdimdense_1898283:@(
gene_decoder_1_1898286:@}$
gene_decoder_1_1898288:}*
batchnormgenedecode1_1898291:}*
batchnormgenedecode1_1898293:}*
batchnormgenedecode1_1898295:}*
batchnormgenedecode1_1898297:}+
protein_decoder_1_1898300:@'
protein_decoder_1_1898302:)
gene_decoder_2_1898305:	}�%
gene_decoder_2_1898307:	�-
batchnormproteindecode1_1898310:-
batchnormproteindecode1_1898312:-
batchnormproteindecode1_1898314:-
batchnormproteindecode1_1898316:+
batchnormgenedecode2_1898319:	�+
batchnormgenedecode2_1898321:	�+
batchnormgenedecode2_1898323:	�+
batchnormgenedecode2_1898325:	�.
protein_decoder_last_1898328:W*
protein_decoder_last_1898330:W-
gene_decoder_last_1898333:
��(
gene_decoder_last_1898335:	�
identity

identity_1��,BatchNormGeneDecode1/StatefulPartitionedCall�,BatchNormGeneDecode2/StatefulPartitionedCall�,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�/BatchNormProteinDecode1/StatefulPartitionedCall�/BatchNormProteinEncode1/StatefulPartitionedCall�)EmbeddingDimDense/StatefulPartitionedCall�&gene_decoder_1/StatefulPartitionedCall�&gene_decoder_2/StatefulPartitionedCall�)gene_decoder_last/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�)protein_decoder_1/StatefulPartitionedCall�,protein_decoder_last/StatefulPartitionedCall�)protein_encoder_1/StatefulPartitionedCall�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallgene_input_layergene_encoder_1_1898238gene_encoder_1_1898240*
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
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_1897459�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_1898243batchnormgeneencode1_1898245batchnormgeneencode1_1898247batchnormgeneencode1_1898249*
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
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1896971�
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallprotein_input_layerprotein_encoder_1_1898252protein_encoder_1_1898254*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_protein_encoder_1_layer_call_and_return_conditional_losses_1897485�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_1898257gene_encoder_2_1898259*
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
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_1897502�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_1898262batchnormgeneencode2_1898264batchnormgeneencode2_1898266batchnormgeneencode2_1898268*
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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1897053�
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_1898271batchnormproteinencode1_1898273batchnormproteinencode1_1898275batchnormproteinencode1_1898277*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1897135�
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
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1897533�
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_1898281embeddingdimdense_1898283*
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
GPU 2J 8� *W
fRRP
N__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_1897546�
&gene_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0gene_decoder_1_1898286gene_decoder_1_1898288*
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
K__inference_gene_decoder_1_layer_call_and_return_conditional_losses_1897563�
,BatchNormGeneDecode1/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_1/StatefulPartitionedCall:output:0batchnormgenedecode1_1898291batchnormgenedecode1_1898293batchnormgenedecode1_1898295batchnormgenedecode1_1898297*
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
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_1897217�
)protein_decoder_1/StatefulPartitionedCallStatefulPartitionedCall2EmbeddingDimDense/StatefulPartitionedCall:output:0protein_decoder_1_1898300protein_decoder_1_1898302*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_protein_decoder_1_layer_call_and_return_conditional_losses_1897589�
&gene_decoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode1/StatefulPartitionedCall:output:0gene_decoder_2_1898305gene_decoder_2_1898307*
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
K__inference_gene_decoder_2_layer_call_and_return_conditional_losses_1897606�
/BatchNormProteinDecode1/StatefulPartitionedCallStatefulPartitionedCall2protein_decoder_1/StatefulPartitionedCall:output:0batchnormproteindecode1_1898310batchnormproteindecode1_1898312batchnormproteindecode1_1898314batchnormproteindecode1_1898316*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_1897381�
,BatchNormGeneDecode2/StatefulPartitionedCallStatefulPartitionedCall/gene_decoder_2/StatefulPartitionedCall:output:0batchnormgenedecode2_1898319batchnormgenedecode2_1898321batchnormgenedecode2_1898323batchnormgenedecode2_1898325*
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
Q__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_1897299�
,protein_decoder_last/StatefulPartitionedCallStatefulPartitionedCall8BatchNormProteinDecode1/StatefulPartitionedCall:output:0protein_decoder_last_1898328protein_decoder_last_1898330*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������W*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_protein_decoder_last_layer_call_and_return_conditional_losses_1897641�
)gene_decoder_last/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneDecode2/StatefulPartitionedCall:output:0gene_decoder_last_1898333gene_decoder_last_1898335*
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
GPU 2J 8� *W
fRRP
N__inference_gene_decoder_last_layer_call_and_return_conditional_losses_1897658�
IdentityIdentity2gene_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:�����������

Identity_1Identity5protein_decoder_last/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������W�
NoOpNoOp-^BatchNormGeneDecode1/StatefulPartitionedCall-^BatchNormGeneDecode2/StatefulPartitionedCall-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinDecode1/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_decoder_1/StatefulPartitionedCall'^gene_decoder_2/StatefulPartitionedCall*^gene_decoder_last/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_decoder_1/StatefulPartitionedCall-^protein_decoder_last/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes}
{:����������:���������W: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
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
_user_specified_nameGene_Input_Layer:\X
'
_output_shapes
:���������W
-
_user_specified_nameProtein_Input_Layer
�
�
Q__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_1899667

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
9__inference_BatchNormProteinEncode1_layer_call_fn_1899381

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1897135o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1899248

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
N__inference_protein_decoder_1_layer_call_and_return_conditional_losses_1899621

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
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
K__inference_gene_decoder_2_layer_call_and_return_conditional_losses_1899601

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
�

�
K__inference_gene_decoder_1_layer_call_and_return_conditional_losses_1897563

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
�

�
N__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_1897546

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
�
�
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1899334

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
�
�
0__inference_gene_encoder_2_layer_call_fn_1899257

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
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_1897502o
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
�
�
6__inference_BatchNormGeneEncode1_layer_call_fn_1899194

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
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1897018p
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
0__inference_gene_decoder_1_layer_call_fn_1899490

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
K__inference_gene_decoder_1_layer_call_and_return_conditional_losses_1897563o
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
�
�
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1896971

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
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1899214

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

�
N__inference_gene_decoder_last_layer_call_and_return_conditional_losses_1897658

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
�
�	
*__inference_model_40_layer_call_fn_1898730
inputs_0
inputs_1
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:W
	unknown_6:
	unknown_7:	�}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:	�@

unknown_18:@

unknown_19:@}

unknown_20:}

unknown_21:}

unknown_22:}

unknown_23:}

unknown_24:}

unknown_25:@

unknown_26:

unknown_27:	}�

unknown_28:	�

unknown_29:

unknown_30:

unknown_31:

unknown_32:

unknown_33:	�

unknown_34:	�

unknown_35:	�

unknown_36:	�

unknown_37:W

unknown_38:W

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
 *;
_output_shapes)
':����������:���������W*@
_read_only_resource_inputs"
 	
"#&'()*+*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_40_layer_call_and_return_conditional_losses_1898053p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:���������W`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes}
{:����������:���������W: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:����������
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������W
"
_user_specified_name
inputs/1
�
�
6__inference_BatchNormGeneDecode2_layer_call_fn_1899647

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
Q__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_1897346p
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
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_1897264

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
N__inference_protein_decoder_1_layer_call_and_return_conditional_losses_1897589

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
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
^
2__inference_ConcatenateLayer_layer_call_fn_1899454
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
 *-
config_proto

CPU

GPU 2J 8� *V
fQRO
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1897533a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������}:���������:Q M
'
_output_shapes
:���������}
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs/1
�

�
N__inference_protein_encoder_1_layer_call_and_return_conditional_losses_1899288

inputs0
matmul_readvariableop_resource:W-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:W*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:���������Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������W: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������W
 
_user_specified_nameinputs
�
�
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1899414

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identity��batchnorm/ReadVariableOp�batchnorm/ReadVariableOp_1�batchnorm/ReadVariableOp_2�batchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
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
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:���������z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:���������b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
6__inference_protein_decoder_last_layer_call_fn_1899810

inputs
unknown:W
	unknown_0:W
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������W*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *Z
fURS
Q__inference_protein_decoder_last_layer_call_and_return_conditional_losses_1897641o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������W`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_1899547

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
9__inference_BatchNormProteinDecode1_layer_call_fn_1899727

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_1897428o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�	L
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
S
Protein_Input_Layer<
%serving_default_Protein_Input_Layer:0���������WF
gene_decoder_last1
StatefulPartitionedCall:0����������H
protein_decoder_last0
StatefulPartitionedCall:1���������Wtensorflow/serving/predict:��
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
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias"
_tf_keras_layer
�
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses
*axis
	+gamma
,beta
-moving_mean
.moving_variance"
_tf_keras_layer
"
_tf_keras_input_layer
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

5kernel
6bias"
_tf_keras_layer
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses

=kernel
>bias"
_tf_keras_layer
�
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses
Eaxis
	Fgamma
Gbeta
Hmoving_mean
Imoving_variance"
_tf_keras_layer
�
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses
Paxis
	Qgamma
Rbeta
Smoving_mean
Tmoving_variance"
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
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses

akernel
bbias"
_tf_keras_layer
�
c	variables
dtrainable_variables
eregularization_losses
f	keras_api
g__call__
*h&call_and_return_all_conditional_losses

ikernel
jbias"
_tf_keras_layer
�
k	variables
ltrainable_variables
mregularization_losses
n	keras_api
o__call__
*p&call_and_return_all_conditional_losses
qaxis
	rgamma
sbeta
tmoving_mean
umoving_variance"
_tf_keras_layer
�
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses

|kernel
}bias"
_tf_keras_layer
�
~	variables
trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�axis

�gamma
	�beta
�moving_mean
�moving_variance"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
"0
#1
+2
,3
-4
.5
56
67
=8
>9
F10
G11
H12
I13
Q14
R15
S16
T17
a18
b19
i20
j21
r22
s23
t24
u25
|26
}27
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
�41"
trackable_list_wrapper
�
"0
#1
+2
,3
54
65
=6
>7
F8
G9
Q10
R11
a12
b13
i14
j15
r16
s17
|18
}19
�20
�21
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
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_1
�trace_2
�trace_32�
*__inference_model_40_layer_call_fn_1897755
*__inference_model_40_layer_call_fn_1898638
*__inference_model_40_layer_call_fn_1898730
*__inference_model_40_layer_call_fn_1898234�
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
E__inference_model_40_layer_call_and_return_conditional_losses_1898897
E__inference_model_40_layer_call_and_return_conditional_losses_1899148
E__inference_model_40_layer_call_and_return_conditional_losses_1898340
E__inference_model_40_layer_call_and_return_conditional_losses_1898446�
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
"__inference__wrapped_model_1896947Gene_Input_LayerProtein_Input_Layer"�
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
�
	�iter
�beta_1
�beta_2

�decay
�learning_rate"m�#m�+m�,m�5m�6m�=m�>m�Fm�Gm�Qm�Rm�am�bm�im�jm�rm�sm�|m�}m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�	�m�"v�#v�+v�,v�5v�6v�=v�>v�Fv�Gv�Qv�Rv�av�bv�iv�jv�rv�sv�|v�}v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�	�v�"
	optimizer
-
�serving_default"
signature_map
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_gene_encoder_1_layer_call_fn_1899157�
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
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_1899168�
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
+0
,1
-2
.3"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_BatchNormGeneEncode1_layer_call_fn_1899181
6__inference_BatchNormGeneEncode1_layer_call_fn_1899194�
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
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1899214
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1899248�
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
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_gene_encoder_2_layer_call_fn_1899257�
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
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_1899268�
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
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_protein_encoder_1_layer_call_fn_1899277�
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
N__inference_protein_encoder_1_layer_call_and_return_conditional_losses_1899288�
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
*:(W2protein_encoder_1/kernel
$:"2protein_encoder_1/bias
<
F0
G1
H2
I3"
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
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_BatchNormGeneEncode2_layer_call_fn_1899301
6__inference_BatchNormGeneEncode2_layer_call_fn_1899314�
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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1899334
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1899368�
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
<
Q0
R1
S2
T3"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
9__inference_BatchNormProteinEncode1_layer_call_fn_1899381
9__inference_BatchNormProteinEncode1_layer_call_fn_1899394�
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
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1899414
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1899448�
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
+:)2BatchNormProteinEncode1/gamma
*:(2BatchNormProteinEncode1/beta
3:1 (2#BatchNormProteinEncode1/moving_mean
7:5 (2'BatchNormProteinEncode1/moving_variance
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
�
�trace_02�
2__inference_ConcatenateLayer_layer_call_fn_1899454�
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1899461�
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
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_EmbeddingDimDense_layer_call_fn_1899470�
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
N__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_1899481�
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
+:)	�@2EmbeddingDimDense/kernel
$:"@2EmbeddingDimDense/bias
.
i0
j1"
trackable_list_wrapper
.
i0
j1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
c	variables
dtrainable_variables
eregularization_losses
g__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_gene_decoder_1_layer_call_fn_1899490�
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
 z�trace_0
�
�trace_02�
K__inference_gene_decoder_1_layer_call_and_return_conditional_losses_1899501�
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
 z�trace_0
':%@}2gene_decoder_1/kernel
!:}2gene_decoder_1/bias
<
r0
s1
t2
u3"
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
k	variables
ltrainable_variables
mregularization_losses
o__call__
*p&call_and_return_all_conditional_losses
&p"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
6__inference_BatchNormGeneDecode1_layer_call_fn_1899514
6__inference_BatchNormGeneDecode1_layer_call_fn_1899527�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_1899547
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_1899581�
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
 z�trace_0z�trace_1
 "
trackable_list_wrapper
(:&}2BatchNormGeneDecode1/gamma
':%}2BatchNormGeneDecode1/beta
0:.} (2 BatchNormGeneDecode1/moving_mean
4:2} (2$BatchNormGeneDecode1/moving_variance
.
|0
}1"
trackable_list_wrapper
.
|0
}1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_gene_decoder_2_layer_call_fn_1899590�
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
 z�trace_0
�
�trace_02�
K__inference_gene_decoder_2_layer_call_and_return_conditional_losses_1899601�
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
 z�trace_0
(:&	}�2gene_decoder_2/kernel
": �2gene_decoder_2/bias
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
~	variables
trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_protein_decoder_1_layer_call_fn_1899610�
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
 z�trace_0
�
�trace_02�
N__inference_protein_decoder_1_layer_call_and_return_conditional_losses_1899621�
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
 z�trace_0
*:(@2protein_decoder_1/kernel
$:"2protein_decoder_1/bias
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
�
�trace_0
�trace_12�
6__inference_BatchNormGeneDecode2_layer_call_fn_1899634
6__inference_BatchNormGeneDecode2_layer_call_fn_1899647�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
Q__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_1899667
Q__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_1899701�
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
 z�trace_0z�trace_1
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
�
�trace_0
�trace_12�
9__inference_BatchNormProteinDecode1_layer_call_fn_1899714
9__inference_BatchNormProteinDecode1_layer_call_fn_1899727�
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
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
T__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_1899747
T__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_1899781�
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
 z�trace_0z�trace_1
 "
trackable_list_wrapper
+:)2BatchNormProteinDecode1/gamma
*:(2BatchNormProteinDecode1/beta
3:1 (2#BatchNormProteinDecode1/moving_mean
7:5 (2'BatchNormProteinDecode1/moving_variance
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
�
�trace_02�
3__inference_gene_decoder_last_layer_call_fn_1899790�
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
 z�trace_0
�
�trace_02�
N__inference_gene_decoder_last_layer_call_and_return_conditional_losses_1899801�
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
 z�trace_0
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
�
�trace_02�
6__inference_protein_decoder_last_layer_call_fn_1899810�
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
 z�trace_0
�
�trace_02�
Q__inference_protein_decoder_last_layer_call_and_return_conditional_losses_1899821�
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
 z�trace_0
-:+W2protein_decoder_last/kernel
':%W2protein_decoder_last/bias
z
-0
.1
H2
I3
S4
T5
t6
u7
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
�B�
*__inference_model_40_layer_call_fn_1897755Gene_Input_LayerProtein_Input_Layer"�
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
*__inference_model_40_layer_call_fn_1898638inputs/0inputs/1"�
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
*__inference_model_40_layer_call_fn_1898730inputs/0inputs/1"�
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
*__inference_model_40_layer_call_fn_1898234Gene_Input_LayerProtein_Input_Layer"�
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
E__inference_model_40_layer_call_and_return_conditional_losses_1898897inputs/0inputs/1"�
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
E__inference_model_40_layer_call_and_return_conditional_losses_1899148inputs/0inputs/1"�
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
E__inference_model_40_layer_call_and_return_conditional_losses_1898340Gene_Input_LayerProtein_Input_Layer"�
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
E__inference_model_40_layer_call_and_return_conditional_losses_1898446Gene_Input_LayerProtein_Input_Layer"�
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
%__inference_signature_wrapper_1898546Gene_Input_LayerProtein_Input_Layer"�
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
0__inference_gene_encoder_1_layer_call_fn_1899157inputs"�
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
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_1899168inputs"�
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
-0
.1"
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
6__inference_BatchNormGeneEncode1_layer_call_fn_1899181inputs"�
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
6__inference_BatchNormGeneEncode1_layer_call_fn_1899194inputs"�
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
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1899214inputs"�
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
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1899248inputs"�
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
0__inference_gene_encoder_2_layer_call_fn_1899257inputs"�
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
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_1899268inputs"�
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_protein_encoder_1_layer_call_fn_1899277inputs"�
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
N__inference_protein_encoder_1_layer_call_and_return_conditional_losses_1899288inputs"�
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
H0
I1"
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
6__inference_BatchNormGeneEncode2_layer_call_fn_1899301inputs"�
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
6__inference_BatchNormGeneEncode2_layer_call_fn_1899314inputs"�
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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1899334inputs"�
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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1899368inputs"�
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
.
S0
T1"
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
9__inference_BatchNormProteinEncode1_layer_call_fn_1899381inputs"�
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
9__inference_BatchNormProteinEncode1_layer_call_fn_1899394inputs"�
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
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1899414inputs"�
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
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1899448inputs"�
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
2__inference_ConcatenateLayer_layer_call_fn_1899454inputs/0inputs/1"�
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
�B�
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1899461inputs/0inputs/1"�
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_EmbeddingDimDense_layer_call_fn_1899470inputs"�
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
N__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_1899481inputs"�
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
0__inference_gene_decoder_1_layer_call_fn_1899490inputs"�
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
K__inference_gene_decoder_1_layer_call_and_return_conditional_losses_1899501inputs"�
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
t0
u1"
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
6__inference_BatchNormGeneDecode1_layer_call_fn_1899514inputs"�
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
6__inference_BatchNormGeneDecode1_layer_call_fn_1899527inputs"�
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
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_1899547inputs"�
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
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_1899581inputs"�
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
0__inference_gene_decoder_2_layer_call_fn_1899590inputs"�
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
K__inference_gene_decoder_2_layer_call_and_return_conditional_losses_1899601inputs"�
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
3__inference_protein_decoder_1_layer_call_fn_1899610inputs"�
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
N__inference_protein_decoder_1_layer_call_and_return_conditional_losses_1899621inputs"�
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
�B�
6__inference_BatchNormGeneDecode2_layer_call_fn_1899634inputs"�
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
6__inference_BatchNormGeneDecode2_layer_call_fn_1899647inputs"�
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
Q__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_1899667inputs"�
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
Q__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_1899701inputs"�
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
�B�
9__inference_BatchNormProteinDecode1_layer_call_fn_1899714inputs"�
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
9__inference_BatchNormProteinDecode1_layer_call_fn_1899727inputs"�
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
T__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_1899747inputs"�
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
T__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_1899781inputs"�
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
3__inference_gene_decoder_last_layer_call_fn_1899790inputs"�
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
N__inference_gene_decoder_last_layer_call_and_return_conditional_losses_1899801inputs"�
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
6__inference_protein_decoder_last_layer_call_fn_1899810inputs"�
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
�B�
Q__inference_protein_decoder_last_layer_call_and_return_conditional_losses_1899821inputs"�
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
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
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
:  (2total
:  (2count
.:,
��2Adam/gene_encoder_1/kernel/m
':%�2Adam/gene_encoder_1/bias/m
.:,�2!Adam/BatchNormGeneEncode1/gamma/m
-:+�2 Adam/BatchNormGeneEncode1/beta/m
-:+	�}2Adam/gene_encoder_2/kernel/m
&:$}2Adam/gene_encoder_2/bias/m
/:-W2Adam/protein_encoder_1/kernel/m
):'2Adam/protein_encoder_1/bias/m
-:+}2!Adam/BatchNormGeneEncode2/gamma/m
,:*}2 Adam/BatchNormGeneEncode2/beta/m
0:.2$Adam/BatchNormProteinEncode1/gamma/m
/:-2#Adam/BatchNormProteinEncode1/beta/m
0:.	�@2Adam/EmbeddingDimDense/kernel/m
):'@2Adam/EmbeddingDimDense/bias/m
,:*@}2Adam/gene_decoder_1/kernel/m
&:$}2Adam/gene_decoder_1/bias/m
-:+}2!Adam/BatchNormGeneDecode1/gamma/m
,:*}2 Adam/BatchNormGeneDecode1/beta/m
-:+	}�2Adam/gene_decoder_2/kernel/m
':%�2Adam/gene_decoder_2/bias/m
/:-@2Adam/protein_decoder_1/kernel/m
):'2Adam/protein_decoder_1/bias/m
.:,�2!Adam/BatchNormGeneDecode2/gamma/m
-:+�2 Adam/BatchNormGeneDecode2/beta/m
0:.2$Adam/BatchNormProteinDecode1/gamma/m
/:-2#Adam/BatchNormProteinDecode1/beta/m
1:/
��2Adam/gene_decoder_last/kernel/m
*:(�2Adam/gene_decoder_last/bias/m
2:0W2"Adam/protein_decoder_last/kernel/m
,:*W2 Adam/protein_decoder_last/bias/m
.:,
��2Adam/gene_encoder_1/kernel/v
':%�2Adam/gene_encoder_1/bias/v
.:,�2!Adam/BatchNormGeneEncode1/gamma/v
-:+�2 Adam/BatchNormGeneEncode1/beta/v
-:+	�}2Adam/gene_encoder_2/kernel/v
&:$}2Adam/gene_encoder_2/bias/v
/:-W2Adam/protein_encoder_1/kernel/v
):'2Adam/protein_encoder_1/bias/v
-:+}2!Adam/BatchNormGeneEncode2/gamma/v
,:*}2 Adam/BatchNormGeneEncode2/beta/v
0:.2$Adam/BatchNormProteinEncode1/gamma/v
/:-2#Adam/BatchNormProteinEncode1/beta/v
0:.	�@2Adam/EmbeddingDimDense/kernel/v
):'@2Adam/EmbeddingDimDense/bias/v
,:*@}2Adam/gene_decoder_1/kernel/v
&:$}2Adam/gene_decoder_1/bias/v
-:+}2!Adam/BatchNormGeneDecode1/gamma/v
,:*}2 Adam/BatchNormGeneDecode1/beta/v
-:+	}�2Adam/gene_decoder_2/kernel/v
':%�2Adam/gene_decoder_2/bias/v
/:-@2Adam/protein_decoder_1/kernel/v
):'2Adam/protein_decoder_1/bias/v
.:,�2!Adam/BatchNormGeneDecode2/gamma/v
-:+�2 Adam/BatchNormGeneDecode2/beta/v
0:.2$Adam/BatchNormProteinDecode1/gamma/v
/:-2#Adam/BatchNormProteinDecode1/beta/v
1:/
��2Adam/gene_decoder_last/kernel/v
*:(�2Adam/gene_decoder_last/bias/v
2:0W2"Adam/protein_decoder_last/kernel/v
,:*W2 Adam/protein_decoder_last/bias/v�
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_1899547burts3�0
)�&
 �
inputs���������}
p 
� "%�"
�
0���������}
� �
Q__inference_BatchNormGeneDecode1_layer_call_and_return_conditional_losses_1899581bturs3�0
)�&
 �
inputs���������}
p
� "%�"
�
0���������}
� �
6__inference_BatchNormGeneDecode1_layer_call_fn_1899514Uurts3�0
)�&
 �
inputs���������}
p 
� "����������}�
6__inference_BatchNormGeneDecode1_layer_call_fn_1899527Uturs3�0
)�&
 �
inputs���������}
p
� "����������}�
Q__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_1899667h����4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
Q__inference_BatchNormGeneDecode2_layer_call_and_return_conditional_losses_1899701h����4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
6__inference_BatchNormGeneDecode2_layer_call_fn_1899634[����4�1
*�'
!�
inputs����������
p 
� "������������
6__inference_BatchNormGeneDecode2_layer_call_fn_1899647[����4�1
*�'
!�
inputs����������
p
� "������������
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1899214d.+-,4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1899248d-.+,4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
6__inference_BatchNormGeneEncode1_layer_call_fn_1899181W.+-,4�1
*�'
!�
inputs����������
p 
� "������������
6__inference_BatchNormGeneEncode1_layer_call_fn_1899194W-.+,4�1
*�'
!�
inputs����������
p
� "������������
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1899334bIFHG3�0
)�&
 �
inputs���������}
p 
� "%�"
�
0���������}
� �
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1899368bHIFG3�0
)�&
 �
inputs���������}
p
� "%�"
�
0���������}
� �
6__inference_BatchNormGeneEncode2_layer_call_fn_1899301UIFHG3�0
)�&
 �
inputs���������}
p 
� "����������}�
6__inference_BatchNormGeneEncode2_layer_call_fn_1899314UHIFG3�0
)�&
 �
inputs���������}
p
� "����������}�
T__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_1899747f����3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
T__inference_BatchNormProteinDecode1_layer_call_and_return_conditional_losses_1899781f����3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
9__inference_BatchNormProteinDecode1_layer_call_fn_1899714Y����3�0
)�&
 �
inputs���������
p 
� "�����������
9__inference_BatchNormProteinDecode1_layer_call_fn_1899727Y����3�0
)�&
 �
inputs���������
p
� "�����������
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1899414bTQSR3�0
)�&
 �
inputs���������
p 
� "%�"
�
0���������
� �
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1899448bSTQR3�0
)�&
 �
inputs���������
p
� "%�"
�
0���������
� �
9__inference_BatchNormProteinEncode1_layer_call_fn_1899381UTQSR3�0
)�&
 �
inputs���������
p 
� "�����������
9__inference_BatchNormProteinEncode1_layer_call_fn_1899394USTQR3�0
)�&
 �
inputs���������
p
� "�����������
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1899461�Z�W
P�M
K�H
"�
inputs/0���������}
"�
inputs/1���������
� "&�#
�
0����������
� �
2__inference_ConcatenateLayer_layer_call_fn_1899454wZ�W
P�M
K�H
"�
inputs/0���������}
"�
inputs/1���������
� "������������
N__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_1899481]ab0�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� �
3__inference_EmbeddingDimDense_layer_call_fn_1899470Pab0�-
&�#
!�
inputs����������
� "����������@�
"__inference__wrapped_model_1896947�8"#.+-,=>56IFHGTQSRabijurts��|}������������n�k
d�a
_�\
+�(
Gene_Input_Layer����������
-�*
Protein_Input_Layer���������W
� "���
A
gene_decoder_last,�)
gene_decoder_last����������
F
protein_decoder_last.�+
protein_decoder_last���������W�
K__inference_gene_decoder_1_layer_call_and_return_conditional_losses_1899501\ij/�,
%�"
 �
inputs���������@
� "%�"
�
0���������}
� �
0__inference_gene_decoder_1_layer_call_fn_1899490Oij/�,
%�"
 �
inputs���������@
� "����������}�
K__inference_gene_decoder_2_layer_call_and_return_conditional_losses_1899601]|}/�,
%�"
 �
inputs���������}
� "&�#
�
0����������
� �
0__inference_gene_decoder_2_layer_call_fn_1899590P|}/�,
%�"
 �
inputs���������}
� "������������
N__inference_gene_decoder_last_layer_call_and_return_conditional_losses_1899801`��0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
3__inference_gene_decoder_last_layer_call_fn_1899790S��0�-
&�#
!�
inputs����������
� "������������
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_1899168^"#0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
0__inference_gene_encoder_1_layer_call_fn_1899157Q"#0�-
&�#
!�
inputs����������
� "������������
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_1899268]560�-
&�#
!�
inputs����������
� "%�"
�
0���������}
� �
0__inference_gene_encoder_2_layer_call_fn_1899257P560�-
&�#
!�
inputs����������
� "����������}�
E__inference_model_40_layer_call_and_return_conditional_losses_1898340�8"#.+-,=>56IFHGTQSRabijurts��|}������������v�s
l�i
_�\
+�(
Gene_Input_Layer����������
-�*
Protein_Input_Layer���������W
p 

 
� "L�I
B�?
�
0/0����������
�
0/1���������W
� �
E__inference_model_40_layer_call_and_return_conditional_losses_1898446�8"#-.+,=>56HIFGSTQRabijturs��|}������������v�s
l�i
_�\
+�(
Gene_Input_Layer����������
-�*
Protein_Input_Layer���������W
p

 
� "L�I
B�?
�
0/0����������
�
0/1���������W
� �
E__inference_model_40_layer_call_and_return_conditional_losses_1898897�8"#.+-,=>56IFHGTQSRabijurts��|}������������c�`
Y�V
L�I
#� 
inputs/0����������
"�
inputs/1���������W
p 

 
� "L�I
B�?
�
0/0����������
�
0/1���������W
� �
E__inference_model_40_layer_call_and_return_conditional_losses_1899148�8"#-.+,=>56HIFGSTQRabijturs��|}������������c�`
Y�V
L�I
#� 
inputs/0����������
"�
inputs/1���������W
p

 
� "L�I
B�?
�
0/0����������
�
0/1���������W
� �
*__inference_model_40_layer_call_fn_1897755�8"#.+-,=>56IFHGTQSRabijurts��|}������������v�s
l�i
_�\
+�(
Gene_Input_Layer����������
-�*
Protein_Input_Layer���������W
p 

 
� ">�;
�
0����������
�
1���������W�
*__inference_model_40_layer_call_fn_1898234�8"#-.+,=>56HIFGSTQRabijturs��|}������������v�s
l�i
_�\
+�(
Gene_Input_Layer����������
-�*
Protein_Input_Layer���������W
p

 
� ">�;
�
0����������
�
1���������W�
*__inference_model_40_layer_call_fn_1898638�8"#.+-,=>56IFHGTQSRabijurts��|}������������c�`
Y�V
L�I
#� 
inputs/0����������
"�
inputs/1���������W
p 

 
� ">�;
�
0����������
�
1���������W�
*__inference_model_40_layer_call_fn_1898730�8"#-.+,=>56HIFGSTQRabijturs��|}������������c�`
Y�V
L�I
#� 
inputs/0����������
"�
inputs/1���������W
p

 
� ">�;
�
0����������
�
1���������W�
N__inference_protein_decoder_1_layer_call_and_return_conditional_losses_1899621^��/�,
%�"
 �
inputs���������@
� "%�"
�
0���������
� �
3__inference_protein_decoder_1_layer_call_fn_1899610Q��/�,
%�"
 �
inputs���������@
� "�����������
Q__inference_protein_decoder_last_layer_call_and_return_conditional_losses_1899821^��/�,
%�"
 �
inputs���������
� "%�"
�
0���������W
� �
6__inference_protein_decoder_last_layer_call_fn_1899810Q��/�,
%�"
 �
inputs���������
� "����������W�
N__inference_protein_encoder_1_layer_call_and_return_conditional_losses_1899288\=>/�,
%�"
 �
inputs���������W
� "%�"
�
0���������
� �
3__inference_protein_encoder_1_layer_call_fn_1899277O=>/�,
%�"
 �
inputs���������W
� "�����������
%__inference_signature_wrapper_1898546�8"#.+-,=>56IFHGTQSRabijurts��|}���������������
� 
���
?
Gene_Input_Layer+�(
Gene_Input_Layer����������
D
Protein_Input_Layer-�*
Protein_Input_Layer���������W"���
A
gene_decoder_last,�)
gene_decoder_last����������
F
protein_decoder_last.�+
protein_decoder_last���������W