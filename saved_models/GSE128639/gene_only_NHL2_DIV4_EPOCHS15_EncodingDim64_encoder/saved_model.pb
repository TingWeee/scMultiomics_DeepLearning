��
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
 �"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68��

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

NoOpNoOp
�5
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�5
value�4B�4 B�4
�
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
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
�
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses*
�

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses*
�
,axis
	-gamma
.beta
/moving_mean
0moving_variance
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
* 
�
0
1
2
3
4
5
$6
%7
-8
.9
/10
011
712
813
@14
A15
B16
C17*
Z
0
1
2
3
$4
%5
-6
.7
78
89
@10
A11*
* 
�
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

Oserving_default* 
e_
VARIABLE_VALUEgene_encoder_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEgene_encoder_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
�
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
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
 
0
1
2
3*

0
1*
* 
�
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
* 
* 
e_
VARIABLE_VALUEgene_encoder_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEgene_encoder_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 
�
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*
* 
* 
* 
ic
VARIABLE_VALUEBatchNormGeneEncode2/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEBatchNormGeneEncode2/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
uo
VARIABLE_VALUE BatchNormGeneEncode2/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE$BatchNormGeneEncode2/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
-0
.1
/2
03*

-0
.1*
* 
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*
* 
* 
ga
VARIABLE_VALUEEmbeddingDimGene/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEEmbeddingDimGene/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

70
81*

70
81*
* 
�
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*
* 
* 
* 
f`
VARIABLE_VALUEEmbeddingDimGene3/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEEmbeddingDimGene3/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEEmbeddingDimGene3/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUE!EmbeddingDimGene3/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
@0
A1
B2
C3*

@0
A1*
* 
�
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*
* 
* 
.
0
1
/2
03
B4
C5*
5
0
1
2
3
4
5
6*

n0*
* 
* 
* 
* 
* 
* 
* 
* 

0
1*
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
/0
01*
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
8
	ototal
	pcount
q	variables
r	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

o0
p1*

q	variables*
�
 serving_default_gene_input_layerPlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCall serving_default_gene_input_layergene_encoder_1/kernelgene_encoder_1/bias$BatchNormGeneEncode1/moving_varianceBatchNormGeneEncode1/gamma BatchNormGeneEncode1/moving_meanBatchNormGeneEncode1/betagene_encoder_2/kernelgene_encoder_2/bias$BatchNormGeneEncode2/moving_varianceBatchNormGeneEncode2/gamma BatchNormGeneEncode2/moving_meanBatchNormGeneEncode2/betaEmbeddingDimGene/kernelEmbeddingDimGene/bias!EmbeddingDimGene3/moving_varianceEmbeddingDimGene3/gammaEmbeddingDimGene3/moving_meanEmbeddingDimGene3/beta*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_149632
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�	
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)gene_encoder_1/kernel/Read/ReadVariableOp'gene_encoder_1/bias/Read/ReadVariableOp.BatchNormGeneEncode1/gamma/Read/ReadVariableOp-BatchNormGeneEncode1/beta/Read/ReadVariableOp4BatchNormGeneEncode1/moving_mean/Read/ReadVariableOp8BatchNormGeneEncode1/moving_variance/Read/ReadVariableOp)gene_encoder_2/kernel/Read/ReadVariableOp'gene_encoder_2/bias/Read/ReadVariableOp.BatchNormGeneEncode2/gamma/Read/ReadVariableOp-BatchNormGeneEncode2/beta/Read/ReadVariableOp4BatchNormGeneEncode2/moving_mean/Read/ReadVariableOp8BatchNormGeneEncode2/moving_variance/Read/ReadVariableOp+EmbeddingDimGene/kernel/Read/ReadVariableOp)EmbeddingDimGene/bias/Read/ReadVariableOp+EmbeddingDimGene3/gamma/Read/ReadVariableOp*EmbeddingDimGene3/beta/Read/ReadVariableOp1EmbeddingDimGene3/moving_mean/Read/ReadVariableOp5EmbeddingDimGene3/moving_variance/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*!
Tin
2*
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
__inference__traced_save_150015
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegene_encoder_1/kernelgene_encoder_1/biasBatchNormGeneEncode1/gammaBatchNormGeneEncode1/beta BatchNormGeneEncode1/moving_mean$BatchNormGeneEncode1/moving_variancegene_encoder_2/kernelgene_encoder_2/biasBatchNormGeneEncode2/gammaBatchNormGeneEncode2/beta BatchNormGeneEncode2/moving_mean$BatchNormGeneEncode2/moving_varianceEmbeddingDimGene/kernelEmbeddingDimGene/biasEmbeddingDimGene3/gammaEmbeddingDimGene3/betaEmbeddingDimGene3/moving_mean!EmbeddingDimGene3/moving_variancetotalcount* 
Tin
2*
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
"__inference__traced_restore_150085��	
�
�
2__inference_EmbeddingDimGene3_layer_call_fn_149865

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
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_148845o
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
�
�
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_148681

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
�(
�
C__inference_model_3_layer_call_and_return_conditional_losses_148989

inputs)
gene_encoder_1_148922:
��$
gene_encoder_1_148924:	�*
batchnormgeneencode1_148927:	�*
batchnormgeneencode1_148929:	�*
batchnormgeneencode1_148931:	�*
batchnormgeneencode1_148933:	�(
gene_encoder_2_148948:	�}#
gene_encoder_2_148950:})
batchnormgeneencode2_148953:})
batchnormgeneencode2_148955:})
batchnormgeneencode2_148957:})
batchnormgeneencode2_148959:})
embeddingdimgene_148974:}@%
embeddingdimgene_148976:@&
embeddingdimgene3_148979:@&
embeddingdimgene3_148981:@&
embeddingdimgene3_148983:@&
embeddingdimgene3_148985:@
identity��,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�(EmbeddingDimGene/StatefulPartitionedCall�)EmbeddingDimGene3/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputsgene_encoder_1_148922gene_encoder_1_148924*
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
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_148921�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_148927batchnormgeneencode1_148929batchnormgeneencode1_148931batchnormgeneencode1_148933*
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
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_148681�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_148948gene_encoder_2_148950*
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
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_148947�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_148953batchnormgeneencode2_148955batchnormgeneencode2_148957batchnormgeneencode2_148959*
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
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_148763�
(EmbeddingDimGene/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:0embeddingdimgene_148974embeddingdimgene_148976*
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
GPU2*0J 8� *U
fPRN
L__inference_EmbeddingDimGene_layer_call_and_return_conditional_losses_148973�
)EmbeddingDimGene3/StatefulPartitionedCallStatefulPartitionedCall1EmbeddingDimGene/StatefulPartitionedCall:output:0embeddingdimgene3_148979embeddingdimgene3_148981embeddingdimgene3_148983embeddingdimgene3_148985*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_148845�
IdentityIdentity2EmbeddingDimGene3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall)^EmbeddingDimGene/StatefulPartitionedCall*^EmbeddingDimGene3/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������: : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2T
(EmbeddingDimGene/StatefulPartitionedCall(EmbeddingDimGene/StatefulPartitionedCall2V
)EmbeddingDimGene3/StatefulPartitionedCall)EmbeddingDimGene3/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�t
�
!__inference__wrapped_model_148657
gene_input_layerI
5model_3_gene_encoder_1_matmul_readvariableop_resource:
��E
6model_3_gene_encoder_1_biasadd_readvariableop_resource:	�M
>model_3_batchnormgeneencode1_batchnorm_readvariableop_resource:	�Q
Bmodel_3_batchnormgeneencode1_batchnorm_mul_readvariableop_resource:	�O
@model_3_batchnormgeneencode1_batchnorm_readvariableop_1_resource:	�O
@model_3_batchnormgeneencode1_batchnorm_readvariableop_2_resource:	�H
5model_3_gene_encoder_2_matmul_readvariableop_resource:	�}D
6model_3_gene_encoder_2_biasadd_readvariableop_resource:}L
>model_3_batchnormgeneencode2_batchnorm_readvariableop_resource:}P
Bmodel_3_batchnormgeneencode2_batchnorm_mul_readvariableop_resource:}N
@model_3_batchnormgeneencode2_batchnorm_readvariableop_1_resource:}N
@model_3_batchnormgeneencode2_batchnorm_readvariableop_2_resource:}I
7model_3_embeddingdimgene_matmul_readvariableop_resource:}@F
8model_3_embeddingdimgene_biasadd_readvariableop_resource:@I
;model_3_embeddingdimgene3_batchnorm_readvariableop_resource:@M
?model_3_embeddingdimgene3_batchnorm_mul_readvariableop_resource:@K
=model_3_embeddingdimgene3_batchnorm_readvariableop_1_resource:@K
=model_3_embeddingdimgene3_batchnorm_readvariableop_2_resource:@
identity��5model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp�7model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1�7model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2�9model_3/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp�5model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp�7model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1�7model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2�9model_3/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp�/model_3/EmbeddingDimGene/BiasAdd/ReadVariableOp�.model_3/EmbeddingDimGene/MatMul/ReadVariableOp�2model_3/EmbeddingDimGene3/batchnorm/ReadVariableOp�4model_3/EmbeddingDimGene3/batchnorm/ReadVariableOp_1�4model_3/EmbeddingDimGene3/batchnorm/ReadVariableOp_2�6model_3/EmbeddingDimGene3/batchnorm/mul/ReadVariableOp�-model_3/gene_encoder_1/BiasAdd/ReadVariableOp�,model_3/gene_encoder_1/MatMul/ReadVariableOp�-model_3/gene_encoder_2/BiasAdd/ReadVariableOp�,model_3/gene_encoder_2/MatMul/ReadVariableOp�
,model_3/gene_encoder_1/MatMul/ReadVariableOpReadVariableOp5model_3_gene_encoder_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model_3/gene_encoder_1/MatMulMatMulgene_input_layer4model_3/gene_encoder_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-model_3/gene_encoder_1/BiasAdd/ReadVariableOpReadVariableOp6model_3_gene_encoder_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_3/gene_encoder_1/BiasAddBiasAdd'model_3/gene_encoder_1/MatMul:product:05model_3/gene_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
model_3/gene_encoder_1/SigmoidSigmoid'model_3/gene_encoder_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
5model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOpReadVariableOp>model_3_batchnormgeneencode1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0q
,model_3/BatchNormGeneEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
*model_3/BatchNormGeneEncode1/batchnorm/addAddV2=model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp:value:05model_3/BatchNormGeneEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
,model_3/BatchNormGeneEncode1/batchnorm/RsqrtRsqrt.model_3/BatchNormGeneEncode1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
9model_3/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOpReadVariableOpBmodel_3_batchnormgeneencode1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
*model_3/BatchNormGeneEncode1/batchnorm/mulMul0model_3/BatchNormGeneEncode1/batchnorm/Rsqrt:y:0Amodel_3/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
,model_3/BatchNormGeneEncode1/batchnorm/mul_1Mul"model_3/gene_encoder_1/Sigmoid:y:0.model_3/BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
7model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1ReadVariableOp@model_3_batchnormgeneencode1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
,model_3/BatchNormGeneEncode1/batchnorm/mul_2Mul?model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1:value:0.model_3/BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
7model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2ReadVariableOp@model_3_batchnormgeneencode1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
*model_3/BatchNormGeneEncode1/batchnorm/subSub?model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2:value:00model_3/BatchNormGeneEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
,model_3/BatchNormGeneEncode1/batchnorm/add_1AddV20model_3/BatchNormGeneEncode1/batchnorm/mul_1:z:0.model_3/BatchNormGeneEncode1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
,model_3/gene_encoder_2/MatMul/ReadVariableOpReadVariableOp5model_3_gene_encoder_2_matmul_readvariableop_resource*
_output_shapes
:	�}*
dtype0�
model_3/gene_encoder_2/MatMulMatMul0model_3/BatchNormGeneEncode1/batchnorm/add_1:z:04model_3/gene_encoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}�
-model_3/gene_encoder_2/BiasAdd/ReadVariableOpReadVariableOp6model_3_gene_encoder_2_biasadd_readvariableop_resource*
_output_shapes
:}*
dtype0�
model_3/gene_encoder_2/BiasAddBiasAdd'model_3/gene_encoder_2/MatMul:product:05model_3/gene_encoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}�
model_3/gene_encoder_2/SigmoidSigmoid'model_3/gene_encoder_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������}�
5model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOpReadVariableOp>model_3_batchnormgeneencode2_batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0q
,model_3/BatchNormGeneEncode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
*model_3/BatchNormGeneEncode2/batchnorm/addAddV2=model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp:value:05model_3/BatchNormGeneEncode2/batchnorm/add/y:output:0*
T0*
_output_shapes
:}�
,model_3/BatchNormGeneEncode2/batchnorm/RsqrtRsqrt.model_3/BatchNormGeneEncode2/batchnorm/add:z:0*
T0*
_output_shapes
:}�
9model_3/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOpReadVariableOpBmodel_3_batchnormgeneencode2_batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0�
*model_3/BatchNormGeneEncode2/batchnorm/mulMul0model_3/BatchNormGeneEncode2/batchnorm/Rsqrt:y:0Amodel_3/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}�
,model_3/BatchNormGeneEncode2/batchnorm/mul_1Mul"model_3/gene_encoder_2/Sigmoid:y:0.model_3/BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������}�
7model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1ReadVariableOp@model_3_batchnormgeneencode2_batchnorm_readvariableop_1_resource*
_output_shapes
:}*
dtype0�
,model_3/BatchNormGeneEncode2/batchnorm/mul_2Mul?model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1:value:0.model_3/BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*
_output_shapes
:}�
7model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2ReadVariableOp@model_3_batchnormgeneencode2_batchnorm_readvariableop_2_resource*
_output_shapes
:}*
dtype0�
*model_3/BatchNormGeneEncode2/batchnorm/subSub?model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2:value:00model_3/BatchNormGeneEncode2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:}�
,model_3/BatchNormGeneEncode2/batchnorm/add_1AddV20model_3/BatchNormGeneEncode2/batchnorm/mul_1:z:0.model_3/BatchNormGeneEncode2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}�
.model_3/EmbeddingDimGene/MatMul/ReadVariableOpReadVariableOp7model_3_embeddingdimgene_matmul_readvariableop_resource*
_output_shapes

:}@*
dtype0�
model_3/EmbeddingDimGene/MatMulMatMul0model_3/BatchNormGeneEncode2/batchnorm/add_1:z:06model_3/EmbeddingDimGene/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
/model_3/EmbeddingDimGene/BiasAdd/ReadVariableOpReadVariableOp8model_3_embeddingdimgene_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
 model_3/EmbeddingDimGene/BiasAddBiasAdd)model_3/EmbeddingDimGene/MatMul:product:07model_3/EmbeddingDimGene/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
 model_3/EmbeddingDimGene/SigmoidSigmoid)model_3/EmbeddingDimGene/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
2model_3/EmbeddingDimGene3/batchnorm/ReadVariableOpReadVariableOp;model_3_embeddingdimgene3_batchnorm_readvariableop_resource*
_output_shapes
:@*
dtype0n
)model_3/EmbeddingDimGene3/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
'model_3/EmbeddingDimGene3/batchnorm/addAddV2:model_3/EmbeddingDimGene3/batchnorm/ReadVariableOp:value:02model_3/EmbeddingDimGene3/batchnorm/add/y:output:0*
T0*
_output_shapes
:@�
)model_3/EmbeddingDimGene3/batchnorm/RsqrtRsqrt+model_3/EmbeddingDimGene3/batchnorm/add:z:0*
T0*
_output_shapes
:@�
6model_3/EmbeddingDimGene3/batchnorm/mul/ReadVariableOpReadVariableOp?model_3_embeddingdimgene3_batchnorm_mul_readvariableop_resource*
_output_shapes
:@*
dtype0�
'model_3/EmbeddingDimGene3/batchnorm/mulMul-model_3/EmbeddingDimGene3/batchnorm/Rsqrt:y:0>model_3/EmbeddingDimGene3/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:@�
)model_3/EmbeddingDimGene3/batchnorm/mul_1Mul$model_3/EmbeddingDimGene/Sigmoid:y:0+model_3/EmbeddingDimGene3/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������@�
4model_3/EmbeddingDimGene3/batchnorm/ReadVariableOp_1ReadVariableOp=model_3_embeddingdimgene3_batchnorm_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
)model_3/EmbeddingDimGene3/batchnorm/mul_2Mul<model_3/EmbeddingDimGene3/batchnorm/ReadVariableOp_1:value:0+model_3/EmbeddingDimGene3/batchnorm/mul:z:0*
T0*
_output_shapes
:@�
4model_3/EmbeddingDimGene3/batchnorm/ReadVariableOp_2ReadVariableOp=model_3_embeddingdimgene3_batchnorm_readvariableop_2_resource*
_output_shapes
:@*
dtype0�
'model_3/EmbeddingDimGene3/batchnorm/subSub<model_3/EmbeddingDimGene3/batchnorm/ReadVariableOp_2:value:0-model_3/EmbeddingDimGene3/batchnorm/mul_2:z:0*
T0*
_output_shapes
:@�
)model_3/EmbeddingDimGene3/batchnorm/add_1AddV2-model_3/EmbeddingDimGene3/batchnorm/mul_1:z:0+model_3/EmbeddingDimGene3/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������@|
IdentityIdentity-model_3/EmbeddingDimGene3/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp6^model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp8^model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_18^model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2:^model_3/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp6^model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp8^model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_18^model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2:^model_3/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp0^model_3/EmbeddingDimGene/BiasAdd/ReadVariableOp/^model_3/EmbeddingDimGene/MatMul/ReadVariableOp3^model_3/EmbeddingDimGene3/batchnorm/ReadVariableOp5^model_3/EmbeddingDimGene3/batchnorm/ReadVariableOp_15^model_3/EmbeddingDimGene3/batchnorm/ReadVariableOp_27^model_3/EmbeddingDimGene3/batchnorm/mul/ReadVariableOp.^model_3/gene_encoder_1/BiasAdd/ReadVariableOp-^model_3/gene_encoder_1/MatMul/ReadVariableOp.^model_3/gene_encoder_2/BiasAdd/ReadVariableOp-^model_3/gene_encoder_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������: : : : : : : : : : : : : : : : : : 2n
5model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp5model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp2r
7model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_17model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_12r
7model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_27model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_22v
9model_3/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp9model_3/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp2n
5model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp5model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp2r
7model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_17model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_12r
7model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_27model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_22v
9model_3/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp9model_3/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp2b
/model_3/EmbeddingDimGene/BiasAdd/ReadVariableOp/model_3/EmbeddingDimGene/BiasAdd/ReadVariableOp2`
.model_3/EmbeddingDimGene/MatMul/ReadVariableOp.model_3/EmbeddingDimGene/MatMul/ReadVariableOp2h
2model_3/EmbeddingDimGene3/batchnorm/ReadVariableOp2model_3/EmbeddingDimGene3/batchnorm/ReadVariableOp2l
4model_3/EmbeddingDimGene3/batchnorm/ReadVariableOp_14model_3/EmbeddingDimGene3/batchnorm/ReadVariableOp_12l
4model_3/EmbeddingDimGene3/batchnorm/ReadVariableOp_24model_3/EmbeddingDimGene3/batchnorm/ReadVariableOp_22p
6model_3/EmbeddingDimGene3/batchnorm/mul/ReadVariableOp6model_3/EmbeddingDimGene3/batchnorm/mul/ReadVariableOp2^
-model_3/gene_encoder_1/BiasAdd/ReadVariableOp-model_3/gene_encoder_1/BiasAdd/ReadVariableOp2\
,model_3/gene_encoder_1/MatMul/ReadVariableOp,model_3/gene_encoder_1/MatMul/ReadVariableOp2^
-model_3/gene_encoder_2/BiasAdd/ReadVariableOp-model_3/gene_encoder_2/BiasAdd/ReadVariableOp2\
,model_3/gene_encoder_2/MatMul/ReadVariableOp,model_3/gene_encoder_2/MatMul/ReadVariableOp:Z V
(
_output_shapes
:����������
*
_user_specified_namegene_input_layer
�

�
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_148921

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
�

�
L__inference_EmbeddingDimGene_layer_call_and_return_conditional_losses_148973

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
/__inference_gene_encoder_1_layer_call_fn_149641

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
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_148921p
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
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_148947

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
�
�
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_148763

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
M__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_148892

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
�f
�
C__inference_model_3_layer_call_and_return_conditional_losses_149474

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
5embeddingdimgene3_batchnorm_readvariableop_2_resource:@
identity��-BatchNormGeneEncode1/batchnorm/ReadVariableOp�/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1�/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2�1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp�-BatchNormGeneEncode2/batchnorm/ReadVariableOp�/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1�/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2�1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp�'EmbeddingDimGene/BiasAdd/ReadVariableOp�&EmbeddingDimGene/MatMul/ReadVariableOp�*EmbeddingDimGene3/batchnorm/ReadVariableOp�,EmbeddingDimGene3/batchnorm/ReadVariableOp_1�,EmbeddingDimGene3/batchnorm/ReadVariableOp_2�.EmbeddingDimGene3/batchnorm/mul/ReadVariableOp�%gene_encoder_1/BiasAdd/ReadVariableOp�$gene_encoder_1/MatMul/ReadVariableOp�%gene_encoder_2/BiasAdd/ReadVariableOp�$gene_encoder_2/MatMul/ReadVariableOp�
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
:���������@t
IdentityIdentity%EmbeddingDimGene3/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp.^BatchNormGeneEncode1/batchnorm/ReadVariableOp0^BatchNormGeneEncode1/batchnorm/ReadVariableOp_10^BatchNormGeneEncode1/batchnorm/ReadVariableOp_22^BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp.^BatchNormGeneEncode2/batchnorm/ReadVariableOp0^BatchNormGeneEncode2/batchnorm/ReadVariableOp_10^BatchNormGeneEncode2/batchnorm/ReadVariableOp_22^BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp(^EmbeddingDimGene/BiasAdd/ReadVariableOp'^EmbeddingDimGene/MatMul/ReadVariableOp+^EmbeddingDimGene3/batchnorm/ReadVariableOp-^EmbeddingDimGene3/batchnorm/ReadVariableOp_1-^EmbeddingDimGene3/batchnorm/ReadVariableOp_2/^EmbeddingDimGene3/batchnorm/mul/ReadVariableOp&^gene_encoder_1/BiasAdd/ReadVariableOp%^gene_encoder_1/MatMul/ReadVariableOp&^gene_encoder_2/BiasAdd/ReadVariableOp%^gene_encoder_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������: : : : : : : : : : : : : : : : : : 2^
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
%gene_encoder_1/BiasAdd/ReadVariableOp%gene_encoder_1/BiasAdd/ReadVariableOp2L
$gene_encoder_1/MatMul/ReadVariableOp$gene_encoder_1/MatMul/ReadVariableOp2N
%gene_encoder_2/BiasAdd/ReadVariableOp%gene_encoder_2/BiasAdd/ReadVariableOp2L
$gene_encoder_2/MatMul/ReadVariableOp$gene_encoder_2/MatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�S
�
"__inference__traced_restore_150085
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
5assignvariableop_17_embeddingdimgene3_moving_variance:@#
assignvariableop_18_total: #
assignvariableop_19_count: 
identity_21��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2[
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
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_21IdentityIdentity_20:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_21Identity_21:output:0*=
_input_shapes,
*: : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�(
�
C__inference_model_3_layer_call_and_return_conditional_losses_149319
gene_input_layer)
gene_encoder_1_149276:
��$
gene_encoder_1_149278:	�*
batchnormgeneencode1_149281:	�*
batchnormgeneencode1_149283:	�*
batchnormgeneencode1_149285:	�*
batchnormgeneencode1_149287:	�(
gene_encoder_2_149290:	�}#
gene_encoder_2_149292:})
batchnormgeneencode2_149295:})
batchnormgeneencode2_149297:})
batchnormgeneencode2_149299:})
batchnormgeneencode2_149301:})
embeddingdimgene_149304:}@%
embeddingdimgene_149306:@&
embeddingdimgene3_149309:@&
embeddingdimgene3_149311:@&
embeddingdimgene3_149313:@&
embeddingdimgene3_149315:@
identity��,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�(EmbeddingDimGene/StatefulPartitionedCall�)EmbeddingDimGene3/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallgene_input_layergene_encoder_1_149276gene_encoder_1_149278*
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
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_148921�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_149281batchnormgeneencode1_149283batchnormgeneencode1_149285batchnormgeneencode1_149287*
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
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_148728�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_149290gene_encoder_2_149292*
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
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_148947�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_149295batchnormgeneencode2_149297batchnormgeneencode2_149299batchnormgeneencode2_149301*
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
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_148810�
(EmbeddingDimGene/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:0embeddingdimgene_149304embeddingdimgene_149306*
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
GPU2*0J 8� *U
fPRN
L__inference_EmbeddingDimGene_layer_call_and_return_conditional_losses_148973�
)EmbeddingDimGene3/StatefulPartitionedCallStatefulPartitionedCall1EmbeddingDimGene/StatefulPartitionedCall:output:0embeddingdimgene3_149309embeddingdimgene3_149311embeddingdimgene3_149313embeddingdimgene3_149315*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_148892�
IdentityIdentity2EmbeddingDimGene3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall)^EmbeddingDimGene/StatefulPartitionedCall*^EmbeddingDimGene3/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������: : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2T
(EmbeddingDimGene/StatefulPartitionedCall(EmbeddingDimGene/StatefulPartitionedCall2V
)EmbeddingDimGene3/StatefulPartitionedCall)EmbeddingDimGene3/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall:Z V
(
_output_shapes
:����������
*
_user_specified_namegene_input_layer
�

�
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_149652

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
�
�
(__inference_model_3_layer_call_fn_149227
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

unknown_16:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgene_input_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*.
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_149147o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:����������
*
_user_specified_namegene_input_layer
�%
�
M__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_149932

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
�(
�
C__inference_model_3_layer_call_and_return_conditional_losses_149147

inputs)
gene_encoder_1_149104:
��$
gene_encoder_1_149106:	�*
batchnormgeneencode1_149109:	�*
batchnormgeneencode1_149111:	�*
batchnormgeneencode1_149113:	�*
batchnormgeneencode1_149115:	�(
gene_encoder_2_149118:	�}#
gene_encoder_2_149120:})
batchnormgeneencode2_149123:})
batchnormgeneencode2_149125:})
batchnormgeneencode2_149127:})
batchnormgeneencode2_149129:})
embeddingdimgene_149132:}@%
embeddingdimgene_149134:@&
embeddingdimgene3_149137:@&
embeddingdimgene3_149139:@&
embeddingdimgene3_149141:@&
embeddingdimgene3_149143:@
identity��,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�(EmbeddingDimGene/StatefulPartitionedCall�)EmbeddingDimGene3/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputsgene_encoder_1_149104gene_encoder_1_149106*
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
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_148921�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_149109batchnormgeneencode1_149111batchnormgeneencode1_149113batchnormgeneencode1_149115*
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
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_148728�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_149118gene_encoder_2_149120*
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
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_148947�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_149123batchnormgeneencode2_149125batchnormgeneencode2_149127batchnormgeneencode2_149129*
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
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_148810�
(EmbeddingDimGene/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:0embeddingdimgene_149132embeddingdimgene_149134*
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
GPU2*0J 8� *U
fPRN
L__inference_EmbeddingDimGene_layer_call_and_return_conditional_losses_148973�
)EmbeddingDimGene3/StatefulPartitionedCallStatefulPartitionedCall1EmbeddingDimGene/StatefulPartitionedCall:output:0embeddingdimgene3_149137embeddingdimgene3_149139embeddingdimgene3_149141embeddingdimgene3_149143*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_148892�
IdentityIdentity2EmbeddingDimGene3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall)^EmbeddingDimGene/StatefulPartitionedCall*^EmbeddingDimGene3/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������: : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2T
(EmbeddingDimGene/StatefulPartitionedCall(EmbeddingDimGene/StatefulPartitionedCall2V
)EmbeddingDimGene3/StatefulPartitionedCall)EmbeddingDimGene3/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
L__inference_EmbeddingDimGene_layer_call_and_return_conditional_losses_149852

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
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_148728

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
�
$__inference_signature_wrapper_149632
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

unknown_16:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgene_input_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_148657o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:����������
*
_user_specified_namegene_input_layer
�%
�
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_149732

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
5__inference_BatchNormGeneEncode1_layer_call_fn_149678

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
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_148728p
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
�
(__inference_model_3_layer_call_fn_149360

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

unknown_16:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_148989o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
M__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_148845

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
�(
�
C__inference_model_3_layer_call_and_return_conditional_losses_149273
gene_input_layer)
gene_encoder_1_149230:
��$
gene_encoder_1_149232:	�*
batchnormgeneencode1_149235:	�*
batchnormgeneencode1_149237:	�*
batchnormgeneencode1_149239:	�*
batchnormgeneencode1_149241:	�(
gene_encoder_2_149244:	�}#
gene_encoder_2_149246:})
batchnormgeneencode2_149249:})
batchnormgeneencode2_149251:})
batchnormgeneencode2_149253:})
batchnormgeneencode2_149255:})
embeddingdimgene_149258:}@%
embeddingdimgene_149260:@&
embeddingdimgene3_149263:@&
embeddingdimgene3_149265:@&
embeddingdimgene3_149267:@&
embeddingdimgene3_149269:@
identity��,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�(EmbeddingDimGene/StatefulPartitionedCall�)EmbeddingDimGene3/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallgene_input_layergene_encoder_1_149230gene_encoder_1_149232*
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
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_148921�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_149235batchnormgeneencode1_149237batchnormgeneencode1_149239batchnormgeneencode1_149241*
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
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_148681�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_149244gene_encoder_2_149246*
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
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_148947�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_149249batchnormgeneencode2_149251batchnormgeneencode2_149253batchnormgeneencode2_149255*
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
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_148763�
(EmbeddingDimGene/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:0embeddingdimgene_149258embeddingdimgene_149260*
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
GPU2*0J 8� *U
fPRN
L__inference_EmbeddingDimGene_layer_call_and_return_conditional_losses_148973�
)EmbeddingDimGene3/StatefulPartitionedCallStatefulPartitionedCall1EmbeddingDimGene/StatefulPartitionedCall:output:0embeddingdimgene3_149263embeddingdimgene3_149265embeddingdimgene3_149267embeddingdimgene3_149269*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_148845�
IdentityIdentity2EmbeddingDimGene3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall)^EmbeddingDimGene/StatefulPartitionedCall*^EmbeddingDimGene3/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������: : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2T
(EmbeddingDimGene/StatefulPartitionedCall(EmbeddingDimGene/StatefulPartitionedCall2V
)EmbeddingDimGene3/StatefulPartitionedCall)EmbeddingDimGene3/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall:Z V
(
_output_shapes
:����������
*
_user_specified_namegene_input_layer
�
�
1__inference_EmbeddingDimGene_layer_call_fn_149841

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
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_EmbeddingDimGene_layer_call_and_return_conditional_losses_148973o
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
�

�
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_149752

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
�
(__inference_model_3_layer_call_fn_149401

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

unknown_16:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*.
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_149147o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
C__inference_model_3_layer_call_and_return_conditional_losses_149589

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
3embeddingdimgene3_batchnorm_readvariableop_resource:@
identity��$BatchNormGeneEncode1/AssignMovingAvg�3BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOp�&BatchNormGeneEncode1/AssignMovingAvg_1�5BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOp�-BatchNormGeneEncode1/batchnorm/ReadVariableOp�1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp�$BatchNormGeneEncode2/AssignMovingAvg�3BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOp�&BatchNormGeneEncode2/AssignMovingAvg_1�5BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOp�-BatchNormGeneEncode2/batchnorm/ReadVariableOp�1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp�'EmbeddingDimGene/BiasAdd/ReadVariableOp�&EmbeddingDimGene/MatMul/ReadVariableOp�!EmbeddingDimGene3/AssignMovingAvg�0EmbeddingDimGene3/AssignMovingAvg/ReadVariableOp�#EmbeddingDimGene3/AssignMovingAvg_1�2EmbeddingDimGene3/AssignMovingAvg_1/ReadVariableOp�*EmbeddingDimGene3/batchnorm/ReadVariableOp�.EmbeddingDimGene3/batchnorm/mul/ReadVariableOp�%gene_encoder_1/BiasAdd/ReadVariableOp�$gene_encoder_1/MatMul/ReadVariableOp�%gene_encoder_2/BiasAdd/ReadVariableOp�$gene_encoder_2/MatMul/ReadVariableOp�
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
:���������@t
IdentityIdentity%EmbeddingDimGene3/batchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:���������@�	
NoOpNoOp%^BatchNormGeneEncode1/AssignMovingAvg4^BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOp'^BatchNormGeneEncode1/AssignMovingAvg_16^BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneEncode1/batchnorm/ReadVariableOp2^BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp%^BatchNormGeneEncode2/AssignMovingAvg4^BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOp'^BatchNormGeneEncode2/AssignMovingAvg_16^BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneEncode2/batchnorm/ReadVariableOp2^BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp(^EmbeddingDimGene/BiasAdd/ReadVariableOp'^EmbeddingDimGene/MatMul/ReadVariableOp"^EmbeddingDimGene3/AssignMovingAvg1^EmbeddingDimGene3/AssignMovingAvg/ReadVariableOp$^EmbeddingDimGene3/AssignMovingAvg_13^EmbeddingDimGene3/AssignMovingAvg_1/ReadVariableOp+^EmbeddingDimGene3/batchnorm/ReadVariableOp/^EmbeddingDimGene3/batchnorm/mul/ReadVariableOp&^gene_encoder_1/BiasAdd/ReadVariableOp%^gene_encoder_1/MatMul/ReadVariableOp&^gene_encoder_2/BiasAdd/ReadVariableOp%^gene_encoder_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������: : : : : : : : : : : : : : : : : : 2L
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
5__inference_BatchNormGeneEncode2_layer_call_fn_149765

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
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_148763o
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
2__inference_EmbeddingDimGene3_layer_call_fn_149878

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
*0
config_proto 

CPU

GPU2*0J 8� *V
fQRO
M__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_148892o
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
�
�
5__inference_BatchNormGeneEncode2_layer_call_fn_149778

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
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_148810o
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
5__inference_BatchNormGeneEncode1_layer_call_fn_149665

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
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_148681p
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
M__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_149898

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
�
(__inference_model_3_layer_call_fn_149028
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

unknown_16:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgene_input_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_148989o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8:����������: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:����������
*
_user_specified_namegene_input_layer
�
�
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_149698

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
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_149798

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
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_148810

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
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_149832

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
�2
�	
__inference__traced_save_150015
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
<savev2_embeddingdimgene3_moving_variance_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
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
: �

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B �	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_gene_encoder_1_kernel_read_readvariableop.savev2_gene_encoder_1_bias_read_readvariableop5savev2_batchnormgeneencode1_gamma_read_readvariableop4savev2_batchnormgeneencode1_beta_read_readvariableop;savev2_batchnormgeneencode1_moving_mean_read_readvariableop?savev2_batchnormgeneencode1_moving_variance_read_readvariableop0savev2_gene_encoder_2_kernel_read_readvariableop.savev2_gene_encoder_2_bias_read_readvariableop5savev2_batchnormgeneencode2_gamma_read_readvariableop4savev2_batchnormgeneencode2_beta_read_readvariableop;savev2_batchnormgeneencode2_moving_mean_read_readvariableop?savev2_batchnormgeneencode2_moving_variance_read_readvariableop2savev2_embeddingdimgene_kernel_read_readvariableop0savev2_embeddingdimgene_bias_read_readvariableop2savev2_embeddingdimgene3_gamma_read_readvariableop1savev2_embeddingdimgene3_beta_read_readvariableop8savev2_embeddingdimgene3_moving_mean_read_readvariableop<savev2_embeddingdimgene3_moving_variance_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *#
dtypes
2�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: :
��:�:�:�:�:�:	�}:}:}:}:}:}:}@:@:@:@:@:@: : : 2(
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
:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
�
�
/__inference_gene_encoder_2_layer_call_fn_149741

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
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_148947o
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
 
_user_specified_nameinputs"�L
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
"serving_default_gene_input_layer:0����������E
EmbeddingDimGene30
StatefulPartitionedCall:0���������@tensorflow/serving/predict:��
�
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
	optimizer
		variables

trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
�
axis
	gamma
beta
moving_mean
moving_variance
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
�

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
�
,axis
	-gamma
.beta
/moving_mean
0moving_variance
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
"
	optimizer
�
0
1
2
3
4
5
$6
%7
-8
.9
/10
011
712
813
@14
A15
B16
C17"
trackable_list_wrapper
v
0
1
2
3
$4
%5
-6
.7
78
89
@10
A11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
		variables

trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
(__inference_model_3_layer_call_fn_149028
(__inference_model_3_layer_call_fn_149360
(__inference_model_3_layer_call_fn_149401
(__inference_model_3_layer_call_fn_149227�
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
C__inference_model_3_layer_call_and_return_conditional_losses_149474
C__inference_model_3_layer_call_and_return_conditional_losses_149589
C__inference_model_3_layer_call_and_return_conditional_losses_149273
C__inference_model_3_layer_call_and_return_conditional_losses_149319�
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
!__inference__wrapped_model_148657gene_input_layer"�
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
,
Oserving_default"
signature_map
):'
��2gene_encoder_1/kernel
": �2gene_encoder_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�2�
/__inference_gene_encoder_1_layer_call_fn_149641�
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
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_149652�
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
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
�2�
5__inference_BatchNormGeneEncode1_layer_call_fn_149665
5__inference_BatchNormGeneEncode1_layer_call_fn_149678�
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
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_149698
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_149732�
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
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Znon_trainable_variables

[layers
\metrics
]layer_regularization_losses
^layer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
�2�
/__inference_gene_encoder_2_layer_call_fn_149741�
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
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_149752�
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
-0
.1
/2
03"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
�2�
5__inference_BatchNormGeneEncode2_layer_call_fn_149765
5__inference_BatchNormGeneEncode2_layer_call_fn_149778�
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
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_149798
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_149832�
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
):'}@2EmbeddingDimGene/kernel
#:!@2EmbeddingDimGene/bias
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
dnon_trainable_variables

elayers
fmetrics
glayer_regularization_losses
hlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�2�
1__inference_EmbeddingDimGene_layer_call_fn_149841�
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
L__inference_EmbeddingDimGene_layer_call_and_return_conditional_losses_149852�
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
%:#@2EmbeddingDimGene3/gamma
$:"@2EmbeddingDimGene3/beta
-:+@ (2EmbeddingDimGene3/moving_mean
1:/@ (2!EmbeddingDimGene3/moving_variance
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
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
�2�
2__inference_EmbeddingDimGene3_layer_call_fn_149865
2__inference_EmbeddingDimGene3_layer_call_fn_149878�
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
M__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_149898
M__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_149932�
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
J
0
1
/2
03
B4
C5"
trackable_list_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
'
n0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_signature_wrapper_149632gene_input_layer"�
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
0
1"
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
/0
01"
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
N
	ototal
	pcount
q	variables
r	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
o0
p1"
trackable_list_wrapper
-
q	variables"
_generic_user_object�
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_149698d4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_149732d4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
5__inference_BatchNormGeneEncode1_layer_call_fn_149665W4�1
*�'
!�
inputs����������
p 
� "������������
5__inference_BatchNormGeneEncode1_layer_call_fn_149678W4�1
*�'
!�
inputs����������
p
� "������������
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_149798b0-/.3�0
)�&
 �
inputs���������}
p 
� "%�"
�
0���������}
� �
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_149832b/0-.3�0
)�&
 �
inputs���������}
p
� "%�"
�
0���������}
� �
5__inference_BatchNormGeneEncode2_layer_call_fn_149765U0-/.3�0
)�&
 �
inputs���������}
p 
� "����������}�
5__inference_BatchNormGeneEncode2_layer_call_fn_149778U/0-.3�0
)�&
 �
inputs���������}
p
� "����������}�
M__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_149898bC@BA3�0
)�&
 �
inputs���������@
p 
� "%�"
�
0���������@
� �
M__inference_EmbeddingDimGene3_layer_call_and_return_conditional_losses_149932bBC@A3�0
)�&
 �
inputs���������@
p
� "%�"
�
0���������@
� �
2__inference_EmbeddingDimGene3_layer_call_fn_149865UC@BA3�0
)�&
 �
inputs���������@
p 
� "����������@�
2__inference_EmbeddingDimGene3_layer_call_fn_149878UBC@A3�0
)�&
 �
inputs���������@
p
� "����������@�
L__inference_EmbeddingDimGene_layer_call_and_return_conditional_losses_149852\78/�,
%�"
 �
inputs���������}
� "%�"
�
0���������@
� �
1__inference_EmbeddingDimGene_layer_call_fn_149841O78/�,
%�"
 �
inputs���������}
� "����������@�
!__inference__wrapped_model_148657�$%0-/.78C@BA:�7
0�-
+�(
gene_input_layer����������
� "E�B
@
EmbeddingDimGene3+�(
EmbeddingDimGene3���������@�
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_149652^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
/__inference_gene_encoder_1_layer_call_fn_149641Q0�-
&�#
!�
inputs����������
� "������������
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_149752]$%0�-
&�#
!�
inputs����������
� "%�"
�
0���������}
� �
/__inference_gene_encoder_2_layer_call_fn_149741P$%0�-
&�#
!�
inputs����������
� "����������}�
C__inference_model_3_layer_call_and_return_conditional_losses_149273$%0-/.78C@BAB�?
8�5
+�(
gene_input_layer����������
p 

 
� "%�"
�
0���������@
� �
C__inference_model_3_layer_call_and_return_conditional_losses_149319$%/0-.78BC@AB�?
8�5
+�(
gene_input_layer����������
p

 
� "%�"
�
0���������@
� �
C__inference_model_3_layer_call_and_return_conditional_losses_149474u$%0-/.78C@BA8�5
.�+
!�
inputs����������
p 

 
� "%�"
�
0���������@
� �
C__inference_model_3_layer_call_and_return_conditional_losses_149589u$%/0-.78BC@A8�5
.�+
!�
inputs����������
p

 
� "%�"
�
0���������@
� �
(__inference_model_3_layer_call_fn_149028r$%0-/.78C@BAB�?
8�5
+�(
gene_input_layer����������
p 

 
� "����������@�
(__inference_model_3_layer_call_fn_149227r$%/0-.78BC@AB�?
8�5
+�(
gene_input_layer����������
p

 
� "����������@�
(__inference_model_3_layer_call_fn_149360h$%0-/.78C@BA8�5
.�+
!�
inputs����������
p 

 
� "����������@�
(__inference_model_3_layer_call_fn_149401h$%/0-.78BC@A8�5
.�+
!�
inputs����������
p

 
� "����������@�
$__inference_signature_wrapper_149632�$%0-/.78C@BAN�K
� 
D�A
?
gene_input_layer+�(
gene_input_layer����������"E�B
@
EmbeddingDimGene3+�(
EmbeddingDimGene3���������@