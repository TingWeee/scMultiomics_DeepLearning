��
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
 �"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8�
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
shape:/*8
shared_name)'BatchNormProteinEncode1/moving_variance
�
;BatchNormProteinEncode1/moving_variance/Read/ReadVariableOpReadVariableOp'BatchNormProteinEncode1/moving_variance*
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
shape:/*'
shared_nameprotein_encoder_1/bias
}
*protein_encoder_1/bias/Read/ReadVariableOpReadVariableOpprotein_encoder_1/bias*
_output_shapes
:/*
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
#serving_default_Protein_Input_LayerPlaceholder*(
_output_shapes
:����������*
dtype0*
shape:����������
�
StatefulPartitionedCallStatefulPartitionedCall serving_default_Gene_Input_Layer#serving_default_Protein_Input_Layergene_encoder_1/kernelgene_encoder_1/bias$BatchNormGeneEncode1/moving_varianceBatchNormGeneEncode1/gamma BatchNormGeneEncode1/moving_meanBatchNormGeneEncode1/betaprotein_encoder_1/kernelprotein_encoder_1/biasgene_encoder_2/kernelgene_encoder_2/bias$BatchNormGeneEncode2/moving_varianceBatchNormGeneEncode2/gamma BatchNormGeneEncode2/moving_meanBatchNormGeneEncode2/beta'BatchNormProteinEncode1/moving_varianceBatchNormProteinEncode1/gamma#BatchNormProteinEncode1/moving_meanBatchNormProteinEncode1/betaEmbeddingDimDense/kernelEmbeddingDimDense/bias*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *.
f)R'
%__inference_signature_wrapper_1278336

NoOpNoOp
�D
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�C
value�CB�C B�C
�
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
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
* 
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"axis
	#gamma
$beta
%moving_mean
&moving_variance*
* 
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias*
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
=axis
	>gamma
?beta
@moving_mean
Amoving_variance*
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
Haxis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance*
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses* 
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias*
�
0
1
#2
$3
%4
&5
-6
.7
58
69
>10
?11
@12
A13
I14
J15
K16
L17
Y18
Z19*
j
0
1
#2
$3
-4
.5
56
67
>8
?9
I10
J11
Y12
Z13*
* 
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
`trace_0
atrace_1
btrace_2
ctrace_3* 
6
dtrace_0
etrace_1
ftrace_2
gtrace_3* 
* 
* 

hserving_default* 

0
1*

0
1*
* 
�
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

ntrace_0* 

otrace_0* 
e_
VARIABLE_VALUEgene_encoder_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEgene_encoder_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
#0
$1
%2
&3*

#0
$1*
* 
�
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

utrace_0
vtrace_1* 

wtrace_0
xtrace_1* 
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
-0
.1*

-0
.1*
* 
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*

~trace_0* 

trace_0* 
e_
VARIABLE_VALUEgene_encoder_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEgene_encoder_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
hb
VARIABLE_VALUEprotein_encoder_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEprotein_encoder_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
 
>0
?1
@2
A3*

>0
?1*
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
 
I0
J1
K2
L3*

I0
J1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

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
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

Y0
Z1*

Y0
Z1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
hb
VARIABLE_VALUEEmbeddingDimDense/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEEmbeddingDimDense/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
.
%0
&1
@2
A3
K4
L5*
J
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
9*

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
* 
* 
* 
* 
* 
* 
* 
* 

%0
&1*
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
@0
A1*
* 
* 
* 
* 
* 
* 
* 
* 

K0
L1*
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
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)gene_encoder_1/kernel/Read/ReadVariableOp'gene_encoder_1/bias/Read/ReadVariableOp.BatchNormGeneEncode1/gamma/Read/ReadVariableOp-BatchNormGeneEncode1/beta/Read/ReadVariableOp4BatchNormGeneEncode1/moving_mean/Read/ReadVariableOp8BatchNormGeneEncode1/moving_variance/Read/ReadVariableOp)gene_encoder_2/kernel/Read/ReadVariableOp'gene_encoder_2/bias/Read/ReadVariableOp,protein_encoder_1/kernel/Read/ReadVariableOp*protein_encoder_1/bias/Read/ReadVariableOp.BatchNormGeneEncode2/gamma/Read/ReadVariableOp-BatchNormGeneEncode2/beta/Read/ReadVariableOp4BatchNormGeneEncode2/moving_mean/Read/ReadVariableOp8BatchNormGeneEncode2/moving_variance/Read/ReadVariableOp1BatchNormProteinEncode1/gamma/Read/ReadVariableOp0BatchNormProteinEncode1/beta/Read/ReadVariableOp7BatchNormProteinEncode1/moving_mean/Read/ReadVariableOp;BatchNormProteinEncode1/moving_variance/Read/ReadVariableOp,EmbeddingDimDense/kernel/Read/ReadVariableOp*EmbeddingDimDense/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*#
Tin
2*
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
 __inference__traced_save_1279059
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegene_encoder_1/kernelgene_encoder_1/biasBatchNormGeneEncode1/gammaBatchNormGeneEncode1/beta BatchNormGeneEncode1/moving_mean$BatchNormGeneEncode1/moving_variancegene_encoder_2/kernelgene_encoder_2/biasprotein_encoder_1/kernelprotein_encoder_1/biasBatchNormGeneEncode2/gammaBatchNormGeneEncode2/beta BatchNormGeneEncode2/moving_mean$BatchNormGeneEncode2/moving_varianceBatchNormProteinEncode1/gammaBatchNormProteinEncode1/beta#BatchNormProteinEncode1/moving_mean'BatchNormProteinEncode1/moving_varianceEmbeddingDimDense/kernelEmbeddingDimDense/biastotalcount*"
Tin
2*
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
#__inference__traced_restore_1279135��

�

�
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_1277850

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
3__inference_EmbeddingDimDense_layer_call_fn_1278958

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
N__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_1277894o
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
�

�
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_1277807

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
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1278736

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
��
�
E__inference_model_29_layer_call_and_return_conditional_losses_1278636
inputs_0
inputs_1A
-gene_encoder_1_matmul_readvariableop_resource:
��=
.gene_encoder_1_biasadd_readvariableop_resource:	�K
<batchnormgeneencode1_assignmovingavg_readvariableop_resource:	�M
>batchnormgeneencode1_assignmovingavg_1_readvariableop_resource:	�I
:batchnormgeneencode1_batchnorm_mul_readvariableop_resource:	�E
6batchnormgeneencode1_batchnorm_readvariableop_resource:	�C
0protein_encoder_1_matmul_readvariableop_resource:	�/?
1protein_encoder_1_biasadd_readvariableop_resource:/@
-gene_encoder_2_matmul_readvariableop_resource:	�}<
.gene_encoder_2_biasadd_readvariableop_resource:}J
<batchnormgeneencode2_assignmovingavg_readvariableop_resource:}L
>batchnormgeneencode2_assignmovingavg_1_readvariableop_resource:}H
:batchnormgeneencode2_batchnorm_mul_readvariableop_resource:}D
6batchnormgeneencode2_batchnorm_readvariableop_resource:}M
?batchnormproteinencode1_assignmovingavg_readvariableop_resource:/O
Abatchnormproteinencode1_assignmovingavg_1_readvariableop_resource:/K
=batchnormproteinencode1_batchnorm_mul_readvariableop_resource:/G
9batchnormproteinencode1_batchnorm_readvariableop_resource:/C
0embeddingdimdense_matmul_readvariableop_resource:	�@?
1embeddingdimdense_biasadd_readvariableop_resource:@
identity��$BatchNormGeneEncode1/AssignMovingAvg�3BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOp�&BatchNormGeneEncode1/AssignMovingAvg_1�5BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOp�-BatchNormGeneEncode1/batchnorm/ReadVariableOp�1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp�$BatchNormGeneEncode2/AssignMovingAvg�3BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOp�&BatchNormGeneEncode2/AssignMovingAvg_1�5BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOp�-BatchNormGeneEncode2/batchnorm/ReadVariableOp�1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp�'BatchNormProteinEncode1/AssignMovingAvg�6BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOp�)BatchNormProteinEncode1/AssignMovingAvg_1�8BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp�0BatchNormProteinEncode1/batchnorm/ReadVariableOp�4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp�(EmbeddingDimDense/BiasAdd/ReadVariableOp�'EmbeddingDimDense/MatMul/ReadVariableOp�%gene_encoder_1/BiasAdd/ReadVariableOp�$gene_encoder_1/MatMul/ReadVariableOp�%gene_encoder_2/BiasAdd/ReadVariableOp�$gene_encoder_2/MatMul/ReadVariableOp�(protein_encoder_1/BiasAdd/ReadVariableOp�'protein_encoder_1/MatMul/ReadVariableOp�
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
:���������/^
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
:���������@l
IdentityIdentityEmbeddingDimDense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������@�

NoOpNoOp%^BatchNormGeneEncode1/AssignMovingAvg4^BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOp'^BatchNormGeneEncode1/AssignMovingAvg_16^BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneEncode1/batchnorm/ReadVariableOp2^BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp%^BatchNormGeneEncode2/AssignMovingAvg4^BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOp'^BatchNormGeneEncode2/AssignMovingAvg_16^BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneEncode2/batchnorm/ReadVariableOp2^BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp(^BatchNormProteinEncode1/AssignMovingAvg7^BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOp*^BatchNormProteinEncode1/AssignMovingAvg_19^BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp1^BatchNormProteinEncode1/batchnorm/ReadVariableOp5^BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp)^EmbeddingDimDense/BiasAdd/ReadVariableOp(^EmbeddingDimDense/MatMul/ReadVariableOp&^gene_encoder_1/BiasAdd/ReadVariableOp%^gene_encoder_1/MatMul/ReadVariableOp&^gene_encoder_2/BiasAdd/ReadVariableOp%^gene_encoder_2/MatMul/ReadVariableOp)^protein_encoder_1/BiasAdd/ReadVariableOp(^protein_encoder_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:����������:����������: : : : : : : : : : : : : : : : : : : : 2L
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
'BatchNormProteinEncode1/AssignMovingAvg'BatchNormProteinEncode1/AssignMovingAvg2p
6BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOp6BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOp2V
)BatchNormProteinEncode1/AssignMovingAvg_1)BatchNormProteinEncode1/AssignMovingAvg_12t
8BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp8BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp2d
0BatchNormProteinEncode1/batchnorm/ReadVariableOp0BatchNormProteinEncode1/batchnorm/ReadVariableOp2l
4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp2T
(EmbeddingDimDense/BiasAdd/ReadVariableOp(EmbeddingDimDense/BiasAdd/ReadVariableOp2R
'EmbeddingDimDense/MatMul/ReadVariableOp'EmbeddingDimDense/MatMul/ReadVariableOp2N
%gene_encoder_1/BiasAdd/ReadVariableOp%gene_encoder_1/BiasAdd/ReadVariableOp2L
$gene_encoder_1/MatMul/ReadVariableOp$gene_encoder_1/MatMul/ReadVariableOp2N
%gene_encoder_2/BiasAdd/ReadVariableOp%gene_encoder_2/BiasAdd/ReadVariableOp2L
$gene_encoder_2/MatMul/ReadVariableOp$gene_encoder_2/MatMul/ReadVariableOp2T
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
�3
�

E__inference_model_29_layer_call_and_return_conditional_losses_1277901

inputs
inputs_1*
gene_encoder_1_1277808:
��%
gene_encoder_1_1277810:	�+
batchnormgeneencode1_1277813:	�+
batchnormgeneencode1_1277815:	�+
batchnormgeneencode1_1277817:	�+
batchnormgeneencode1_1277819:	�,
protein_encoder_1_1277834:	�/'
protein_encoder_1_1277836:/)
gene_encoder_2_1277851:	�}$
gene_encoder_2_1277853:}*
batchnormgeneencode2_1277856:}*
batchnormgeneencode2_1277858:}*
batchnormgeneencode2_1277860:}*
batchnormgeneencode2_1277862:}-
batchnormproteinencode1_1277865:/-
batchnormproteinencode1_1277867:/-
batchnormproteinencode1_1277869:/-
batchnormproteinencode1_1277871:/,
embeddingdimdense_1277895:	�@'
embeddingdimdense_1277897:@
identity��,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�/BatchNormProteinEncode1/StatefulPartitionedCall�)EmbeddingDimDense/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�)protein_encoder_1/StatefulPartitionedCall�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputsgene_encoder_1_1277808gene_encoder_1_1277810*
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
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_1277807�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_1277813batchnormgeneencode1_1277815batchnormgeneencode1_1277817batchnormgeneencode1_1277819*
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
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1277565�
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1protein_encoder_1_1277834protein_encoder_1_1277836*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_protein_encoder_1_layer_call_and_return_conditional_losses_1277833�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_1277851gene_encoder_2_1277853*
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
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_1277850�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_1277856batchnormgeneencode2_1277858batchnormgeneencode2_1277860batchnormgeneencode2_1277862*
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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1277647�
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_1277865batchnormproteinencode1_1277867batchnormproteinencode1_1277869batchnormproteinencode1_1277871*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1277729�
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1277881�
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_1277895embeddingdimdense_1277897*
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
N__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_1277894�
IdentityIdentity2EmbeddingDimDense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:����������:����������: : : : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2b
/BatchNormProteinEncode1/StatefulPartitionedCall/BatchNormProteinEncode1/StatefulPartitionedCall2V
)EmbeddingDimDense/StatefulPartitionedCall)EmbeddingDimDense/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall2V
)protein_encoder_1/StatefulPartitionedCall)protein_encoder_1/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
*__inference_model_29_layer_call_fn_1278182
gene_input_layer
protein_input_layer
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�/
	unknown_6:/
	unknown_7:	�}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:/

unknown_14:/

unknown_15:/

unknown_16:/

unknown_17:	�@

unknown_18:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgene_input_layerprotein_input_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_29_layer_call_and_return_conditional_losses_1278093o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:����������:����������: : : : : : : : : : : : : : : : : : : : 22
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
�5
�

 __inference__traced_save_1279059
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
1savev2_embeddingdimdense_bias_read_readvariableop$
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
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B �

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_gene_encoder_1_kernel_read_readvariableop.savev2_gene_encoder_1_bias_read_readvariableop5savev2_batchnormgeneencode1_gamma_read_readvariableop4savev2_batchnormgeneencode1_beta_read_readvariableop;savev2_batchnormgeneencode1_moving_mean_read_readvariableop?savev2_batchnormgeneencode1_moving_variance_read_readvariableop0savev2_gene_encoder_2_kernel_read_readvariableop.savev2_gene_encoder_2_bias_read_readvariableop3savev2_protein_encoder_1_kernel_read_readvariableop1savev2_protein_encoder_1_bias_read_readvariableop5savev2_batchnormgeneencode2_gamma_read_readvariableop4savev2_batchnormgeneencode2_beta_read_readvariableop;savev2_batchnormgeneencode2_moving_mean_read_readvariableop?savev2_batchnormgeneencode2_moving_variance_read_readvariableop8savev2_batchnormproteinencode1_gamma_read_readvariableop7savev2_batchnormproteinencode1_beta_read_readvariableop>savev2_batchnormproteinencode1_moving_mean_read_readvariableopBsavev2_batchnormproteinencode1_moving_variance_read_readvariableop3savev2_embeddingdimdense_kernel_read_readvariableop1savev2_embeddingdimdense_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2�
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
��:�:�:�:�:�:	�}:}:	�/:/:}:}:}:}:/:/:/:/:	�@:@: : : 2(
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
:	�/: 


_output_shapes
:/: 
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
:/: 

_output_shapes
:/: 

_output_shapes
:/: 

_output_shapes
:/:%!

_output_shapes
:	�@: 

_output_shapes
:@:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
��
�
"__inference__wrapped_model_1277541
gene_input_layer
protein_input_layerJ
6model_29_gene_encoder_1_matmul_readvariableop_resource:
��F
7model_29_gene_encoder_1_biasadd_readvariableop_resource:	�N
?model_29_batchnormgeneencode1_batchnorm_readvariableop_resource:	�R
Cmodel_29_batchnormgeneencode1_batchnorm_mul_readvariableop_resource:	�P
Amodel_29_batchnormgeneencode1_batchnorm_readvariableop_1_resource:	�P
Amodel_29_batchnormgeneencode1_batchnorm_readvariableop_2_resource:	�L
9model_29_protein_encoder_1_matmul_readvariableop_resource:	�/H
:model_29_protein_encoder_1_biasadd_readvariableop_resource:/I
6model_29_gene_encoder_2_matmul_readvariableop_resource:	�}E
7model_29_gene_encoder_2_biasadd_readvariableop_resource:}M
?model_29_batchnormgeneencode2_batchnorm_readvariableop_resource:}Q
Cmodel_29_batchnormgeneencode2_batchnorm_mul_readvariableop_resource:}O
Amodel_29_batchnormgeneencode2_batchnorm_readvariableop_1_resource:}O
Amodel_29_batchnormgeneencode2_batchnorm_readvariableop_2_resource:}P
Bmodel_29_batchnormproteinencode1_batchnorm_readvariableop_resource:/T
Fmodel_29_batchnormproteinencode1_batchnorm_mul_readvariableop_resource:/R
Dmodel_29_batchnormproteinencode1_batchnorm_readvariableop_1_resource:/R
Dmodel_29_batchnormproteinencode1_batchnorm_readvariableop_2_resource:/L
9model_29_embeddingdimdense_matmul_readvariableop_resource:	�@H
:model_29_embeddingdimdense_biasadd_readvariableop_resource:@
identity��6model_29/BatchNormGeneEncode1/batchnorm/ReadVariableOp�8model_29/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1�8model_29/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2�:model_29/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp�6model_29/BatchNormGeneEncode2/batchnorm/ReadVariableOp�8model_29/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1�8model_29/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2�:model_29/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp�9model_29/BatchNormProteinEncode1/batchnorm/ReadVariableOp�;model_29/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1�;model_29/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2�=model_29/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp�1model_29/EmbeddingDimDense/BiasAdd/ReadVariableOp�0model_29/EmbeddingDimDense/MatMul/ReadVariableOp�.model_29/gene_encoder_1/BiasAdd/ReadVariableOp�-model_29/gene_encoder_1/MatMul/ReadVariableOp�.model_29/gene_encoder_2/BiasAdd/ReadVariableOp�-model_29/gene_encoder_2/MatMul/ReadVariableOp�1model_29/protein_encoder_1/BiasAdd/ReadVariableOp�0model_29/protein_encoder_1/MatMul/ReadVariableOp�
-model_29/gene_encoder_1/MatMul/ReadVariableOpReadVariableOp6model_29_gene_encoder_1_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
model_29/gene_encoder_1/MatMulMatMulgene_input_layer5model_29/gene_encoder_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
.model_29/gene_encoder_1/BiasAdd/ReadVariableOpReadVariableOp7model_29_gene_encoder_1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
model_29/gene_encoder_1/BiasAddBiasAdd(model_29/gene_encoder_1/MatMul:product:06model_29/gene_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
model_29/gene_encoder_1/SigmoidSigmoid(model_29/gene_encoder_1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
6model_29/BatchNormGeneEncode1/batchnorm/ReadVariableOpReadVariableOp?model_29_batchnormgeneencode1_batchnorm_readvariableop_resource*
_output_shapes	
:�*
dtype0r
-model_29/BatchNormGeneEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
+model_29/BatchNormGeneEncode1/batchnorm/addAddV2>model_29/BatchNormGeneEncode1/batchnorm/ReadVariableOp:value:06model_29/BatchNormGeneEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:��
-model_29/BatchNormGeneEncode1/batchnorm/RsqrtRsqrt/model_29/BatchNormGeneEncode1/batchnorm/add:z:0*
T0*
_output_shapes	
:��
:model_29/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_29_batchnormgeneencode1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:�*
dtype0�
+model_29/BatchNormGeneEncode1/batchnorm/mulMul1model_29/BatchNormGeneEncode1/batchnorm/Rsqrt:y:0Bmodel_29/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:��
-model_29/BatchNormGeneEncode1/batchnorm/mul_1Mul#model_29/gene_encoder_1/Sigmoid:y:0/model_29/BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*(
_output_shapes
:�����������
8model_29/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_29_batchnormgeneencode1_batchnorm_readvariableop_1_resource*
_output_shapes	
:�*
dtype0�
-model_29/BatchNormGeneEncode1/batchnorm/mul_2Mul@model_29/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1:value:0/model_29/BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*
_output_shapes	
:��
8model_29/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_29_batchnormgeneencode1_batchnorm_readvariableop_2_resource*
_output_shapes	
:�*
dtype0�
+model_29/BatchNormGeneEncode1/batchnorm/subSub@model_29/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2:value:01model_29/BatchNormGeneEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:��
-model_29/BatchNormGeneEncode1/batchnorm/add_1AddV21model_29/BatchNormGeneEncode1/batchnorm/mul_1:z:0/model_29/BatchNormGeneEncode1/batchnorm/sub:z:0*
T0*(
_output_shapes
:�����������
0model_29/protein_encoder_1/MatMul/ReadVariableOpReadVariableOp9model_29_protein_encoder_1_matmul_readvariableop_resource*
_output_shapes
:	�/*
dtype0�
!model_29/protein_encoder_1/MatMulMatMulprotein_input_layer8model_29/protein_encoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������/�
1model_29/protein_encoder_1/BiasAdd/ReadVariableOpReadVariableOp:model_29_protein_encoder_1_biasadd_readvariableop_resource*
_output_shapes
:/*
dtype0�
"model_29/protein_encoder_1/BiasAddBiasAdd+model_29/protein_encoder_1/MatMul:product:09model_29/protein_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������/�
"model_29/protein_encoder_1/SigmoidSigmoid+model_29/protein_encoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������/�
-model_29/gene_encoder_2/MatMul/ReadVariableOpReadVariableOp6model_29_gene_encoder_2_matmul_readvariableop_resource*
_output_shapes
:	�}*
dtype0�
model_29/gene_encoder_2/MatMulMatMul1model_29/BatchNormGeneEncode1/batchnorm/add_1:z:05model_29/gene_encoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}�
.model_29/gene_encoder_2/BiasAdd/ReadVariableOpReadVariableOp7model_29_gene_encoder_2_biasadd_readvariableop_resource*
_output_shapes
:}*
dtype0�
model_29/gene_encoder_2/BiasAddBiasAdd(model_29/gene_encoder_2/MatMul:product:06model_29/gene_encoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������}�
model_29/gene_encoder_2/SigmoidSigmoid(model_29/gene_encoder_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������}�
6model_29/BatchNormGeneEncode2/batchnorm/ReadVariableOpReadVariableOp?model_29_batchnormgeneencode2_batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0r
-model_29/BatchNormGeneEncode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
+model_29/BatchNormGeneEncode2/batchnorm/addAddV2>model_29/BatchNormGeneEncode2/batchnorm/ReadVariableOp:value:06model_29/BatchNormGeneEncode2/batchnorm/add/y:output:0*
T0*
_output_shapes
:}�
-model_29/BatchNormGeneEncode2/batchnorm/RsqrtRsqrt/model_29/BatchNormGeneEncode2/batchnorm/add:z:0*
T0*
_output_shapes
:}�
:model_29/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_29_batchnormgeneencode2_batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0�
+model_29/BatchNormGeneEncode2/batchnorm/mulMul1model_29/BatchNormGeneEncode2/batchnorm/Rsqrt:y:0Bmodel_29/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}�
-model_29/BatchNormGeneEncode2/batchnorm/mul_1Mul#model_29/gene_encoder_2/Sigmoid:y:0/model_29/BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������}�
8model_29/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_29_batchnormgeneencode2_batchnorm_readvariableop_1_resource*
_output_shapes
:}*
dtype0�
-model_29/BatchNormGeneEncode2/batchnorm/mul_2Mul@model_29/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1:value:0/model_29/BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*
_output_shapes
:}�
8model_29/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_29_batchnormgeneencode2_batchnorm_readvariableop_2_resource*
_output_shapes
:}*
dtype0�
+model_29/BatchNormGeneEncode2/batchnorm/subSub@model_29/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2:value:01model_29/BatchNormGeneEncode2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:}�
-model_29/BatchNormGeneEncode2/batchnorm/add_1AddV21model_29/BatchNormGeneEncode2/batchnorm/mul_1:z:0/model_29/BatchNormGeneEncode2/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������}�
9model_29/BatchNormProteinEncode1/batchnorm/ReadVariableOpReadVariableOpBmodel_29_batchnormproteinencode1_batchnorm_readvariableop_resource*
_output_shapes
:/*
dtype0u
0model_29/BatchNormProteinEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *o�:�
.model_29/BatchNormProteinEncode1/batchnorm/addAddV2Amodel_29/BatchNormProteinEncode1/batchnorm/ReadVariableOp:value:09model_29/BatchNormProteinEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:/�
0model_29/BatchNormProteinEncode1/batchnorm/RsqrtRsqrt2model_29/BatchNormProteinEncode1/batchnorm/add:z:0*
T0*
_output_shapes
:/�
=model_29/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOpReadVariableOpFmodel_29_batchnormproteinencode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:/*
dtype0�
.model_29/BatchNormProteinEncode1/batchnorm/mulMul4model_29/BatchNormProteinEncode1/batchnorm/Rsqrt:y:0Emodel_29/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:/�
0model_29/BatchNormProteinEncode1/batchnorm/mul_1Mul&model_29/protein_encoder_1/Sigmoid:y:02model_29/BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:���������/�
;model_29/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1ReadVariableOpDmodel_29_batchnormproteinencode1_batchnorm_readvariableop_1_resource*
_output_shapes
:/*
dtype0�
0model_29/BatchNormProteinEncode1/batchnorm/mul_2MulCmodel_29/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1:value:02model_29/BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*
_output_shapes
:/�
;model_29/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2ReadVariableOpDmodel_29_batchnormproteinencode1_batchnorm_readvariableop_2_resource*
_output_shapes
:/*
dtype0�
.model_29/BatchNormProteinEncode1/batchnorm/subSubCmodel_29/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2:value:04model_29/BatchNormProteinEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:/�
0model_29/BatchNormProteinEncode1/batchnorm/add_1AddV24model_29/BatchNormProteinEncode1/batchnorm/mul_1:z:02model_29/BatchNormProteinEncode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:���������/g
%model_29/ConcatenateLayer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
 model_29/ConcatenateLayer/concatConcatV21model_29/BatchNormGeneEncode2/batchnorm/add_1:z:04model_29/BatchNormProteinEncode1/batchnorm/add_1:z:0.model_29/ConcatenateLayer/concat/axis:output:0*
N*
T0*(
_output_shapes
:�����������
0model_29/EmbeddingDimDense/MatMul/ReadVariableOpReadVariableOp9model_29_embeddingdimdense_matmul_readvariableop_resource*
_output_shapes
:	�@*
dtype0�
!model_29/EmbeddingDimDense/MatMulMatMul)model_29/ConcatenateLayer/concat:output:08model_29/EmbeddingDimDense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
1model_29/EmbeddingDimDense/BiasAdd/ReadVariableOpReadVariableOp:model_29_embeddingdimdense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
"model_29/EmbeddingDimDense/BiasAddBiasAdd+model_29/EmbeddingDimDense/MatMul:product:09model_29/EmbeddingDimDense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
"model_29/EmbeddingDimDense/SigmoidSigmoid+model_29/EmbeddingDimDense/BiasAdd:output:0*
T0*'
_output_shapes
:���������@u
IdentityIdentity&model_29/EmbeddingDimDense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������@�	
NoOpNoOp7^model_29/BatchNormGeneEncode1/batchnorm/ReadVariableOp9^model_29/BatchNormGeneEncode1/batchnorm/ReadVariableOp_19^model_29/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2;^model_29/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp7^model_29/BatchNormGeneEncode2/batchnorm/ReadVariableOp9^model_29/BatchNormGeneEncode2/batchnorm/ReadVariableOp_19^model_29/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2;^model_29/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:^model_29/BatchNormProteinEncode1/batchnorm/ReadVariableOp<^model_29/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1<^model_29/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2>^model_29/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp2^model_29/EmbeddingDimDense/BiasAdd/ReadVariableOp1^model_29/EmbeddingDimDense/MatMul/ReadVariableOp/^model_29/gene_encoder_1/BiasAdd/ReadVariableOp.^model_29/gene_encoder_1/MatMul/ReadVariableOp/^model_29/gene_encoder_2/BiasAdd/ReadVariableOp.^model_29/gene_encoder_2/MatMul/ReadVariableOp2^model_29/protein_encoder_1/BiasAdd/ReadVariableOp1^model_29/protein_encoder_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:����������:����������: : : : : : : : : : : : : : : : : : : : 2p
6model_29/BatchNormGeneEncode1/batchnorm/ReadVariableOp6model_29/BatchNormGeneEncode1/batchnorm/ReadVariableOp2t
8model_29/BatchNormGeneEncode1/batchnorm/ReadVariableOp_18model_29/BatchNormGeneEncode1/batchnorm/ReadVariableOp_12t
8model_29/BatchNormGeneEncode1/batchnorm/ReadVariableOp_28model_29/BatchNormGeneEncode1/batchnorm/ReadVariableOp_22x
:model_29/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp:model_29/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp2p
6model_29/BatchNormGeneEncode2/batchnorm/ReadVariableOp6model_29/BatchNormGeneEncode2/batchnorm/ReadVariableOp2t
8model_29/BatchNormGeneEncode2/batchnorm/ReadVariableOp_18model_29/BatchNormGeneEncode2/batchnorm/ReadVariableOp_12t
8model_29/BatchNormGeneEncode2/batchnorm/ReadVariableOp_28model_29/BatchNormGeneEncode2/batchnorm/ReadVariableOp_22x
:model_29/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:model_29/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp2v
9model_29/BatchNormProteinEncode1/batchnorm/ReadVariableOp9model_29/BatchNormProteinEncode1/batchnorm/ReadVariableOp2z
;model_29/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1;model_29/BatchNormProteinEncode1/batchnorm/ReadVariableOp_12z
;model_29/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2;model_29/BatchNormProteinEncode1/batchnorm/ReadVariableOp_22~
=model_29/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp=model_29/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp2f
1model_29/EmbeddingDimDense/BiasAdd/ReadVariableOp1model_29/EmbeddingDimDense/BiasAdd/ReadVariableOp2d
0model_29/EmbeddingDimDense/MatMul/ReadVariableOp0model_29/EmbeddingDimDense/MatMul/ReadVariableOp2`
.model_29/gene_encoder_1/BiasAdd/ReadVariableOp.model_29/gene_encoder_1/BiasAdd/ReadVariableOp2^
-model_29/gene_encoder_1/MatMul/ReadVariableOp-model_29/gene_encoder_1/MatMul/ReadVariableOp2`
.model_29/gene_encoder_2/BiasAdd/ReadVariableOp.model_29/gene_encoder_2/BiasAdd/ReadVariableOp2^
-model_29/gene_encoder_2/MatMul/ReadVariableOp-model_29/gene_encoder_2/MatMul/ReadVariableOp2f
1model_29/protein_encoder_1/BiasAdd/ReadVariableOp1model_29/protein_encoder_1/BiasAdd/ReadVariableOp2d
0model_29/protein_encoder_1/MatMul/ReadVariableOp0model_29/protein_encoder_1/MatMul/ReadVariableOp:Z V
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
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1277612

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
�3
�

E__inference_model_29_layer_call_and_return_conditional_losses_1278093

inputs
inputs_1*
gene_encoder_1_1278044:
��%
gene_encoder_1_1278046:	�+
batchnormgeneencode1_1278049:	�+
batchnormgeneencode1_1278051:	�+
batchnormgeneencode1_1278053:	�+
batchnormgeneencode1_1278055:	�,
protein_encoder_1_1278058:	�/'
protein_encoder_1_1278060:/)
gene_encoder_2_1278063:	�}$
gene_encoder_2_1278065:}*
batchnormgeneencode2_1278068:}*
batchnormgeneencode2_1278070:}*
batchnormgeneencode2_1278072:}*
batchnormgeneencode2_1278074:}-
batchnormproteinencode1_1278077:/-
batchnormproteinencode1_1278079:/-
batchnormproteinencode1_1278081:/-
batchnormproteinencode1_1278083:/,
embeddingdimdense_1278087:	�@'
embeddingdimdense_1278089:@
identity��,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�/BatchNormProteinEncode1/StatefulPartitionedCall�)EmbeddingDimDense/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�)protein_encoder_1/StatefulPartitionedCall�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputsgene_encoder_1_1278044gene_encoder_1_1278046*
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
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_1277807�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_1278049batchnormgeneencode1_1278051batchnormgeneencode1_1278053batchnormgeneencode1_1278055*
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
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1277612�
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1protein_encoder_1_1278058protein_encoder_1_1278060*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_protein_encoder_1_layer_call_and_return_conditional_losses_1277833�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_1278063gene_encoder_2_1278065*
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
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_1277850�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_1278068batchnormgeneencode2_1278070batchnormgeneencode2_1278072batchnormgeneencode2_1278074*
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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1277694�
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_1278077batchnormproteinencode1_1278079batchnormproteinencode1_1278081batchnormproteinencode1_1278083*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1277776�
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1277881�
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_1278087embeddingdimdense_1278089*
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
N__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_1277894�
IdentityIdentity2EmbeddingDimDense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:����������:����������: : : : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2b
/BatchNormProteinEncode1/StatefulPartitionedCall/BatchNormProteinEncode1/StatefulPartitionedCall2V
)EmbeddingDimDense/StatefulPartitionedCall)EmbeddingDimDense/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall2V
)protein_encoder_1/StatefulPartitionedCall)protein_encoder_1/StatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs:PL
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
6__inference_BatchNormGeneEncode2_layer_call_fn_1278789

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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1277647o
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
N__inference_protein_encoder_1_layer_call_and_return_conditional_losses_1278776

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
�
�
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1278702

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
0__inference_gene_encoder_2_layer_call_fn_1278745

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
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_1277850o
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
�
^
2__inference_ConcatenateLayer_layer_call_fn_1278942
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1277881a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������}:���������/:Q M
'
_output_shapes
:���������}
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������/
"
_user_specified_name
inputs/1
�
�
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1277565

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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1278822

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
�
�
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1277729

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
�%
�
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1277694

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

�
N__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_1277894

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
�
�
9__inference_BatchNormProteinEncode1_layer_call_fn_1278869

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
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1277729o
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
�

�
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_1278656

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
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1277776

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
�
�
%__inference_signature_wrapper_1278336
gene_input_layer
protein_input_layer
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�/
	unknown_6:/
	unknown_7:	�}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:/

unknown_14:/

unknown_15:/

unknown_16:/

unknown_17:	�@

unknown_18:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgene_input_layerprotein_input_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__wrapped_model_1277541o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:����������:����������: : : : : : : : : : : : : : : : : : : : 22
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
�
�
*__inference_model_29_layer_call_fn_1278428
inputs_0
inputs_1
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�/
	unknown_6:/
	unknown_7:	�}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:/

unknown_14:/

unknown_15:/

unknown_16:/

unknown_17:	�@

unknown_18:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_29_layer_call_and_return_conditional_losses_1278093o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:����������:����������: : : : : : : : : : : : : : : : : : : : 22
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
�
�
*__inference_model_29_layer_call_fn_1278382
inputs_0
inputs_1
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�/
	unknown_6:/
	unknown_7:	�}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:/

unknown_14:/

unknown_15:/

unknown_16:/

unknown_17:	�@

unknown_18:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_29_layer_call_and_return_conditional_losses_1277901o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:����������:����������: : : : : : : : : : : : : : : : : : : : 22
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
�
�
*__inference_model_29_layer_call_fn_1277944
gene_input_layer
protein_input_layer
unknown:
��
	unknown_0:	�
	unknown_1:	�
	unknown_2:	�
	unknown_3:	�
	unknown_4:	�
	unknown_5:	�/
	unknown_6:/
	unknown_7:	�}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:/

unknown_14:/

unknown_15:/

unknown_16:/

unknown_17:	�@

unknown_18:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallgene_input_layerprotein_input_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*6
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_model_29_layer_call_and_return_conditional_losses_1277901o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:����������:����������: : : : : : : : : : : : : : : : : : : : 22
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

�
N__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_1278969

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
�[
�
#__inference__traced_restore_1279135
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
+assignvariableop_8_protein_encoder_1_kernel:	�/7
)assignvariableop_9_protein_encoder_1_bias:/<
.assignvariableop_10_batchnormgeneencode2_gamma:};
-assignvariableop_11_batchnormgeneencode2_beta:}B
4assignvariableop_12_batchnormgeneencode2_moving_mean:}F
8assignvariableop_13_batchnormgeneencode2_moving_variance:}?
1assignvariableop_14_batchnormproteinencode1_gamma:/>
0assignvariableop_15_batchnormproteinencode1_beta:/E
7assignvariableop_16_batchnormproteinencode1_moving_mean:/I
;assignvariableop_17_batchnormproteinencode1_moving_variance:/?
,assignvariableop_18_embeddingdimdense_kernel:	�@8
*assignvariableop_19_embeddingdimdense_bias:@#
assignvariableop_20_total: #
assignvariableop_21_count: 
identity_23��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�

value�
B�
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2[
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
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_23Identity_23:output:0*A
_input_shapes0
.: : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_21AssignVariableOp_212(
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
�3
�

E__inference_model_29_layer_call_and_return_conditional_losses_1278235
gene_input_layer
protein_input_layer*
gene_encoder_1_1278186:
��%
gene_encoder_1_1278188:	�+
batchnormgeneencode1_1278191:	�+
batchnormgeneencode1_1278193:	�+
batchnormgeneencode1_1278195:	�+
batchnormgeneencode1_1278197:	�,
protein_encoder_1_1278200:	�/'
protein_encoder_1_1278202:/)
gene_encoder_2_1278205:	�}$
gene_encoder_2_1278207:}*
batchnormgeneencode2_1278210:}*
batchnormgeneencode2_1278212:}*
batchnormgeneencode2_1278214:}*
batchnormgeneencode2_1278216:}-
batchnormproteinencode1_1278219:/-
batchnormproteinencode1_1278221:/-
batchnormproteinencode1_1278223:/-
batchnormproteinencode1_1278225:/,
embeddingdimdense_1278229:	�@'
embeddingdimdense_1278231:@
identity��,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�/BatchNormProteinEncode1/StatefulPartitionedCall�)EmbeddingDimDense/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�)protein_encoder_1/StatefulPartitionedCall�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallgene_input_layergene_encoder_1_1278186gene_encoder_1_1278188*
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
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_1277807�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_1278191batchnormgeneencode1_1278193batchnormgeneencode1_1278195batchnormgeneencode1_1278197*
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
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1277565�
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallprotein_input_layerprotein_encoder_1_1278200protein_encoder_1_1278202*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_protein_encoder_1_layer_call_and_return_conditional_losses_1277833�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_1278205gene_encoder_2_1278207*
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
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_1277850�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_1278210batchnormgeneencode2_1278212batchnormgeneencode2_1278214batchnormgeneencode2_1278216*
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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1277647�
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_1278219batchnormproteinencode1_1278221batchnormproteinencode1_1278223batchnormproteinencode1_1278225*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1277729�
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1277881�
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_1278229embeddingdimdense_1278231*
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
N__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_1277894�
IdentityIdentity2EmbeddingDimDense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:����������:����������: : : : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2b
/BatchNormProteinEncode1/StatefulPartitionedCall/BatchNormProteinEncode1/StatefulPartitionedCall2V
)EmbeddingDimDense/StatefulPartitionedCall)EmbeddingDimDense/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall2V
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
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1278902

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
�
�
0__inference_gene_encoder_1_layer_call_fn_1278645

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
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_1277807p
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
�
�
6__inference_BatchNormGeneEncode1_layer_call_fn_1278669

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
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1277565p
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
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_1278756

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
9__inference_BatchNormProteinEncode1_layer_call_fn_1278882

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
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1277776o
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
�
w
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1277881

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
&:���������}:���������/:O K
'
_output_shapes
:���������}
 
_user_specified_nameinputs:OK
'
_output_shapes
:���������/
 
_user_specified_nameinputs
�
y
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1278949
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
&:���������}:���������/:Q M
'
_output_shapes
:���������}
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:���������/
"
_user_specified_name
inputs/1
�
�
6__inference_BatchNormGeneEncode2_layer_call_fn_1278802

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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1277694o
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
�3
�

E__inference_model_29_layer_call_and_return_conditional_losses_1278288
gene_input_layer
protein_input_layer*
gene_encoder_1_1278239:
��%
gene_encoder_1_1278241:	�+
batchnormgeneencode1_1278244:	�+
batchnormgeneencode1_1278246:	�+
batchnormgeneencode1_1278248:	�+
batchnormgeneencode1_1278250:	�,
protein_encoder_1_1278253:	�/'
protein_encoder_1_1278255:/)
gene_encoder_2_1278258:	�}$
gene_encoder_2_1278260:}*
batchnormgeneencode2_1278263:}*
batchnormgeneencode2_1278265:}*
batchnormgeneencode2_1278267:}*
batchnormgeneencode2_1278269:}-
batchnormproteinencode1_1278272:/-
batchnormproteinencode1_1278274:/-
batchnormproteinencode1_1278276:/-
batchnormproteinencode1_1278278:/,
embeddingdimdense_1278282:	�@'
embeddingdimdense_1278284:@
identity��,BatchNormGeneEncode1/StatefulPartitionedCall�,BatchNormGeneEncode2/StatefulPartitionedCall�/BatchNormProteinEncode1/StatefulPartitionedCall�)EmbeddingDimDense/StatefulPartitionedCall�&gene_encoder_1/StatefulPartitionedCall�&gene_encoder_2/StatefulPartitionedCall�)protein_encoder_1/StatefulPartitionedCall�
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallgene_input_layergene_encoder_1_1278239gene_encoder_1_1278241*
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
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_1277807�
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_1278244batchnormgeneencode1_1278246batchnormgeneencode1_1278248batchnormgeneencode1_1278250*
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
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1277612�
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallprotein_input_layerprotein_encoder_1_1278253protein_encoder_1_1278255*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_protein_encoder_1_layer_call_and_return_conditional_losses_1277833�
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_1278258gene_encoder_2_1278260*
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
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_1277850�
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_1278263batchnormgeneencode2_1278265batchnormgeneencode2_1278267batchnormgeneencode2_1278269*
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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1277694�
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_1278272batchnormproteinencode1_1278274batchnormproteinencode1_1278276batchnormproteinencode1_1278278*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������/*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *]
fXRV
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1277776�
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1277881�
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_1278282embeddingdimdense_1278284*
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
N__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_1277894�
IdentityIdentity2EmbeddingDimDense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:����������:����������: : : : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2b
/BatchNormProteinEncode1/StatefulPartitionedCall/BatchNormProteinEncode1/StatefulPartitionedCall2V
)EmbeddingDimDense/StatefulPartitionedCall)EmbeddingDimDense/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall2V
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
�
�
6__inference_BatchNormGeneEncode1_layer_call_fn_1278682

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
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1277612p
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
3__inference_protein_encoder_1_layer_call_fn_1278765

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
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_protein_encoder_1_layer_call_and_return_conditional_losses_1277833o
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
�
�
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1277647

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
�v
�
E__inference_model_29_layer_call_and_return_conditional_losses_1278511
inputs_0
inputs_1A
-gene_encoder_1_matmul_readvariableop_resource:
��=
.gene_encoder_1_biasadd_readvariableop_resource:	�E
6batchnormgeneencode1_batchnorm_readvariableop_resource:	�I
:batchnormgeneencode1_batchnorm_mul_readvariableop_resource:	�G
8batchnormgeneencode1_batchnorm_readvariableop_1_resource:	�G
8batchnormgeneencode1_batchnorm_readvariableop_2_resource:	�C
0protein_encoder_1_matmul_readvariableop_resource:	�/?
1protein_encoder_1_biasadd_readvariableop_resource:/@
-gene_encoder_2_matmul_readvariableop_resource:	�}<
.gene_encoder_2_biasadd_readvariableop_resource:}D
6batchnormgeneencode2_batchnorm_readvariableop_resource:}H
:batchnormgeneencode2_batchnorm_mul_readvariableop_resource:}F
8batchnormgeneencode2_batchnorm_readvariableop_1_resource:}F
8batchnormgeneencode2_batchnorm_readvariableop_2_resource:}G
9batchnormproteinencode1_batchnorm_readvariableop_resource:/K
=batchnormproteinencode1_batchnorm_mul_readvariableop_resource:/I
;batchnormproteinencode1_batchnorm_readvariableop_1_resource:/I
;batchnormproteinencode1_batchnorm_readvariableop_2_resource:/C
0embeddingdimdense_matmul_readvariableop_resource:	�@?
1embeddingdimdense_biasadd_readvariableop_resource:@
identity��-BatchNormGeneEncode1/batchnorm/ReadVariableOp�/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1�/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2�1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp�-BatchNormGeneEncode2/batchnorm/ReadVariableOp�/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1�/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2�1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp�0BatchNormProteinEncode1/batchnorm/ReadVariableOp�2BatchNormProteinEncode1/batchnorm/ReadVariableOp_1�2BatchNormProteinEncode1/batchnorm/ReadVariableOp_2�4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp�(EmbeddingDimDense/BiasAdd/ReadVariableOp�'EmbeddingDimDense/MatMul/ReadVariableOp�%gene_encoder_1/BiasAdd/ReadVariableOp�$gene_encoder_1/MatMul/ReadVariableOp�%gene_encoder_2/BiasAdd/ReadVariableOp�$gene_encoder_2/MatMul/ReadVariableOp�(protein_encoder_1/BiasAdd/ReadVariableOp�'protein_encoder_1/MatMul/ReadVariableOp�
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
:���������/^
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
:���������@l
IdentityIdentityEmbeddingDimDense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:���������@�
NoOpNoOp.^BatchNormGeneEncode1/batchnorm/ReadVariableOp0^BatchNormGeneEncode1/batchnorm/ReadVariableOp_10^BatchNormGeneEncode1/batchnorm/ReadVariableOp_22^BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp.^BatchNormGeneEncode2/batchnorm/ReadVariableOp0^BatchNormGeneEncode2/batchnorm/ReadVariableOp_10^BatchNormGeneEncode2/batchnorm/ReadVariableOp_22^BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp1^BatchNormProteinEncode1/batchnorm/ReadVariableOp3^BatchNormProteinEncode1/batchnorm/ReadVariableOp_13^BatchNormProteinEncode1/batchnorm/ReadVariableOp_25^BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp)^EmbeddingDimDense/BiasAdd/ReadVariableOp(^EmbeddingDimDense/MatMul/ReadVariableOp&^gene_encoder_1/BiasAdd/ReadVariableOp%^gene_encoder_1/MatMul/ReadVariableOp&^gene_encoder_2/BiasAdd/ReadVariableOp%^gene_encoder_2/MatMul/ReadVariableOp)^protein_encoder_1/BiasAdd/ReadVariableOp(^protein_encoder_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*c
_input_shapesR
P:����������:����������: : : : : : : : : : : : : : : : : : : : 2^
-BatchNormGeneEncode1/batchnorm/ReadVariableOp-BatchNormGeneEncode1/batchnorm/ReadVariableOp2b
/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1/BatchNormGeneEncode1/batchnorm/ReadVariableOp_12b
/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2/BatchNormGeneEncode1/batchnorm/ReadVariableOp_22f
1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp2^
-BatchNormGeneEncode2/batchnorm/ReadVariableOp-BatchNormGeneEncode2/batchnorm/ReadVariableOp2b
/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1/BatchNormGeneEncode2/batchnorm/ReadVariableOp_12b
/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2/BatchNormGeneEncode2/batchnorm/ReadVariableOp_22f
1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp2d
0BatchNormProteinEncode1/batchnorm/ReadVariableOp0BatchNormProteinEncode1/batchnorm/ReadVariableOp2h
2BatchNormProteinEncode1/batchnorm/ReadVariableOp_12BatchNormProteinEncode1/batchnorm/ReadVariableOp_12h
2BatchNormProteinEncode1/batchnorm/ReadVariableOp_22BatchNormProteinEncode1/batchnorm/ReadVariableOp_22l
4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp2T
(EmbeddingDimDense/BiasAdd/ReadVariableOp(EmbeddingDimDense/BiasAdd/ReadVariableOp2R
'EmbeddingDimDense/MatMul/ReadVariableOp'EmbeddingDimDense/MatMul/ReadVariableOp2N
%gene_encoder_1/BiasAdd/ReadVariableOp%gene_encoder_1/BiasAdd/ReadVariableOp2L
$gene_encoder_1/MatMul/ReadVariableOp$gene_encoder_1/MatMul/ReadVariableOp2N
%gene_encoder_2/BiasAdd/ReadVariableOp%gene_encoder_2/BiasAdd/ReadVariableOp2L
$gene_encoder_2/MatMul/ReadVariableOp$gene_encoder_2/MatMul/ReadVariableOp2T
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

�
N__inference_protein_encoder_1_layer_call_and_return_conditional_losses_1277833

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
�%
�
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1278936

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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1278856

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
T
Protein_Input_Layer=
%serving_default_Protein_Input_Layer:0����������E
EmbeddingDimDense0
StatefulPartitionedCall:0���������@tensorflow/serving/predict:��
�
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
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_network
"
_tf_keras_input_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
�
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"axis
	#gamma
$beta
%moving_mean
&moving_variance"
_tf_keras_layer
"
_tf_keras_input_layer
�
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses

-kernel
.bias"
_tf_keras_layer
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
=axis
	>gamma
?beta
@moving_mean
Amoving_variance"
_tf_keras_layer
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
Haxis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance"
_tf_keras_layer
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias"
_tf_keras_layer
�
0
1
#2
$3
%4
&5
-6
.7
58
69
>10
?11
@12
A13
I14
J15
K16
L17
Y18
Z19"
trackable_list_wrapper
�
0
1
#2
$3
-4
.5
56
67
>8
?9
I10
J11
Y12
Z13"
trackable_list_wrapper
 "
trackable_list_wrapper
�
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
`trace_0
atrace_1
btrace_2
ctrace_32�
*__inference_model_29_layer_call_fn_1277944
*__inference_model_29_layer_call_fn_1278382
*__inference_model_29_layer_call_fn_1278428
*__inference_model_29_layer_call_fn_1278182�
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
 z`trace_0zatrace_1zbtrace_2zctrace_3
�
dtrace_0
etrace_1
ftrace_2
gtrace_32�
E__inference_model_29_layer_call_and_return_conditional_losses_1278511
E__inference_model_29_layer_call_and_return_conditional_losses_1278636
E__inference_model_29_layer_call_and_return_conditional_losses_1278235
E__inference_model_29_layer_call_and_return_conditional_losses_1278288�
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
 zdtrace_0zetrace_1zftrace_2zgtrace_3
�B�
"__inference__wrapped_model_1277541Gene_Input_LayerProtein_Input_Layer"�
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
"
	optimizer
,
hserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
inon_trainable_variables

jlayers
kmetrics
llayer_regularization_losses
mlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
ntrace_02�
0__inference_gene_encoder_1_layer_call_fn_1278645�
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
 zntrace_0
�
otrace_02�
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_1278656�
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
 zotrace_0
):'
��2gene_encoder_1/kernel
": �2gene_encoder_1/bias
<
#0
$1
%2
&3"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
�
utrace_0
vtrace_12�
6__inference_BatchNormGeneEncode1_layer_call_fn_1278669
6__inference_BatchNormGeneEncode1_layer_call_fn_1278682�
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
 zutrace_0zvtrace_1
�
wtrace_0
xtrace_12�
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1278702
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1278736�
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
 zwtrace_0zxtrace_1
 "
trackable_list_wrapper
):'�2BatchNormGeneEncode1/gamma
(:&�2BatchNormGeneEncode1/beta
1:/� (2 BatchNormGeneEncode1/moving_mean
5:3� (2$BatchNormGeneEncode1/moving_variance
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
ynon_trainable_variables

zlayers
{metrics
|layer_regularization_losses
}layer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
�
~trace_02�
0__inference_gene_encoder_2_layer_call_fn_1278745�
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
 z~trace_0
�
trace_02�
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_1278756�
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
 ztrace_0
(:&	�}2gene_encoder_2/kernel
!:}2gene_encoder_2/bias
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
3__inference_protein_encoder_1_layer_call_fn_1278765�
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
N__inference_protein_encoder_1_layer_call_and_return_conditional_losses_1278776�
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
+:)	�/2protein_encoder_1/kernel
$:"/2protein_encoder_1/bias
<
>0
?1
@2
A3"
trackable_list_wrapper
.
>0
?1"
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
�
�trace_0
�trace_12�
6__inference_BatchNormGeneEncode2_layer_call_fn_1278789
6__inference_BatchNormGeneEncode2_layer_call_fn_1278802�
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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1278822
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1278856�
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
I0
J1
K2
L3"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
9__inference_BatchNormProteinEncode1_layer_call_fn_1278869
9__inference_BatchNormProteinEncode1_layer_call_fn_1278882�
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
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1278902
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1278936�
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
+:)/2BatchNormProteinEncode1/gamma
*:(/2BatchNormProteinEncode1/beta
3:1/ (2#BatchNormProteinEncode1/moving_mean
7:5/ (2'BatchNormProteinEncode1/moving_variance
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
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
2__inference_ConcatenateLayer_layer_call_fn_1278942�
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1278949�
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
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
3__inference_EmbeddingDimDense_layer_call_fn_1278958�
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
N__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_1278969�
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
J
%0
&1
@2
A3
K4
L5"
trackable_list_wrapper
f
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
9"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_model_29_layer_call_fn_1277944Gene_Input_LayerProtein_Input_Layer"�
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
*__inference_model_29_layer_call_fn_1278382inputs/0inputs/1"�
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
*__inference_model_29_layer_call_fn_1278428inputs/0inputs/1"�
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
*__inference_model_29_layer_call_fn_1278182Gene_Input_LayerProtein_Input_Layer"�
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
E__inference_model_29_layer_call_and_return_conditional_losses_1278511inputs/0inputs/1"�
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
E__inference_model_29_layer_call_and_return_conditional_losses_1278636inputs/0inputs/1"�
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
E__inference_model_29_layer_call_and_return_conditional_losses_1278235Gene_Input_LayerProtein_Input_Layer"�
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
E__inference_model_29_layer_call_and_return_conditional_losses_1278288Gene_Input_LayerProtein_Input_Layer"�
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
%__inference_signature_wrapper_1278336Gene_Input_LayerProtein_Input_Layer"�
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
0__inference_gene_encoder_1_layer_call_fn_1278645inputs"�
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
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_1278656inputs"�
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
%0
&1"
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
6__inference_BatchNormGeneEncode1_layer_call_fn_1278669inputs"�
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
6__inference_BatchNormGeneEncode1_layer_call_fn_1278682inputs"�
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
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1278702inputs"�
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
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1278736inputs"�
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
0__inference_gene_encoder_2_layer_call_fn_1278745inputs"�
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
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_1278756inputs"�
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
3__inference_protein_encoder_1_layer_call_fn_1278765inputs"�
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
N__inference_protein_encoder_1_layer_call_and_return_conditional_losses_1278776inputs"�
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
@0
A1"
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
6__inference_BatchNormGeneEncode2_layer_call_fn_1278789inputs"�
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
6__inference_BatchNormGeneEncode2_layer_call_fn_1278802inputs"�
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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1278822inputs"�
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
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1278856inputs"�
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
K0
L1"
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
9__inference_BatchNormProteinEncode1_layer_call_fn_1278869inputs"�
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
9__inference_BatchNormProteinEncode1_layer_call_fn_1278882inputs"�
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
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1278902inputs"�
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
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1278936inputs"�
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
2__inference_ConcatenateLayer_layer_call_fn_1278942inputs/0inputs/1"�
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
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1278949inputs/0inputs/1"�
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
3__inference_EmbeddingDimDense_layer_call_fn_1278958inputs"�
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
N__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_1278969inputs"�
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
:  (2count�
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1278702d&#%$4�1
*�'
!�
inputs����������
p 
� "&�#
�
0����������
� �
Q__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_1278736d%&#$4�1
*�'
!�
inputs����������
p
� "&�#
�
0����������
� �
6__inference_BatchNormGeneEncode1_layer_call_fn_1278669W&#%$4�1
*�'
!�
inputs����������
p 
� "������������
6__inference_BatchNormGeneEncode1_layer_call_fn_1278682W%&#$4�1
*�'
!�
inputs����������
p
� "������������
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1278822bA>@?3�0
)�&
 �
inputs���������}
p 
� "%�"
�
0���������}
� �
Q__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_1278856b@A>?3�0
)�&
 �
inputs���������}
p
� "%�"
�
0���������}
� �
6__inference_BatchNormGeneEncode2_layer_call_fn_1278789UA>@?3�0
)�&
 �
inputs���������}
p 
� "����������}�
6__inference_BatchNormGeneEncode2_layer_call_fn_1278802U@A>?3�0
)�&
 �
inputs���������}
p
� "����������}�
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1278902bLIKJ3�0
)�&
 �
inputs���������/
p 
� "%�"
�
0���������/
� �
T__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_1278936bKLIJ3�0
)�&
 �
inputs���������/
p
� "%�"
�
0���������/
� �
9__inference_BatchNormProteinEncode1_layer_call_fn_1278869ULIKJ3�0
)�&
 �
inputs���������/
p 
� "����������/�
9__inference_BatchNormProteinEncode1_layer_call_fn_1278882UKLIJ3�0
)�&
 �
inputs���������/
p
� "����������/�
M__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_1278949�Z�W
P�M
K�H
"�
inputs/0���������}
"�
inputs/1���������/
� "&�#
�
0����������
� �
2__inference_ConcatenateLayer_layer_call_fn_1278942wZ�W
P�M
K�H
"�
inputs/0���������}
"�
inputs/1���������/
� "������������
N__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_1278969]YZ0�-
&�#
!�
inputs����������
� "%�"
�
0���������@
� �
3__inference_EmbeddingDimDense_layer_call_fn_1278958PYZ0�-
&�#
!�
inputs����������
� "����������@�
"__inference__wrapped_model_1277541�&#%$56-.A>@?LIKJYZo�l
e�b
`�]
+�(
Gene_Input_Layer����������
.�+
Protein_Input_Layer����������
� "E�B
@
EmbeddingDimDense+�(
EmbeddingDimDense���������@�
K__inference_gene_encoder_1_layer_call_and_return_conditional_losses_1278656^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
0__inference_gene_encoder_1_layer_call_fn_1278645Q0�-
&�#
!�
inputs����������
� "������������
K__inference_gene_encoder_2_layer_call_and_return_conditional_losses_1278756]-.0�-
&�#
!�
inputs����������
� "%�"
�
0���������}
� �
0__inference_gene_encoder_2_layer_call_fn_1278745P-.0�-
&�#
!�
inputs����������
� "����������}�
E__inference_model_29_layer_call_and_return_conditional_losses_1278235�&#%$56-.A>@?LIKJYZw�t
m�j
`�]
+�(
Gene_Input_Layer����������
.�+
Protein_Input_Layer����������
p 

 
� "%�"
�
0���������@
� �
E__inference_model_29_layer_call_and_return_conditional_losses_1278288�%&#$56-.@A>?KLIJYZw�t
m�j
`�]
+�(
Gene_Input_Layer����������
.�+
Protein_Input_Layer����������
p

 
� "%�"
�
0���������@
� �
E__inference_model_29_layer_call_and_return_conditional_losses_1278511�&#%$56-.A>@?LIKJYZd�a
Z�W
M�J
#� 
inputs/0����������
#� 
inputs/1����������
p 

 
� "%�"
�
0���������@
� �
E__inference_model_29_layer_call_and_return_conditional_losses_1278636�%&#$56-.@A>?KLIJYZd�a
Z�W
M�J
#� 
inputs/0����������
#� 
inputs/1����������
p

 
� "%�"
�
0���������@
� �
*__inference_model_29_layer_call_fn_1277944�&#%$56-.A>@?LIKJYZw�t
m�j
`�]
+�(
Gene_Input_Layer����������
.�+
Protein_Input_Layer����������
p 

 
� "����������@�
*__inference_model_29_layer_call_fn_1278182�%&#$56-.@A>?KLIJYZw�t
m�j
`�]
+�(
Gene_Input_Layer����������
.�+
Protein_Input_Layer����������
p

 
� "����������@�
*__inference_model_29_layer_call_fn_1278382�&#%$56-.A>@?LIKJYZd�a
Z�W
M�J
#� 
inputs/0����������
#� 
inputs/1����������
p 

 
� "����������@�
*__inference_model_29_layer_call_fn_1278428�%&#$56-.@A>?KLIJYZd�a
Z�W
M�J
#� 
inputs/0����������
#� 
inputs/1����������
p

 
� "����������@�
N__inference_protein_encoder_1_layer_call_and_return_conditional_losses_1278776]560�-
&�#
!�
inputs����������
� "%�"
�
0���������/
� �
3__inference_protein_encoder_1_layer_call_fn_1278765P560�-
&�#
!�
inputs����������
� "����������/�
%__inference_signature_wrapper_1278336�&#%$56-.A>@?LIKJYZ���
� 
���
?
Gene_Input_Layer+�(
Gene_Input_Layer����������
E
Protein_Input_Layer.�+
Protein_Input_Layer����������"E�B
@
EmbeddingDimDense+�(
EmbeddingDimDense���������@