ли
ЏВ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
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
delete_old_dirsbool(ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ
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
dtypetypeѕ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
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
list(type)(0ѕ
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
┴
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
executor_typestring ѕе
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
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68д 
ѕ
gene_encoder_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
лЗ*&
shared_namegene_encoder_1/kernel
Ђ
)gene_encoder_1/kernel/Read/ReadVariableOpReadVariableOpgene_encoder_1/kernel* 
_output_shapes
:
лЗ*
dtype0

gene_encoder_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:З*$
shared_namegene_encoder_1/bias
x
'gene_encoder_1/bias/Read/ReadVariableOpReadVariableOpgene_encoder_1/bias*
_output_shapes	
:З*
dtype0
Ї
BatchNormGeneEncode1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:З*+
shared_nameBatchNormGeneEncode1/gamma
є
.BatchNormGeneEncode1/gamma/Read/ReadVariableOpReadVariableOpBatchNormGeneEncode1/gamma*
_output_shapes	
:З*
dtype0
І
BatchNormGeneEncode1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:З**
shared_nameBatchNormGeneEncode1/beta
ё
-BatchNormGeneEncode1/beta/Read/ReadVariableOpReadVariableOpBatchNormGeneEncode1/beta*
_output_shapes	
:З*
dtype0
Ў
 BatchNormGeneEncode1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:З*1
shared_name" BatchNormGeneEncode1/moving_mean
њ
4BatchNormGeneEncode1/moving_mean/Read/ReadVariableOpReadVariableOp BatchNormGeneEncode1/moving_mean*
_output_shapes	
:З*
dtype0
А
$BatchNormGeneEncode1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:З*5
shared_name&$BatchNormGeneEncode1/moving_variance
џ
8BatchNormGeneEncode1/moving_variance/Read/ReadVariableOpReadVariableOp$BatchNormGeneEncode1/moving_variance*
_output_shapes	
:З*
dtype0
Є
gene_encoder_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	З}*&
shared_namegene_encoder_2/kernel
ђ
)gene_encoder_2/kernel/Read/ReadVariableOpReadVariableOpgene_encoder_2/kernel*
_output_shapes
:	З}*
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
ї
protein_encoder_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:W*)
shared_nameprotein_encoder_1/kernel
Ё
,protein_encoder_1/kernel/Read/ReadVariableOpReadVariableOpprotein_encoder_1/kernel*
_output_shapes

:W*
dtype0
ё
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
ї
BatchNormGeneEncode2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*+
shared_nameBatchNormGeneEncode2/gamma
Ё
.BatchNormGeneEncode2/gamma/Read/ReadVariableOpReadVariableOpBatchNormGeneEncode2/gamma*
_output_shapes
:}*
dtype0
і
BatchNormGeneEncode2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:}**
shared_nameBatchNormGeneEncode2/beta
Ѓ
-BatchNormGeneEncode2/beta/Read/ReadVariableOpReadVariableOpBatchNormGeneEncode2/beta*
_output_shapes
:}*
dtype0
ў
 BatchNormGeneEncode2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*1
shared_name" BatchNormGeneEncode2/moving_mean
Љ
4BatchNormGeneEncode2/moving_mean/Read/ReadVariableOpReadVariableOp BatchNormGeneEncode2/moving_mean*
_output_shapes
:}*
dtype0
а
$BatchNormGeneEncode2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*5
shared_name&$BatchNormGeneEncode2/moving_variance
Ў
8BatchNormGeneEncode2/moving_variance/Read/ReadVariableOpReadVariableOp$BatchNormGeneEncode2/moving_variance*
_output_shapes
:}*
dtype0
њ
BatchNormProteinEncode1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameBatchNormProteinEncode1/gamma
І
1BatchNormProteinEncode1/gamma/Read/ReadVariableOpReadVariableOpBatchNormProteinEncode1/gamma*
_output_shapes
:*
dtype0
љ
BatchNormProteinEncode1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameBatchNormProteinEncode1/beta
Ѕ
0BatchNormProteinEncode1/beta/Read/ReadVariableOpReadVariableOpBatchNormProteinEncode1/beta*
_output_shapes
:*
dtype0
ъ
#BatchNormProteinEncode1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#BatchNormProteinEncode1/moving_mean
Ќ
7BatchNormProteinEncode1/moving_mean/Read/ReadVariableOpReadVariableOp#BatchNormProteinEncode1/moving_mean*
_output_shapes
:*
dtype0
д
'BatchNormProteinEncode1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'BatchNormProteinEncode1/moving_variance
Ъ
;BatchNormProteinEncode1/moving_variance/Read/ReadVariableOpReadVariableOp'BatchNormProteinEncode1/moving_variance*
_output_shapes
:*
dtype0
Ї
EmbeddingDimDense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	њ@*)
shared_nameEmbeddingDimDense/kernel
є
,EmbeddingDimDense/kernel/Read/ReadVariableOpReadVariableOpEmbeddingDimDense/kernel*
_output_shapes
:	њ@*
dtype0
ё
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
ћ@
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¤?
value┼?B┬? B╗?
ё
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
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
д

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
Н
axis
	gamma
beta
moving_mean
 moving_variance
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses*
* 
д

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
д

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses*
Н
7axis
	8gamma
9beta
:moving_mean
;moving_variance
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses*
Н
Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses*
ј
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses* 
д

Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses*
* 
џ
0
1
2
3
4
 5
'6
(7
/8
09
810
911
:12
;13
C14
D15
E16
F17
S18
T19*
j
0
1
2
3
'4
(5
/6
07
88
99
C10
D11
S12
T13*
* 
░
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

`serving_default* 
e_
VARIABLE_VALUEgene_encoder_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEgene_encoder_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
Њ
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
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
 
0
1
2
 3*

0
1*
* 
Њ
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
* 
* 
e_
VARIABLE_VALUEgene_encoder_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEgene_encoder_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

'0
(1*

'0
(1*
* 
Њ
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*
* 
* 
hb
VARIABLE_VALUEprotein_encoder_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEprotein_encoder_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

/0
01*

/0
01*
* 
Њ
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*
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
 
80
91
:2
;3*

80
91*
* 
Њ
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses*
* 
* 
* 
lf
VARIABLE_VALUEBatchNormProteinEncode1/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
jd
VARIABLE_VALUEBatchNormProteinEncode1/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE#BatchNormProteinEncode1/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
ђz
VARIABLE_VALUE'BatchNormProteinEncode1/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
C0
D1
E2
F3*

C0
D1*
* 
Њ
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
Ћ
non_trainable_variables
ђlayers
Ђmetrics
 ѓlayer_regularization_losses
Ѓlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses* 
* 
* 
hb
VARIABLE_VALUEEmbeddingDimDense/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEEmbeddingDimDense/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*

S0
T1*

S0
T1*
* 
ў
ёnon_trainable_variables
Ёlayers
єmetrics
 Єlayer_regularization_losses
ѕlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*
* 
* 
.
0
 1
:2
;3
E4
F5*
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

Ѕ0*
* 
* 
* 
* 
* 
* 
* 
* 

0
 1*
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

E0
F1*
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

іtotal

Іcount
ї	variables
Ї	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

і0
І1*

ї	variables*
Ё
 serving_default_Gene_Input_LayerPlaceholder*(
_output_shapes
:         л*
dtype0*
shape:         л
є
#serving_default_Protein_Input_LayerPlaceholder*'
_output_shapes
:         W*
dtype0*
shape:         W
Л
StatefulPartitionedCallStatefulPartitionedCall serving_default_Gene_Input_Layer#serving_default_Protein_Input_Layergene_encoder_1/kernelgene_encoder_1/bias$BatchNormGeneEncode1/moving_varianceBatchNormGeneEncode1/gamma BatchNormGeneEncode1/moving_meanBatchNormGeneEncode1/betaprotein_encoder_1/kernelprotein_encoder_1/biasgene_encoder_2/kernelgene_encoder_2/bias$BatchNormGeneEncode2/moving_varianceBatchNormGeneEncode2/gamma BatchNormGeneEncode2/moving_meanBatchNormGeneEncode2/beta'BatchNormProteinEncode1/moving_varianceBatchNormProteinEncode1/gamma#BatchNormProteinEncode1/moving_meanBatchNormProteinEncode1/betaEmbeddingDimDense/kernelEmbeddingDimDense/bias*!
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *-
f(R&
$__inference_signature_wrapper_748360
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
┐

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
 *0
config_proto 

CPU

GPU2*0J 8ѓ *(
f#R!
__inference__traced_save_748783
ѓ
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
 *0
config_proto 

CPU

GPU2*0J 8ѓ *+
f&R$
"__inference__traced_restore_748859╗з

г

■
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_747531

inputs2
matmul_readvariableop_resource:
лЗ.
biasadd_readvariableop_resource:	З
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
лЗ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Зs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:З*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ЗW
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:         З[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:         Зw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         л: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         л
 
_user_specified_nameinputs
Ч
Ў
)__inference_model_23_layer_call_fn_747668
gene_input_layer
protein_input_layer
unknown:
лЗ
	unknown_0:	З
	unknown_1:	З
	unknown_2:	З
	unknown_3:	З
	unknown_4:	З
	unknown_5:W
	unknown_6:
	unknown_7:	З}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:	њ@

unknown_18:@
identityѕбStatefulPartitionedCall№
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
:         @*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_model_23_layer_call_and_return_conditional_losses_747625o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         л:         W: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:         л
*
_user_specified_nameGene_Input_Layer:\X
'
_output_shapes
:         W
-
_user_specified_nameProtein_Input_Layer
▀
│
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_747289

inputs0
!batchnorm_readvariableop_resource:	З4
%batchnorm_mul_readvariableop_resource:	З2
#batchnorm_readvariableop_1_resource:	З2
#batchnorm_readvariableop_2_resource:	З
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:З*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ЗQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:З
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:З*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Зd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         З{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:З*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:З{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:З*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Зs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         Зc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         З║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         З: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         З
 
_user_specified_nameinputs
═
»
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_748546

inputs/
!batchnorm_readvariableop_resource:}3
%batchnorm_mul_readvariableop_resource:}1
#batchnorm_readvariableop_1_resource:}1
#batchnorm_readvariableop_2_resource:}
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:w
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
:         }z
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
:         }b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         }║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         }: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         }
 
_user_specified_nameinputs
л
▓
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_747453

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:w
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
:         z
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
:         b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         ║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Еv
ю
D__inference_model_23_layer_call_and_return_conditional_losses_748187
inputs_0
inputs_1A
-gene_encoder_1_matmul_readvariableop_resource:
лЗ=
.gene_encoder_1_biasadd_readvariableop_resource:	ЗE
6batchnormgeneencode1_batchnorm_readvariableop_resource:	ЗI
:batchnormgeneencode1_batchnorm_mul_readvariableop_resource:	ЗG
8batchnormgeneencode1_batchnorm_readvariableop_1_resource:	ЗG
8batchnormgeneencode1_batchnorm_readvariableop_2_resource:	ЗB
0protein_encoder_1_matmul_readvariableop_resource:W?
1protein_encoder_1_biasadd_readvariableop_resource:@
-gene_encoder_2_matmul_readvariableop_resource:	З}<
.gene_encoder_2_biasadd_readvariableop_resource:}D
6batchnormgeneencode2_batchnorm_readvariableop_resource:}H
:batchnormgeneencode2_batchnorm_mul_readvariableop_resource:}F
8batchnormgeneencode2_batchnorm_readvariableop_1_resource:}F
8batchnormgeneencode2_batchnorm_readvariableop_2_resource:}G
9batchnormproteinencode1_batchnorm_readvariableop_resource:K
=batchnormproteinencode1_batchnorm_mul_readvariableop_resource:I
;batchnormproteinencode1_batchnorm_readvariableop_1_resource:I
;batchnormproteinencode1_batchnorm_readvariableop_2_resource:C
0embeddingdimdense_matmul_readvariableop_resource:	њ@?
1embeddingdimdense_biasadd_readvariableop_resource:@
identityѕб-BatchNormGeneEncode1/batchnorm/ReadVariableOpб/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1б/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2б1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOpб-BatchNormGeneEncode2/batchnorm/ReadVariableOpб/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1б/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2б1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOpб0BatchNormProteinEncode1/batchnorm/ReadVariableOpб2BatchNormProteinEncode1/batchnorm/ReadVariableOp_1б2BatchNormProteinEncode1/batchnorm/ReadVariableOp_2б4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOpб(EmbeddingDimDense/BiasAdd/ReadVariableOpб'EmbeddingDimDense/MatMul/ReadVariableOpб%gene_encoder_1/BiasAdd/ReadVariableOpб$gene_encoder_1/MatMul/ReadVariableOpб%gene_encoder_2/BiasAdd/ReadVariableOpб$gene_encoder_2/MatMul/ReadVariableOpб(protein_encoder_1/BiasAdd/ReadVariableOpб'protein_encoder_1/MatMul/ReadVariableOpћ
$gene_encoder_1/MatMul/ReadVariableOpReadVariableOp-gene_encoder_1_matmul_readvariableop_resource* 
_output_shapes
:
лЗ*
dtype0і
gene_encoder_1/MatMulMatMulinputs_0,gene_encoder_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ЗЉ
%gene_encoder_1/BiasAdd/ReadVariableOpReadVariableOp.gene_encoder_1_biasadd_readvariableop_resource*
_output_shapes	
:З*
dtype0ц
gene_encoder_1/BiasAddBiasAddgene_encoder_1/MatMul:product:0-gene_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Зu
gene_encoder_1/SigmoidSigmoidgene_encoder_1/BiasAdd:output:0*
T0*(
_output_shapes
:         ЗА
-BatchNormGeneEncode1/batchnorm/ReadVariableOpReadVariableOp6batchnormgeneencode1_batchnorm_readvariableop_resource*
_output_shapes	
:З*
dtype0i
$BatchNormGeneEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:и
"BatchNormGeneEncode1/batchnorm/addAddV25BatchNormGeneEncode1/batchnorm/ReadVariableOp:value:0-BatchNormGeneEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:З{
$BatchNormGeneEncode1/batchnorm/RsqrtRsqrt&BatchNormGeneEncode1/batchnorm/add:z:0*
T0*
_output_shapes	
:ЗЕ
1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgeneencode1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:З*
dtype0┤
"BatchNormGeneEncode1/batchnorm/mulMul(BatchNormGeneEncode1/batchnorm/Rsqrt:y:09BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Зб
$BatchNormGeneEncode1/batchnorm/mul_1Mulgene_encoder_1/Sigmoid:y:0&BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*(
_output_shapes
:         ЗЦ
/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1ReadVariableOp8batchnormgeneencode1_batchnorm_readvariableop_1_resource*
_output_shapes	
:З*
dtype0▓
$BatchNormGeneEncode1/batchnorm/mul_2Mul7BatchNormGeneEncode1/batchnorm/ReadVariableOp_1:value:0&BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ЗЦ
/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2ReadVariableOp8batchnormgeneencode1_batchnorm_readvariableop_2_resource*
_output_shapes	
:З*
dtype0▓
"BatchNormGeneEncode1/batchnorm/subSub7BatchNormGeneEncode1/batchnorm/ReadVariableOp_2:value:0(BatchNormGeneEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:З▓
$BatchNormGeneEncode1/batchnorm/add_1AddV2(BatchNormGeneEncode1/batchnorm/mul_1:z:0&BatchNormGeneEncode1/batchnorm/sub:z:0*
T0*(
_output_shapes
:         Зў
'protein_encoder_1/MatMul/ReadVariableOpReadVariableOp0protein_encoder_1_matmul_readvariableop_resource*
_output_shapes

:W*
dtype0Ј
protein_encoder_1/MatMulMatMulinputs_1/protein_encoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ќ
(protein_encoder_1/BiasAdd/ReadVariableOpReadVariableOp1protein_encoder_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0г
protein_encoder_1/BiasAddBiasAdd"protein_encoder_1/MatMul:product:00protein_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         z
protein_encoder_1/SigmoidSigmoid"protein_encoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:         Њ
$gene_encoder_2/MatMul/ReadVariableOpReadVariableOp-gene_encoder_2_matmul_readvariableop_resource*
_output_shapes
:	З}*
dtype0Е
gene_encoder_2/MatMulMatMul(BatchNormGeneEncode1/batchnorm/add_1:z:0,gene_encoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         }љ
%gene_encoder_2/BiasAdd/ReadVariableOpReadVariableOp.gene_encoder_2_biasadd_readvariableop_resource*
_output_shapes
:}*
dtype0Б
gene_encoder_2/BiasAddBiasAddgene_encoder_2/MatMul:product:0-gene_encoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         }t
gene_encoder_2/SigmoidSigmoidgene_encoder_2/BiasAdd:output:0*
T0*'
_output_shapes
:         }а
-BatchNormGeneEncode2/batchnorm/ReadVariableOpReadVariableOp6batchnormgeneencode2_batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0i
$BatchNormGeneEncode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:Х
"BatchNormGeneEncode2/batchnorm/addAddV25BatchNormGeneEncode2/batchnorm/ReadVariableOp:value:0-BatchNormGeneEncode2/batchnorm/add/y:output:0*
T0*
_output_shapes
:}z
$BatchNormGeneEncode2/batchnorm/RsqrtRsqrt&BatchNormGeneEncode2/batchnorm/add:z:0*
T0*
_output_shapes
:}е
1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgeneencode2_batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0│
"BatchNormGeneEncode2/batchnorm/mulMul(BatchNormGeneEncode2/batchnorm/Rsqrt:y:09BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}А
$BatchNormGeneEncode2/batchnorm/mul_1Mulgene_encoder_2/Sigmoid:y:0&BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         }ц
/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1ReadVariableOp8batchnormgeneencode2_batchnorm_readvariableop_1_resource*
_output_shapes
:}*
dtype0▒
$BatchNormGeneEncode2/batchnorm/mul_2Mul7BatchNormGeneEncode2/batchnorm/ReadVariableOp_1:value:0&BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*
_output_shapes
:}ц
/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2ReadVariableOp8batchnormgeneencode2_batchnorm_readvariableop_2_resource*
_output_shapes
:}*
dtype0▒
"BatchNormGeneEncode2/batchnorm/subSub7BatchNormGeneEncode2/batchnorm/ReadVariableOp_2:value:0(BatchNormGeneEncode2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:}▒
$BatchNormGeneEncode2/batchnorm/add_1AddV2(BatchNormGeneEncode2/batchnorm/mul_1:z:0&BatchNormGeneEncode2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         }д
0BatchNormProteinEncode1/batchnorm/ReadVariableOpReadVariableOp9batchnormproteinencode1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'BatchNormProteinEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:┐
%BatchNormProteinEncode1/batchnorm/addAddV28BatchNormProteinEncode1/batchnorm/ReadVariableOp:value:00BatchNormProteinEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:ђ
'BatchNormProteinEncode1/batchnorm/RsqrtRsqrt)BatchNormProteinEncode1/batchnorm/add:z:0*
T0*
_output_shapes
:«
4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOpReadVariableOp=batchnormproteinencode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╝
%BatchNormProteinEncode1/batchnorm/mulMul+BatchNormProteinEncode1/batchnorm/Rsqrt:y:0<BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ф
'BatchNormProteinEncode1/batchnorm/mul_1Mulprotein_encoder_1/Sigmoid:y:0)BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:         ф
2BatchNormProteinEncode1/batchnorm/ReadVariableOp_1ReadVariableOp;batchnormproteinencode1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0║
'BatchNormProteinEncode1/batchnorm/mul_2Mul:BatchNormProteinEncode1/batchnorm/ReadVariableOp_1:value:0)BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*
_output_shapes
:ф
2BatchNormProteinEncode1/batchnorm/ReadVariableOp_2ReadVariableOp;batchnormproteinencode1_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0║
%BatchNormProteinEncode1/batchnorm/subSub:BatchNormProteinEncode1/batchnorm/ReadVariableOp_2:value:0+BatchNormProteinEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:║
'BatchNormProteinEncode1/batchnorm/add_1AddV2+BatchNormProteinEncode1/batchnorm/mul_1:z:0)BatchNormProteinEncode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:         ^
ConcatenateLayer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :П
ConcatenateLayer/concatConcatV2(BatchNormGeneEncode2/batchnorm/add_1:z:0+BatchNormProteinEncode1/batchnorm/add_1:z:0%ConcatenateLayer/concat/axis:output:0*
N*
T0*(
_output_shapes
:         њЎ
'EmbeddingDimDense/MatMul/ReadVariableOpReadVariableOp0embeddingdimdense_matmul_readvariableop_resource*
_output_shapes
:	њ@*
dtype0Д
EmbeddingDimDense/MatMulMatMul ConcatenateLayer/concat:output:0/EmbeddingDimDense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ќ
(EmbeddingDimDense/BiasAdd/ReadVariableOpReadVariableOp1embeddingdimdense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0г
EmbeddingDimDense/BiasAddBiasAdd"EmbeddingDimDense/MatMul:product:00EmbeddingDimDense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @z
EmbeddingDimDense/SigmoidSigmoid"EmbeddingDimDense/BiasAdd:output:0*
T0*'
_output_shapes
:         @l
IdentityIdentityEmbeddingDimDense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         @Ы
NoOpNoOp.^BatchNormGeneEncode1/batchnorm/ReadVariableOp0^BatchNormGeneEncode1/batchnorm/ReadVariableOp_10^BatchNormGeneEncode1/batchnorm/ReadVariableOp_22^BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp.^BatchNormGeneEncode2/batchnorm/ReadVariableOp0^BatchNormGeneEncode2/batchnorm/ReadVariableOp_10^BatchNormGeneEncode2/batchnorm/ReadVariableOp_22^BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp1^BatchNormProteinEncode1/batchnorm/ReadVariableOp3^BatchNormProteinEncode1/batchnorm/ReadVariableOp_13^BatchNormProteinEncode1/batchnorm/ReadVariableOp_25^BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp)^EmbeddingDimDense/BiasAdd/ReadVariableOp(^EmbeddingDimDense/MatMul/ReadVariableOp&^gene_encoder_1/BiasAdd/ReadVariableOp%^gene_encoder_1/MatMul/ReadVariableOp&^gene_encoder_2/BiasAdd/ReadVariableOp%^gene_encoder_2/MatMul/ReadVariableOp)^protein_encoder_1/BiasAdd/ReadVariableOp(^protein_encoder_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         л:         W: : : : : : : : : : : : : : : : : : : : 2^
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
:         л
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         W
"
_user_specified_name
inputs/1
Љ%
ж
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_748580

inputs5
'assignmovingavg_readvariableop_resource:}7
)assignmovingavg_1_readvariableop_resource:}3
%batchnorm_mul_readvariableop_resource:}/
!batchnorm_readvariableop_resource:}
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOph
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

:}Є
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         }l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ъ
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
О#<ѓ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:}*
dtype0Ђ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:}x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:}г
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
О#<є
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:}*
dtype0Є
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:}~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:}┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:q
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
:         }h
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
:         }b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         }Ж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         }: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         }
 
_user_specified_nameinputs
┌
а
2__inference_EmbeddingDimDense_layer_call_fn_748682

inputs
unknown:	њ@
	unknown_0:@
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_747618o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         њ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         њ
 
_user_specified_nameinputs
О
Ъ
2__inference_protein_encoder_1_layer_call_fn_748489

inputs
unknown:W
	unknown_0:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_747557o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         W: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         W
 
_user_specified_nameinputs
Б

■
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_748500

inputs0
matmul_readvariableop_resource:W-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:W*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         W: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         W
 
_user_specified_nameinputs
л
▓
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_748626

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:w
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
:         z
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
:         b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         ║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Д

 
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_748693

inputs1
matmul_readvariableop_resource:	њ@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	њ@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         @Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         њ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         њ
 
_user_specified_nameinputs
▒
н
5__inference_BatchNormGeneEncode1_layer_call_fn_748393

inputs
unknown:	З
	unknown_0:	З
	unknown_1:	З
	unknown_2:	З
identityѕбStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         З*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_747289p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         З`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         З: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         З
 
_user_specified_nameinputs
»
н
5__inference_BatchNormGeneEncode1_layer_call_fn_748406

inputs
unknown:	З
	unknown_0:	З
	unknown_1:	З
	unknown_2:	З
identityѕбStatefulPartitionedCallЂ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         З*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_747336p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         З`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         З: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         З
 
_user_specified_nameinputs
Д├
╬
D__inference_model_23_layer_call_and_return_conditional_losses_748312
inputs_0
inputs_1A
-gene_encoder_1_matmul_readvariableop_resource:
лЗ=
.gene_encoder_1_biasadd_readvariableop_resource:	ЗK
<batchnormgeneencode1_assignmovingavg_readvariableop_resource:	ЗM
>batchnormgeneencode1_assignmovingavg_1_readvariableop_resource:	ЗI
:batchnormgeneencode1_batchnorm_mul_readvariableop_resource:	ЗE
6batchnormgeneencode1_batchnorm_readvariableop_resource:	ЗB
0protein_encoder_1_matmul_readvariableop_resource:W?
1protein_encoder_1_biasadd_readvariableop_resource:@
-gene_encoder_2_matmul_readvariableop_resource:	З}<
.gene_encoder_2_biasadd_readvariableop_resource:}J
<batchnormgeneencode2_assignmovingavg_readvariableop_resource:}L
>batchnormgeneencode2_assignmovingavg_1_readvariableop_resource:}H
:batchnormgeneencode2_batchnorm_mul_readvariableop_resource:}D
6batchnormgeneencode2_batchnorm_readvariableop_resource:}M
?batchnormproteinencode1_assignmovingavg_readvariableop_resource:O
Abatchnormproteinencode1_assignmovingavg_1_readvariableop_resource:K
=batchnormproteinencode1_batchnorm_mul_readvariableop_resource:G
9batchnormproteinencode1_batchnorm_readvariableop_resource:C
0embeddingdimdense_matmul_readvariableop_resource:	њ@?
1embeddingdimdense_biasadd_readvariableop_resource:@
identityѕб$BatchNormGeneEncode1/AssignMovingAvgб3BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOpб&BatchNormGeneEncode1/AssignMovingAvg_1б5BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOpб-BatchNormGeneEncode1/batchnorm/ReadVariableOpб1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOpб$BatchNormGeneEncode2/AssignMovingAvgб3BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOpб&BatchNormGeneEncode2/AssignMovingAvg_1б5BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOpб-BatchNormGeneEncode2/batchnorm/ReadVariableOpб1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOpб'BatchNormProteinEncode1/AssignMovingAvgб6BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOpб)BatchNormProteinEncode1/AssignMovingAvg_1б8BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOpб0BatchNormProteinEncode1/batchnorm/ReadVariableOpб4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOpб(EmbeddingDimDense/BiasAdd/ReadVariableOpб'EmbeddingDimDense/MatMul/ReadVariableOpб%gene_encoder_1/BiasAdd/ReadVariableOpб$gene_encoder_1/MatMul/ReadVariableOpб%gene_encoder_2/BiasAdd/ReadVariableOpб$gene_encoder_2/MatMul/ReadVariableOpб(protein_encoder_1/BiasAdd/ReadVariableOpб'protein_encoder_1/MatMul/ReadVariableOpћ
$gene_encoder_1/MatMul/ReadVariableOpReadVariableOp-gene_encoder_1_matmul_readvariableop_resource* 
_output_shapes
:
лЗ*
dtype0і
gene_encoder_1/MatMulMatMulinputs_0,gene_encoder_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ЗЉ
%gene_encoder_1/BiasAdd/ReadVariableOpReadVariableOp.gene_encoder_1_biasadd_readvariableop_resource*
_output_shapes	
:З*
dtype0ц
gene_encoder_1/BiasAddBiasAddgene_encoder_1/MatMul:product:0-gene_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Зu
gene_encoder_1/SigmoidSigmoidgene_encoder_1/BiasAdd:output:0*
T0*(
_output_shapes
:         З}
3BatchNormGeneEncode1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Й
!BatchNormGeneEncode1/moments/meanMeangene_encoder_1/Sigmoid:y:0<BatchNormGeneEncode1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	З*
	keep_dims(Ј
)BatchNormGeneEncode1/moments/StopGradientStopGradient*BatchNormGeneEncode1/moments/mean:output:0*
T0*
_output_shapes
:	Зк
.BatchNormGeneEncode1/moments/SquaredDifferenceSquaredDifferencegene_encoder_1/Sigmoid:y:02BatchNormGeneEncode1/moments/StopGradient:output:0*
T0*(
_output_shapes
:         ЗЂ
7BatchNormGeneEncode1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: я
%BatchNormGeneEncode1/moments/varianceMean2BatchNormGeneEncode1/moments/SquaredDifference:z:0@BatchNormGeneEncode1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	З*
	keep_dims(ў
$BatchNormGeneEncode1/moments/SqueezeSqueeze*BatchNormGeneEncode1/moments/mean:output:0*
T0*
_output_shapes	
:З*
squeeze_dims
 ъ
&BatchNormGeneEncode1/moments/Squeeze_1Squeeze.BatchNormGeneEncode1/moments/variance:output:0*
T0*
_output_shapes	
:З*
squeeze_dims
 o
*BatchNormGeneEncode1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<Г
3BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOpReadVariableOp<batchnormgeneencode1_assignmovingavg_readvariableop_resource*
_output_shapes	
:З*
dtype0┴
(BatchNormGeneEncode1/AssignMovingAvg/subSub;BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOp:value:0-BatchNormGeneEncode1/moments/Squeeze:output:0*
T0*
_output_shapes	
:ЗИ
(BatchNormGeneEncode1/AssignMovingAvg/mulMul,BatchNormGeneEncode1/AssignMovingAvg/sub:z:03BatchNormGeneEncode1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:Зђ
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
О#<▒
5BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOpReadVariableOp>batchnormgeneencode1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:З*
dtype0К
*BatchNormGeneEncode1/AssignMovingAvg_1/subSub=BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOp:value:0/BatchNormGeneEncode1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:ЗЙ
*BatchNormGeneEncode1/AssignMovingAvg_1/mulMul.BatchNormGeneEncode1/AssignMovingAvg_1/sub:z:05BatchNormGeneEncode1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:Зѕ
&BatchNormGeneEncode1/AssignMovingAvg_1AssignSubVariableOp>batchnormgeneencode1_assignmovingavg_1_readvariableop_resource.BatchNormGeneEncode1/AssignMovingAvg_1/mul:z:06^BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0i
$BatchNormGeneEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:▒
"BatchNormGeneEncode1/batchnorm/addAddV2/BatchNormGeneEncode1/moments/Squeeze_1:output:0-BatchNormGeneEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:З{
$BatchNormGeneEncode1/batchnorm/RsqrtRsqrt&BatchNormGeneEncode1/batchnorm/add:z:0*
T0*
_output_shapes	
:ЗЕ
1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgeneencode1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:З*
dtype0┤
"BatchNormGeneEncode1/batchnorm/mulMul(BatchNormGeneEncode1/batchnorm/Rsqrt:y:09BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Зб
$BatchNormGeneEncode1/batchnorm/mul_1Mulgene_encoder_1/Sigmoid:y:0&BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*(
_output_shapes
:         Зе
$BatchNormGeneEncode1/batchnorm/mul_2Mul-BatchNormGeneEncode1/moments/Squeeze:output:0&BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*
_output_shapes	
:ЗА
-BatchNormGeneEncode1/batchnorm/ReadVariableOpReadVariableOp6batchnormgeneencode1_batchnorm_readvariableop_resource*
_output_shapes	
:З*
dtype0░
"BatchNormGeneEncode1/batchnorm/subSub5BatchNormGeneEncode1/batchnorm/ReadVariableOp:value:0(BatchNormGeneEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:З▓
$BatchNormGeneEncode1/batchnorm/add_1AddV2(BatchNormGeneEncode1/batchnorm/mul_1:z:0&BatchNormGeneEncode1/batchnorm/sub:z:0*
T0*(
_output_shapes
:         Зў
'protein_encoder_1/MatMul/ReadVariableOpReadVariableOp0protein_encoder_1_matmul_readvariableop_resource*
_output_shapes

:W*
dtype0Ј
protein_encoder_1/MatMulMatMulinputs_1/protein_encoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ќ
(protein_encoder_1/BiasAdd/ReadVariableOpReadVariableOp1protein_encoder_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0г
protein_encoder_1/BiasAddBiasAdd"protein_encoder_1/MatMul:product:00protein_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         z
protein_encoder_1/SigmoidSigmoid"protein_encoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:         Њ
$gene_encoder_2/MatMul/ReadVariableOpReadVariableOp-gene_encoder_2_matmul_readvariableop_resource*
_output_shapes
:	З}*
dtype0Е
gene_encoder_2/MatMulMatMul(BatchNormGeneEncode1/batchnorm/add_1:z:0,gene_encoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         }љ
%gene_encoder_2/BiasAdd/ReadVariableOpReadVariableOp.gene_encoder_2_biasadd_readvariableop_resource*
_output_shapes
:}*
dtype0Б
gene_encoder_2/BiasAddBiasAddgene_encoder_2/MatMul:product:0-gene_encoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         }t
gene_encoder_2/SigmoidSigmoidgene_encoder_2/BiasAdd:output:0*
T0*'
_output_shapes
:         }}
3BatchNormGeneEncode2/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: й
!BatchNormGeneEncode2/moments/meanMeangene_encoder_2/Sigmoid:y:0<BatchNormGeneEncode2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:}*
	keep_dims(ј
)BatchNormGeneEncode2/moments/StopGradientStopGradient*BatchNormGeneEncode2/moments/mean:output:0*
T0*
_output_shapes

:}┼
.BatchNormGeneEncode2/moments/SquaredDifferenceSquaredDifferencegene_encoder_2/Sigmoid:y:02BatchNormGeneEncode2/moments/StopGradient:output:0*
T0*'
_output_shapes
:         }Ђ
7BatchNormGeneEncode2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: П
%BatchNormGeneEncode2/moments/varianceMean2BatchNormGeneEncode2/moments/SquaredDifference:z:0@BatchNormGeneEncode2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:}*
	keep_dims(Ќ
$BatchNormGeneEncode2/moments/SqueezeSqueeze*BatchNormGeneEncode2/moments/mean:output:0*
T0*
_output_shapes
:}*
squeeze_dims
 Ю
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
О#<г
3BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOpReadVariableOp<batchnormgeneencode2_assignmovingavg_readvariableop_resource*
_output_shapes
:}*
dtype0└
(BatchNormGeneEncode2/AssignMovingAvg/subSub;BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOp:value:0-BatchNormGeneEncode2/moments/Squeeze:output:0*
T0*
_output_shapes
:}и
(BatchNormGeneEncode2/AssignMovingAvg/mulMul,BatchNormGeneEncode2/AssignMovingAvg/sub:z:03BatchNormGeneEncode2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:}ђ
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
О#<░
5BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOpReadVariableOp>batchnormgeneencode2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:}*
dtype0к
*BatchNormGeneEncode2/AssignMovingAvg_1/subSub=BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOp:value:0/BatchNormGeneEncode2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:}й
*BatchNormGeneEncode2/AssignMovingAvg_1/mulMul.BatchNormGeneEncode2/AssignMovingAvg_1/sub:z:05BatchNormGeneEncode2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:}ѕ
&BatchNormGeneEncode2/AssignMovingAvg_1AssignSubVariableOp>batchnormgeneencode2_assignmovingavg_1_readvariableop_resource.BatchNormGeneEncode2/AssignMovingAvg_1/mul:z:06^BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0i
$BatchNormGeneEncode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:░
"BatchNormGeneEncode2/batchnorm/addAddV2/BatchNormGeneEncode2/moments/Squeeze_1:output:0-BatchNormGeneEncode2/batchnorm/add/y:output:0*
T0*
_output_shapes
:}z
$BatchNormGeneEncode2/batchnorm/RsqrtRsqrt&BatchNormGeneEncode2/batchnorm/add:z:0*
T0*
_output_shapes
:}е
1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgeneencode2_batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0│
"BatchNormGeneEncode2/batchnorm/mulMul(BatchNormGeneEncode2/batchnorm/Rsqrt:y:09BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}А
$BatchNormGeneEncode2/batchnorm/mul_1Mulgene_encoder_2/Sigmoid:y:0&BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         }Д
$BatchNormGeneEncode2/batchnorm/mul_2Mul-BatchNormGeneEncode2/moments/Squeeze:output:0&BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*
_output_shapes
:}а
-BatchNormGeneEncode2/batchnorm/ReadVariableOpReadVariableOp6batchnormgeneencode2_batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0»
"BatchNormGeneEncode2/batchnorm/subSub5BatchNormGeneEncode2/batchnorm/ReadVariableOp:value:0(BatchNormGeneEncode2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:}▒
$BatchNormGeneEncode2/batchnorm/add_1AddV2(BatchNormGeneEncode2/batchnorm/mul_1:z:0&BatchNormGeneEncode2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         }ђ
6BatchNormProteinEncode1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: к
$BatchNormProteinEncode1/moments/meanMeanprotein_encoder_1/Sigmoid:y:0?BatchNormProteinEncode1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(ћ
,BatchNormProteinEncode1/moments/StopGradientStopGradient-BatchNormProteinEncode1/moments/mean:output:0*
T0*
_output_shapes

:╬
1BatchNormProteinEncode1/moments/SquaredDifferenceSquaredDifferenceprotein_encoder_1/Sigmoid:y:05BatchNormProteinEncode1/moments/StopGradient:output:0*
T0*'
_output_shapes
:         ё
:BatchNormProteinEncode1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Т
(BatchNormProteinEncode1/moments/varianceMean5BatchNormProteinEncode1/moments/SquaredDifference:z:0CBatchNormProteinEncode1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(Ю
'BatchNormProteinEncode1/moments/SqueezeSqueeze-BatchNormProteinEncode1/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Б
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
О#<▓
6BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOpReadVariableOp?batchnormproteinencode1_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0╔
+BatchNormProteinEncode1/AssignMovingAvg/subSub>BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOp:value:00BatchNormProteinEncode1/moments/Squeeze:output:0*
T0*
_output_shapes
:└
+BatchNormProteinEncode1/AssignMovingAvg/mulMul/BatchNormProteinEncode1/AssignMovingAvg/sub:z:06BatchNormProteinEncode1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:ї
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
О#<Х
8BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatchnormproteinencode1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0¤
-BatchNormProteinEncode1/AssignMovingAvg_1/subSub@BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp:value:02BatchNormProteinEncode1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:к
-BatchNormProteinEncode1/AssignMovingAvg_1/mulMul1BatchNormProteinEncode1/AssignMovingAvg_1/sub:z:08BatchNormProteinEncode1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:ћ
)BatchNormProteinEncode1/AssignMovingAvg_1AssignSubVariableOpAbatchnormproteinencode1_assignmovingavg_1_readvariableop_resource1BatchNormProteinEncode1/AssignMovingAvg_1/mul:z:09^BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'BatchNormProteinEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:╣
%BatchNormProteinEncode1/batchnorm/addAddV22BatchNormProteinEncode1/moments/Squeeze_1:output:00BatchNormProteinEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:ђ
'BatchNormProteinEncode1/batchnorm/RsqrtRsqrt)BatchNormProteinEncode1/batchnorm/add:z:0*
T0*
_output_shapes
:«
4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOpReadVariableOp=batchnormproteinencode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╝
%BatchNormProteinEncode1/batchnorm/mulMul+BatchNormProteinEncode1/batchnorm/Rsqrt:y:0<BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:ф
'BatchNormProteinEncode1/batchnorm/mul_1Mulprotein_encoder_1/Sigmoid:y:0)BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:         ░
'BatchNormProteinEncode1/batchnorm/mul_2Mul0BatchNormProteinEncode1/moments/Squeeze:output:0)BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*
_output_shapes
:д
0BatchNormProteinEncode1/batchnorm/ReadVariableOpReadVariableOp9batchnormproteinencode1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0И
%BatchNormProteinEncode1/batchnorm/subSub8BatchNormProteinEncode1/batchnorm/ReadVariableOp:value:0+BatchNormProteinEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:║
'BatchNormProteinEncode1/batchnorm/add_1AddV2+BatchNormProteinEncode1/batchnorm/mul_1:z:0)BatchNormProteinEncode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:         ^
ConcatenateLayer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :П
ConcatenateLayer/concatConcatV2(BatchNormGeneEncode2/batchnorm/add_1:z:0+BatchNormProteinEncode1/batchnorm/add_1:z:0%ConcatenateLayer/concat/axis:output:0*
N*
T0*(
_output_shapes
:         њЎ
'EmbeddingDimDense/MatMul/ReadVariableOpReadVariableOp0embeddingdimdense_matmul_readvariableop_resource*
_output_shapes
:	њ@*
dtype0Д
EmbeddingDimDense/MatMulMatMul ConcatenateLayer/concat:output:0/EmbeddingDimDense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ќ
(EmbeddingDimDense/BiasAdd/ReadVariableOpReadVariableOp1embeddingdimdense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0г
EmbeddingDimDense/BiasAddBiasAdd"EmbeddingDimDense/MatMul:product:00EmbeddingDimDense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @z
EmbeddingDimDense/SigmoidSigmoid"EmbeddingDimDense/BiasAdd:output:0*
T0*'
_output_shapes
:         @l
IdentityIdentityEmbeddingDimDense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         @є

NoOpNoOp%^BatchNormGeneEncode1/AssignMovingAvg4^BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOp'^BatchNormGeneEncode1/AssignMovingAvg_16^BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneEncode1/batchnorm/ReadVariableOp2^BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp%^BatchNormGeneEncode2/AssignMovingAvg4^BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOp'^BatchNormGeneEncode2/AssignMovingAvg_16^BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneEncode2/batchnorm/ReadVariableOp2^BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp(^BatchNormProteinEncode1/AssignMovingAvg7^BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOp*^BatchNormProteinEncode1/AssignMovingAvg_19^BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp1^BatchNormProteinEncode1/batchnorm/ReadVariableOp5^BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp)^EmbeddingDimDense/BiasAdd/ReadVariableOp(^EmbeddingDimDense/MatMul/ReadVariableOp&^gene_encoder_1/BiasAdd/ReadVariableOp%^gene_encoder_1/MatMul/ReadVariableOp&^gene_encoder_2/BiasAdd/ReadVariableOp%^gene_encoder_2/MatMul/ReadVariableOp)^protein_encoder_1/BiasAdd/ReadVariableOp(^protein_encoder_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         л:         W: : : : : : : : : : : : : : : : : : : : 2L
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
:         л
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         W
"
_user_specified_name
inputs/1
«%
ь
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_747336

inputs6
'assignmovingavg_readvariableop_resource:	З8
)assignmovingavg_1_readvariableop_resource:	З4
%batchnorm_mul_readvariableop_resource:	З0
!batchnorm_readvariableop_resource:	З
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ђ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	З*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	Зѕ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         Зl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ъ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	З*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:З*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:З*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<Ѓ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:З*
dtype0ѓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:Зy
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:Зг
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
О#<Є
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:З*
dtype0ѕ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:З
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:З┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ЗQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:З
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:З*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Зd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         Зi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Зw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:З*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Зs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         Зc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         ЗЖ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         З: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         З
 
_user_specified_nameinputs
Б

■
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_747557

inputs0
matmul_readvariableop_resource:W-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:W*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         W: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         W
 
_user_specified_nameinputs
К[
т
"__inference__traced_restore_748859
file_prefix:
&assignvariableop_gene_encoder_1_kernel:
лЗ5
&assignvariableop_1_gene_encoder_1_bias:	З<
-assignvariableop_2_batchnormgeneencode1_gamma:	З;
,assignvariableop_3_batchnormgeneencode1_beta:	ЗB
3assignvariableop_4_batchnormgeneencode1_moving_mean:	ЗF
7assignvariableop_5_batchnormgeneencode1_moving_variance:	З;
(assignvariableop_6_gene_encoder_2_kernel:	З}4
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
,assignvariableop_18_embeddingdimdense_kernel:	њ@8
*assignvariableop_19_embeddingdimdense_bias:@#
assignvariableop_20_total: #
assignvariableop_21_count: 
identity_23ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9Ы

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ў

valueј
BІ
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHъ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B Љ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOpAssignVariableOp&assignvariableop_gene_encoder_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_1AssignVariableOp&assignvariableop_1_gene_encoder_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:ю
AssignVariableOp_2AssignVariableOp-assignvariableop_2_batchnormgeneencode1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_3AssignVariableOp,assignvariableop_3_batchnormgeneencode1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_4AssignVariableOp3assignvariableop_4_batchnormgeneencode1_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:д
AssignVariableOp_5AssignVariableOp7assignvariableop_5_batchnormgeneencode1_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_6AssignVariableOp(assignvariableop_6_gene_encoder_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ћ
AssignVariableOp_7AssignVariableOp&assignvariableop_7_gene_encoder_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:џ
AssignVariableOp_8AssignVariableOp+assignvariableop_8_protein_encoder_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:ў
AssignVariableOp_9AssignVariableOp)assignvariableop_9_protein_encoder_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_10AssignVariableOp.assignvariableop_10_batchnormgeneencode2_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:ъ
AssignVariableOp_11AssignVariableOp-assignvariableop_11_batchnormgeneencode2_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_12AssignVariableOp4assignvariableop_12_batchnormgeneencode2_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_13AssignVariableOp8assignvariableop_13_batchnormgeneencode2_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_14AssignVariableOp1assignvariableop_14_batchnormproteinencode1_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOp_15AssignVariableOp0assignvariableop_15_batchnormproteinencode1_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_16AssignVariableOp7assignvariableop_16_batchnormproteinencode1_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:г
AssignVariableOp_17AssignVariableOp;assignvariableop_17_batchnormproteinencode1_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_18AssignVariableOp,assignvariableop_18_embeddingdimdense_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_19AssignVariableOp*assignvariableop_19_embeddingdimdense_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_21AssignVariableOpassignvariableop_21_countIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 │
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_23IdentityIdentity_22:output:0^NoOp_1*
T0*
_output_shapes
: а
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
Д
л
5__inference_BatchNormGeneEncode2_layer_call_fn_748526

inputs
unknown:}
	unknown_0:}
	unknown_1:}
	unknown_2:}
identityѕбStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         }*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_747418o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         }`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         }: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         }
 
_user_specified_nameinputs
Ш
Ў
)__inference_model_23_layer_call_fn_747906
gene_input_layer
protein_input_layer
unknown:
лЗ
	unknown_0:	З
	unknown_1:	З
	unknown_2:	З
	unknown_3:	З
	unknown_4:	З
	unknown_5:W
	unknown_6:
	unknown_7:	З}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:	њ@

unknown_18:@
identityѕбStatefulPartitionedCallж
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
:         @*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_model_23_layer_call_and_return_conditional_losses_747817o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         л:         W: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:         л
*
_user_specified_nameGene_Input_Layer:\X
'
_output_shapes
:         W
-
_user_specified_nameProtein_Input_Layer
ы2
ѕ

D__inference_model_23_layer_call_and_return_conditional_losses_747625

inputs
inputs_1)
gene_encoder_1_747532:
лЗ$
gene_encoder_1_747534:	З*
batchnormgeneencode1_747537:	З*
batchnormgeneencode1_747539:	З*
batchnormgeneencode1_747541:	З*
batchnormgeneencode1_747543:	З*
protein_encoder_1_747558:W&
protein_encoder_1_747560:(
gene_encoder_2_747575:	З}#
gene_encoder_2_747577:})
batchnormgeneencode2_747580:})
batchnormgeneencode2_747582:})
batchnormgeneencode2_747584:})
batchnormgeneencode2_747586:},
batchnormproteinencode1_747589:,
batchnormproteinencode1_747591:,
batchnormproteinencode1_747593:,
batchnormproteinencode1_747595:+
embeddingdimdense_747619:	њ@&
embeddingdimdense_747621:@
identityѕб,BatchNormGeneEncode1/StatefulPartitionedCallб,BatchNormGeneEncode2/StatefulPartitionedCallб/BatchNormProteinEncode1/StatefulPartitionedCallб)EmbeddingDimDense/StatefulPartitionedCallб&gene_encoder_1/StatefulPartitionedCallб&gene_encoder_2/StatefulPartitionedCallб)protein_encoder_1/StatefulPartitionedCallї
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputsgene_encoder_1_747532gene_encoder_1_747534*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         З*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_747531І
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_747537batchnormgeneencode1_747539batchnormgeneencode1_747541batchnormgeneencode1_747543*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         З*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_747289Ў
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1protein_encoder_1_747558protein_encoder_1_747560*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_747557║
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_747575gene_encoder_2_747577*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         }*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_747574і
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_747580batchnormgeneencode2_747582batchnormgeneencode2_747584batchnormgeneencode2_747586*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         }*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_747371Ъ
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_747589batchnormproteinencode1_747591batchnormproteinencode1_747593batchnormproteinencode1_747595*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_747453Х
 ConcatenateLayer/PartitionedCallPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:08BatchNormProteinEncode1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         њ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_747605║
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_747619embeddingdimdense_747621*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_747618Ђ
IdentityIdentity2EmbeddingDimDense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @ђ
NoOpNoOp-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         л:         W: : : : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2b
/BatchNormProteinEncode1/StatefulPartitionedCall/BatchNormProteinEncode1/StatefulPartitionedCall2V
)EmbeddingDimDense/StatefulPartitionedCall)EmbeddingDimDense/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall2V
)protein_encoder_1/StatefulPartitionedCall)protein_encoder_1/StatefulPartitionedCall:P L
(
_output_shapes
:         л
 
_user_specified_nameinputs:OK
'
_output_shapes
:         W
 
_user_specified_nameinputs
ц

Ч
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_747574

inputs1
matmul_readvariableop_resource:	З}-
biasadd_readvariableop_resource:}
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	З}*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         }r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:}*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         }V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         }Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         }w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         З: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         З
 
_user_specified_nameinputs
«%
ь
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_748460

inputs6
'assignmovingavg_readvariableop_resource:	З8
)assignmovingavg_1_readvariableop_resource:	З4
%batchnorm_mul_readvariableop_resource:	З0
!batchnorm_readvariableop_resource:	З
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ђ
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	З*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	Зѕ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         Зl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ъ
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	З*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:З*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:З*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
О#<Ѓ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:З*
dtype0ѓ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:Зy
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:Зг
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
О#<Є
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:З*
dtype0ѕ
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:З
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:З┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ЗQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:З
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:З*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Зd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         Зi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Зw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:З*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Зs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         Зc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         ЗЖ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         З: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         З
 
_user_specified_nameinputs
├
є
)__inference_model_23_layer_call_fn_748058
inputs_0
inputs_1
unknown:
лЗ
	unknown_0:	З
	unknown_1:	З
	unknown_2:	З
	unknown_3:	З
	unknown_4:	З
	unknown_5:W
	unknown_6:
	unknown_7:	З}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:	њ@

unknown_18:@
identityѕбStatefulPartitionedCall▄
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
:         @*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_model_23_layer_call_and_return_conditional_losses_747625o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         л:         W: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:         л
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         W
"
_user_specified_name
inputs/1
в2
ѕ

D__inference_model_23_layer_call_and_return_conditional_losses_747817

inputs
inputs_1)
gene_encoder_1_747768:
лЗ$
gene_encoder_1_747770:	З*
batchnormgeneencode1_747773:	З*
batchnormgeneencode1_747775:	З*
batchnormgeneencode1_747777:	З*
batchnormgeneencode1_747779:	З*
protein_encoder_1_747782:W&
protein_encoder_1_747784:(
gene_encoder_2_747787:	З}#
gene_encoder_2_747789:})
batchnormgeneencode2_747792:})
batchnormgeneencode2_747794:})
batchnormgeneencode2_747796:})
batchnormgeneencode2_747798:},
batchnormproteinencode1_747801:,
batchnormproteinencode1_747803:,
batchnormproteinencode1_747805:,
batchnormproteinencode1_747807:+
embeddingdimdense_747811:	њ@&
embeddingdimdense_747813:@
identityѕб,BatchNormGeneEncode1/StatefulPartitionedCallб,BatchNormGeneEncode2/StatefulPartitionedCallб/BatchNormProteinEncode1/StatefulPartitionedCallб)EmbeddingDimDense/StatefulPartitionedCallб&gene_encoder_1/StatefulPartitionedCallб&gene_encoder_2/StatefulPartitionedCallб)protein_encoder_1/StatefulPartitionedCallї
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputsgene_encoder_1_747768gene_encoder_1_747770*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         З*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_747531Ѕ
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_747773batchnormgeneencode1_747775batchnormgeneencode1_747777batchnormgeneencode1_747779*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         З*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_747336Ў
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1protein_encoder_1_747782protein_encoder_1_747784*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_747557║
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_747787gene_encoder_2_747789*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         }*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_747574ѕ
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_747792batchnormgeneencode2_747794batchnormgeneencode2_747796batchnormgeneencode2_747798*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         }*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_747418Ю
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_747801batchnormproteinencode1_747803batchnormproteinencode1_747805batchnormproteinencode1_747807*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_747500Х
 ConcatenateLayer/PartitionedCallPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:08BatchNormProteinEncode1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         њ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_747605║
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_747811embeddingdimdense_747813*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_747618Ђ
IdentityIdentity2EmbeddingDimDense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @ђ
NoOpNoOp-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         л:         W: : : : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2b
/BatchNormProteinEncode1/StatefulPartitionedCall/BatchNormProteinEncode1/StatefulPartitionedCall2V
)EmbeddingDimDense/StatefulPartitionedCall)EmbeddingDimDense/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall2V
)protein_encoder_1/StatefulPartitionedCall)protein_encoder_1/StatefulPartitionedCall:P L
(
_output_shapes
:         л
 
_user_specified_nameinputs:OK
'
_output_shapes
:         W
 
_user_specified_nameinputs
ћ%
В
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_748660

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOph
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

:Є
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ъ
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
О#<ѓ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Ђ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:г
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
О#<є
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Є
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:q
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
:         h
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
:         b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         Ж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ћ%
В
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_747500

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOph
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

:Є
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ъ
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
О#<ѓ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Ђ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:г
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
О#<є
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0Є
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:q
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
:         h
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
:         b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         Ж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
н
Ю
/__inference_gene_encoder_2_layer_call_fn_748469

inputs
unknown:	З}
	unknown_0:}
identityѕбStatefulPartitionedCallР
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         }*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_747574o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         }`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         З: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         З
 
_user_specified_nameinputs
═
»
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_747371

inputs/
!batchnorm_readvariableop_resource:}3
%batchnorm_mul_readvariableop_resource:}1
#batchnorm_readvariableop_1_resource:}1
#batchnorm_readvariableop_2_resource:}
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:w
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
:         }z
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
:         }b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         }║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         }: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         }
 
_user_specified_nameinputs
Е
л
5__inference_BatchNormGeneEncode2_layer_call_fn_748513

inputs
unknown:}
	unknown_0:}
	unknown_1:}
	unknown_2:}
identityѕбStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         }*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_747371o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         }`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         }: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         }
 
_user_specified_nameinputs
ы5
ж

__inference__traced_save_748783
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

identity_1ѕбMergeV2Checkpointsw
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
_temp/partЂ
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
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: №

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*ў

valueј
BІ
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЏ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B в

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_gene_encoder_1_kernel_read_readvariableop.savev2_gene_encoder_1_bias_read_readvariableop5savev2_batchnormgeneencode1_gamma_read_readvariableop4savev2_batchnormgeneencode1_beta_read_readvariableop;savev2_batchnormgeneencode1_moving_mean_read_readvariableop?savev2_batchnormgeneencode1_moving_variance_read_readvariableop0savev2_gene_encoder_2_kernel_read_readvariableop.savev2_gene_encoder_2_bias_read_readvariableop3savev2_protein_encoder_1_kernel_read_readvariableop1savev2_protein_encoder_1_bias_read_readvariableop5savev2_batchnormgeneencode2_gamma_read_readvariableop4savev2_batchnormgeneencode2_beta_read_readvariableop;savev2_batchnormgeneencode2_moving_mean_read_readvariableop?savev2_batchnormgeneencode2_moving_variance_read_readvariableop8savev2_batchnormproteinencode1_gamma_read_readvariableop7savev2_batchnormproteinencode1_beta_read_readvariableop>savev2_batchnormproteinencode1_moving_mean_read_readvariableopBsavev2_batchnormproteinencode1_moving_variance_read_readvariableop3savev2_embeddingdimdense_kernel_read_readvariableop1savev2_embeddingdimdense_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:І
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

identity_1Identity_1:output:0*«
_input_shapesю
Ў: :
лЗ:З:З:З:З:З:	З}:}:W::}:}:}:}:::::	њ@:@: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
лЗ:!

_output_shapes	
:З:!

_output_shapes	
:З:!

_output_shapes	
:З:!

_output_shapes	
:З:!

_output_shapes	
:З:%!

_output_shapes
:	З}: 
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
:	њ@: 
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
гЄ
З
!__inference__wrapped_model_747265
gene_input_layer
protein_input_layerJ
6model_23_gene_encoder_1_matmul_readvariableop_resource:
лЗF
7model_23_gene_encoder_1_biasadd_readvariableop_resource:	ЗN
?model_23_batchnormgeneencode1_batchnorm_readvariableop_resource:	ЗR
Cmodel_23_batchnormgeneencode1_batchnorm_mul_readvariableop_resource:	ЗP
Amodel_23_batchnormgeneencode1_batchnorm_readvariableop_1_resource:	ЗP
Amodel_23_batchnormgeneencode1_batchnorm_readvariableop_2_resource:	ЗK
9model_23_protein_encoder_1_matmul_readvariableop_resource:WH
:model_23_protein_encoder_1_biasadd_readvariableop_resource:I
6model_23_gene_encoder_2_matmul_readvariableop_resource:	З}E
7model_23_gene_encoder_2_biasadd_readvariableop_resource:}M
?model_23_batchnormgeneencode2_batchnorm_readvariableop_resource:}Q
Cmodel_23_batchnormgeneencode2_batchnorm_mul_readvariableop_resource:}O
Amodel_23_batchnormgeneencode2_batchnorm_readvariableop_1_resource:}O
Amodel_23_batchnormgeneencode2_batchnorm_readvariableop_2_resource:}P
Bmodel_23_batchnormproteinencode1_batchnorm_readvariableop_resource:T
Fmodel_23_batchnormproteinencode1_batchnorm_mul_readvariableop_resource:R
Dmodel_23_batchnormproteinencode1_batchnorm_readvariableop_1_resource:R
Dmodel_23_batchnormproteinencode1_batchnorm_readvariableop_2_resource:L
9model_23_embeddingdimdense_matmul_readvariableop_resource:	њ@H
:model_23_embeddingdimdense_biasadd_readvariableop_resource:@
identityѕб6model_23/BatchNormGeneEncode1/batchnorm/ReadVariableOpб8model_23/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1б8model_23/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2б:model_23/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOpб6model_23/BatchNormGeneEncode2/batchnorm/ReadVariableOpб8model_23/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1б8model_23/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2б:model_23/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOpб9model_23/BatchNormProteinEncode1/batchnorm/ReadVariableOpб;model_23/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1б;model_23/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2б=model_23/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOpб1model_23/EmbeddingDimDense/BiasAdd/ReadVariableOpб0model_23/EmbeddingDimDense/MatMul/ReadVariableOpб.model_23/gene_encoder_1/BiasAdd/ReadVariableOpб-model_23/gene_encoder_1/MatMul/ReadVariableOpб.model_23/gene_encoder_2/BiasAdd/ReadVariableOpб-model_23/gene_encoder_2/MatMul/ReadVariableOpб1model_23/protein_encoder_1/BiasAdd/ReadVariableOpб0model_23/protein_encoder_1/MatMul/ReadVariableOpд
-model_23/gene_encoder_1/MatMul/ReadVariableOpReadVariableOp6model_23_gene_encoder_1_matmul_readvariableop_resource* 
_output_shapes
:
лЗ*
dtype0ц
model_23/gene_encoder_1/MatMulMatMulgene_input_layer5model_23/gene_encoder_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ЗБ
.model_23/gene_encoder_1/BiasAdd/ReadVariableOpReadVariableOp7model_23_gene_encoder_1_biasadd_readvariableop_resource*
_output_shapes	
:З*
dtype0┐
model_23/gene_encoder_1/BiasAddBiasAdd(model_23/gene_encoder_1/MatMul:product:06model_23/gene_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ЗЄ
model_23/gene_encoder_1/SigmoidSigmoid(model_23/gene_encoder_1/BiasAdd:output:0*
T0*(
_output_shapes
:         З│
6model_23/BatchNormGeneEncode1/batchnorm/ReadVariableOpReadVariableOp?model_23_batchnormgeneencode1_batchnorm_readvariableop_resource*
_output_shapes	
:З*
dtype0r
-model_23/BatchNormGeneEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:м
+model_23/BatchNormGeneEncode1/batchnorm/addAddV2>model_23/BatchNormGeneEncode1/batchnorm/ReadVariableOp:value:06model_23/BatchNormGeneEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ЗЇ
-model_23/BatchNormGeneEncode1/batchnorm/RsqrtRsqrt/model_23/BatchNormGeneEncode1/batchnorm/add:z:0*
T0*
_output_shapes	
:З╗
:model_23/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_23_batchnormgeneencode1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:З*
dtype0¤
+model_23/BatchNormGeneEncode1/batchnorm/mulMul1model_23/BatchNormGeneEncode1/batchnorm/Rsqrt:y:0Bmodel_23/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Зй
-model_23/BatchNormGeneEncode1/batchnorm/mul_1Mul#model_23/gene_encoder_1/Sigmoid:y:0/model_23/BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*(
_output_shapes
:         Зи
8model_23/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_23_batchnormgeneencode1_batchnorm_readvariableop_1_resource*
_output_shapes	
:З*
dtype0═
-model_23/BatchNormGeneEncode1/batchnorm/mul_2Mul@model_23/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1:value:0/model_23/BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*
_output_shapes	
:Зи
8model_23/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_23_batchnormgeneencode1_batchnorm_readvariableop_2_resource*
_output_shapes	
:З*
dtype0═
+model_23/BatchNormGeneEncode1/batchnorm/subSub@model_23/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2:value:01model_23/BatchNormGeneEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:З═
-model_23/BatchNormGeneEncode1/batchnorm/add_1AddV21model_23/BatchNormGeneEncode1/batchnorm/mul_1:z:0/model_23/BatchNormGeneEncode1/batchnorm/sub:z:0*
T0*(
_output_shapes
:         Зф
0model_23/protein_encoder_1/MatMul/ReadVariableOpReadVariableOp9model_23_protein_encoder_1_matmul_readvariableop_resource*
_output_shapes

:W*
dtype0г
!model_23/protein_encoder_1/MatMulMatMulprotein_input_layer8model_23/protein_encoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         е
1model_23/protein_encoder_1/BiasAdd/ReadVariableOpReadVariableOp:model_23_protein_encoder_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0К
"model_23/protein_encoder_1/BiasAddBiasAdd+model_23/protein_encoder_1/MatMul:product:09model_23/protein_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ї
"model_23/protein_encoder_1/SigmoidSigmoid+model_23/protein_encoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:         Ц
-model_23/gene_encoder_2/MatMul/ReadVariableOpReadVariableOp6model_23_gene_encoder_2_matmul_readvariableop_resource*
_output_shapes
:	З}*
dtype0─
model_23/gene_encoder_2/MatMulMatMul1model_23/BatchNormGeneEncode1/batchnorm/add_1:z:05model_23/gene_encoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         }б
.model_23/gene_encoder_2/BiasAdd/ReadVariableOpReadVariableOp7model_23_gene_encoder_2_biasadd_readvariableop_resource*
_output_shapes
:}*
dtype0Й
model_23/gene_encoder_2/BiasAddBiasAdd(model_23/gene_encoder_2/MatMul:product:06model_23/gene_encoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         }є
model_23/gene_encoder_2/SigmoidSigmoid(model_23/gene_encoder_2/BiasAdd:output:0*
T0*'
_output_shapes
:         }▓
6model_23/BatchNormGeneEncode2/batchnorm/ReadVariableOpReadVariableOp?model_23_batchnormgeneencode2_batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0r
-model_23/BatchNormGeneEncode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:Л
+model_23/BatchNormGeneEncode2/batchnorm/addAddV2>model_23/BatchNormGeneEncode2/batchnorm/ReadVariableOp:value:06model_23/BatchNormGeneEncode2/batchnorm/add/y:output:0*
T0*
_output_shapes
:}ї
-model_23/BatchNormGeneEncode2/batchnorm/RsqrtRsqrt/model_23/BatchNormGeneEncode2/batchnorm/add:z:0*
T0*
_output_shapes
:}║
:model_23/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOpReadVariableOpCmodel_23_batchnormgeneencode2_batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0╬
+model_23/BatchNormGeneEncode2/batchnorm/mulMul1model_23/BatchNormGeneEncode2/batchnorm/Rsqrt:y:0Bmodel_23/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}╝
-model_23/BatchNormGeneEncode2/batchnorm/mul_1Mul#model_23/gene_encoder_2/Sigmoid:y:0/model_23/BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         }Х
8model_23/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1ReadVariableOpAmodel_23_batchnormgeneencode2_batchnorm_readvariableop_1_resource*
_output_shapes
:}*
dtype0╠
-model_23/BatchNormGeneEncode2/batchnorm/mul_2Mul@model_23/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1:value:0/model_23/BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*
_output_shapes
:}Х
8model_23/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2ReadVariableOpAmodel_23_batchnormgeneencode2_batchnorm_readvariableop_2_resource*
_output_shapes
:}*
dtype0╠
+model_23/BatchNormGeneEncode2/batchnorm/subSub@model_23/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2:value:01model_23/BatchNormGeneEncode2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:}╠
-model_23/BatchNormGeneEncode2/batchnorm/add_1AddV21model_23/BatchNormGeneEncode2/batchnorm/mul_1:z:0/model_23/BatchNormGeneEncode2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         }И
9model_23/BatchNormProteinEncode1/batchnorm/ReadVariableOpReadVariableOpBmodel_23_batchnormproteinencode1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0u
0model_23/BatchNormProteinEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:┌
.model_23/BatchNormProteinEncode1/batchnorm/addAddV2Amodel_23/BatchNormProteinEncode1/batchnorm/ReadVariableOp:value:09model_23/BatchNormProteinEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:њ
0model_23/BatchNormProteinEncode1/batchnorm/RsqrtRsqrt2model_23/BatchNormProteinEncode1/batchnorm/add:z:0*
T0*
_output_shapes
:└
=model_23/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOpReadVariableOpFmodel_23_batchnormproteinencode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0О
.model_23/BatchNormProteinEncode1/batchnorm/mulMul4model_23/BatchNormProteinEncode1/batchnorm/Rsqrt:y:0Emodel_23/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:┼
0model_23/BatchNormProteinEncode1/batchnorm/mul_1Mul&model_23/protein_encoder_1/Sigmoid:y:02model_23/BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:         ╝
;model_23/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1ReadVariableOpDmodel_23_batchnormproteinencode1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0Н
0model_23/BatchNormProteinEncode1/batchnorm/mul_2MulCmodel_23/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1:value:02model_23/BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*
_output_shapes
:╝
;model_23/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2ReadVariableOpDmodel_23_batchnormproteinencode1_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0Н
.model_23/BatchNormProteinEncode1/batchnorm/subSubCmodel_23/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2:value:04model_23/BatchNormProteinEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:Н
0model_23/BatchNormProteinEncode1/batchnorm/add_1AddV24model_23/BatchNormProteinEncode1/batchnorm/mul_1:z:02model_23/BatchNormProteinEncode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:         g
%model_23/ConcatenateLayer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ђ
 model_23/ConcatenateLayer/concatConcatV21model_23/BatchNormGeneEncode2/batchnorm/add_1:z:04model_23/BatchNormProteinEncode1/batchnorm/add_1:z:0.model_23/ConcatenateLayer/concat/axis:output:0*
N*
T0*(
_output_shapes
:         њФ
0model_23/EmbeddingDimDense/MatMul/ReadVariableOpReadVariableOp9model_23_embeddingdimdense_matmul_readvariableop_resource*
_output_shapes
:	њ@*
dtype0┬
!model_23/EmbeddingDimDense/MatMulMatMul)model_23/ConcatenateLayer/concat:output:08model_23/EmbeddingDimDense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @е
1model_23/EmbeddingDimDense/BiasAdd/ReadVariableOpReadVariableOp:model_23_embeddingdimdense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0К
"model_23/EmbeddingDimDense/BiasAddBiasAdd+model_23/EmbeddingDimDense/MatMul:product:09model_23/EmbeddingDimDense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ї
"model_23/EmbeddingDimDense/SigmoidSigmoid+model_23/EmbeddingDimDense/BiasAdd:output:0*
T0*'
_output_shapes
:         @u
IdentityIdentity&model_23/EmbeddingDimDense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         @д	
NoOpNoOp7^model_23/BatchNormGeneEncode1/batchnorm/ReadVariableOp9^model_23/BatchNormGeneEncode1/batchnorm/ReadVariableOp_19^model_23/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2;^model_23/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp7^model_23/BatchNormGeneEncode2/batchnorm/ReadVariableOp9^model_23/BatchNormGeneEncode2/batchnorm/ReadVariableOp_19^model_23/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2;^model_23/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:^model_23/BatchNormProteinEncode1/batchnorm/ReadVariableOp<^model_23/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1<^model_23/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2>^model_23/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp2^model_23/EmbeddingDimDense/BiasAdd/ReadVariableOp1^model_23/EmbeddingDimDense/MatMul/ReadVariableOp/^model_23/gene_encoder_1/BiasAdd/ReadVariableOp.^model_23/gene_encoder_1/MatMul/ReadVariableOp/^model_23/gene_encoder_2/BiasAdd/ReadVariableOp.^model_23/gene_encoder_2/MatMul/ReadVariableOp2^model_23/protein_encoder_1/BiasAdd/ReadVariableOp1^model_23/protein_encoder_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         л:         W: : : : : : : : : : : : : : : : : : : : 2p
6model_23/BatchNormGeneEncode1/batchnorm/ReadVariableOp6model_23/BatchNormGeneEncode1/batchnorm/ReadVariableOp2t
8model_23/BatchNormGeneEncode1/batchnorm/ReadVariableOp_18model_23/BatchNormGeneEncode1/batchnorm/ReadVariableOp_12t
8model_23/BatchNormGeneEncode1/batchnorm/ReadVariableOp_28model_23/BatchNormGeneEncode1/batchnorm/ReadVariableOp_22x
:model_23/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp:model_23/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp2p
6model_23/BatchNormGeneEncode2/batchnorm/ReadVariableOp6model_23/BatchNormGeneEncode2/batchnorm/ReadVariableOp2t
8model_23/BatchNormGeneEncode2/batchnorm/ReadVariableOp_18model_23/BatchNormGeneEncode2/batchnorm/ReadVariableOp_12t
8model_23/BatchNormGeneEncode2/batchnorm/ReadVariableOp_28model_23/BatchNormGeneEncode2/batchnorm/ReadVariableOp_22x
:model_23/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:model_23/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp2v
9model_23/BatchNormProteinEncode1/batchnorm/ReadVariableOp9model_23/BatchNormProteinEncode1/batchnorm/ReadVariableOp2z
;model_23/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1;model_23/BatchNormProteinEncode1/batchnorm/ReadVariableOp_12z
;model_23/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2;model_23/BatchNormProteinEncode1/batchnorm/ReadVariableOp_22~
=model_23/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp=model_23/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp2f
1model_23/EmbeddingDimDense/BiasAdd/ReadVariableOp1model_23/EmbeddingDimDense/BiasAdd/ReadVariableOp2d
0model_23/EmbeddingDimDense/MatMul/ReadVariableOp0model_23/EmbeddingDimDense/MatMul/ReadVariableOp2`
.model_23/gene_encoder_1/BiasAdd/ReadVariableOp.model_23/gene_encoder_1/BiasAdd/ReadVariableOp2^
-model_23/gene_encoder_1/MatMul/ReadVariableOp-model_23/gene_encoder_1/MatMul/ReadVariableOp2`
.model_23/gene_encoder_2/BiasAdd/ReadVariableOp.model_23/gene_encoder_2/BiasAdd/ReadVariableOp2^
-model_23/gene_encoder_2/MatMul/ReadVariableOp-model_23/gene_encoder_2/MatMul/ReadVariableOp2f
1model_23/protein_encoder_1/BiasAdd/ReadVariableOp1model_23/protein_encoder_1/BiasAdd/ReadVariableOp2d
0model_23/protein_encoder_1/MatMul/ReadVariableOp0model_23/protein_encoder_1/MatMul/ReadVariableOp:Z V
(
_output_shapes
:         л
*
_user_specified_nameGene_Input_Layer:\X
'
_output_shapes
:         W
-
_user_specified_nameProtein_Input_Layer
г3
Ю

D__inference_model_23_layer_call_and_return_conditional_losses_748012
gene_input_layer
protein_input_layer)
gene_encoder_1_747963:
лЗ$
gene_encoder_1_747965:	З*
batchnormgeneencode1_747968:	З*
batchnormgeneencode1_747970:	З*
batchnormgeneencode1_747972:	З*
batchnormgeneencode1_747974:	З*
protein_encoder_1_747977:W&
protein_encoder_1_747979:(
gene_encoder_2_747982:	З}#
gene_encoder_2_747984:})
batchnormgeneencode2_747987:})
batchnormgeneencode2_747989:})
batchnormgeneencode2_747991:})
batchnormgeneencode2_747993:},
batchnormproteinencode1_747996:,
batchnormproteinencode1_747998:,
batchnormproteinencode1_748000:,
batchnormproteinencode1_748002:+
embeddingdimdense_748006:	њ@&
embeddingdimdense_748008:@
identityѕб,BatchNormGeneEncode1/StatefulPartitionedCallб,BatchNormGeneEncode2/StatefulPartitionedCallб/BatchNormProteinEncode1/StatefulPartitionedCallб)EmbeddingDimDense/StatefulPartitionedCallб&gene_encoder_1/StatefulPartitionedCallб&gene_encoder_2/StatefulPartitionedCallб)protein_encoder_1/StatefulPartitionedCallќ
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallgene_input_layergene_encoder_1_747963gene_encoder_1_747965*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         З*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_747531Ѕ
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_747968batchnormgeneencode1_747970batchnormgeneencode1_747972batchnormgeneencode1_747974*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         З*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_747336ц
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallprotein_input_layerprotein_encoder_1_747977protein_encoder_1_747979*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_747557║
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_747982gene_encoder_2_747984*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         }*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_747574ѕ
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_747987batchnormgeneencode2_747989batchnormgeneencode2_747991batchnormgeneencode2_747993*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         }*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_747418Ю
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_747996batchnormproteinencode1_747998batchnormproteinencode1_748000batchnormproteinencode1_748002*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_747500Х
 ConcatenateLayer/PartitionedCallPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:08BatchNormProteinEncode1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         њ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_747605║
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_748006embeddingdimdense_748008*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_747618Ђ
IdentityIdentity2EmbeddingDimDense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @ђ
NoOpNoOp-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         л:         W: : : : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2b
/BatchNormProteinEncode1/StatefulPartitionedCall/BatchNormProteinEncode1/StatefulPartitionedCall2V
)EmbeddingDimDense/StatefulPartitionedCall)EmbeddingDimDense/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall2V
)protein_encoder_1/StatefulPartitionedCall)protein_encoder_1/StatefulPartitionedCall:Z V
(
_output_shapes
:         л
*
_user_specified_nameGene_Input_Layer:\X
'
_output_shapes
:         W
-
_user_specified_nameProtein_Input_Layer
Д

 
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_747618

inputs1
matmul_readvariableop_resource:	њ@-
biasadd_readvariableop_resource:@
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	њ@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         @Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         @w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         њ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         њ
 
_user_specified_nameinputs
Г
М
8__inference_BatchNormProteinEncode1_layer_call_fn_748606

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallЃ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_747500o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
К
x
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_748673
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
:         њX
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:         њ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         }:         :Q M
'
_output_shapes
:         }
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
п
Ъ
/__inference_gene_encoder_1_layer_call_fn_748369

inputs
unknown:
лЗ
	unknown_0:	З
identityѕбStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         З*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_747531p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         З`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         л: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         л
 
_user_specified_nameinputs
и
]
1__inference_ConcatenateLayer_layer_call_fn_748666
inputs_0
inputs_1
identity╚
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         њ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_747605a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         њ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         }:         :Q M
'
_output_shapes
:         }
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
н
ћ
$__inference_signature_wrapper_748360
gene_input_layer
protein_input_layer
unknown:
лЗ
	unknown_0:	З
	unknown_1:	З
	unknown_2:	З
	unknown_3:	З
	unknown_4:	З
	unknown_5:W
	unknown_6:
	unknown_7:	З}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:	њ@

unknown_18:@
identityѕбStatefulPartitionedCall╠
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
:         @*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ **
f%R#
!__inference__wrapped_model_747265o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         л:         W: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:         л
*
_user_specified_nameGene_Input_Layer:\X
'
_output_shapes
:         W
-
_user_specified_nameProtein_Input_Layer
»
М
8__inference_BatchNormProteinEncode1_layer_call_fn_748593

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityѕбStatefulPartitionedCallЁ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_747453o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
й
є
)__inference_model_23_layer_call_fn_748104
inputs_0
inputs_1
unknown:
лЗ
	unknown_0:	З
	unknown_1:	З
	unknown_2:	З
	unknown_3:	З
	unknown_4:	З
	unknown_5:W
	unknown_6:
	unknown_7:	З}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:	њ@

unknown_18:@
identityѕбStatefulPartitionedCallо
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
:         @*0
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8ѓ *M
fHRF
D__inference_model_23_layer_call_and_return_conditional_losses_747817o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         л:         W: : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:         л
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         W
"
_user_specified_name
inputs/1
▀
│
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_748426

inputs0
!batchnorm_readvariableop_resource:	З4
%batchnorm_mul_readvariableop_resource:	З2
#batchnorm_readvariableop_1_resource:	З2
#batchnorm_readvariableop_2_resource:	З
identityѕбbatchnorm/ReadVariableOpбbatchnorm/ReadVariableOp_1бbatchnorm/ReadVariableOp_2бbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:З*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ЗQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:З
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:З*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Зd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         З{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:З*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:З{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:З*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Зs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         Зc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         З║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         З: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         З
 
_user_specified_nameinputs
Љ%
ж
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_747418

inputs5
'assignmovingavg_readvariableop_resource:}7
)assignmovingavg_1_readvariableop_resource:}3
%batchnorm_mul_readvariableop_resource:}/
!batchnorm_readvariableop_resource:}
identityѕбAssignMovingAvgбAssignMovingAvg/ReadVariableOpбAssignMovingAvg_1б AssignMovingAvg_1/ReadVariableOpбbatchnorm/ReadVariableOpбbatchnorm/mul/ReadVariableOph
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

:}Є
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         }l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ъ
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
О#<ѓ
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:}*
dtype0Ђ
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:}x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:}г
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
О#<є
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:}*
dtype0Є
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:}~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:}┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oЃ:q
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
:         }h
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
:         }b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         }Ж
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         }: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         }
 
_user_specified_nameinputs
ц

Ч
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_748480

inputs1
matmul_readvariableop_resource:	З}-
biasadd_readvariableop_resource:}
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	З}*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         }r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:}*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         }V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         }Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         }w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         З: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         З
 
_user_specified_nameinputs
г

■
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_748380

inputs2
matmul_readvariableop_resource:
лЗ.
biasadd_readvariableop_resource:	З
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
лЗ*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Зs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:З*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ЗW
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:         З[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:         Зw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         л: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         л
 
_user_specified_nameinputs
┐
v
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_747605

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
:         њX
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:         њ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         }:         :O K
'
_output_shapes
:         }
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs
▓3
Ю

D__inference_model_23_layer_call_and_return_conditional_losses_747959
gene_input_layer
protein_input_layer)
gene_encoder_1_747910:
лЗ$
gene_encoder_1_747912:	З*
batchnormgeneencode1_747915:	З*
batchnormgeneencode1_747917:	З*
batchnormgeneencode1_747919:	З*
batchnormgeneencode1_747921:	З*
protein_encoder_1_747924:W&
protein_encoder_1_747926:(
gene_encoder_2_747929:	З}#
gene_encoder_2_747931:})
batchnormgeneencode2_747934:})
batchnormgeneencode2_747936:})
batchnormgeneencode2_747938:})
batchnormgeneencode2_747940:},
batchnormproteinencode1_747943:,
batchnormproteinencode1_747945:,
batchnormproteinencode1_747947:,
batchnormproteinencode1_747949:+
embeddingdimdense_747953:	њ@&
embeddingdimdense_747955:@
identityѕб,BatchNormGeneEncode1/StatefulPartitionedCallб,BatchNormGeneEncode2/StatefulPartitionedCallб/BatchNormProteinEncode1/StatefulPartitionedCallб)EmbeddingDimDense/StatefulPartitionedCallб&gene_encoder_1/StatefulPartitionedCallб&gene_encoder_2/StatefulPartitionedCallб)protein_encoder_1/StatefulPartitionedCallќ
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallgene_input_layergene_encoder_1_747910gene_encoder_1_747912*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         З*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_747531І
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_747915batchnormgeneencode1_747917batchnormgeneencode1_747919batchnormgeneencode1_747921*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         З*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_747289ц
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallprotein_input_layerprotein_encoder_1_747924protein_encoder_1_747926*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_747557║
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_747929gene_encoder_2_747931*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         }*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *S
fNRL
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_747574і
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_747934batchnormgeneencode2_747936batchnormgeneencode2_747938batchnormgeneencode2_747940*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         }*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_747371Ъ
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_747943batchnormproteinencode1_747945batchnormproteinencode1_747947batchnormproteinencode1_747949*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_747453Х
 ConcatenateLayer/PartitionedCallPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:08BatchNormProteinEncode1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         њ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8ѓ *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_747605║
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_747953embeddingdimdense_747955*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8ѓ *V
fQRO
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_747618Ђ
IdentityIdentity2EmbeddingDimDense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @ђ
NoOpNoOp-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         л:         W: : : : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2b
/BatchNormProteinEncode1/StatefulPartitionedCall/BatchNormProteinEncode1/StatefulPartitionedCall2V
)EmbeddingDimDense/StatefulPartitionedCall)EmbeddingDimDense/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall2V
)protein_encoder_1/StatefulPartitionedCall)protein_encoder_1/StatefulPartitionedCall:Z V
(
_output_shapes
:         л
*
_user_specified_nameGene_Input_Layer:\X
'
_output_shapes
:         W
-
_user_specified_nameProtein_Input_Layer"█L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ю
serving_defaultѕ
N
Gene_Input_Layer:
"serving_default_Gene_Input_Layer:0         л
S
Protein_Input_Layer<
%serving_default_Protein_Input_Layer:0         WE
EmbeddingDimDense0
StatefulPartitionedCall:0         @tensorflow/serving/predict:Хц
Џ
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
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
╗

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Ж
axis
	gamma
beta
moving_mean
 moving_variance
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
╗

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_layer
Ж
7axis
	8gamma
9beta
:moving_mean
;moving_variance
<	variables
=trainable_variables
>regularization_losses
?	keras_api
@__call__
*A&call_and_return_all_conditional_losses"
_tf_keras_layer
Ж
Baxis
	Cgamma
Dbeta
Emoving_mean
Fmoving_variance
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses"
_tf_keras_layer
Ц
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"
_tf_keras_layer
"
	optimizer
Х
0
1
2
3
4
 5
'6
(7
/8
09
810
911
:12
;13
C14
D15
E16
F17
S18
T19"
trackable_list_wrapper
є
0
1
2
3
'4
(5
/6
07
88
99
C10
D11
S12
T13"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
[non_trainable_variables

\layers
]metrics
^layer_regularization_losses
_layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ы2№
)__inference_model_23_layer_call_fn_747668
)__inference_model_23_layer_call_fn_748058
)__inference_model_23_layer_call_fn_748104
)__inference_model_23_layer_call_fn_747906└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
я2█
D__inference_model_23_layer_call_and_return_conditional_losses_748187
D__inference_model_23_layer_call_and_return_conditional_losses_748312
D__inference_model_23_layer_call_and_return_conditional_losses_747959
D__inference_model_23_layer_call_and_return_conditional_losses_748012└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ЖBу
!__inference__wrapped_model_747265Gene_Input_LayerProtein_Input_Layer"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
,
`serving_default"
signature_map
):'
лЗ2gene_encoder_1/kernel
": З2gene_encoder_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
anon_trainable_variables

blayers
cmetrics
dlayer_regularization_losses
elayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
┘2о
/__inference_gene_encoder_1_layer_call_fn_748369б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
З2ы
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_748380б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
):'З2BatchNormGeneEncode1/gamma
(:&З2BatchNormGeneEncode1/beta
1:/З (2 BatchNormGeneEncode1/moving_mean
5:3З (2$BatchNormGeneEncode1/moving_variance
<
0
1
2
 3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
fnon_trainable_variables

glayers
hmetrics
ilayer_regularization_losses
jlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
е2Ц
5__inference_BatchNormGeneEncode1_layer_call_fn_748393
5__inference_BatchNormGeneEncode1_layer_call_fn_748406┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
я2█
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_748426
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_748460┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
(:&	З}2gene_encoder_2/kernel
!:}2gene_encoder_2/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
knon_trainable_variables

llayers
mmetrics
nlayer_regularization_losses
olayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
┘2о
/__inference_gene_encoder_2_layer_call_fn_748469б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
З2ы
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_748480б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
*:(W2protein_encoder_1/kernel
$:"2protein_encoder_1/bias
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
Г
pnon_trainable_variables

qlayers
rmetrics
slayer_regularization_losses
tlayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
▄2┘
2__inference_protein_encoder_1_layer_call_fn_748489б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
э2З
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_748500б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
(:&}2BatchNormGeneEncode2/gamma
':%}2BatchNormGeneEncode2/beta
0:.} (2 BatchNormGeneEncode2/moving_mean
4:2} (2$BatchNormGeneEncode2/moving_variance
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
Г
unon_trainable_variables

vlayers
wmetrics
xlayer_regularization_losses
ylayer_metrics
<	variables
=trainable_variables
>regularization_losses
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
е2Ц
5__inference_BatchNormGeneEncode2_layer_call_fn_748513
5__inference_BatchNormGeneEncode2_layer_call_fn_748526┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
я2█
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_748546
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_748580┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
 "
trackable_list_wrapper
+:)2BatchNormProteinEncode1/gamma
*:(2BatchNormProteinEncode1/beta
3:1 (2#BatchNormProteinEncode1/moving_mean
7:5 (2'BatchNormProteinEncode1/moving_variance
<
C0
D1
E2
F3"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
znon_trainable_variables

{layers
|metrics
}layer_regularization_losses
~layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
«2Ф
8__inference_BatchNormProteinEncode1_layer_call_fn_748593
8__inference_BatchNormProteinEncode1_layer_call_fn_748606┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
С2р
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_748626
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_748660┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▒
non_trainable_variables
ђlayers
Ђmetrics
 ѓlayer_regularization_losses
Ѓlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
█2п
1__inference_ConcatenateLayer_layer_call_fn_748666б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ш2з
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_748673б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
+:)	њ@2EmbeddingDimDense/kernel
$:"@2EmbeddingDimDense/bias
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
ёnon_trainable_variables
Ёlayers
єmetrics
 Єlayer_regularization_losses
ѕlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
▄2┘
2__inference_EmbeddingDimDense_layer_call_fn_748682б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
э2З
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_748693б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
J
0
 1
:2
;3
E4
F5"
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
Ѕ0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
уBС
$__inference_signature_wrapper_748360Gene_Input_LayerProtein_Input_Layer"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
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
0
 1"
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
.
E0
F1"
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

іtotal

Іcount
ї	variables
Ї	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
і0
І1"
trackable_list_wrapper
.
ї	variables"
_generic_user_objectИ
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_748426d 4б1
*б'
!і
inputs         З
p 
ф "&б#
і
0         З
џ И
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_748460d 4б1
*б'
!і
inputs         З
p
ф "&б#
і
0         З
џ љ
5__inference_BatchNormGeneEncode1_layer_call_fn_748393W 4б1
*б'
!і
inputs         З
p 
ф "і         Зљ
5__inference_BatchNormGeneEncode1_layer_call_fn_748406W 4б1
*б'
!і
inputs         З
p
ф "і         ЗХ
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_748546b;8:93б0
)б&
 і
inputs         }
p 
ф "%б"
і
0         }
џ Х
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_748580b:;893б0
)б&
 і
inputs         }
p
ф "%б"
і
0         }
џ ј
5__inference_BatchNormGeneEncode2_layer_call_fn_748513U;8:93б0
)б&
 і
inputs         }
p 
ф "і         }ј
5__inference_BatchNormGeneEncode2_layer_call_fn_748526U:;893б0
)б&
 і
inputs         }
p
ф "і         }╣
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_748626bFCED3б0
)б&
 і
inputs         
p 
ф "%б"
і
0         
џ ╣
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_748660bEFCD3б0
)б&
 і
inputs         
p
ф "%б"
і
0         
џ Љ
8__inference_BatchNormProteinEncode1_layer_call_fn_748593UFCED3б0
)б&
 і
inputs         
p 
ф "і         Љ
8__inference_BatchNormProteinEncode1_layer_call_fn_748606UEFCD3б0
)б&
 і
inputs         
p
ф "і         Н
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_748673ёZбW
PбM
KџH
"і
inputs/0         }
"і
inputs/1         
ф "&б#
і
0         њ
џ г
1__inference_ConcatenateLayer_layer_call_fn_748666wZбW
PбM
KџH
"і
inputs/0         }
"і
inputs/1         
ф "і         њ«
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_748693]ST0б-
&б#
!і
inputs         њ
ф "%б"
і
0         @
џ є
2__inference_EmbeddingDimDense_layer_call_fn_748682PST0б-
&б#
!і
inputs         њ
ф "і         @з
!__inference__wrapped_model_747265═ /0'(;8:9FCEDSTnбk
dбa
_џ\
+і(
Gene_Input_Layer         л
-і*
Protein_Input_Layer         W
ф "EфB
@
EmbeddingDimDense+і(
EmbeddingDimDense         @г
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_748380^0б-
&б#
!і
inputs         л
ф "&б#
і
0         З
џ ё
/__inference_gene_encoder_1_layer_call_fn_748369Q0б-
&б#
!і
inputs         л
ф "і         ЗФ
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_748480]'(0б-
&б#
!і
inputs         З
ф "%б"
і
0         }
џ Ѓ
/__inference_gene_encoder_2_layer_call_fn_748469P'(0б-
&б#
!і
inputs         З
ф "і         }■
D__inference_model_23_layer_call_and_return_conditional_losses_747959х /0'(;8:9FCEDSTvбs
lбi
_џ\
+і(
Gene_Input_Layer         л
-і*
Protein_Input_Layer         W
p 

 
ф "%б"
і
0         @
џ ■
D__inference_model_23_layer_call_and_return_conditional_losses_748012х /0'(:;89EFCDSTvбs
lбi
_џ\
+і(
Gene_Input_Layer         л
-і*
Protein_Input_Layer         W
p

 
ф "%б"
і
0         @
џ в
D__inference_model_23_layer_call_and_return_conditional_losses_748187б /0'(;8:9FCEDSTcб`
YбV
LџI
#і 
inputs/0         л
"і
inputs/1         W
p 

 
ф "%б"
і
0         @
џ в
D__inference_model_23_layer_call_and_return_conditional_losses_748312б /0'(:;89EFCDSTcб`
YбV
LџI
#і 
inputs/0         л
"і
inputs/1         W
p

 
ф "%б"
і
0         @
џ о
)__inference_model_23_layer_call_fn_747668е /0'(;8:9FCEDSTvбs
lбi
_џ\
+і(
Gene_Input_Layer         л
-і*
Protein_Input_Layer         W
p 

 
ф "і         @о
)__inference_model_23_layer_call_fn_747906е /0'(:;89EFCDSTvбs
lбi
_џ\
+і(
Gene_Input_Layer         л
-і*
Protein_Input_Layer         W
p

 
ф "і         @├
)__inference_model_23_layer_call_fn_748058Ћ /0'(;8:9FCEDSTcб`
YбV
LџI
#і 
inputs/0         л
"і
inputs/1         W
p 

 
ф "і         @├
)__inference_model_23_layer_call_fn_748104Ћ /0'(:;89EFCDSTcб`
YбV
LџI
#і 
inputs/0         л
"і
inputs/1         W
p

 
ф "і         @Г
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_748500\/0/б,
%б"
 і
inputs         W
ф "%б"
і
0         
џ Ё
2__inference_protein_encoder_1_layer_call_fn_748489O/0/б,
%б"
 і
inputs         W
ф "і         а
$__inference_signature_wrapper_748360э /0'(;8:9FCEDSTЌбЊ
б 
ІфЄ
?
Gene_Input_Layer+і(
Gene_Input_Layer         л
D
Protein_Input_Layer-і*
Protein_Input_Layer         W"EфB
@
EmbeddingDimDense+і(
EmbeddingDimDense         @