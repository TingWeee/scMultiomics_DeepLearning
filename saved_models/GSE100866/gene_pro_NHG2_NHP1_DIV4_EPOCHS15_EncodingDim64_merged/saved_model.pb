├╡
Ыь
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
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
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
dtypetypeИ
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
list(type)(0И
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
executor_typestring Ии
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
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68й¤
И
gene_encoder_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
╨Ї*&
shared_namegene_encoder_1/kernel
Б
)gene_encoder_1/kernel/Read/ReadVariableOpReadVariableOpgene_encoder_1/kernel* 
_output_shapes
:
╨Ї*
dtype0

gene_encoder_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ї*$
shared_namegene_encoder_1/bias
x
'gene_encoder_1/bias/Read/ReadVariableOpReadVariableOpgene_encoder_1/bias*
_output_shapes	
:Ї*
dtype0
Н
BatchNormGeneEncode1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ї*+
shared_nameBatchNormGeneEncode1/gamma
Ж
.BatchNormGeneEncode1/gamma/Read/ReadVariableOpReadVariableOpBatchNormGeneEncode1/gamma*
_output_shapes	
:Ї*
dtype0
Л
BatchNormGeneEncode1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ї**
shared_nameBatchNormGeneEncode1/beta
Д
-BatchNormGeneEncode1/beta/Read/ReadVariableOpReadVariableOpBatchNormGeneEncode1/beta*
_output_shapes	
:Ї*
dtype0
Щ
 BatchNormGeneEncode1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ї*1
shared_name" BatchNormGeneEncode1/moving_mean
Т
4BatchNormGeneEncode1/moving_mean/Read/ReadVariableOpReadVariableOp BatchNormGeneEncode1/moving_mean*
_output_shapes	
:Ї*
dtype0
б
$BatchNormGeneEncode1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:Ї*5
shared_name&$BatchNormGeneEncode1/moving_variance
Ъ
8BatchNormGeneEncode1/moving_variance/Read/ReadVariableOpReadVariableOp$BatchNormGeneEncode1/moving_variance*
_output_shapes	
:Ї*
dtype0
З
gene_encoder_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Ї}*&
shared_namegene_encoder_2/kernel
А
)gene_encoder_2/kernel/Read/ReadVariableOpReadVariableOpgene_encoder_2/kernel*
_output_shapes
:	Ї}*
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
М
protein_encoder_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*)
shared_nameprotein_encoder_1/kernel
Е
,protein_encoder_1/kernel/Read/ReadVariableOpReadVariableOpprotein_encoder_1/kernel*
_output_shapes

:*
dtype0
Д
protein_encoder_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameprotein_encoder_1/bias
}
*protein_encoder_1/bias/Read/ReadVariableOpReadVariableOpprotein_encoder_1/bias*
_output_shapes
:*
dtype0
М
BatchNormGeneEncode2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*+
shared_nameBatchNormGeneEncode2/gamma
Е
.BatchNormGeneEncode2/gamma/Read/ReadVariableOpReadVariableOpBatchNormGeneEncode2/gamma*
_output_shapes
:}*
dtype0
К
BatchNormGeneEncode2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:}**
shared_nameBatchNormGeneEncode2/beta
Г
-BatchNormGeneEncode2/beta/Read/ReadVariableOpReadVariableOpBatchNormGeneEncode2/beta*
_output_shapes
:}*
dtype0
Ш
 BatchNormGeneEncode2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:}*1
shared_name" BatchNormGeneEncode2/moving_mean
С
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
Щ
8BatchNormGeneEncode2/moving_variance/Read/ReadVariableOpReadVariableOp$BatchNormGeneEncode2/moving_variance*
_output_shapes
:}*
dtype0
Т
BatchNormProteinEncode1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*.
shared_nameBatchNormProteinEncode1/gamma
Л
1BatchNormProteinEncode1/gamma/Read/ReadVariableOpReadVariableOpBatchNormProteinEncode1/gamma*
_output_shapes
:*
dtype0
Р
BatchNormProteinEncode1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameBatchNormProteinEncode1/beta
Й
0BatchNormProteinEncode1/beta/Read/ReadVariableOpReadVariableOpBatchNormProteinEncode1/beta*
_output_shapes
:*
dtype0
Ю
#BatchNormProteinEncode1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#BatchNormProteinEncode1/moving_mean
Ч
7BatchNormProteinEncode1/moving_mean/Read/ReadVariableOpReadVariableOp#BatchNormProteinEncode1/moving_mean*
_output_shapes
:*
dtype0
ж
'BatchNormProteinEncode1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'BatchNormProteinEncode1/moving_variance
Я
;BatchNormProteinEncode1/moving_variance/Read/ReadVariableOpReadVariableOp'BatchNormProteinEncode1/moving_variance*
_output_shapes
:*
dtype0
Н
EmbeddingDimDense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Г@*)
shared_nameEmbeddingDimDense/kernel
Ж
,EmbeddingDimDense/kernel/Read/ReadVariableOpReadVariableOpEmbeddingDimDense/kernel*
_output_shapes
:	Г@*
dtype0
Д
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
Ф@
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╧?
value┼?B┬? B╗?
Д
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
ж

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
╒
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
ж

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
ж

/kernel
0bias
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses*
╒
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
╒
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
О
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses* 
ж

Skernel
Tbias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses*
* 
Ъ
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
У
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
У
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
У
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
У
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
У
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
Аz
VARIABLE_VALUE'BatchNormProteinEncode1/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
 
C0
D1
E2
F3*

C0
D1*
* 
У
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
Х
non_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
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
Ш
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
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

Й0*
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

Кtotal

Лcount
М	variables
Н	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

К0
Л1*

М	variables*
Е
 serving_default_Gene_Input_LayerPlaceholder*(
_output_shapes
:         ╨*
dtype0*
shape:         ╨
Ж
#serving_default_Protein_Input_LayerPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
╤
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
GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_132474
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
GPU2*0J 8В *(
f#R!
__inference__traced_save_132897
В
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
GPU2*0J 8В *+
f&R$
"__inference__traced_restore_132973╛ё

г

■
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_132614

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╪
Я
/__inference_gene_encoder_1_layer_call_fn_132483

inputs
unknown:
╨Ї
	unknown_0:	Ї
identityИвStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_131645p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         Ї`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╨: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ╨
 
_user_specified_nameinputs
╨
▓
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_131567

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         ║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┐
v
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_131719

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
:         ГX
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:         Г"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         }:         :O K
'
_output_shapes
:         }
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs
С%
щ
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_131532

inputs5
'assignmovingavg_readvariableop_resource:}7
)assignmovingavg_1_readvariableop_resource:}3
%batchnorm_mul_readvariableop_resource:}/
!batchnorm_readvariableop_resource:}
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
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

:}З
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         }l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ю
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
╫#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:}*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:}x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:}м
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
╫#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:}*
dtype0З
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
 *oГ:q
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
:         }ъ
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
╖
]
1__inference_ConcatenateLayer_layer_call_fn_132780
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
:         Г* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_131719a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         Г"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         }:         :Q M
'
_output_shapes
:         }
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
ж├
═
C__inference_model_3_layer_call_and_return_conditional_losses_132426
inputs_0
inputs_1A
-gene_encoder_1_matmul_readvariableop_resource:
╨Ї=
.gene_encoder_1_biasadd_readvariableop_resource:	ЇK
<batchnormgeneencode1_assignmovingavg_readvariableop_resource:	ЇM
>batchnormgeneencode1_assignmovingavg_1_readvariableop_resource:	ЇI
:batchnormgeneencode1_batchnorm_mul_readvariableop_resource:	ЇE
6batchnormgeneencode1_batchnorm_readvariableop_resource:	ЇB
0protein_encoder_1_matmul_readvariableop_resource:?
1protein_encoder_1_biasadd_readvariableop_resource:@
-gene_encoder_2_matmul_readvariableop_resource:	Ї}<
.gene_encoder_2_biasadd_readvariableop_resource:}J
<batchnormgeneencode2_assignmovingavg_readvariableop_resource:}L
>batchnormgeneencode2_assignmovingavg_1_readvariableop_resource:}H
:batchnormgeneencode2_batchnorm_mul_readvariableop_resource:}D
6batchnormgeneencode2_batchnorm_readvariableop_resource:}M
?batchnormproteinencode1_assignmovingavg_readvariableop_resource:O
Abatchnormproteinencode1_assignmovingavg_1_readvariableop_resource:K
=batchnormproteinencode1_batchnorm_mul_readvariableop_resource:G
9batchnormproteinencode1_batchnorm_readvariableop_resource:C
0embeddingdimdense_matmul_readvariableop_resource:	Г@?
1embeddingdimdense_biasadd_readvariableop_resource:@
identityИв$BatchNormGeneEncode1/AssignMovingAvgв3BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOpв&BatchNormGeneEncode1/AssignMovingAvg_1в5BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOpв-BatchNormGeneEncode1/batchnorm/ReadVariableOpв1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOpв$BatchNormGeneEncode2/AssignMovingAvgв3BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOpв&BatchNormGeneEncode2/AssignMovingAvg_1в5BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOpв-BatchNormGeneEncode2/batchnorm/ReadVariableOpв1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOpв'BatchNormProteinEncode1/AssignMovingAvgв6BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOpв)BatchNormProteinEncode1/AssignMovingAvg_1в8BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOpв0BatchNormProteinEncode1/batchnorm/ReadVariableOpв4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOpв(EmbeddingDimDense/BiasAdd/ReadVariableOpв'EmbeddingDimDense/MatMul/ReadVariableOpв%gene_encoder_1/BiasAdd/ReadVariableOpв$gene_encoder_1/MatMul/ReadVariableOpв%gene_encoder_2/BiasAdd/ReadVariableOpв$gene_encoder_2/MatMul/ReadVariableOpв(protein_encoder_1/BiasAdd/ReadVariableOpв'protein_encoder_1/MatMul/ReadVariableOpФ
$gene_encoder_1/MatMul/ReadVariableOpReadVariableOp-gene_encoder_1_matmul_readvariableop_resource* 
_output_shapes
:
╨Ї*
dtype0К
gene_encoder_1/MatMulMatMulinputs_0,gene_encoder_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ЇС
%gene_encoder_1/BiasAdd/ReadVariableOpReadVariableOp.gene_encoder_1_biasadd_readvariableop_resource*
_output_shapes	
:Ї*
dtype0д
gene_encoder_1/BiasAddBiasAddgene_encoder_1/MatMul:product:0-gene_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Їu
gene_encoder_1/SigmoidSigmoidgene_encoder_1/BiasAdd:output:0*
T0*(
_output_shapes
:         Ї}
3BatchNormGeneEncode1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ╛
!BatchNormGeneEncode1/moments/meanMeangene_encoder_1/Sigmoid:y:0<BatchNormGeneEncode1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Ї*
	keep_dims(П
)BatchNormGeneEncode1/moments/StopGradientStopGradient*BatchNormGeneEncode1/moments/mean:output:0*
T0*
_output_shapes
:	Ї╞
.BatchNormGeneEncode1/moments/SquaredDifferenceSquaredDifferencegene_encoder_1/Sigmoid:y:02BatchNormGeneEncode1/moments/StopGradient:output:0*
T0*(
_output_shapes
:         ЇБ
7BatchNormGeneEncode1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ▐
%BatchNormGeneEncode1/moments/varianceMean2BatchNormGeneEncode1/moments/SquaredDifference:z:0@BatchNormGeneEncode1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Ї*
	keep_dims(Ш
$BatchNormGeneEncode1/moments/SqueezeSqueeze*BatchNormGeneEncode1/moments/mean:output:0*
T0*
_output_shapes	
:Ї*
squeeze_dims
 Ю
&BatchNormGeneEncode1/moments/Squeeze_1Squeeze.BatchNormGeneEncode1/moments/variance:output:0*
T0*
_output_shapes	
:Ї*
squeeze_dims
 o
*BatchNormGeneEncode1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<н
3BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOpReadVariableOp<batchnormgeneencode1_assignmovingavg_readvariableop_resource*
_output_shapes	
:Ї*
dtype0┴
(BatchNormGeneEncode1/AssignMovingAvg/subSub;BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOp:value:0-BatchNormGeneEncode1/moments/Squeeze:output:0*
T0*
_output_shapes	
:Ї╕
(BatchNormGeneEncode1/AssignMovingAvg/mulMul,BatchNormGeneEncode1/AssignMovingAvg/sub:z:03BatchNormGeneEncode1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:ЇА
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
╫#<▒
5BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOpReadVariableOp>batchnormgeneencode1_assignmovingavg_1_readvariableop_resource*
_output_shapes	
:Ї*
dtype0╟
*BatchNormGeneEncode1/AssignMovingAvg_1/subSub=BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOp:value:0/BatchNormGeneEncode1/moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ї╛
*BatchNormGeneEncode1/AssignMovingAvg_1/mulMul.BatchNormGeneEncode1/AssignMovingAvg_1/sub:z:05BatchNormGeneEncode1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:ЇИ
&BatchNormGeneEncode1/AssignMovingAvg_1AssignSubVariableOp>batchnormgeneencode1_assignmovingavg_1_readvariableop_resource.BatchNormGeneEncode1/AssignMovingAvg_1/mul:z:06^BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0i
$BatchNormGeneEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:▒
"BatchNormGeneEncode1/batchnorm/addAddV2/BatchNormGeneEncode1/moments/Squeeze_1:output:0-BatchNormGeneEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ї{
$BatchNormGeneEncode1/batchnorm/RsqrtRsqrt&BatchNormGeneEncode1/batchnorm/add:z:0*
T0*
_output_shapes	
:Їй
1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgeneencode1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ї*
dtype0┤
"BatchNormGeneEncode1/batchnorm/mulMul(BatchNormGeneEncode1/batchnorm/Rsqrt:y:09BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Їв
$BatchNormGeneEncode1/batchnorm/mul_1Mulgene_encoder_1/Sigmoid:y:0&BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*(
_output_shapes
:         Їи
$BatchNormGeneEncode1/batchnorm/mul_2Mul-BatchNormGeneEncode1/moments/Squeeze:output:0&BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*
_output_shapes	
:Їб
-BatchNormGeneEncode1/batchnorm/ReadVariableOpReadVariableOp6batchnormgeneencode1_batchnorm_readvariableop_resource*
_output_shapes	
:Ї*
dtype0░
"BatchNormGeneEncode1/batchnorm/subSub5BatchNormGeneEncode1/batchnorm/ReadVariableOp:value:0(BatchNormGeneEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ї▓
$BatchNormGeneEncode1/batchnorm/add_1AddV2(BatchNormGeneEncode1/batchnorm/mul_1:z:0&BatchNormGeneEncode1/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ЇШ
'protein_encoder_1/MatMul/ReadVariableOpReadVariableOp0protein_encoder_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0П
protein_encoder_1/MatMulMatMulinputs_1/protein_encoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ц
(protein_encoder_1/BiasAdd/ReadVariableOpReadVariableOp1protein_encoder_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0м
protein_encoder_1/BiasAddBiasAdd"protein_encoder_1/MatMul:product:00protein_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         z
protein_encoder_1/SigmoidSigmoid"protein_encoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:         У
$gene_encoder_2/MatMul/ReadVariableOpReadVariableOp-gene_encoder_2_matmul_readvariableop_resource*
_output_shapes
:	Ї}*
dtype0й
gene_encoder_2/MatMulMatMul(BatchNormGeneEncode1/batchnorm/add_1:z:0,gene_encoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         }Р
%gene_encoder_2/BiasAdd/ReadVariableOpReadVariableOp.gene_encoder_2_biasadd_readvariableop_resource*
_output_shapes
:}*
dtype0г
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
valueB: ╜
!BatchNormGeneEncode2/moments/meanMeangene_encoder_2/Sigmoid:y:0<BatchNormGeneEncode2/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:}*
	keep_dims(О
)BatchNormGeneEncode2/moments/StopGradientStopGradient*BatchNormGeneEncode2/moments/mean:output:0*
T0*
_output_shapes

:}┼
.BatchNormGeneEncode2/moments/SquaredDifferenceSquaredDifferencegene_encoder_2/Sigmoid:y:02BatchNormGeneEncode2/moments/StopGradient:output:0*
T0*'
_output_shapes
:         }Б
7BatchNormGeneEncode2/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ▌
%BatchNormGeneEncode2/moments/varianceMean2BatchNormGeneEncode2/moments/SquaredDifference:z:0@BatchNormGeneEncode2/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:}*
	keep_dims(Ч
$BatchNormGeneEncode2/moments/SqueezeSqueeze*BatchNormGeneEncode2/moments/mean:output:0*
T0*
_output_shapes
:}*
squeeze_dims
 Э
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
╫#<м
3BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOpReadVariableOp<batchnormgeneencode2_assignmovingavg_readvariableop_resource*
_output_shapes
:}*
dtype0└
(BatchNormGeneEncode2/AssignMovingAvg/subSub;BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOp:value:0-BatchNormGeneEncode2/moments/Squeeze:output:0*
T0*
_output_shapes
:}╖
(BatchNormGeneEncode2/AssignMovingAvg/mulMul,BatchNormGeneEncode2/AssignMovingAvg/sub:z:03BatchNormGeneEncode2/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:}А
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
╫#<░
5BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOpReadVariableOp>batchnormgeneencode2_assignmovingavg_1_readvariableop_resource*
_output_shapes
:}*
dtype0╞
*BatchNormGeneEncode2/AssignMovingAvg_1/subSub=BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOp:value:0/BatchNormGeneEncode2/moments/Squeeze_1:output:0*
T0*
_output_shapes
:}╜
*BatchNormGeneEncode2/AssignMovingAvg_1/mulMul.BatchNormGeneEncode2/AssignMovingAvg_1/sub:z:05BatchNormGeneEncode2/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:}И
&BatchNormGeneEncode2/AssignMovingAvg_1AssignSubVariableOp>batchnormgeneencode2_assignmovingavg_1_readvariableop_resource.BatchNormGeneEncode2/AssignMovingAvg_1/mul:z:06^BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0i
$BatchNormGeneEncode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:░
"BatchNormGeneEncode2/batchnorm/addAddV2/BatchNormGeneEncode2/moments/Squeeze_1:output:0-BatchNormGeneEncode2/batchnorm/add/y:output:0*
T0*
_output_shapes
:}z
$BatchNormGeneEncode2/batchnorm/RsqrtRsqrt&BatchNormGeneEncode2/batchnorm/add:z:0*
T0*
_output_shapes
:}и
1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgeneencode2_batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0│
"BatchNormGeneEncode2/batchnorm/mulMul(BatchNormGeneEncode2/batchnorm/Rsqrt:y:09BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}б
$BatchNormGeneEncode2/batchnorm/mul_1Mulgene_encoder_2/Sigmoid:y:0&BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         }з
$BatchNormGeneEncode2/batchnorm/mul_2Mul-BatchNormGeneEncode2/moments/Squeeze:output:0&BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*
_output_shapes
:}а
-BatchNormGeneEncode2/batchnorm/ReadVariableOpReadVariableOp6batchnormgeneencode2_batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0п
"BatchNormGeneEncode2/batchnorm/subSub5BatchNormGeneEncode2/batchnorm/ReadVariableOp:value:0(BatchNormGeneEncode2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:}▒
$BatchNormGeneEncode2/batchnorm/add_1AddV2(BatchNormGeneEncode2/batchnorm/mul_1:z:0&BatchNormGeneEncode2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         }А
6BatchNormProteinEncode1/moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ╞
$BatchNormProteinEncode1/moments/meanMeanprotein_encoder_1/Sigmoid:y:0?BatchNormProteinEncode1/moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(Ф
,BatchNormProteinEncode1/moments/StopGradientStopGradient-BatchNormProteinEncode1/moments/mean:output:0*
T0*
_output_shapes

:╬
1BatchNormProteinEncode1/moments/SquaredDifferenceSquaredDifferenceprotein_encoder_1/Sigmoid:y:05BatchNormProteinEncode1/moments/StopGradient:output:0*
T0*'
_output_shapes
:         Д
:BatchNormProteinEncode1/moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: ц
(BatchNormProteinEncode1/moments/varianceMean5BatchNormProteinEncode1/moments/SquaredDifference:z:0CBatchNormProteinEncode1/moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(Э
'BatchNormProteinEncode1/moments/SqueezeSqueeze-BatchNormProteinEncode1/moments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 г
)BatchNormProteinEncode1/moments/Squeeze_1Squeeze1BatchNormProteinEncode1/moments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 r
-BatchNormProteinEncode1/AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<▓
6BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOpReadVariableOp?batchnormproteinencode1_assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0╔
+BatchNormProteinEncode1/AssignMovingAvg/subSub>BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOp:value:00BatchNormProteinEncode1/moments/Squeeze:output:0*
T0*
_output_shapes
:└
+BatchNormProteinEncode1/AssignMovingAvg/mulMul/BatchNormProteinEncode1/AssignMovingAvg/sub:z:06BatchNormProteinEncode1/AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:М
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
╫#<╢
8BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOpReadVariableOpAbatchnormproteinencode1_assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0╧
-BatchNormProteinEncode1/AssignMovingAvg_1/subSub@BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp:value:02BatchNormProteinEncode1/moments/Squeeze_1:output:0*
T0*
_output_shapes
:╞
-BatchNormProteinEncode1/AssignMovingAvg_1/mulMul1BatchNormProteinEncode1/AssignMovingAvg_1/sub:z:08BatchNormProteinEncode1/AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:Ф
)BatchNormProteinEncode1/AssignMovingAvg_1AssignSubVariableOpAbatchnormproteinencode1_assignmovingavg_1_readvariableop_resource1BatchNormProteinEncode1/AssignMovingAvg_1/mul:z:09^BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0l
'BatchNormProteinEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╣
%BatchNormProteinEncode1/batchnorm/addAddV22BatchNormProteinEncode1/moments/Squeeze_1:output:00BatchNormProteinEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:А
'BatchNormProteinEncode1/batchnorm/RsqrtRsqrt)BatchNormProteinEncode1/batchnorm/add:z:0*
T0*
_output_shapes
:о
4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOpReadVariableOp=batchnormproteinencode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╝
%BatchNormProteinEncode1/batchnorm/mulMul+BatchNormProteinEncode1/batchnorm/Rsqrt:y:0<BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:к
'BatchNormProteinEncode1/batchnorm/mul_1Mulprotein_encoder_1/Sigmoid:y:0)BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:         ░
'BatchNormProteinEncode1/batchnorm/mul_2Mul0BatchNormProteinEncode1/moments/Squeeze:output:0)BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*
_output_shapes
:ж
0BatchNormProteinEncode1/batchnorm/ReadVariableOpReadVariableOp9batchnormproteinencode1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0╕
%BatchNormProteinEncode1/batchnorm/subSub8BatchNormProteinEncode1/batchnorm/ReadVariableOp:value:0+BatchNormProteinEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:║
'BatchNormProteinEncode1/batchnorm/add_1AddV2+BatchNormProteinEncode1/batchnorm/mul_1:z:0)BatchNormProteinEncode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:         ^
ConcatenateLayer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :▌
ConcatenateLayer/concatConcatV2(BatchNormGeneEncode2/batchnorm/add_1:z:0+BatchNormProteinEncode1/batchnorm/add_1:z:0%ConcatenateLayer/concat/axis:output:0*
N*
T0*(
_output_shapes
:         ГЩ
'EmbeddingDimDense/MatMul/ReadVariableOpReadVariableOp0embeddingdimdense_matmul_readvariableop_resource*
_output_shapes
:	Г@*
dtype0з
EmbeddingDimDense/MatMulMatMul ConcatenateLayer/concat:output:0/EmbeddingDimDense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ц
(EmbeddingDimDense/BiasAdd/ReadVariableOpReadVariableOp1embeddingdimdense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0м
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
:         @Ж

NoOpNoOp%^BatchNormGeneEncode1/AssignMovingAvg4^BatchNormGeneEncode1/AssignMovingAvg/ReadVariableOp'^BatchNormGeneEncode1/AssignMovingAvg_16^BatchNormGeneEncode1/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneEncode1/batchnorm/ReadVariableOp2^BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp%^BatchNormGeneEncode2/AssignMovingAvg4^BatchNormGeneEncode2/AssignMovingAvg/ReadVariableOp'^BatchNormGeneEncode2/AssignMovingAvg_16^BatchNormGeneEncode2/AssignMovingAvg_1/ReadVariableOp.^BatchNormGeneEncode2/batchnorm/ReadVariableOp2^BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp(^BatchNormProteinEncode1/AssignMovingAvg7^BatchNormProteinEncode1/AssignMovingAvg/ReadVariableOp*^BatchNormProteinEncode1/AssignMovingAvg_19^BatchNormProteinEncode1/AssignMovingAvg_1/ReadVariableOp1^BatchNormProteinEncode1/batchnorm/ReadVariableOp5^BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp)^EmbeddingDimDense/BiasAdd/ReadVariableOp(^EmbeddingDimDense/MatMul/ReadVariableOp&^gene_encoder_1/BiasAdd/ReadVariableOp%^gene_encoder_1/MatMul/ReadVariableOp&^gene_encoder_2/BiasAdd/ReadVariableOp%^gene_encoder_2/MatMul/ReadVariableOp)^protein_encoder_1/BiasAdd/ReadVariableOp(^protein_encoder_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         ╨:         : : : : : : : : : : : : : : : : : : : : 2L
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
:         ╨
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
ё5
щ

__inference__traced_save_132897
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

identity_1ИвMergeV2Checkpointsw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: я

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ш

valueО
BЛ
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЫ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B ы

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_gene_encoder_1_kernel_read_readvariableop.savev2_gene_encoder_1_bias_read_readvariableop5savev2_batchnormgeneencode1_gamma_read_readvariableop4savev2_batchnormgeneencode1_beta_read_readvariableop;savev2_batchnormgeneencode1_moving_mean_read_readvariableop?savev2_batchnormgeneencode1_moving_variance_read_readvariableop0savev2_gene_encoder_2_kernel_read_readvariableop.savev2_gene_encoder_2_bias_read_readvariableop3savev2_protein_encoder_1_kernel_read_readvariableop1savev2_protein_encoder_1_bias_read_readvariableop5savev2_batchnormgeneencode2_gamma_read_readvariableop4savev2_batchnormgeneencode2_beta_read_readvariableop;savev2_batchnormgeneencode2_moving_mean_read_readvariableop?savev2_batchnormgeneencode2_moving_variance_read_readvariableop8savev2_batchnormproteinencode1_gamma_read_readvariableop7savev2_batchnormproteinencode1_beta_read_readvariableop>savev2_batchnormproteinencode1_moving_mean_read_readvariableopBsavev2_batchnormproteinencode1_moving_variance_read_readvariableop3savev2_embeddingdimdense_kernel_read_readvariableop1savev2_embeddingdimdense_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *%
dtypes
2Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
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

identity_1Identity_1:output:0*о
_input_shapesЬ
Щ: :
╨Ї:Ї:Ї:Ї:Ї:Ї:	Ї}:}:::}:}:}:}:::::	Г@:@: : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
╨Ї:!

_output_shapes	
:Ї:!

_output_shapes	
:Ї:!

_output_shapes	
:Ї:!

_output_shapes	
:Ї:!

_output_shapes	
:Ї:%!

_output_shapes
:	Ї}: 

_output_shapes
:}:$	 

_output_shapes

:: 


_output_shapes
:: 
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
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::%!

_output_shapes
:	Г@: 
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
╘
Э
/__inference_gene_encoder_2_layer_call_fn_132583

inputs
unknown:	Ї}
	unknown_0:}
identityИвStatefulPartitionedCallт
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
GPU2*0J 8В *S
fNRL
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_131688o
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
:         Ї: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
═
п
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_132660

inputs/
!batchnorm_readvariableop_resource:}3
%batchnorm_mul_readvariableop_resource:}1
#batchnorm_readvariableop_1_resource:}1
#batchnorm_readvariableop_2_resource:}
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
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
┌
а
2__inference_EmbeddingDimDense_layer_call_fn_132796

inputs
unknown:	Г@
	unknown_0:@
identityИвStatefulPartitionedCallх
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
GPU2*0J 8В *V
fQRO
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_131732o
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
:         Г: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Г
 
_user_specified_nameinputs
м

■
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_131645

inputs2
matmul_readvariableop_resource:
╨Ї.
biasadd_readvariableop_resource:	Ї
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
╨Ї*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Їs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ї*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ЇW
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:         Ї[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:         Їw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╨: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ╨
 
_user_specified_nameinputs
┴
Е
(__inference_model_3_layer_call_fn_132172
inputs_0
inputs_1
unknown:
╨Ї
	unknown_0:	Ї
	unknown_1:	Ї
	unknown_2:	Ї
	unknown_3:	Ї
	unknown_4:	Ї
	unknown_5:
	unknown_6:
	unknown_7:	Ї}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:	Г@

unknown_18:@
identityИвStatefulPartitionedCall█
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
GPU2*0J 8В *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_131739o
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
O:         ╨:         : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:         ╨
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
═
п
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_131485

inputs/
!batchnorm_readvariableop_resource:}3
%batchnorm_mul_readvariableop_resource:}1
#batchnorm_readvariableop_1_resource:}1
#batchnorm_readvariableop_2_resource:}
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
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
╗
Е
(__inference_model_3_layer_call_fn_132218
inputs_0
inputs_1
unknown:
╨Ї
	unknown_0:	Ї
	unknown_1:	Ї
	unknown_2:	Ї
	unknown_3:	Ї
	unknown_4:	Ї
	unknown_5:
	unknown_6:
	unknown_7:	Ї}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:	Г@

unknown_18:@
identityИвStatefulPartitionedCall╒
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
GPU2*0J 8В *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_131931o
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
O:         ╨:         : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
(
_output_shapes
:         ╨
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
ъ2
З

C__inference_model_3_layer_call_and_return_conditional_losses_131931

inputs
inputs_1)
gene_encoder_1_131882:
╨Ї$
gene_encoder_1_131884:	Ї*
batchnormgeneencode1_131887:	Ї*
batchnormgeneencode1_131889:	Ї*
batchnormgeneencode1_131891:	Ї*
batchnormgeneencode1_131893:	Ї*
protein_encoder_1_131896:&
protein_encoder_1_131898:(
gene_encoder_2_131901:	Ї}#
gene_encoder_2_131903:})
batchnormgeneencode2_131906:})
batchnormgeneencode2_131908:})
batchnormgeneencode2_131910:})
batchnormgeneencode2_131912:},
batchnormproteinencode1_131915:,
batchnormproteinencode1_131917:,
batchnormproteinencode1_131919:,
batchnormproteinencode1_131921:+
embeddingdimdense_131925:	Г@&
embeddingdimdense_131927:@
identityИв,BatchNormGeneEncode1/StatefulPartitionedCallв,BatchNormGeneEncode2/StatefulPartitionedCallв/BatchNormProteinEncode1/StatefulPartitionedCallв)EmbeddingDimDense/StatefulPartitionedCallв&gene_encoder_1/StatefulPartitionedCallв&gene_encoder_2/StatefulPartitionedCallв)protein_encoder_1/StatefulPartitionedCallМ
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputsgene_encoder_1_131882gene_encoder_1_131884*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_131645Й
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_131887batchnormgeneencode1_131889batchnormgeneencode1_131891batchnormgeneencode1_131893*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_131450Щ
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1protein_encoder_1_131896protein_encoder_1_131898*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_131671║
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_131901gene_encoder_2_131903*
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
GPU2*0J 8В *S
fNRL
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_131688И
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_131906batchnormgeneencode2_131908batchnormgeneencode2_131910batchnormgeneencode2_131912*
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
GPU2*0J 8В *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_131532Э
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_131915batchnormproteinencode1_131917batchnormproteinencode1_131919batchnormproteinencode1_131921*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_131614╢
 ConcatenateLayer/PartitionedCallPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:08BatchNormProteinEncode1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Г* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_131719║
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_131925embeddingdimdense_131927*
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
GPU2*0J 8В *V
fQRO
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_131732Б
IdentityIdentity2EmbeddingDimDense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @А
NoOpNoOp-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         ╨:         : : : : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2b
/BatchNormProteinEncode1/StatefulPartitionedCall/BatchNormProteinEncode1/StatefulPartitionedCall2V
)EmbeddingDimDense/StatefulPartitionedCall)EmbeddingDimDense/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall2V
)protein_encoder_1/StatefulPartitionedCall)protein_encoder_1/StatefulPartitionedCall:P L
(
_output_shapes
:         ╨
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs
з

 
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_131732

inputs1
matmul_readvariableop_resource:	Г@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Г@*
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
:         Г: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         Г
 
_user_specified_nameinputs
С%
щ
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_132694

inputs5
'assignmovingavg_readvariableop_resource:}7
)assignmovingavg_1_readvariableop_resource:}3
%batchnorm_mul_readvariableop_resource:}/
!batchnorm_readvariableop_resource:}
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
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

:}З
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         }l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ю
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
╫#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:}*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:}x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:}м
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
╫#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:}*
dtype0З
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
 *oГ:q
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
:         }ъ
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
Ф%
ь
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_131614

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:З
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ю
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:м
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
╫#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         ъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
д

№
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_132594

inputs1
matmul_readvariableop_resource:	Ї}-
biasadd_readvariableop_resource:}
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ї}*
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
:         Ї: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
й
╨
5__inference_BatchNormGeneEncode2_layer_call_fn_132627

inputs
unknown:}
	unknown_0:}
	unknown_1:}
	unknown_2:}
identityИвStatefulPartitionedCallВ
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
GPU2*0J 8В *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_131485o
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
н
╙
8__inference_BatchNormProteinEncode1_layer_call_fn_132720

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_131614o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╜Е
╠
!__inference__wrapped_model_131379
gene_input_layer
protein_input_layerI
5model_3_gene_encoder_1_matmul_readvariableop_resource:
╨ЇE
6model_3_gene_encoder_1_biasadd_readvariableop_resource:	ЇM
>model_3_batchnormgeneencode1_batchnorm_readvariableop_resource:	ЇQ
Bmodel_3_batchnormgeneencode1_batchnorm_mul_readvariableop_resource:	ЇO
@model_3_batchnormgeneencode1_batchnorm_readvariableop_1_resource:	ЇO
@model_3_batchnormgeneencode1_batchnorm_readvariableop_2_resource:	ЇJ
8model_3_protein_encoder_1_matmul_readvariableop_resource:G
9model_3_protein_encoder_1_biasadd_readvariableop_resource:H
5model_3_gene_encoder_2_matmul_readvariableop_resource:	Ї}D
6model_3_gene_encoder_2_biasadd_readvariableop_resource:}L
>model_3_batchnormgeneencode2_batchnorm_readvariableop_resource:}P
Bmodel_3_batchnormgeneencode2_batchnorm_mul_readvariableop_resource:}N
@model_3_batchnormgeneencode2_batchnorm_readvariableop_1_resource:}N
@model_3_batchnormgeneencode2_batchnorm_readvariableop_2_resource:}O
Amodel_3_batchnormproteinencode1_batchnorm_readvariableop_resource:S
Emodel_3_batchnormproteinencode1_batchnorm_mul_readvariableop_resource:Q
Cmodel_3_batchnormproteinencode1_batchnorm_readvariableop_1_resource:Q
Cmodel_3_batchnormproteinencode1_batchnorm_readvariableop_2_resource:K
8model_3_embeddingdimdense_matmul_readvariableop_resource:	Г@G
9model_3_embeddingdimdense_biasadd_readvariableop_resource:@
identityИв5model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOpв7model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1в7model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2в9model_3/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOpв5model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOpв7model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1в7model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2в9model_3/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOpв8model_3/BatchNormProteinEncode1/batchnorm/ReadVariableOpв:model_3/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1в:model_3/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2в<model_3/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOpв0model_3/EmbeddingDimDense/BiasAdd/ReadVariableOpв/model_3/EmbeddingDimDense/MatMul/ReadVariableOpв-model_3/gene_encoder_1/BiasAdd/ReadVariableOpв,model_3/gene_encoder_1/MatMul/ReadVariableOpв-model_3/gene_encoder_2/BiasAdd/ReadVariableOpв,model_3/gene_encoder_2/MatMul/ReadVariableOpв0model_3/protein_encoder_1/BiasAdd/ReadVariableOpв/model_3/protein_encoder_1/MatMul/ReadVariableOpд
,model_3/gene_encoder_1/MatMul/ReadVariableOpReadVariableOp5model_3_gene_encoder_1_matmul_readvariableop_resource* 
_output_shapes
:
╨Ї*
dtype0в
model_3/gene_encoder_1/MatMulMatMulgene_input_layer4model_3/gene_encoder_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Їб
-model_3/gene_encoder_1/BiasAdd/ReadVariableOpReadVariableOp6model_3_gene_encoder_1_biasadd_readvariableop_resource*
_output_shapes	
:Ї*
dtype0╝
model_3/gene_encoder_1/BiasAddBiasAdd'model_3/gene_encoder_1/MatMul:product:05model_3/gene_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ЇЕ
model_3/gene_encoder_1/SigmoidSigmoid'model_3/gene_encoder_1/BiasAdd:output:0*
T0*(
_output_shapes
:         Ї▒
5model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOpReadVariableOp>model_3_batchnormgeneencode1_batchnorm_readvariableop_resource*
_output_shapes	
:Ї*
dtype0q
,model_3/BatchNormGeneEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╧
*model_3/BatchNormGeneEncode1/batchnorm/addAddV2=model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp:value:05model_3/BatchNormGeneEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:ЇЛ
,model_3/BatchNormGeneEncode1/batchnorm/RsqrtRsqrt.model_3/BatchNormGeneEncode1/batchnorm/add:z:0*
T0*
_output_shapes	
:Ї╣
9model_3/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOpReadVariableOpBmodel_3_batchnormgeneencode1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ї*
dtype0╠
*model_3/BatchNormGeneEncode1/batchnorm/mulMul0model_3/BatchNormGeneEncode1/batchnorm/Rsqrt:y:0Amodel_3/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Ї║
,model_3/BatchNormGeneEncode1/batchnorm/mul_1Mul"model_3/gene_encoder_1/Sigmoid:y:0.model_3/BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*(
_output_shapes
:         Ї╡
7model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1ReadVariableOp@model_3_batchnormgeneencode1_batchnorm_readvariableop_1_resource*
_output_shapes	
:Ї*
dtype0╩
,model_3/BatchNormGeneEncode1/batchnorm/mul_2Mul?model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1:value:0.model_3/BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*
_output_shapes	
:Ї╡
7model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2ReadVariableOp@model_3_batchnormgeneencode1_batchnorm_readvariableop_2_resource*
_output_shapes	
:Ї*
dtype0╩
*model_3/BatchNormGeneEncode1/batchnorm/subSub?model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2:value:00model_3/BatchNormGeneEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ї╩
,model_3/BatchNormGeneEncode1/batchnorm/add_1AddV20model_3/BatchNormGeneEncode1/batchnorm/mul_1:z:0.model_3/BatchNormGeneEncode1/batchnorm/sub:z:0*
T0*(
_output_shapes
:         Їи
/model_3/protein_encoder_1/MatMul/ReadVariableOpReadVariableOp8model_3_protein_encoder_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0к
 model_3/protein_encoder_1/MatMulMatMulprotein_input_layer7model_3/protein_encoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ж
0model_3/protein_encoder_1/BiasAdd/ReadVariableOpReadVariableOp9model_3_protein_encoder_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0─
!model_3/protein_encoder_1/BiasAddBiasAdd*model_3/protein_encoder_1/MatMul:product:08model_3/protein_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         К
!model_3/protein_encoder_1/SigmoidSigmoid*model_3/protein_encoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:         г
,model_3/gene_encoder_2/MatMul/ReadVariableOpReadVariableOp5model_3_gene_encoder_2_matmul_readvariableop_resource*
_output_shapes
:	Ї}*
dtype0┴
model_3/gene_encoder_2/MatMulMatMul0model_3/BatchNormGeneEncode1/batchnorm/add_1:z:04model_3/gene_encoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         }а
-model_3/gene_encoder_2/BiasAdd/ReadVariableOpReadVariableOp6model_3_gene_encoder_2_biasadd_readvariableop_resource*
_output_shapes
:}*
dtype0╗
model_3/gene_encoder_2/BiasAddBiasAdd'model_3/gene_encoder_2/MatMul:product:05model_3/gene_encoder_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         }Д
model_3/gene_encoder_2/SigmoidSigmoid'model_3/gene_encoder_2/BiasAdd:output:0*
T0*'
_output_shapes
:         }░
5model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOpReadVariableOp>model_3_batchnormgeneencode2_batchnorm_readvariableop_resource*
_output_shapes
:}*
dtype0q
,model_3/BatchNormGeneEncode2/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╬
*model_3/BatchNormGeneEncode2/batchnorm/addAddV2=model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp:value:05model_3/BatchNormGeneEncode2/batchnorm/add/y:output:0*
T0*
_output_shapes
:}К
,model_3/BatchNormGeneEncode2/batchnorm/RsqrtRsqrt.model_3/BatchNormGeneEncode2/batchnorm/add:z:0*
T0*
_output_shapes
:}╕
9model_3/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOpReadVariableOpBmodel_3_batchnormgeneencode2_batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0╦
*model_3/BatchNormGeneEncode2/batchnorm/mulMul0model_3/BatchNormGeneEncode2/batchnorm/Rsqrt:y:0Amodel_3/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}╣
,model_3/BatchNormGeneEncode2/batchnorm/mul_1Mul"model_3/gene_encoder_2/Sigmoid:y:0.model_3/BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         }┤
7model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1ReadVariableOp@model_3_batchnormgeneencode2_batchnorm_readvariableop_1_resource*
_output_shapes
:}*
dtype0╔
,model_3/BatchNormGeneEncode2/batchnorm/mul_2Mul?model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1:value:0.model_3/BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*
_output_shapes
:}┤
7model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2ReadVariableOp@model_3_batchnormgeneencode2_batchnorm_readvariableop_2_resource*
_output_shapes
:}*
dtype0╔
*model_3/BatchNormGeneEncode2/batchnorm/subSub?model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2:value:00model_3/BatchNormGeneEncode2/batchnorm/mul_2:z:0*
T0*
_output_shapes
:}╔
,model_3/BatchNormGeneEncode2/batchnorm/add_1AddV20model_3/BatchNormGeneEncode2/batchnorm/mul_1:z:0.model_3/BatchNormGeneEncode2/batchnorm/sub:z:0*
T0*'
_output_shapes
:         }╢
8model_3/BatchNormProteinEncode1/batchnorm/ReadVariableOpReadVariableOpAmodel_3_batchnormproteinencode1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0t
/model_3/BatchNormProteinEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╫
-model_3/BatchNormProteinEncode1/batchnorm/addAddV2@model_3/BatchNormProteinEncode1/batchnorm/ReadVariableOp:value:08model_3/BatchNormProteinEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:Р
/model_3/BatchNormProteinEncode1/batchnorm/RsqrtRsqrt1model_3/BatchNormProteinEncode1/batchnorm/add:z:0*
T0*
_output_shapes
:╛
<model_3/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOpReadVariableOpEmodel_3_batchnormproteinencode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╘
-model_3/BatchNormProteinEncode1/batchnorm/mulMul3model_3/BatchNormProteinEncode1/batchnorm/Rsqrt:y:0Dmodel_3/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:┬
/model_3/BatchNormProteinEncode1/batchnorm/mul_1Mul%model_3/protein_encoder_1/Sigmoid:y:01model_3/BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:         ║
:model_3/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1ReadVariableOpCmodel_3_batchnormproteinencode1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0╥
/model_3/BatchNormProteinEncode1/batchnorm/mul_2MulBmodel_3/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1:value:01model_3/BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*
_output_shapes
:║
:model_3/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2ReadVariableOpCmodel_3_batchnormproteinencode1_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0╥
-model_3/BatchNormProteinEncode1/batchnorm/subSubBmodel_3/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2:value:03model_3/BatchNormProteinEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:╥
/model_3/BatchNormProteinEncode1/batchnorm/add_1AddV23model_3/BatchNormProteinEncode1/batchnorm/mul_1:z:01model_3/BatchNormProteinEncode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:         f
$model_3/ConcatenateLayer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :¤
model_3/ConcatenateLayer/concatConcatV20model_3/BatchNormGeneEncode2/batchnorm/add_1:z:03model_3/BatchNormProteinEncode1/batchnorm/add_1:z:0-model_3/ConcatenateLayer/concat/axis:output:0*
N*
T0*(
_output_shapes
:         Гй
/model_3/EmbeddingDimDense/MatMul/ReadVariableOpReadVariableOp8model_3_embeddingdimdense_matmul_readvariableop_resource*
_output_shapes
:	Г@*
dtype0┐
 model_3/EmbeddingDimDense/MatMulMatMul(model_3/ConcatenateLayer/concat:output:07model_3/EmbeddingDimDense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @ж
0model_3/EmbeddingDimDense/BiasAdd/ReadVariableOpReadVariableOp9model_3_embeddingdimdense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0─
!model_3/EmbeddingDimDense/BiasAddBiasAdd*model_3/EmbeddingDimDense/MatMul:product:08model_3/EmbeddingDimDense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @К
!model_3/EmbeddingDimDense/SigmoidSigmoid*model_3/EmbeddingDimDense/BiasAdd:output:0*
T0*'
_output_shapes
:         @t
IdentityIdentity%model_3/EmbeddingDimDense/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:         @Т	
NoOpNoOp6^model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp8^model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_18^model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2:^model_3/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp6^model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp8^model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_18^model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2:^model_3/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp9^model_3/BatchNormProteinEncode1/batchnorm/ReadVariableOp;^model_3/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1;^model_3/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2=^model_3/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp1^model_3/EmbeddingDimDense/BiasAdd/ReadVariableOp0^model_3/EmbeddingDimDense/MatMul/ReadVariableOp.^model_3/gene_encoder_1/BiasAdd/ReadVariableOp-^model_3/gene_encoder_1/MatMul/ReadVariableOp.^model_3/gene_encoder_2/BiasAdd/ReadVariableOp-^model_3/gene_encoder_2/MatMul/ReadVariableOp1^model_3/protein_encoder_1/BiasAdd/ReadVariableOp0^model_3/protein_encoder_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         ╨:         : : : : : : : : : : : : : : : : : : : : 2n
5model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp5model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp2r
7model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_17model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_12r
7model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_27model_3/BatchNormGeneEncode1/batchnorm/ReadVariableOp_22v
9model_3/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp9model_3/BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp2n
5model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp5model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp2r
7model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_17model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_12r
7model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_27model_3/BatchNormGeneEncode2/batchnorm/ReadVariableOp_22v
9model_3/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp9model_3/BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp2t
8model_3/BatchNormProteinEncode1/batchnorm/ReadVariableOp8model_3/BatchNormProteinEncode1/batchnorm/ReadVariableOp2x
:model_3/BatchNormProteinEncode1/batchnorm/ReadVariableOp_1:model_3/BatchNormProteinEncode1/batchnorm/ReadVariableOp_12x
:model_3/BatchNormProteinEncode1/batchnorm/ReadVariableOp_2:model_3/BatchNormProteinEncode1/batchnorm/ReadVariableOp_22|
<model_3/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp<model_3/BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp2d
0model_3/EmbeddingDimDense/BiasAdd/ReadVariableOp0model_3/EmbeddingDimDense/BiasAdd/ReadVariableOp2b
/model_3/EmbeddingDimDense/MatMul/ReadVariableOp/model_3/EmbeddingDimDense/MatMul/ReadVariableOp2^
-model_3/gene_encoder_1/BiasAdd/ReadVariableOp-model_3/gene_encoder_1/BiasAdd/ReadVariableOp2\
,model_3/gene_encoder_1/MatMul/ReadVariableOp,model_3/gene_encoder_1/MatMul/ReadVariableOp2^
-model_3/gene_encoder_2/BiasAdd/ReadVariableOp-model_3/gene_encoder_2/BiasAdd/ReadVariableOp2\
,model_3/gene_encoder_2/MatMul/ReadVariableOp,model_3/gene_encoder_2/MatMul/ReadVariableOp2d
0model_3/protein_encoder_1/BiasAdd/ReadVariableOp0model_3/protein_encoder_1/BiasAdd/ReadVariableOp2b
/model_3/protein_encoder_1/MatMul/ReadVariableOp/model_3/protein_encoder_1/MatMul/ReadVariableOp:Z V
(
_output_shapes
:         ╨
*
_user_specified_nameGene_Input_Layer:\X
'
_output_shapes
:         
-
_user_specified_nameProtein_Input_Layer
╟[
х
"__inference__traced_restore_132973
file_prefix:
&assignvariableop_gene_encoder_1_kernel:
╨Ї5
&assignvariableop_1_gene_encoder_1_bias:	Ї<
-assignvariableop_2_batchnormgeneencode1_gamma:	Ї;
,assignvariableop_3_batchnormgeneencode1_beta:	ЇB
3assignvariableop_4_batchnormgeneencode1_moving_mean:	ЇF
7assignvariableop_5_batchnormgeneencode1_moving_variance:	Ї;
(assignvariableop_6_gene_encoder_2_kernel:	Ї}4
&assignvariableop_7_gene_encoder_2_bias:}=
+assignvariableop_8_protein_encoder_1_kernel:7
)assignvariableop_9_protein_encoder_1_bias:<
.assignvariableop_10_batchnormgeneencode2_gamma:};
-assignvariableop_11_batchnormgeneencode2_beta:}B
4assignvariableop_12_batchnormgeneencode2_moving_mean:}F
8assignvariableop_13_batchnormgeneencode2_moving_variance:}?
1assignvariableop_14_batchnormproteinencode1_gamma:>
0assignvariableop_15_batchnormproteinencode1_beta:E
7assignvariableop_16_batchnormproteinencode1_moving_mean:I
;assignvariableop_17_batchnormproteinencode1_moving_variance:?
,assignvariableop_18_embeddingdimdense_kernel:	Г@8
*assignvariableop_19_embeddingdimdense_bias:@#
assignvariableop_20_total: #
assignvariableop_21_count: 
identity_23ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9Є

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ш

valueО
BЛ
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHЮ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*A
value8B6B B B B B B B B B B B B B B B B B B B B B B B С
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*p
_output_shapes^
\:::::::::::::::::::::::*%
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOpAssignVariableOp&assignvariableop_gene_encoder_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_1AssignVariableOp&assignvariableop_1_gene_encoder_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_2AssignVariableOp-assignvariableop_2_batchnormgeneencode1_gammaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_3AssignVariableOp,assignvariableop_3_batchnormgeneencode1_betaIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_4AssignVariableOp3assignvariableop_4_batchnormgeneencode1_moving_meanIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:ж
AssignVariableOp_5AssignVariableOp7assignvariableop_5_batchnormgeneencode1_moving_varianceIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_6AssignVariableOp(assignvariableop_6_gene_encoder_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_7AssignVariableOp&assignvariableop_7_gene_encoder_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_8AssignVariableOp+assignvariableop_8_protein_encoder_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_9AssignVariableOp)assignvariableop_9_protein_encoder_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Я
AssignVariableOp_10AssignVariableOp.assignvariableop_10_batchnormgeneencode2_gammaIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_11AssignVariableOp-assignvariableop_11_batchnormgeneencode2_betaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:е
AssignVariableOp_12AssignVariableOp4assignvariableop_12_batchnormgeneencode2_moving_meanIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_13AssignVariableOp8assignvariableop_13_batchnormgeneencode2_moving_varianceIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:в
AssignVariableOp_14AssignVariableOp1assignvariableop_14_batchnormproteinencode1_gammaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:б
AssignVariableOp_15AssignVariableOp0assignvariableop_15_batchnormproteinencode1_betaIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:и
AssignVariableOp_16AssignVariableOp7assignvariableop_16_batchnormproteinencode1_moving_meanIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:м
AssignVariableOp_17AssignVariableOp;assignvariableop_17_batchnormproteinencode1_moving_varianceIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_18AssignVariableOp,assignvariableop_18_embeddingdimdense_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_19AssignVariableOp*assignvariableop_19_embeddingdimdense_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:К
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
╫
Я
2__inference_protein_encoder_1_layer_call_fn_132603

inputs
unknown:
	unknown_0:
identityИвStatefulPartitionedCallх
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_131671o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╨
▓
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_132740

inputs/
!batchnorm_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:1
#batchnorm_readvariableop_1_resource:1
#batchnorm_readvariableop_2_resource:
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpv
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:w
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         z
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0r
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes
:z
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0r
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         ║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
з

 
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_132807

inputs1
matmul_readvariableop_resource:	Г@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Г@*
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
:         Г: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         Г
 
_user_specified_nameinputs
╟
x
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_132787
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
:         ГX
IdentityIdentityconcat:output:0*
T0*(
_output_shapes
:         Г"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         }:         :Q M
'
_output_shapes
:         }
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
о%
э
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_131450

inputs6
'assignmovingavg_readvariableop_resource:	Ї8
)assignmovingavg_1_readvariableop_resource:	Ї4
%batchnorm_mul_readvariableop_resource:	Ї0
!batchnorm_readvariableop_resource:	Ї
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: А
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Ї*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ЇИ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         Їl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Ї*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:Ї*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:Ї*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<Г
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:Ї*
dtype0В
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:Їy
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:Їм
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
╫#<З
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:Ї*
dtype0И
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ї
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:Ї┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ЇQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Ї
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ї*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Їd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         Їi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Їw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Ї*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Їs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         Їc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         Їъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         Ї: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
▀
│
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_132540

inputs0
!batchnorm_readvariableop_resource:	Ї4
%batchnorm_mul_readvariableop_resource:	Ї2
#batchnorm_readvariableop_1_resource:	Ї2
#batchnorm_readvariableop_2_resource:	Ї
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Ї*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ЇQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Ї
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ї*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Їd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         Ї{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:Ї*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Ї{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:Ї*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Їs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         Їc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         Ї║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         Ї: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
Ё2
З

C__inference_model_3_layer_call_and_return_conditional_losses_131739

inputs
inputs_1)
gene_encoder_1_131646:
╨Ї$
gene_encoder_1_131648:	Ї*
batchnormgeneencode1_131651:	Ї*
batchnormgeneencode1_131653:	Ї*
batchnormgeneencode1_131655:	Ї*
batchnormgeneencode1_131657:	Ї*
protein_encoder_1_131672:&
protein_encoder_1_131674:(
gene_encoder_2_131689:	Ї}#
gene_encoder_2_131691:})
batchnormgeneencode2_131694:})
batchnormgeneencode2_131696:})
batchnormgeneencode2_131698:})
batchnormgeneencode2_131700:},
batchnormproteinencode1_131703:,
batchnormproteinencode1_131705:,
batchnormproteinencode1_131707:,
batchnormproteinencode1_131709:+
embeddingdimdense_131733:	Г@&
embeddingdimdense_131735:@
identityИв,BatchNormGeneEncode1/StatefulPartitionedCallв,BatchNormGeneEncode2/StatefulPartitionedCallв/BatchNormProteinEncode1/StatefulPartitionedCallв)EmbeddingDimDense/StatefulPartitionedCallв&gene_encoder_1/StatefulPartitionedCallв&gene_encoder_2/StatefulPartitionedCallв)protein_encoder_1/StatefulPartitionedCallМ
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputsgene_encoder_1_131646gene_encoder_1_131648*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_131645Л
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_131651batchnormgeneencode1_131653batchnormgeneencode1_131655batchnormgeneencode1_131657*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_131403Щ
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1protein_encoder_1_131672protein_encoder_1_131674*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_131671║
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_131689gene_encoder_2_131691*
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
GPU2*0J 8В *S
fNRL
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_131688К
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_131694batchnormgeneencode2_131696batchnormgeneencode2_131698batchnormgeneencode2_131700*
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
GPU2*0J 8В *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_131485Я
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_131703batchnormproteinencode1_131705batchnormproteinencode1_131707batchnormproteinencode1_131709*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_131567╢
 ConcatenateLayer/PartitionedCallPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:08BatchNormProteinEncode1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Г* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_131719║
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_131733embeddingdimdense_131735*
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
GPU2*0J 8В *V
fQRO
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_131732Б
IdentityIdentity2EmbeddingDimDense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @А
NoOpNoOp-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         ╨:         : : : : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2b
/BatchNormProteinEncode1/StatefulPartitionedCall/BatchNormProteinEncode1/StatefulPartitionedCall2V
)EmbeddingDimDense/StatefulPartitionedCall)EmbeddingDimDense/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall2V
)protein_encoder_1/StatefulPartitionedCall)protein_encoder_1/StatefulPartitionedCall:P L
(
_output_shapes
:         ╨
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs
о%
э
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_132574

inputs6
'assignmovingavg_readvariableop_resource:	Ї8
)assignmovingavg_1_readvariableop_resource:	Ї4
%batchnorm_mul_readvariableop_resource:	Ї0
!batchnorm_readvariableop_resource:	Ї
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: А
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes
:	Ї*
	keep_dims(e
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes
:	ЇИ
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*(
_output_shapes
:         Їl
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Я
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes
:	Ї*
	keep_dims(n
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes	
:Ї*
squeeze_dims
 t
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes	
:Ї*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<Г
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes	
:Ї*
dtype0В
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes	
:Їy
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes	
:Їм
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
╫#<З
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes	
:Ї*
dtype0И
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes	
:Ї
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes	
:Ї┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:r
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ЇQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Ї
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ї*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Їd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         Їi
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Їw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Ї*
dtype0q
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Їs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         Їc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         Їъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         Ї: : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
з
╨
5__inference_BatchNormGeneEncode2_layer_call_fn_132640

inputs
unknown:}
	unknown_0:}
	unknown_1:}
	unknown_2:}
identityИвStatefulPartitionedCallА
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
GPU2*0J 8В *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_131532o
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
д

№
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_131688

inputs1
matmul_readvariableop_resource:	Ї}-
biasadd_readvariableop_resource:}
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	Ї}*
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
:         Ї: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
▒
╘
5__inference_BatchNormGeneEncode1_layer_call_fn_132507

inputs
unknown:	Ї
	unknown_0:	Ї
	unknown_1:	Ї
	unknown_2:	Ї
identityИвStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_131403p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         Ї`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         Ї: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
Ф%
ь
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_132774

inputs5
'assignmovingavg_readvariableop_resource:7
)assignmovingavg_1_readvariableop_resource:3
%batchnorm_mul_readvariableop_resource:/
!batchnorm_readvariableop_resource:
identityИвAssignMovingAvgвAssignMovingAvg/ReadVariableOpвAssignMovingAvg_1в AssignMovingAvg_1/ReadVariableOpвbatchnorm/ReadVariableOpвbatchnorm/mul/ReadVariableOph
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 
moments/meanMeaninputs'moments/mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(d
moments/StopGradientStopGradientmoments/mean:output:0*
T0*
_output_shapes

:З
moments/SquaredDifferenceSquaredDifferenceinputsmoments/StopGradient:output:0*
T0*'
_output_shapes
:         l
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: Ю
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(m
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 s
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 Z
AssignMovingAvg/decayConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<В
AssignMovingAvg/ReadVariableOpReadVariableOp'assignmovingavg_readvariableop_resource*
_output_shapes
:*
dtype0Б
AssignMovingAvg/subSub&AssignMovingAvg/ReadVariableOp:value:0moments/Squeeze:output:0*
T0*
_output_shapes
:x
AssignMovingAvg/mulMulAssignMovingAvg/sub:z:0AssignMovingAvg/decay:output:0*
T0*
_output_shapes
:м
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
╫#<Ж
 AssignMovingAvg_1/ReadVariableOpReadVariableOp)assignmovingavg_1_readvariableop_resource*
_output_shapes
:*
dtype0З
AssignMovingAvg_1/subSub(AssignMovingAvg_1/ReadVariableOp:value:0moments/Squeeze_1:output:0*
T0*
_output_shapes
:~
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub:z:0 AssignMovingAvg_1/decay:output:0*
T0*
_output_shapes
:┤
AssignMovingAvg_1AssignSubVariableOp)assignmovingavg_1_readvariableop_resourceAssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp*
_output_shapes
 *
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:q
batchnorm/addAddV2moments/Squeeze_1:output:0batchnorm/add/y:output:0*
T0*
_output_shapes
:P
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes
:~
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0t
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:c
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*'
_output_shapes
:         h
batchnorm/mul_2Mulmoments/Squeeze:output:0batchnorm/mul:z:0*
T0*
_output_shapes
:v
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0p
batchnorm/subSub batchnorm/ReadVariableOp:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes
:r
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*'
_output_shapes
:         b
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*'
_output_shapes
:         ъ
NoOpNoOp^AssignMovingAvg^AssignMovingAvg/ReadVariableOp^AssignMovingAvg_1!^AssignMovingAvg_1/ReadVariableOp^batchnorm/ReadVariableOp^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 2"
AssignMovingAvgAssignMovingAvg2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2&
AssignMovingAvg_1AssignMovingAvg_12D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp2<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
п
╙
8__inference_BatchNormProteinEncode1_layer_call_fn_132707

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identityИвStatefulPartitionedCallЕ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_131567o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
л3
Ь

C__inference_model_3_layer_call_and_return_conditional_losses_132126
gene_input_layer
protein_input_layer)
gene_encoder_1_132077:
╨Ї$
gene_encoder_1_132079:	Ї*
batchnormgeneencode1_132082:	Ї*
batchnormgeneencode1_132084:	Ї*
batchnormgeneencode1_132086:	Ї*
batchnormgeneencode1_132088:	Ї*
protein_encoder_1_132091:&
protein_encoder_1_132093:(
gene_encoder_2_132096:	Ї}#
gene_encoder_2_132098:})
batchnormgeneencode2_132101:})
batchnormgeneencode2_132103:})
batchnormgeneencode2_132105:})
batchnormgeneencode2_132107:},
batchnormproteinencode1_132110:,
batchnormproteinencode1_132112:,
batchnormproteinencode1_132114:,
batchnormproteinencode1_132116:+
embeddingdimdense_132120:	Г@&
embeddingdimdense_132122:@
identityИв,BatchNormGeneEncode1/StatefulPartitionedCallв,BatchNormGeneEncode2/StatefulPartitionedCallв/BatchNormProteinEncode1/StatefulPartitionedCallв)EmbeddingDimDense/StatefulPartitionedCallв&gene_encoder_1/StatefulPartitionedCallв&gene_encoder_2/StatefulPartitionedCallв)protein_encoder_1/StatefulPartitionedCallЦ
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallgene_input_layergene_encoder_1_132077gene_encoder_1_132079*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_131645Й
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_132082batchnormgeneencode1_132084batchnormgeneencode1_132086batchnormgeneencode1_132088*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_131450д
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallprotein_input_layerprotein_encoder_1_132091protein_encoder_1_132093*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_131671║
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_132096gene_encoder_2_132098*
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
GPU2*0J 8В *S
fNRL
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_131688И
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_132101batchnormgeneencode2_132103batchnormgeneencode2_132105batchnormgeneencode2_132107*
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
GPU2*0J 8В *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_131532Э
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_132110batchnormproteinencode1_132112batchnormproteinencode1_132114batchnormproteinencode1_132116*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_131614╢
 ConcatenateLayer/PartitionedCallPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:08BatchNormProteinEncode1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Г* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_131719║
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_132120embeddingdimdense_132122*
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
GPU2*0J 8В *V
fQRO
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_131732Б
IdentityIdentity2EmbeddingDimDense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @А
NoOpNoOp-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         ╨:         : : : : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2b
/BatchNormProteinEncode1/StatefulPartitionedCall/BatchNormProteinEncode1/StatefulPartitionedCall2V
)EmbeddingDimDense/StatefulPartitionedCall)EmbeddingDimDense/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall2V
)protein_encoder_1/StatefulPartitionedCall)protein_encoder_1/StatefulPartitionedCall:Z V
(
_output_shapes
:         ╨
*
_user_specified_nameGene_Input_Layer:\X
'
_output_shapes
:         
-
_user_specified_nameProtein_Input_Layer
·
Ш
(__inference_model_3_layer_call_fn_131782
gene_input_layer
protein_input_layer
unknown:
╨Ї
	unknown_0:	Ї
	unknown_1:	Ї
	unknown_2:	Ї
	unknown_3:	Ї
	unknown_4:	Ї
	unknown_5:
	unknown_6:
	unknown_7:	Ї}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:	Г@

unknown_18:@
identityИвStatefulPartitionedCallю
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
GPU2*0J 8В *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_131739o
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
O:         ╨:         : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:         ╨
*
_user_specified_nameGene_Input_Layer:\X
'
_output_shapes
:         
-
_user_specified_nameProtein_Input_Layer
п
╘
5__inference_BatchNormGeneEncode1_layer_call_fn_132520

inputs
unknown:	Ї
	unknown_0:	Ї
	unknown_1:	Ї
	unknown_2:	Ї
identityИвStatefulPartitionedCallБ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_131450p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         Ї`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         Ї: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
иv
Ы
C__inference_model_3_layer_call_and_return_conditional_losses_132301
inputs_0
inputs_1A
-gene_encoder_1_matmul_readvariableop_resource:
╨Ї=
.gene_encoder_1_biasadd_readvariableop_resource:	ЇE
6batchnormgeneencode1_batchnorm_readvariableop_resource:	ЇI
:batchnormgeneencode1_batchnorm_mul_readvariableop_resource:	ЇG
8batchnormgeneencode1_batchnorm_readvariableop_1_resource:	ЇG
8batchnormgeneencode1_batchnorm_readvariableop_2_resource:	ЇB
0protein_encoder_1_matmul_readvariableop_resource:?
1protein_encoder_1_biasadd_readvariableop_resource:@
-gene_encoder_2_matmul_readvariableop_resource:	Ї}<
.gene_encoder_2_biasadd_readvariableop_resource:}D
6batchnormgeneencode2_batchnorm_readvariableop_resource:}H
:batchnormgeneencode2_batchnorm_mul_readvariableop_resource:}F
8batchnormgeneencode2_batchnorm_readvariableop_1_resource:}F
8batchnormgeneencode2_batchnorm_readvariableop_2_resource:}G
9batchnormproteinencode1_batchnorm_readvariableop_resource:K
=batchnormproteinencode1_batchnorm_mul_readvariableop_resource:I
;batchnormproteinencode1_batchnorm_readvariableop_1_resource:I
;batchnormproteinencode1_batchnorm_readvariableop_2_resource:C
0embeddingdimdense_matmul_readvariableop_resource:	Г@?
1embeddingdimdense_biasadd_readvariableop_resource:@
identityИв-BatchNormGeneEncode1/batchnorm/ReadVariableOpв/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1в/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2в1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOpв-BatchNormGeneEncode2/batchnorm/ReadVariableOpв/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1в/BatchNormGeneEncode2/batchnorm/ReadVariableOp_2в1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOpв0BatchNormProteinEncode1/batchnorm/ReadVariableOpв2BatchNormProteinEncode1/batchnorm/ReadVariableOp_1в2BatchNormProteinEncode1/batchnorm/ReadVariableOp_2в4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOpв(EmbeddingDimDense/BiasAdd/ReadVariableOpв'EmbeddingDimDense/MatMul/ReadVariableOpв%gene_encoder_1/BiasAdd/ReadVariableOpв$gene_encoder_1/MatMul/ReadVariableOpв%gene_encoder_2/BiasAdd/ReadVariableOpв$gene_encoder_2/MatMul/ReadVariableOpв(protein_encoder_1/BiasAdd/ReadVariableOpв'protein_encoder_1/MatMul/ReadVariableOpФ
$gene_encoder_1/MatMul/ReadVariableOpReadVariableOp-gene_encoder_1_matmul_readvariableop_resource* 
_output_shapes
:
╨Ї*
dtype0К
gene_encoder_1/MatMulMatMulinputs_0,gene_encoder_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ЇС
%gene_encoder_1/BiasAdd/ReadVariableOpReadVariableOp.gene_encoder_1_biasadd_readvariableop_resource*
_output_shapes	
:Ї*
dtype0д
gene_encoder_1/BiasAddBiasAddgene_encoder_1/MatMul:product:0-gene_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Їu
gene_encoder_1/SigmoidSigmoidgene_encoder_1/BiasAdd:output:0*
T0*(
_output_shapes
:         Їб
-BatchNormGeneEncode1/batchnorm/ReadVariableOpReadVariableOp6batchnormgeneencode1_batchnorm_readvariableop_resource*
_output_shapes	
:Ї*
dtype0i
$BatchNormGeneEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:╖
"BatchNormGeneEncode1/batchnorm/addAddV25BatchNormGeneEncode1/batchnorm/ReadVariableOp:value:0-BatchNormGeneEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes	
:Ї{
$BatchNormGeneEncode1/batchnorm/RsqrtRsqrt&BatchNormGeneEncode1/batchnorm/add:z:0*
T0*
_output_shapes	
:Їй
1BatchNormGeneEncode1/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgeneencode1_batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ї*
dtype0┤
"BatchNormGeneEncode1/batchnorm/mulMul(BatchNormGeneEncode1/batchnorm/Rsqrt:y:09BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Їв
$BatchNormGeneEncode1/batchnorm/mul_1Mulgene_encoder_1/Sigmoid:y:0&BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*(
_output_shapes
:         Їе
/BatchNormGeneEncode1/batchnorm/ReadVariableOp_1ReadVariableOp8batchnormgeneencode1_batchnorm_readvariableop_1_resource*
_output_shapes	
:Ї*
dtype0▓
$BatchNormGeneEncode1/batchnorm/mul_2Mul7BatchNormGeneEncode1/batchnorm/ReadVariableOp_1:value:0&BatchNormGeneEncode1/batchnorm/mul:z:0*
T0*
_output_shapes	
:Їе
/BatchNormGeneEncode1/batchnorm/ReadVariableOp_2ReadVariableOp8batchnormgeneencode1_batchnorm_readvariableop_2_resource*
_output_shapes	
:Ї*
dtype0▓
"BatchNormGeneEncode1/batchnorm/subSub7BatchNormGeneEncode1/batchnorm/ReadVariableOp_2:value:0(BatchNormGeneEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Ї▓
$BatchNormGeneEncode1/batchnorm/add_1AddV2(BatchNormGeneEncode1/batchnorm/mul_1:z:0&BatchNormGeneEncode1/batchnorm/sub:z:0*
T0*(
_output_shapes
:         ЇШ
'protein_encoder_1/MatMul/ReadVariableOpReadVariableOp0protein_encoder_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0П
protein_encoder_1/MatMulMatMulinputs_1/protein_encoder_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ц
(protein_encoder_1/BiasAdd/ReadVariableOpReadVariableOp1protein_encoder_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0м
protein_encoder_1/BiasAddBiasAdd"protein_encoder_1/MatMul:product:00protein_encoder_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         z
protein_encoder_1/SigmoidSigmoid"protein_encoder_1/BiasAdd:output:0*
T0*'
_output_shapes
:         У
$gene_encoder_2/MatMul/ReadVariableOpReadVariableOp-gene_encoder_2_matmul_readvariableop_resource*
_output_shapes
:	Ї}*
dtype0й
gene_encoder_2/MatMulMatMul(BatchNormGeneEncode1/batchnorm/add_1:z:0,gene_encoder_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         }Р
%gene_encoder_2/BiasAdd/ReadVariableOpReadVariableOp.gene_encoder_2_biasadd_readvariableop_resource*
_output_shapes
:}*
dtype0г
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
 *oГ:╢
"BatchNormGeneEncode2/batchnorm/addAddV25BatchNormGeneEncode2/batchnorm/ReadVariableOp:value:0-BatchNormGeneEncode2/batchnorm/add/y:output:0*
T0*
_output_shapes
:}z
$BatchNormGeneEncode2/batchnorm/RsqrtRsqrt&BatchNormGeneEncode2/batchnorm/add:z:0*
T0*
_output_shapes
:}и
1BatchNormGeneEncode2/batchnorm/mul/ReadVariableOpReadVariableOp:batchnormgeneencode2_batchnorm_mul_readvariableop_resource*
_output_shapes
:}*
dtype0│
"BatchNormGeneEncode2/batchnorm/mulMul(BatchNormGeneEncode2/batchnorm/Rsqrt:y:09BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:}б
$BatchNormGeneEncode2/batchnorm/mul_1Mulgene_encoder_2/Sigmoid:y:0&BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*'
_output_shapes
:         }д
/BatchNormGeneEncode2/batchnorm/ReadVariableOp_1ReadVariableOp8batchnormgeneencode2_batchnorm_readvariableop_1_resource*
_output_shapes
:}*
dtype0▒
$BatchNormGeneEncode2/batchnorm/mul_2Mul7BatchNormGeneEncode2/batchnorm/ReadVariableOp_1:value:0&BatchNormGeneEncode2/batchnorm/mul:z:0*
T0*
_output_shapes
:}д
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
:         }ж
0BatchNormProteinEncode1/batchnorm/ReadVariableOpReadVariableOp9batchnormproteinencode1_batchnorm_readvariableop_resource*
_output_shapes
:*
dtype0l
'BatchNormProteinEncode1/batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:┐
%BatchNormProteinEncode1/batchnorm/addAddV28BatchNormProteinEncode1/batchnorm/ReadVariableOp:value:00BatchNormProteinEncode1/batchnorm/add/y:output:0*
T0*
_output_shapes
:А
'BatchNormProteinEncode1/batchnorm/RsqrtRsqrt)BatchNormProteinEncode1/batchnorm/add:z:0*
T0*
_output_shapes
:о
4BatchNormProteinEncode1/batchnorm/mul/ReadVariableOpReadVariableOp=batchnormproteinencode1_batchnorm_mul_readvariableop_resource*
_output_shapes
:*
dtype0╝
%BatchNormProteinEncode1/batchnorm/mulMul+BatchNormProteinEncode1/batchnorm/Rsqrt:y:0<BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes
:к
'BatchNormProteinEncode1/batchnorm/mul_1Mulprotein_encoder_1/Sigmoid:y:0)BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*'
_output_shapes
:         к
2BatchNormProteinEncode1/batchnorm/ReadVariableOp_1ReadVariableOp;batchnormproteinencode1_batchnorm_readvariableop_1_resource*
_output_shapes
:*
dtype0║
'BatchNormProteinEncode1/batchnorm/mul_2Mul:BatchNormProteinEncode1/batchnorm/ReadVariableOp_1:value:0)BatchNormProteinEncode1/batchnorm/mul:z:0*
T0*
_output_shapes
:к
2BatchNormProteinEncode1/batchnorm/ReadVariableOp_2ReadVariableOp;batchnormproteinencode1_batchnorm_readvariableop_2_resource*
_output_shapes
:*
dtype0║
%BatchNormProteinEncode1/batchnorm/subSub:BatchNormProteinEncode1/batchnorm/ReadVariableOp_2:value:0+BatchNormProteinEncode1/batchnorm/mul_2:z:0*
T0*
_output_shapes
:║
'BatchNormProteinEncode1/batchnorm/add_1AddV2+BatchNormProteinEncode1/batchnorm/mul_1:z:0)BatchNormProteinEncode1/batchnorm/sub:z:0*
T0*'
_output_shapes
:         ^
ConcatenateLayer/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :▌
ConcatenateLayer/concatConcatV2(BatchNormGeneEncode2/batchnorm/add_1:z:0+BatchNormProteinEncode1/batchnorm/add_1:z:0%ConcatenateLayer/concat/axis:output:0*
N*
T0*(
_output_shapes
:         ГЩ
'EmbeddingDimDense/MatMul/ReadVariableOpReadVariableOp0embeddingdimdense_matmul_readvariableop_resource*
_output_shapes
:	Г@*
dtype0з
EmbeddingDimDense/MatMulMatMul ConcatenateLayer/concat:output:0/EmbeddingDimDense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @Ц
(EmbeddingDimDense/BiasAdd/ReadVariableOpReadVariableOp1embeddingdimdense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0м
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
:         @Є
NoOpNoOp.^BatchNormGeneEncode1/batchnorm/ReadVariableOp0^BatchNormGeneEncode1/batchnorm/ReadVariableOp_10^BatchNormGeneEncode1/batchnorm/ReadVariableOp_22^BatchNormGeneEncode1/batchnorm/mul/ReadVariableOp.^BatchNormGeneEncode2/batchnorm/ReadVariableOp0^BatchNormGeneEncode2/batchnorm/ReadVariableOp_10^BatchNormGeneEncode2/batchnorm/ReadVariableOp_22^BatchNormGeneEncode2/batchnorm/mul/ReadVariableOp1^BatchNormProteinEncode1/batchnorm/ReadVariableOp3^BatchNormProteinEncode1/batchnorm/ReadVariableOp_13^BatchNormProteinEncode1/batchnorm/ReadVariableOp_25^BatchNormProteinEncode1/batchnorm/mul/ReadVariableOp)^EmbeddingDimDense/BiasAdd/ReadVariableOp(^EmbeddingDimDense/MatMul/ReadVariableOp&^gene_encoder_1/BiasAdd/ReadVariableOp%^gene_encoder_1/MatMul/ReadVariableOp&^gene_encoder_2/BiasAdd/ReadVariableOp%^gene_encoder_2/MatMul/ReadVariableOp)^protein_encoder_1/BiasAdd/ReadVariableOp(^protein_encoder_1/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         ╨:         : : : : : : : : : : : : : : : : : : : : 2^
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
:         ╨
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:         
"
_user_specified_name
inputs/1
▀
│
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_131403

inputs0
!batchnorm_readvariableop_resource:	Ї4
%batchnorm_mul_readvariableop_resource:	Ї2
#batchnorm_readvariableop_1_resource:	Ї2
#batchnorm_readvariableop_2_resource:	Ї
identityИвbatchnorm/ReadVariableOpвbatchnorm/ReadVariableOp_1вbatchnorm/ReadVariableOp_2вbatchnorm/mul/ReadVariableOpw
batchnorm/ReadVariableOpReadVariableOp!batchnorm_readvariableop_resource*
_output_shapes	
:Ї*
dtype0T
batchnorm/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *oГ:x
batchnorm/addAddV2 batchnorm/ReadVariableOp:value:0batchnorm/add/y:output:0*
T0*
_output_shapes	
:ЇQ
batchnorm/RsqrtRsqrtbatchnorm/add:z:0*
T0*
_output_shapes	
:Ї
batchnorm/mul/ReadVariableOpReadVariableOp%batchnorm_mul_readvariableop_resource*
_output_shapes	
:Ї*
dtype0u
batchnorm/mulMulbatchnorm/Rsqrt:y:0$batchnorm/mul/ReadVariableOp:value:0*
T0*
_output_shapes	
:Їd
batchnorm/mul_1Mulinputsbatchnorm/mul:z:0*
T0*(
_output_shapes
:         Ї{
batchnorm/ReadVariableOp_1ReadVariableOp#batchnorm_readvariableop_1_resource*
_output_shapes	
:Ї*
dtype0s
batchnorm/mul_2Mul"batchnorm/ReadVariableOp_1:value:0batchnorm/mul:z:0*
T0*
_output_shapes	
:Ї{
batchnorm/ReadVariableOp_2ReadVariableOp#batchnorm_readvariableop_2_resource*
_output_shapes	
:Ї*
dtype0s
batchnorm/subSub"batchnorm/ReadVariableOp_2:value:0batchnorm/mul_2:z:0*
T0*
_output_shapes	
:Їs
batchnorm/add_1AddV2batchnorm/mul_1:z:0batchnorm/sub:z:0*
T0*(
_output_shapes
:         Їc
IdentityIdentitybatchnorm/add_1:z:0^NoOp*
T0*(
_output_shapes
:         Ї║
NoOpNoOp^batchnorm/ReadVariableOp^batchnorm/ReadVariableOp_1^batchnorm/ReadVariableOp_2^batchnorm/mul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:         Ї: : : : 24
batchnorm/ReadVariableOpbatchnorm/ReadVariableOp28
batchnorm/ReadVariableOp_1batchnorm/ReadVariableOp_128
batchnorm/ReadVariableOp_2batchnorm/ReadVariableOp_22<
batchnorm/mul/ReadVariableOpbatchnorm/mul/ReadVariableOp:P L
(
_output_shapes
:         Ї
 
_user_specified_nameinputs
г

■
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_131671

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         V
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         Z
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
м

■
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_132494

inputs2
matmul_readvariableop_resource:
╨Ї.
biasadd_readvariableop_resource:	Ї
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
╨Ї*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Їs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:Ї*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ЇW
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:         Ї[
IdentityIdentitySigmoid:y:0^NoOp*
T0*(
_output_shapes
:         Їw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         ╨: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         ╨
 
_user_specified_nameinputs
Ї
Ш
(__inference_model_3_layer_call_fn_132020
gene_input_layer
protein_input_layer
unknown:
╨Ї
	unknown_0:	Ї
	unknown_1:	Ї
	unknown_2:	Ї
	unknown_3:	Ї
	unknown_4:	Ї
	unknown_5:
	unknown_6:
	unknown_7:	Ї}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:	Г@

unknown_18:@
identityИвStatefulPartitionedCallш
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
GPU2*0J 8В *L
fGRE
C__inference_model_3_layer_call_and_return_conditional_losses_131931o
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
O:         ╨:         : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:         ╨
*
_user_specified_nameGene_Input_Layer:\X
'
_output_shapes
:         
-
_user_specified_nameProtein_Input_Layer
▒3
Ь

C__inference_model_3_layer_call_and_return_conditional_losses_132073
gene_input_layer
protein_input_layer)
gene_encoder_1_132024:
╨Ї$
gene_encoder_1_132026:	Ї*
batchnormgeneencode1_132029:	Ї*
batchnormgeneencode1_132031:	Ї*
batchnormgeneencode1_132033:	Ї*
batchnormgeneencode1_132035:	Ї*
protein_encoder_1_132038:&
protein_encoder_1_132040:(
gene_encoder_2_132043:	Ї}#
gene_encoder_2_132045:})
batchnormgeneencode2_132048:})
batchnormgeneencode2_132050:})
batchnormgeneencode2_132052:})
batchnormgeneencode2_132054:},
batchnormproteinencode1_132057:,
batchnormproteinencode1_132059:,
batchnormproteinencode1_132061:,
batchnormproteinencode1_132063:+
embeddingdimdense_132067:	Г@&
embeddingdimdense_132069:@
identityИв,BatchNormGeneEncode1/StatefulPartitionedCallв,BatchNormGeneEncode2/StatefulPartitionedCallв/BatchNormProteinEncode1/StatefulPartitionedCallв)EmbeddingDimDense/StatefulPartitionedCallв&gene_encoder_1/StatefulPartitionedCallв&gene_encoder_2/StatefulPartitionedCallв)protein_encoder_1/StatefulPartitionedCallЦ
&gene_encoder_1/StatefulPartitionedCallStatefulPartitionedCallgene_input_layergene_encoder_1_132024gene_encoder_1_132026*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_131645Л
,BatchNormGeneEncode1/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_1/StatefulPartitionedCall:output:0batchnormgeneencode1_132029batchnormgeneencode1_132031batchnormgeneencode1_132033batchnormgeneencode1_132035*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Ї*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *Y
fTRR
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_131403д
)protein_encoder_1/StatefulPartitionedCallStatefulPartitionedCallprotein_input_layerprotein_encoder_1_132038protein_encoder_1_132040*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *V
fQRO
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_131671║
&gene_encoder_2/StatefulPartitionedCallStatefulPartitionedCall5BatchNormGeneEncode1/StatefulPartitionedCall:output:0gene_encoder_2_132043gene_encoder_2_132045*
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
GPU2*0J 8В *S
fNRL
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_131688К
,BatchNormGeneEncode2/StatefulPartitionedCallStatefulPartitionedCall/gene_encoder_2/StatefulPartitionedCall:output:0batchnormgeneencode2_132048batchnormgeneencode2_132050batchnormgeneencode2_132052batchnormgeneencode2_132054*
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
GPU2*0J 8В *Y
fTRR
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_131485Я
/BatchNormProteinEncode1/StatefulPartitionedCallStatefulPartitionedCall2protein_encoder_1/StatefulPartitionedCall:output:0batchnormproteinencode1_132057batchnormproteinencode1_132059batchnormproteinencode1_132061batchnormproteinencode1_132063*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_131567╢
 ConcatenateLayer/PartitionedCallPartitionedCall5BatchNormGeneEncode2/StatefulPartitionedCall:output:08BatchNormProteinEncode1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         Г* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *U
fPRN
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_131719║
)EmbeddingDimDense/StatefulPartitionedCallStatefulPartitionedCall)ConcatenateLayer/PartitionedCall:output:0embeddingdimdense_132067embeddingdimdense_132069*
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
GPU2*0J 8В *V
fQRO
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_131732Б
IdentityIdentity2EmbeddingDimDense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @А
NoOpNoOp-^BatchNormGeneEncode1/StatefulPartitionedCall-^BatchNormGeneEncode2/StatefulPartitionedCall0^BatchNormProteinEncode1/StatefulPartitionedCall*^EmbeddingDimDense/StatefulPartitionedCall'^gene_encoder_1/StatefulPartitionedCall'^gene_encoder_2/StatefulPartitionedCall*^protein_encoder_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:         ╨:         : : : : : : : : : : : : : : : : : : : : 2\
,BatchNormGeneEncode1/StatefulPartitionedCall,BatchNormGeneEncode1/StatefulPartitionedCall2\
,BatchNormGeneEncode2/StatefulPartitionedCall,BatchNormGeneEncode2/StatefulPartitionedCall2b
/BatchNormProteinEncode1/StatefulPartitionedCall/BatchNormProteinEncode1/StatefulPartitionedCall2V
)EmbeddingDimDense/StatefulPartitionedCall)EmbeddingDimDense/StatefulPartitionedCall2P
&gene_encoder_1/StatefulPartitionedCall&gene_encoder_1/StatefulPartitionedCall2P
&gene_encoder_2/StatefulPartitionedCall&gene_encoder_2/StatefulPartitionedCall2V
)protein_encoder_1/StatefulPartitionedCall)protein_encoder_1/StatefulPartitionedCall:Z V
(
_output_shapes
:         ╨
*
_user_specified_nameGene_Input_Layer:\X
'
_output_shapes
:         
-
_user_specified_nameProtein_Input_Layer
╘
Ф
$__inference_signature_wrapper_132474
gene_input_layer
protein_input_layer
unknown:
╨Ї
	unknown_0:	Ї
	unknown_1:	Ї
	unknown_2:	Ї
	unknown_3:	Ї
	unknown_4:	Ї
	unknown_5:
	unknown_6:
	unknown_7:	Ї}
	unknown_8:}
	unknown_9:}

unknown_10:}

unknown_11:}

unknown_12:}

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:	Г@

unknown_18:@
identityИвStatefulPartitionedCall╠
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
GPU2*0J 8В **
f%R#
!__inference__wrapped_model_131379o
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
O:         ╨:         : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
(
_output_shapes
:         ╨
*
_user_specified_nameGene_Input_Layer:\X
'
_output_shapes
:         
-
_user_specified_nameProtein_Input_Layer"█L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ь
serving_defaultИ
N
Gene_Input_Layer:
"serving_default_Gene_Input_Layer:0         ╨
S
Protein_Input_Layer<
%serving_default_Protein_Input_Layer:0         E
EmbeddingDimDense0
StatefulPartitionedCall:0         @tensorflow/serving/predict:жд
Ы
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
ъ
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
ъ
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
ъ
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
е
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
╢
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
Ж
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
ю2ы
(__inference_model_3_layer_call_fn_131782
(__inference_model_3_layer_call_fn_132172
(__inference_model_3_layer_call_fn_132218
(__inference_model_3_layer_call_fn_132020└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
┌2╫
C__inference_model_3_layer_call_and_return_conditional_losses_132301
C__inference_model_3_layer_call_and_return_conditional_losses_132426
C__inference_model_3_layer_call_and_return_conditional_losses_132073
C__inference_model_3_layer_call_and_return_conditional_losses_132126└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ъBч
!__inference__wrapped_model_131379Gene_Input_LayerProtein_Input_Layer"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
,
`serving_default"
signature_map
):'
╨Ї2gene_encoder_1/kernel
": Ї2gene_encoder_1/bias
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
н
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
┘2╓
/__inference_gene_encoder_1_layer_call_fn_132483в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ї2ё
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_132494в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
):'Ї2BatchNormGeneEncode1/gamma
(:&Ї2BatchNormGeneEncode1/beta
1:/Ї (2 BatchNormGeneEncode1/moving_mean
5:3Ї (2$BatchNormGeneEncode1/moving_variance
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
н
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
и2е
5__inference_BatchNormGeneEncode1_layer_call_fn_132507
5__inference_BatchNormGeneEncode1_layer_call_fn_132520┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▐2█
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_132540
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_132574┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
(:&	Ї}2gene_encoder_2/kernel
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
н
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
┘2╓
/__inference_gene_encoder_2_layer_call_fn_132583в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ї2ё
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_132594в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
*:(2protein_encoder_1/kernel
$:"2protein_encoder_1/bias
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
н
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
2__inference_protein_encoder_1_layer_call_fn_132603в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ў2Ї
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_132614в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
н
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
и2е
5__inference_BatchNormGeneEncode2_layer_call_fn_132627
5__inference_BatchNormGeneEncode2_layer_call_fn_132640┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▐2█
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_132660
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_132694┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_list_wrapper
+:)2BatchNormProteinEncode1/gamma
*:(2BatchNormProteinEncode1/beta
3:1 (2#BatchNormProteinEncode1/moving_mean
7:5 (2'BatchNormProteinEncode1/moving_variance
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
н
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
о2л
8__inference_BatchNormProteinEncode1_layer_call_fn_132707
8__inference_BatchNormProteinEncode1_layer_call_fn_132720┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ф2с
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_132740
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_132774┤
л▓з
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▒
non_trainable_variables
Аlayers
Бmetrics
 Вlayer_regularization_losses
Гlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
█2╪
1__inference_ConcatenateLayer_layer_call_fn_132780в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ў2є
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_132787в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
+:)	Г@2EmbeddingDimDense/kernel
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
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
▄2┘
2__inference_EmbeddingDimDense_layer_call_fn_132796в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ў2Ї
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_132807в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
Й0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
чBф
$__inference_signature_wrapper_132474Gene_Input_LayerProtein_Input_Layer"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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

Кtotal

Лcount
М	variables
Н	keras_api"
_tf_keras_metric
:  (2total
:  (2count
0
К0
Л1"
trackable_list_wrapper
.
М	variables"
_generic_user_object╕
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_132540d 4в1
*в'
!К
inputs         Ї
p 
к "&в#
К
0         Ї
Ъ ╕
P__inference_BatchNormGeneEncode1_layer_call_and_return_conditional_losses_132574d 4в1
*в'
!К
inputs         Ї
p
к "&в#
К
0         Ї
Ъ Р
5__inference_BatchNormGeneEncode1_layer_call_fn_132507W 4в1
*в'
!К
inputs         Ї
p 
к "К         ЇР
5__inference_BatchNormGeneEncode1_layer_call_fn_132520W 4в1
*в'
!К
inputs         Ї
p
к "К         Ї╢
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_132660b;8:93в0
)в&
 К
inputs         }
p 
к "%в"
К
0         }
Ъ ╢
P__inference_BatchNormGeneEncode2_layer_call_and_return_conditional_losses_132694b:;893в0
)в&
 К
inputs         }
p
к "%в"
К
0         }
Ъ О
5__inference_BatchNormGeneEncode2_layer_call_fn_132627U;8:93в0
)в&
 К
inputs         }
p 
к "К         }О
5__inference_BatchNormGeneEncode2_layer_call_fn_132640U:;893в0
)в&
 К
inputs         }
p
к "К         }╣
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_132740bFCED3в0
)в&
 К
inputs         
p 
к "%в"
К
0         
Ъ ╣
S__inference_BatchNormProteinEncode1_layer_call_and_return_conditional_losses_132774bEFCD3в0
)в&
 К
inputs         
p
к "%в"
К
0         
Ъ С
8__inference_BatchNormProteinEncode1_layer_call_fn_132707UFCED3в0
)в&
 К
inputs         
p 
к "К         С
8__inference_BatchNormProteinEncode1_layer_call_fn_132720UEFCD3в0
)в&
 К
inputs         
p
к "К         ╒
L__inference_ConcatenateLayer_layer_call_and_return_conditional_losses_132787ДZвW
PвM
KЪH
"К
inputs/0         }
"К
inputs/1         
к "&в#
К
0         Г
Ъ м
1__inference_ConcatenateLayer_layer_call_fn_132780wZвW
PвM
KЪH
"К
inputs/0         }
"К
inputs/1         
к "К         Го
M__inference_EmbeddingDimDense_layer_call_and_return_conditional_losses_132807]ST0в-
&в#
!К
inputs         Г
к "%в"
К
0         @
Ъ Ж
2__inference_EmbeddingDimDense_layer_call_fn_132796PST0в-
&в#
!К
inputs         Г
к "К         @є
!__inference__wrapped_model_131379═ /0'(;8:9FCEDSTnвk
dвa
_Ъ\
+К(
Gene_Input_Layer         ╨
-К*
Protein_Input_Layer         
к "EкB
@
EmbeddingDimDense+К(
EmbeddingDimDense         @м
J__inference_gene_encoder_1_layer_call_and_return_conditional_losses_132494^0в-
&в#
!К
inputs         ╨
к "&в#
К
0         Ї
Ъ Д
/__inference_gene_encoder_1_layer_call_fn_132483Q0в-
&в#
!К
inputs         ╨
к "К         Їл
J__inference_gene_encoder_2_layer_call_and_return_conditional_losses_132594]'(0в-
&в#
!К
inputs         Ї
к "%в"
К
0         }
Ъ Г
/__inference_gene_encoder_2_layer_call_fn_132583P'(0в-
&в#
!К
inputs         Ї
к "К         }¤
C__inference_model_3_layer_call_and_return_conditional_losses_132073╡ /0'(;8:9FCEDSTvвs
lвi
_Ъ\
+К(
Gene_Input_Layer         ╨
-К*
Protein_Input_Layer         
p 

 
к "%в"
К
0         @
Ъ ¤
C__inference_model_3_layer_call_and_return_conditional_losses_132126╡ /0'(:;89EFCDSTvвs
lвi
_Ъ\
+К(
Gene_Input_Layer         ╨
-К*
Protein_Input_Layer         
p

 
к "%в"
К
0         @
Ъ ъ
C__inference_model_3_layer_call_and_return_conditional_losses_132301в /0'(;8:9FCEDSTcв`
YвV
LЪI
#К 
inputs/0         ╨
"К
inputs/1         
p 

 
к "%в"
К
0         @
Ъ ъ
C__inference_model_3_layer_call_and_return_conditional_losses_132426в /0'(:;89EFCDSTcв`
YвV
LЪI
#К 
inputs/0         ╨
"К
inputs/1         
p

 
к "%в"
К
0         @
Ъ ╒
(__inference_model_3_layer_call_fn_131782и /0'(;8:9FCEDSTvвs
lвi
_Ъ\
+К(
Gene_Input_Layer         ╨
-К*
Protein_Input_Layer         
p 

 
к "К         @╒
(__inference_model_3_layer_call_fn_132020и /0'(:;89EFCDSTvвs
lвi
_Ъ\
+К(
Gene_Input_Layer         ╨
-К*
Protein_Input_Layer         
p

 
к "К         @┬
(__inference_model_3_layer_call_fn_132172Х /0'(;8:9FCEDSTcв`
YвV
LЪI
#К 
inputs/0         ╨
"К
inputs/1         
p 

 
к "К         @┬
(__inference_model_3_layer_call_fn_132218Х /0'(:;89EFCDSTcв`
YвV
LЪI
#К 
inputs/0         ╨
"К
inputs/1         
p

 
к "К         @н
M__inference_protein_encoder_1_layer_call_and_return_conditional_losses_132614\/0/в,
%в"
 К
inputs         
к "%в"
К
0         
Ъ Е
2__inference_protein_encoder_1_layer_call_fn_132603O/0/в,
%в"
 К
inputs         
к "К         а
$__inference_signature_wrapper_132474ў /0'(;8:9FCEDSTЧвУ
в 
ЛкЗ
?
Gene_Input_Layer+К(
Gene_Input_Layer         ╨
D
Protein_Input_Layer-К*
Protein_Input_Layer         "EкB
@
EmbeddingDimDense+К(
EmbeddingDimDense         @