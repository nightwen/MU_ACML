ђ
ЬЃ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02v2.3.0-0-gb36436b0878ЭМ

conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_25/kernel
}
$conv2d_25/kernel/Read/ReadVariableOpReadVariableOpconv2d_25/kernel*&
_output_shapes
:*
dtype0
t
conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_25/bias
m
"conv2d_25/bias/Read/ReadVariableOpReadVariableOpconv2d_25/bias*
_output_shapes
:*
dtype0

conv2d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_27/kernel
}
$conv2d_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_27/kernel*&
_output_shapes
:*
dtype0
t
conv2d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_27/bias
m
"conv2d_27/bias/Read/ReadVariableOpReadVariableOpconv2d_27/bias*
_output_shapes
:*
dtype0
l
RMSprop/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameRMSprop/iter
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
_output_shapes
: *
dtype0	
n
RMSprop/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
_output_shapes
: *
dtype0
~
RMSprop/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameRMSprop/learning_rate
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
_output_shapes
: *
dtype0
t
RMSprop/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
_output_shapes
: *
dtype0
j
RMSprop/rhoVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
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

RMSprop/conv2d_25/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameRMSprop/conv2d_25/kernel/rms

0RMSprop/conv2d_25/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_25/kernel/rms*&
_output_shapes
:*
dtype0

RMSprop/conv2d_25/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameRMSprop/conv2d_25/bias/rms

.RMSprop/conv2d_25/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_25/bias/rms*
_output_shapes
:*
dtype0

RMSprop/conv2d_27/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameRMSprop/conv2d_27/kernel/rms

0RMSprop/conv2d_27/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_27/kernel/rms*&
_output_shapes
:*
dtype0

RMSprop/conv2d_27/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameRMSprop/conv2d_27/bias/rms

.RMSprop/conv2d_27/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_27/bias/rms*
_output_shapes
:*
dtype0

NoOpNoOp
"
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ю!
valueФ!BС! BК!

layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer-5
layer_with_weights-1
layer-6
	optimizer
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
 
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
R
 regularization_losses
!	variables
"trainable_variables
#	keras_api
h

$kernel
%bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
k
*iter
	+decay
,learning_rate
-momentum
.rho	rms]	rms^	$rms_	%rms`
 

0
1
$2
%3

0
1
$2
%3
­
	regularization_losses
/layer_regularization_losses
0non_trainable_variables

	variables
1metrics

2layers
trainable_variables
3layer_metrics
 
 
 
 
­
regularization_losses
4layer_regularization_losses
5non_trainable_variables
	variables
6metrics

7layers
trainable_variables
8layer_metrics
\Z
VARIABLE_VALUEconv2d_25/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_25/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
­
regularization_losses
9layer_regularization_losses
:non_trainable_variables
	variables
;metrics

<layers
trainable_variables
=layer_metrics
 
 
 
­
regularization_losses
>layer_regularization_losses
?non_trainable_variables
	variables
@metrics

Alayers
trainable_variables
Blayer_metrics
 
 
 
­
regularization_losses
Clayer_regularization_losses
Dnon_trainable_variables
	variables
Emetrics

Flayers
trainable_variables
Glayer_metrics
 
 
 
­
 regularization_losses
Hlayer_regularization_losses
Inon_trainable_variables
!	variables
Jmetrics

Klayers
"trainable_variables
Llayer_metrics
\Z
VARIABLE_VALUEconv2d_27/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_27/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

$0
%1

$0
%1
­
&regularization_losses
Mlayer_regularization_losses
Nnon_trainable_variables
'	variables
Ometrics

Players
(trainable_variables
Qlayer_metrics
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
 
 

R0
S1
1
0
1
2
3
4
5
6
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ttotal
	Ucount
V	variables
W	keras_api
D
	Xtotal
	Ycount
Z
_fn_kwargs
[	variables
\	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

T0
U1

V	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

X0
Y1

[	variables

VARIABLE_VALUERMSprop/conv2d_25/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/conv2d_25/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/conv2d_27/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUERMSprop/conv2d_27/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE

serving_default_input_8Placeholder*/
_output_shapes
:џџџџџџџџџ  *
dtype0*$
shape:џџџџџџџџџ  

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_8conv2d_25/kernelconv2d_25/biasconv2d_27/kernelconv2d_27/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_41742
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Њ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_25/kernel/Read/ReadVariableOp"conv2d_25/bias/Read/ReadVariableOp$conv2d_27/kernel/Read/ReadVariableOp"conv2d_27/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp0RMSprop/conv2d_25/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_25/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_27/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_27/bias/rms/Read/ReadVariableOpConst*
Tin
2	*
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
GPU 2J 8 *'
f"R 
__inference__traced_save_41944
б
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_25/kernelconv2d_25/biasconv2d_27/kernelconv2d_27/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcounttotal_1count_1RMSprop/conv2d_25/kernel/rmsRMSprop/conv2d_25/bias/rmsRMSprop/conv2d_27/kernel/rmsRMSprop/conv2d_27/bias/rms*
Tin
2*
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
GPU 2J 8 **
f%R#
!__inference__traced_restore_42005єщ
ъI
щ
!__inference__traced_restore_42005
file_prefix%
!assignvariableop_conv2d_25_kernel%
!assignvariableop_1_conv2d_25_bias'
#assignvariableop_2_conv2d_27_kernel%
!assignvariableop_3_conv2d_27_bias#
assignvariableop_4_rmsprop_iter$
 assignvariableop_5_rmsprop_decay,
(assignvariableop_6_rmsprop_learning_rate'
#assignvariableop_7_rmsprop_momentum"
assignvariableop_8_rmsprop_rho
assignvariableop_9_total
assignvariableop_10_count
assignvariableop_11_total_1
assignvariableop_12_count_14
0assignvariableop_13_rmsprop_conv2d_25_kernel_rms2
.assignvariableop_14_rmsprop_conv2d_25_bias_rms4
0assignvariableop_15_rmsprop_conv2d_27_kernel_rms2
.assignvariableop_16_rmsprop_conv2d_27_bias_rms
identity_18ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_2ЂAssignVariableOp_3ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesВ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*\
_output_shapesJ
H::::::::::::::::::* 
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity 
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_25_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1І
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_25_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2Ј
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_27_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3І
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_27_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4Є
AssignVariableOp_4AssignVariableOpassignvariableop_4_rmsprop_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5Ѕ
AssignVariableOp_5AssignVariableOp assignvariableop_5_rmsprop_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6­
AssignVariableOp_6AssignVariableOp(assignvariableop_6_rmsprop_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7Ј
AssignVariableOp_7AssignVariableOp#assignvariableop_7_rmsprop_momentumIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8Ѓ
AssignVariableOp_8AssignVariableOpassignvariableop_8_rmsprop_rhoIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ё
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11Ѓ
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12Ѓ
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13И
AssignVariableOp_13AssignVariableOp0assignvariableop_13_rmsprop_conv2d_25_kernel_rmsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ж
AssignVariableOp_14AssignVariableOp.assignvariableop_14_rmsprop_conv2d_25_bias_rmsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15И
AssignVariableOp_15AssignVariableOp0assignvariableop_15_rmsprop_conv2d_27_kernel_rmsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ж
AssignVariableOp_16AssignVariableOp.assignvariableop_16_rmsprop_conv2d_27_bias_rmsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_169
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpд
Identity_17Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_17Ч
Identity_18IdentityIdentity_17:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_18"#
identity_18Identity_18:output:0*Y
_input_shapesH
F: :::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162(
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

f
J__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_41556

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2Ю
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mulе
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
half_pixel_centers(2
resize/ResizeNearestNeighborЄ
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
%
Ё
H__inference_functional_15_layer_call_and_return_conditional_losses_41773

inputs,
(conv2d_25_conv2d_readvariableop_resource-
)conv2d_25_biasadd_readvariableop_resource,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource
identityБ
zero_padding2d_15/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2 
zero_padding2d_15/Pad/paddings 
zero_padding2d_15/PadPadinputs'zero_padding2d_15/Pad/paddings:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$$2
zero_padding2d_15/PadГ
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_25/Conv2D/ReadVariableOpк
conv2d_25/Conv2DConv2Dzero_padding2d_15/Pad:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  *
paddingVALID*
strides
2
conv2d_25/Conv2DЊ
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_25/BiasAdd/ReadVariableOpА
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  2
conv2d_25/BiasAdd~
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  2
conv2d_25/ReluЧ
max_pooling2d_9/MaxPoolMaxPoolconv2d_25/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
2
max_pooling2d_9/MaxPool~
up_sampling2d_9/ShapeShape max_pooling2d_9/MaxPool:output:0*
T0*
_output_shapes
:2
up_sampling2d_9/Shape
#up_sampling2d_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_9/strided_slice/stack
%up_sampling2d_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_9/strided_slice/stack_1
%up_sampling2d_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_9/strided_slice/stack_2Ў
up_sampling2d_9/strided_sliceStridedSliceup_sampling2d_9/Shape:output:0,up_sampling2d_9/strided_slice/stack:output:0.up_sampling2d_9/strided_slice/stack_1:output:0.up_sampling2d_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_9/strided_slice
up_sampling2d_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_9/Const
up_sampling2d_9/mulMul&up_sampling2d_9/strided_slice:output:0up_sampling2d_9/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_9/mul
,up_sampling2d_9/resize/ResizeNearestNeighborResizeNearestNeighbor max_pooling2d_9/MaxPool:output:0up_sampling2d_9/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ  *
half_pixel_centers(2.
,up_sampling2d_9/resize/ResizeNearestNeighborБ
zero_padding2d_17/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2 
zero_padding2d_17/Pad/paddingsз
zero_padding2d_17/PadPad=up_sampling2d_9/resize/ResizeNearestNeighbor:resized_images:0'zero_padding2d_17/Pad/paddings:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$$2
zero_padding2d_17/PadГ
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_27/Conv2D/ReadVariableOpк
conv2d_27/Conv2DConv2Dzero_padding2d_17/Pad:output:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  *
paddingVALID*
strides
2
conv2d_27/Conv2DЊ
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOpА
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  2
conv2d_27/BiasAdd~
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  2
conv2d_27/Relux
IdentityIdentityconv2d_27/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ  :::::W S
/
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs
Д

H__inference_functional_15_layer_call_and_return_conditional_losses_41638
input_8
conv2d_25_41602
conv2d_25_41604
conv2d_27_41632
conv2d_27_41634
identityЂ!conv2d_25/StatefulPartitionedCallЂ!conv2d_27/StatefulPartitionedCallї
!zero_padding2d_15/PartitionedCallPartitionedCallinput_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ$$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_zero_padding2d_15_layer_call_and_return_conditional_losses_415252#
!zero_padding2d_15/PartitionedCallТ
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_15/PartitionedCall:output:0conv2d_25_41602conv2d_25_41604*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_415912#
!conv2d_25/StatefulPartitionedCall
max_pooling2d_9/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_415372!
max_pooling2d_9/PartitionedCallЄ
up_sampling2d_9/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_415562!
up_sampling2d_9/PartitionedCallЊ
!zero_padding2d_17/PartitionedCallPartitionedCall(up_sampling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_zero_padding2d_17_layer_call_and_return_conditional_losses_415692#
!zero_padding2d_17/PartitionedCallд
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_17/PartitionedCall:output:0conv2d_27_41632conv2d_27_41634*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_416212#
!conv2d_27/StatefulPartitionedCallр
IdentityIdentity*conv2d_27/StatefulPartitionedCall:output:0"^conv2d_25/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ  ::::2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ  
!
_user_specified_name	input_8
-
Ѕ
__inference__traced_save_41944
file_prefix/
+savev2_conv2d_25_kernel_read_readvariableop-
)savev2_conv2d_25_bias_read_readvariableop/
+savev2_conv2d_27_kernel_read_readvariableop-
)savev2_conv2d_27_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop;
7savev2_rmsprop_conv2d_25_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_25_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_27_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_27_bias_rms_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_a7d09b9879464cc0be23b979d6ea94ec/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesЌ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesС
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_25_kernel_read_readvariableop)savev2_conv2d_25_bias_read_readvariableop+savev2_conv2d_27_kernel_read_readvariableop)savev2_conv2d_27_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop7savev2_rmsprop_conv2d_25_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_25_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_27_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_27_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 * 
dtypes
2	2
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapesx
v: ::::: : : : : : : : : ::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
Ж-
В
 __inference__wrapped_model_41518
input_8:
6functional_15_conv2d_25_conv2d_readvariableop_resource;
7functional_15_conv2d_25_biasadd_readvariableop_resource:
6functional_15_conv2d_27_conv2d_readvariableop_resource;
7functional_15_conv2d_27_biasadd_readvariableop_resource
identityЭ
,functional_15/zero_padding2d_15/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2.
,functional_15/zero_padding2d_15/Pad/paddingsЫ
#functional_15/zero_padding2d_15/PadPadinput_85functional_15/zero_padding2d_15/Pad/paddings:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$$2%
#functional_15/zero_padding2d_15/Padн
-functional_15/conv2d_25/Conv2D/ReadVariableOpReadVariableOp6functional_15_conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-functional_15/conv2d_25/Conv2D/ReadVariableOp
functional_15/conv2d_25/Conv2DConv2D,functional_15/zero_padding2d_15/Pad:output:05functional_15/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  *
paddingVALID*
strides
2 
functional_15/conv2d_25/Conv2Dд
.functional_15/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp7functional_15_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.functional_15/conv2d_25/BiasAdd/ReadVariableOpш
functional_15/conv2d_25/BiasAddBiasAdd'functional_15/conv2d_25/Conv2D:output:06functional_15/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  2!
functional_15/conv2d_25/BiasAddЈ
functional_15/conv2d_25/ReluRelu(functional_15/conv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  2
functional_15/conv2d_25/Reluё
%functional_15/max_pooling2d_9/MaxPoolMaxPool*functional_15/conv2d_25/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
2'
%functional_15/max_pooling2d_9/MaxPoolЈ
#functional_15/up_sampling2d_9/ShapeShape.functional_15/max_pooling2d_9/MaxPool:output:0*
T0*
_output_shapes
:2%
#functional_15/up_sampling2d_9/ShapeА
1functional_15/up_sampling2d_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:23
1functional_15/up_sampling2d_9/strided_slice/stackД
3functional_15/up_sampling2d_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:25
3functional_15/up_sampling2d_9/strided_slice/stack_1Д
3functional_15/up_sampling2d_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:25
3functional_15/up_sampling2d_9/strided_slice/stack_2
+functional_15/up_sampling2d_9/strided_sliceStridedSlice,functional_15/up_sampling2d_9/Shape:output:0:functional_15/up_sampling2d_9/strided_slice/stack:output:0<functional_15/up_sampling2d_9/strided_slice/stack_1:output:0<functional_15/up_sampling2d_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2-
+functional_15/up_sampling2d_9/strided_slice
#functional_15/up_sampling2d_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2%
#functional_15/up_sampling2d_9/Constж
!functional_15/up_sampling2d_9/mulMul4functional_15/up_sampling2d_9/strided_slice:output:0,functional_15/up_sampling2d_9/Const:output:0*
T0*
_output_shapes
:2#
!functional_15/up_sampling2d_9/mulМ
:functional_15/up_sampling2d_9/resize/ResizeNearestNeighborResizeNearestNeighbor.functional_15/max_pooling2d_9/MaxPool:output:0%functional_15/up_sampling2d_9/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ  *
half_pixel_centers(2<
:functional_15/up_sampling2d_9/resize/ResizeNearestNeighborЭ
,functional_15/zero_padding2d_17/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2.
,functional_15/zero_padding2d_17/Pad/paddings
#functional_15/zero_padding2d_17/PadPadKfunctional_15/up_sampling2d_9/resize/ResizeNearestNeighbor:resized_images:05functional_15/zero_padding2d_17/Pad/paddings:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$$2%
#functional_15/zero_padding2d_17/Padн
-functional_15/conv2d_27/Conv2D/ReadVariableOpReadVariableOp6functional_15_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-functional_15/conv2d_27/Conv2D/ReadVariableOp
functional_15/conv2d_27/Conv2DConv2D,functional_15/zero_padding2d_17/Pad:output:05functional_15/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  *
paddingVALID*
strides
2 
functional_15/conv2d_27/Conv2Dд
.functional_15/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp7functional_15_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.functional_15/conv2d_27/BiasAdd/ReadVariableOpш
functional_15/conv2d_27/BiasAddBiasAdd'functional_15/conv2d_27/Conv2D:output:06functional_15/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  2!
functional_15/conv2d_27/BiasAddЈ
functional_15/conv2d_27/ReluRelu(functional_15/conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  2
functional_15/conv2d_27/Relu
IdentityIdentity*functional_15/conv2d_27/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ  :::::X T
/
_output_shapes
:џџџџџџџџџ  
!
_user_specified_name	input_8
ц
h
L__inference_zero_padding2d_17_layer_call_and_return_conditional_losses_41569

inputs
identity
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Pad/paddings
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Pad
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ь
Ё
-__inference_functional_15_layer_call_fn_41688
input_8
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_functional_15_layer_call_and_return_conditional_losses_416772
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ  ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ  
!
_user_specified_name	input_8
Б

H__inference_functional_15_layer_call_and_return_conditional_losses_41708

inputs
conv2d_25_41694
conv2d_25_41696
conv2d_27_41702
conv2d_27_41704
identityЂ!conv2d_25/StatefulPartitionedCallЂ!conv2d_27/StatefulPartitionedCallі
!zero_padding2d_15/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ$$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_zero_padding2d_15_layer_call_and_return_conditional_losses_415252#
!zero_padding2d_15/PartitionedCallТ
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_15/PartitionedCall:output:0conv2d_25_41694conv2d_25_41696*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_415912#
!conv2d_25/StatefulPartitionedCall
max_pooling2d_9/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_415372!
max_pooling2d_9/PartitionedCallЄ
up_sampling2d_9/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_415562!
up_sampling2d_9/PartitionedCallЊ
!zero_padding2d_17/PartitionedCallPartitionedCall(up_sampling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_zero_padding2d_17_layer_call_and_return_conditional_losses_415692#
!zero_padding2d_17/PartitionedCallд
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_17/PartitionedCall:output:0conv2d_27_41702conv2d_27_41704*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_416212#
!conv2d_27/StatefulPartitionedCallр
IdentityIdentity*conv2d_27/StatefulPartitionedCall:output:0"^conv2d_25/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ  ::::2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs
ь
Ё
-__inference_functional_15_layer_call_fn_41719
input_8
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall­
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_functional_15_layer_call_and_return_conditional_losses_417082
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ  ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ  
!
_user_specified_name	input_8
	
Ќ
D__inference_conv2d_25_layer_call_and_return_conditional_losses_41591

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ$$:::W S
/
_output_shapes
:џџџџџџџџџ$$
 
_user_specified_nameinputs
ќ
~
)__inference_conv2d_25_layer_call_fn_41850

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallќ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_415912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ$$::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ$$
 
_user_specified_nameinputs
щ
 
-__inference_functional_15_layer_call_fn_41830

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_functional_15_layer_call_and_return_conditional_losses_417082
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ  ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs
Д

H__inference_functional_15_layer_call_and_return_conditional_losses_41656
input_8
conv2d_25_41642
conv2d_25_41644
conv2d_27_41650
conv2d_27_41652
identityЂ!conv2d_25/StatefulPartitionedCallЂ!conv2d_27/StatefulPartitionedCallї
!zero_padding2d_15/PartitionedCallPartitionedCallinput_8*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ$$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_zero_padding2d_15_layer_call_and_return_conditional_losses_415252#
!zero_padding2d_15/PartitionedCallТ
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_15/PartitionedCall:output:0conv2d_25_41642conv2d_25_41644*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_415912#
!conv2d_25/StatefulPartitionedCall
max_pooling2d_9/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_415372!
max_pooling2d_9/PartitionedCallЄ
up_sampling2d_9/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_415562!
up_sampling2d_9/PartitionedCallЊ
!zero_padding2d_17/PartitionedCallPartitionedCall(up_sampling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_zero_padding2d_17_layer_call_and_return_conditional_losses_415692#
!zero_padding2d_17/PartitionedCallд
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_17/PartitionedCall:output:0conv2d_27_41650conv2d_27_41652*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_416212#
!conv2d_27/StatefulPartitionedCallр
IdentityIdentity*conv2d_27/StatefulPartitionedCall:output:0"^conv2d_25/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ  ::::2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ  
!
_user_specified_name	input_8
щ
 
-__inference_functional_15_layer_call_fn_41817

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallЌ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_functional_15_layer_call_and_return_conditional_losses_416772
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ  ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs
Ф
~
)__inference_conv2d_27_layer_call_fn_41870

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_416212
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ц
h
L__inference_zero_padding2d_15_layer_call_and_return_conditional_losses_41525

inputs
identity
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Pad/paddings
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Pad
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
%
Ё
H__inference_functional_15_layer_call_and_return_conditional_losses_41804

inputs,
(conv2d_25_conv2d_readvariableop_resource-
)conv2d_25_biasadd_readvariableop_resource,
(conv2d_27_conv2d_readvariableop_resource-
)conv2d_27_biasadd_readvariableop_resource
identityБ
zero_padding2d_15/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2 
zero_padding2d_15/Pad/paddings 
zero_padding2d_15/PadPadinputs'zero_padding2d_15/Pad/paddings:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$$2
zero_padding2d_15/PadГ
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_25/Conv2D/ReadVariableOpк
conv2d_25/Conv2DConv2Dzero_padding2d_15/Pad:output:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  *
paddingVALID*
strides
2
conv2d_25/Conv2DЊ
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_25/BiasAdd/ReadVariableOpА
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  2
conv2d_25/BiasAdd~
conv2d_25/ReluReluconv2d_25/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  2
conv2d_25/ReluЧ
max_pooling2d_9/MaxPoolMaxPoolconv2d_25/Relu:activations:0*/
_output_shapes
:џџџџџџџџџ*
ksize
*
paddingSAME*
strides
2
max_pooling2d_9/MaxPool~
up_sampling2d_9/ShapeShape max_pooling2d_9/MaxPool:output:0*
T0*
_output_shapes
:2
up_sampling2d_9/Shape
#up_sampling2d_9/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_9/strided_slice/stack
%up_sampling2d_9/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_9/strided_slice/stack_1
%up_sampling2d_9/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_9/strided_slice/stack_2Ў
up_sampling2d_9/strided_sliceStridedSliceup_sampling2d_9/Shape:output:0,up_sampling2d_9/strided_slice/stack:output:0.up_sampling2d_9/strided_slice/stack_1:output:0.up_sampling2d_9/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_9/strided_slice
up_sampling2d_9/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_9/Const
up_sampling2d_9/mulMul&up_sampling2d_9/strided_slice:output:0up_sampling2d_9/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_9/mul
,up_sampling2d_9/resize/ResizeNearestNeighborResizeNearestNeighbor max_pooling2d_9/MaxPool:output:0up_sampling2d_9/mul:z:0*
T0*/
_output_shapes
:џџџџџџџџџ  *
half_pixel_centers(2.
,up_sampling2d_9/resize/ResizeNearestNeighborБ
zero_padding2d_17/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2 
zero_padding2d_17/Pad/paddingsз
zero_padding2d_17/PadPad=up_sampling2d_9/resize/ResizeNearestNeighbor:resized_images:0'zero_padding2d_17/Pad/paddings:output:0*
T0*/
_output_shapes
:џџџџџџџџџ$$2
zero_padding2d_17/PadГ
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_27/Conv2D/ReadVariableOpк
conv2d_27/Conv2DConv2Dzero_padding2d_17/Pad:output:0'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  *
paddingVALID*
strides
2
conv2d_27/Conv2DЊ
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_27/BiasAdd/ReadVariableOpА
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  2
conv2d_27/BiasAdd~
conv2d_27/ReluReluconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  2
conv2d_27/Relux
IdentityIdentityconv2d_27/Relu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ  :::::W S
/
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs
Џ
M
1__inference_zero_padding2d_15_layer_call_fn_41531

inputs
identityэ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_zero_padding2d_15_layer_call_and_return_conditional_losses_415252
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ
K
/__inference_max_pooling2d_9_layer_call_fn_41543

inputs
identityы
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_415372
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs


#__inference_signature_wrapper_41742
input_8
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCallinput_8unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_415182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:џџџџџџџџџ  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ  ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:џџџџџџџџџ  
!
_user_specified_name	input_8
ї	
Ќ
D__inference_conv2d_27_layer_call_and_return_conditional_losses_41861

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
	
Ќ
D__inference_conv2d_25_layer_call_and_return_conditional_losses_41841

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЄ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  *
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:џџџџџџџџџ  2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:џџџџџџџџџ  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:џџџџџџџџџ$$:::W S
/
_output_shapes
:џџџџџџџџџ$$
 
_user_specified_nameinputs
Ћ
K
/__inference_up_sampling2d_9_layer_call_fn_41562

inputs
identityы
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_415562
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
џ
f
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_41537

inputs
identityЌ
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ*
ksize
*
paddingSAME*
strides
2	
MaxPool
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Б

H__inference_functional_15_layer_call_and_return_conditional_losses_41677

inputs
conv2d_25_41663
conv2d_25_41665
conv2d_27_41671
conv2d_27_41673
identityЂ!conv2d_25/StatefulPartitionedCallЂ!conv2d_27/StatefulPartitionedCallі
!zero_padding2d_15/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ$$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_zero_padding2d_15_layer_call_and_return_conditional_losses_415252#
!zero_padding2d_15/PartitionedCallТ
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_15/PartitionedCall:output:0conv2d_25_41663conv2d_25_41665*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_25_layer_call_and_return_conditional_losses_415912#
!conv2d_25/StatefulPartitionedCall
max_pooling2d_9/PartitionedCallPartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_415372!
max_pooling2d_9/PartitionedCallЄ
up_sampling2d_9/PartitionedCallPartitionedCall(max_pooling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_415562!
up_sampling2d_9/PartitionedCallЊ
!zero_padding2d_17/PartitionedCallPartitionedCall(up_sampling2d_9/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_zero_padding2d_17_layer_call_and_return_conditional_losses_415692#
!zero_padding2d_17/PartitionedCallд
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_17/PartitionedCall:output:0conv2d_27_41671conv2d_27_41673*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_conv2d_27_layer_call_and_return_conditional_losses_416212#
!conv2d_27/StatefulPartitionedCallр
IdentityIdentity*conv2d_27/StatefulPartitionedCall:output:0"^conv2d_25/StatefulPartitionedCall"^conv2d_27/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:џџџџџџџџџ  ::::2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall:W S
/
_output_shapes
:џџџџџџџџџ  
 
_user_specified_nameinputs
ї	
Ќ
D__inference_conv2d_27_layer_call_and_return_conditional_losses_41621

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOpЖ
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ*
paddingVALID*
strides
2
Conv2D
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2
Relu
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ:::i e
A
_output_shapes/
-:+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Џ
M
1__inference_zero_padding2d_17_layer_call_fn_41575

inputs
identityэ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_zero_padding2d_17_layer_call_and_return_conditional_losses_415692
PartitionedCall
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ:r n
J
_output_shapes8
6:4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*М
serving_defaultЈ
C
input_88
serving_default_input_8:0џџџџџџџџџ  E
	conv2d_278
StatefulPartitionedCall:0џџџџџџџџџ  tensorflow/serving/predict:рк
;
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer-4
layer-5
layer_with_weights-1
layer-6
	optimizer
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
a__call__
*b&call_and_return_all_conditional_losses
c_default_save_signature"С8
_tf_keras_networkЅ8{"class_name": "Functional", "name": "functional_15", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_15", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}, "name": "input_8", "inbound_nodes": []}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_15", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [2, 2]}, {"class_name": "__tuple__", "items": [2, 2]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_15", "inbound_nodes": [[["input_8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_25", "inbound_nodes": [[["zero_padding2d_15", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last"}, "name": "max_pooling2d_9", "inbound_nodes": [[["conv2d_25", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_9", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_9", "inbound_nodes": [[["max_pooling2d_9", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_17", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [2, 2]}, {"class_name": "__tuple__", "items": [2, 2]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_17", "inbound_nodes": [[["up_sampling2d_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_27", "inbound_nodes": [[["zero_padding2d_17", 0, 0, {}]]]}], "input_layers": [["input_8", 0, 0]], "output_layers": [["conv2d_27", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_15", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}, "name": "input_8", "inbound_nodes": []}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_15", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [2, 2]}, {"class_name": "__tuple__", "items": [2, 2]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_15", "inbound_nodes": [[["input_8", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_25", "inbound_nodes": [[["zero_padding2d_15", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last"}, "name": "max_pooling2d_9", "inbound_nodes": [[["conv2d_25", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_9", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_9", "inbound_nodes": [[["max_pooling2d_9", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_17", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [2, 2]}, {"class_name": "__tuple__", "items": [2, 2]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_17", "inbound_nodes": [[["up_sampling2d_9", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_27", "inbound_nodes": [[["zero_padding2d_17", 0, 0, {}]]]}], "input_layers": [["input_8", 0, 0]], "output_layers": [["conv2d_27", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
љ"і
_tf_keras_input_layerж{"class_name": "InputLayer", "name": "input_8", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_8"}}

regularization_losses
	variables
trainable_variables
	keras_api
d__call__
*e&call_and_return_all_conditional_losses"ќ
_tf_keras_layerт{"class_name": "ZeroPadding2D", "name": "zero_padding2d_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "zero_padding2d_15", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [2, 2]}, {"class_name": "__tuple__", "items": [2, 2]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ђ	

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
f__call__
*g&call_and_return_all_conditional_losses"Э
_tf_keras_layerГ{"class_name": "Conv2D", "name": "conv2d_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36, 36, 3]}}
ў
regularization_losses
	variables
trainable_variables
	keras_api
h__call__
*i&call_and_return_all_conditional_losses"я
_tf_keras_layerе{"class_name": "MaxPooling2D", "name": "max_pooling2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_9", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [4, 4]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
Щ
regularization_losses
	variables
trainable_variables
	keras_api
j__call__
*k&call_and_return_all_conditional_losses"К
_tf_keras_layer {"class_name": "UpSampling2D", "name": "up_sampling2d_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_9", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [4, 4]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}

 regularization_losses
!	variables
"trainable_variables
#	keras_api
l__call__
*m&call_and_return_all_conditional_losses"ќ
_tf_keras_layerт{"class_name": "ZeroPadding2D", "name": "zero_padding2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "zero_padding2d_17", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [2, 2]}, {"class_name": "__tuple__", "items": [2, 2]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ђ	

$kernel
%bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
n__call__
*o&call_and_return_all_conditional_losses"Э
_tf_keras_layerГ{"class_name": "Conv2D", "name": "conv2d_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [5, 5]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36, 36, 3]}}
~
*iter
	+decay
,learning_rate
-momentum
.rho	rms]	rms^	$rms_	%rms`"
	optimizer
 "
trackable_list_wrapper
<
0
1
$2
%3"
trackable_list_wrapper
<
0
1
$2
%3"
trackable_list_wrapper
Ъ
	regularization_losses
/layer_regularization_losses
0non_trainable_variables

	variables
1metrics

2layers
trainable_variables
3layer_metrics
a__call__
c_default_save_signature
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
,
pserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
regularization_losses
4layer_regularization_losses
5non_trainable_variables
	variables
6metrics

7layers
trainable_variables
8layer_metrics
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_25/kernel
:2conv2d_25/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
regularization_losses
9layer_regularization_losses
:non_trainable_variables
	variables
;metrics

<layers
trainable_variables
=layer_metrics
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
regularization_losses
>layer_regularization_losses
?non_trainable_variables
	variables
@metrics

Alayers
trainable_variables
Blayer_metrics
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
regularization_losses
Clayer_regularization_losses
Dnon_trainable_variables
	variables
Emetrics

Flayers
trainable_variables
Glayer_metrics
j__call__
*k&call_and_return_all_conditional_losses
&k"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
 regularization_losses
Hlayer_regularization_losses
Inon_trainable_variables
!	variables
Jmetrics

Klayers
"trainable_variables
Llayer_metrics
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_27/kernel
:2conv2d_27/bias
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
­
&regularization_losses
Mlayer_regularization_losses
Nnon_trainable_variables
'	variables
Ometrics

Players
(trainable_variables
Qlayer_metrics
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
R0
S1"
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
Л
	Ttotal
	Ucount
V	variables
W	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
џ
	Xtotal
	Ycount
Z
_fn_kwargs
[	variables
\	keras_api"И
_tf_keras_metric{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
:  (2total
:  (2count
.
T0
U1"
trackable_list_wrapper
-
V	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
X0
Y1"
trackable_list_wrapper
-
[	variables"
_generic_user_object
4:22RMSprop/conv2d_25/kernel/rms
&:$2RMSprop/conv2d_25/bias/rms
4:22RMSprop/conv2d_27/kernel/rms
&:$2RMSprop/conv2d_27/bias/rms
2џ
-__inference_functional_15_layer_call_fn_41719
-__inference_functional_15_layer_call_fn_41817
-__inference_functional_15_layer_call_fn_41830
-__inference_functional_15_layer_call_fn_41688Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ю2ы
H__inference_functional_15_layer_call_and_return_conditional_losses_41773
H__inference_functional_15_layer_call_and_return_conditional_losses_41638
H__inference_functional_15_layer_call_and_return_conditional_losses_41804
H__inference_functional_15_layer_call_and_return_conditional_losses_41656Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
ц2у
 __inference__wrapped_model_41518О
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *.Ђ+
)&
input_8џџџџџџџџџ  
2
1__inference_zero_padding2d_15_layer_call_fn_41531р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Д2Б
L__inference_zero_padding2d_15_layer_call_and_return_conditional_losses_41525р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
г2а
)__inference_conv2d_25_layer_call_fn_41850Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_conv2d_25_layer_call_and_return_conditional_losses_41841Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2
/__inference_max_pooling2d_9_layer_call_fn_41543р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
В2Џ
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_41537р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
/__inference_up_sampling2d_9_layer_call_fn_41562р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
В2Џ
J__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_41556р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
2
1__inference_zero_padding2d_17_layer_call_fn_41575р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Д2Б
L__inference_zero_padding2d_17_layer_call_and_return_conditional_losses_41569р
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *@Ђ=
;84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
г2а
)__inference_conv2d_27_layer_call_fn_41870Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ю2ы
D__inference_conv2d_27_layer_call_and_return_conditional_losses_41861Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
2B0
#__inference_signature_wrapper_41742input_8Ѓ
 __inference__wrapped_model_41518$%8Ђ5
.Ђ+
)&
input_8џџџџџџџџџ  
Њ "=Њ:
8
	conv2d_27+(
	conv2d_27џџџџџџџџџ  Д
D__inference_conv2d_25_layer_call_and_return_conditional_losses_41841l7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ$$
Њ "-Ђ*
# 
0џџџџџџџџџ  
 
)__inference_conv2d_25_layer_call_fn_41850_7Ђ4
-Ђ*
(%
inputsџџџџџџџџџ$$
Њ " џџџџџџџџџ  й
D__inference_conv2d_27_layer_call_and_return_conditional_losses_41861$%IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Б
)__inference_conv2d_27_layer_call_fn_41870$%IЂF
?Ђ<
:7
inputs+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџж
H__inference_functional_15_layer_call_and_return_conditional_losses_41638$%@Ђ=
6Ђ3
)&
input_8џџџџџџџџџ  
p

 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 ж
H__inference_functional_15_layer_call_and_return_conditional_losses_41656$%@Ђ=
6Ђ3
)&
input_8џџџџџџџџџ  
p 

 
Њ "?Ђ<
52
0+џџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Т
H__inference_functional_15_layer_call_and_return_conditional_losses_41773v$%?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ  
p

 
Њ "-Ђ*
# 
0џџџџџџџџџ  
 Т
H__inference_functional_15_layer_call_and_return_conditional_losses_41804v$%?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ  
p 

 
Њ "-Ђ*
# 
0џџџџџџџџџ  
 ­
-__inference_functional_15_layer_call_fn_41688|$%@Ђ=
6Ђ3
)&
input_8џџџџџџџџџ  
p

 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџ­
-__inference_functional_15_layer_call_fn_41719|$%@Ђ=
6Ђ3
)&
input_8џџџџџџџџџ  
p 

 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЌ
-__inference_functional_15_layer_call_fn_41817{$%?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ  
p

 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџЌ
-__inference_functional_15_layer_call_fn_41830{$%?Ђ<
5Ђ2
(%
inputsџџџџџџџџџ  
p 

 
Њ "2/+џџџџџџџџџџџџџџџџџџџџџџџџџџџэ
J__inference_max_pooling2d_9_layer_call_and_return_conditional_losses_41537RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Х
/__inference_max_pooling2d_9_layer_call_fn_41543RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџВ
#__inference_signature_wrapper_41742$%CЂ@
Ђ 
9Њ6
4
input_8)&
input_8џџџџџџџџџ  "=Њ:
8
	conv2d_27+(
	conv2d_27џџџџџџџџџ  э
J__inference_up_sampling2d_9_layer_call_and_return_conditional_losses_41556RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Х
/__inference_up_sampling2d_9_layer_call_fn_41562RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџя
L__inference_zero_padding2d_15_layer_call_and_return_conditional_losses_41525RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ч
1__inference_zero_padding2d_15_layer_call_fn_41531RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџя
L__inference_zero_padding2d_17_layer_call_and_return_conditional_losses_41569RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ "HЂE
>;
04џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
 Ч
1__inference_zero_padding2d_17_layer_call_fn_41575RЂO
HЂE
C@
inputs4џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ
Њ ";84џџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџџ