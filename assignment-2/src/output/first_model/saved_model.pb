��
��
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
dtypetype�
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
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.02v2.3.0-0-gb36436b0878��
�
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
:*
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:*
dtype0
�
conv2d_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_10/kernel
}
$conv2d_10/kernel/Read/ReadVariableOpReadVariableOpconv2d_10/kernel*&
_output_shapes
:*
dtype0
t
conv2d_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_10/bias
m
"conv2d_10/bias/Read/ReadVariableOpReadVariableOpconv2d_10/bias*
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
�
RMSprop/conv2d_8/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameRMSprop/conv2d_8/kernel/rms
�
/RMSprop/conv2d_8/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_8/kernel/rms*&
_output_shapes
:*
dtype0
�
RMSprop/conv2d_8/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameRMSprop/conv2d_8/bias/rms
�
-RMSprop/conv2d_8/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_8/bias/rms*
_output_shapes
:*
dtype0
�
RMSprop/conv2d_10/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameRMSprop/conv2d_10/kernel/rms
�
0RMSprop/conv2d_10/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_10/kernel/rms*&
_output_shapes
:*
dtype0
�
RMSprop/conv2d_10/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameRMSprop/conv2d_10/bias/rms
�
.RMSprop/conv2d_10/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_10/bias/rms*
_output_shapes
:*
dtype0

NoOpNoOp
�"
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�!
value�!B�! B�!
�
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
	trainable_variables

regularization_losses
	variables
	keras_api

signatures
 
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
R
 trainable_variables
!regularization_losses
"	variables
#	keras_api
h

$kernel
%bias
&trainable_variables
'regularization_losses
(	variables
)	keras_api
k
*iter
	+decay
,learning_rate
-momentum
.rho	rms]	rms^	$rms_	%rms`

0
1
$2
%3
 

0
1
$2
%3
�
/non_trainable_variables
	trainable_variables

0layers
1layer_metrics

regularization_losses
	variables
2layer_regularization_losses
3metrics
 
 
 
 
�
4non_trainable_variables

5layers
trainable_variables
6layer_metrics
regularization_losses
	variables
7layer_regularization_losses
8metrics
[Y
VARIABLE_VALUEconv2d_8/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_8/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
9non_trainable_variables

:layers
trainable_variables
;layer_metrics
regularization_losses
	variables
<layer_regularization_losses
=metrics
 
 
 
�
>non_trainable_variables

?layers
trainable_variables
@layer_metrics
regularization_losses
	variables
Alayer_regularization_losses
Bmetrics
 
 
 
�
Cnon_trainable_variables

Dlayers
trainable_variables
Elayer_metrics
regularization_losses
	variables
Flayer_regularization_losses
Gmetrics
 
 
 
�
Hnon_trainable_variables

Ilayers
 trainable_variables
Jlayer_metrics
!regularization_losses
"	variables
Klayer_regularization_losses
Lmetrics
\Z
VARIABLE_VALUEconv2d_10/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_10/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1
 

$0
%1
�
Mnon_trainable_variables

Nlayers
&trainable_variables
Olayer_metrics
'regularization_losses
(	variables
Player_regularization_losses
Qmetrics
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

R0
S1
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
��
VARIABLE_VALUERMSprop/conv2d_8/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
�
VARIABLE_VALUERMSprop/conv2d_8/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/conv2d_10/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/conv2d_10/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_3Placeholder*/
_output_shapes
:���������  *
dtype0*$
shape:���������  
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3conv2d_8/kernelconv2d_8/biasconv2d_10/kernelconv2d_10/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_23472
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOp$conv2d_10/kernel/Read/ReadVariableOp"conv2d_10/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp/RMSprop/conv2d_8/kernel/rms/Read/ReadVariableOp-RMSprop/conv2d_8/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_10/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_10/bias/rms/Read/ReadVariableOpConst*
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
GPU 2J 8� *'
f"R 
__inference__traced_save_23674
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_8/kernelconv2d_8/biasconv2d_10/kernelconv2d_10/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcounttotal_1count_1RMSprop/conv2d_8/kernel/rmsRMSprop/conv2d_8/bias/rmsRMSprop/conv2d_10/kernel/rmsRMSprop/conv2d_10/bias/rms*
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
GPU 2J 8� **
f%R#
!__inference__traced_restore_23735��
�$
�
G__inference_functional_5_layer_call_and_return_conditional_losses_23503

inputs+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource
identity��
zero_padding2d_3/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d_3/Pad/paddings�
zero_padding2d_3/PadPadinputs&zero_padding2d_3/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2
zero_padding2d_3/Pad�
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_8/Conv2D/ReadVariableOp�
conv2d_8/Conv2DConv2Dzero_padding2d_3/Pad:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
conv2d_8/Conv2D�
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp�
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2
conv2d_8/BiasAdd{
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
conv2d_8/Relu�
max_pooling2d_3/MaxPoolMaxPoolconv2d_8/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
2
max_pooling2d_3/MaxPool~
up_sampling2d_3/ShapeShape max_pooling2d_3/MaxPool:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/Shape�
#up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_3/strided_slice/stack�
%up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_1�
%up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_2�
up_sampling2d_3/strided_sliceStridedSliceup_sampling2d_3/Shape:output:0,up_sampling2d_3/strided_slice/stack:output:0.up_sampling2d_3/strided_slice/stack_1:output:0.up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_3/strided_slice
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/Const�
up_sampling2d_3/mulMul&up_sampling2d_3/strided_slice:output:0up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/mul�
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor max_pooling2d_3/MaxPool:output:0up_sampling2d_3/mul:z:0*
T0*/
_output_shapes
:���������  *
half_pixel_centers(2.
,up_sampling2d_3/resize/ResizeNearestNeighbor�
zero_padding2d_5/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d_5/Pad/paddings�
zero_padding2d_5/PadPad=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0&zero_padding2d_5/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2
zero_padding2d_5/Pad�
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_10/Conv2D/ReadVariableOp�
conv2d_10/Conv2DConv2Dzero_padding2d_5/Pad:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
conv2d_10/Conv2D�
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp�
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2
conv2d_10/BiasAdd~
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
conv2d_10/Relux
IdentityIdentityconv2d_10/Relu:activations:0*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������  :::::W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�	
�
D__inference_conv2d_10_layer_call_and_return_conditional_losses_23591

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2
Relu�
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������:::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
,__inference_functional_5_layer_call_fn_23418
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_functional_5_layer_call_and_return_conditional_losses_234072
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������  ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_3
�
g
K__inference_zero_padding2d_3_layer_call_and_return_conditional_losses_23255

inputs
identity�
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Pad/paddings�
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4������������������������������������2
Pad�
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
f
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_23286

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
strided_slice/stack_2�
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
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul�
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4������������������������������������*
half_pixel_centers(2
resize/ResizeNearestNeighbor�
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
,__inference_functional_5_layer_call_fn_23449
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_functional_5_layer_call_and_return_conditional_losses_234382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������  ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_3
�	
�
C__inference_conv2d_8_layer_call_and_return_conditional_losses_23571

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������"":::W S
/
_output_shapes
:���������""
 
_user_specified_nameinputs
�
L
0__inference_zero_padding2d_5_layer_call_fn_23305

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_zero_padding2d_5_layer_call_and_return_conditional_losses_232992
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�I
�
!__inference__traced_restore_23735
file_prefix$
 assignvariableop_conv2d_8_kernel$
 assignvariableop_1_conv2d_8_bias'
#assignvariableop_2_conv2d_10_kernel%
!assignvariableop_3_conv2d_10_bias#
assignvariableop_4_rmsprop_iter$
 assignvariableop_5_rmsprop_decay,
(assignvariableop_6_rmsprop_learning_rate'
#assignvariableop_7_rmsprop_momentum"
assignvariableop_8_rmsprop_rho
assignvariableop_9_total
assignvariableop_10_count
assignvariableop_11_total_1
assignvariableop_12_count_13
/assignvariableop_13_rmsprop_conv2d_8_kernel_rms1
-assignvariableop_14_rmsprop_conv2d_8_bias_rms4
0assignvariableop_15_rmsprop_conv2d_10_kernel_rms2
.assignvariableop_16_rmsprop_conv2d_10_bias_rms
identity_18��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
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

Identity�
AssignVariableOpAssignVariableOp assignvariableop_conv2d_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_10_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_10_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_rmsprop_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp assignvariableop_5_rmsprop_decayIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp(assignvariableop_6_rmsprop_learning_rateIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp#assignvariableop_7_rmsprop_momentumIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOpassignvariableop_8_rmsprop_rhoIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp/assignvariableop_13_rmsprop_conv2d_8_kernel_rmsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp-assignvariableop_14_rmsprop_conv2d_8_bias_rmsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp0assignvariableop_15_rmsprop_conv2d_10_kernel_rmsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp.assignvariableop_16_rmsprop_conv2d_10_bias_rmsIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_169
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_17Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_17�
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
�
�
G__inference_functional_5_layer_call_and_return_conditional_losses_23386
input_3
conv2d_8_23372
conv2d_8_23374
conv2d_10_23380
conv2d_10_23382
identity��!conv2d_10/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall�
 zero_padding2d_3/PartitionedCallPartitionedCallinput_3*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������""* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_zero_padding2d_3_layer_call_and_return_conditional_losses_232552"
 zero_padding2d_3/PartitionedCall�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_3/PartitionedCall:output:0conv2d_8_23372conv2d_8_23374*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_233212"
 conv2d_8/StatefulPartitionedCall�
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_232672!
max_pooling2d_3/PartitionedCall�
up_sampling2d_3/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_232862!
up_sampling2d_3/PartitionedCall�
 zero_padding2d_5/PartitionedCallPartitionedCall(up_sampling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_zero_padding2d_5_layer_call_and_return_conditional_losses_232992"
 zero_padding2d_5/PartitionedCall�
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_5/PartitionedCall:output:0conv2d_10_23380conv2d_10_23382*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_233512#
!conv2d_10/StatefulPartitionedCall�
IdentityIdentity*conv2d_10/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������  ::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_3
�
}
(__inference_conv2d_8_layer_call_fn_23580

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_233212
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������""::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������""
 
_user_specified_nameinputs
�-
�
__inference__traced_save_23674
file_prefix.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop/
+savev2_conv2d_10_kernel_read_readvariableop-
)savev2_conv2d_10_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop:
6savev2_rmsprop_conv2d_8_kernel_rms_read_readvariableop8
4savev2_rmsprop_conv2d_8_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_10_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_10_bias_rms_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
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
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_9542ee80c80947d98b90567740b948bd/part2	
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*7
value.B,B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableop+savev2_conv2d_10_kernel_read_readvariableop)savev2_conv2d_10_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop6savev2_rmsprop_conv2d_8_kernel_rms_read_readvariableop4savev2_rmsprop_conv2d_8_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_10_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_10_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 * 
dtypes
2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
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

identity_1Identity_1:output:0*�
_input_shapesx
v: ::::: : : : : : : : : ::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 
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
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::

_output_shapes
: 
�$
�
G__inference_functional_5_layer_call_and_return_conditional_losses_23534

inputs+
'conv2d_8_conv2d_readvariableop_resource,
(conv2d_8_biasadd_readvariableop_resource,
(conv2d_10_conv2d_readvariableop_resource-
)conv2d_10_biasadd_readvariableop_resource
identity��
zero_padding2d_3/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d_3/Pad/paddings�
zero_padding2d_3/PadPadinputs&zero_padding2d_3/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2
zero_padding2d_3/Pad�
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02 
conv2d_8/Conv2D/ReadVariableOp�
conv2d_8/Conv2DConv2Dzero_padding2d_3/Pad:output:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
conv2d_8/Conv2D�
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp�
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2
conv2d_8/BiasAdd{
conv2d_8/ReluReluconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
conv2d_8/Relu�
max_pooling2d_3/MaxPoolMaxPoolconv2d_8/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
2
max_pooling2d_3/MaxPool~
up_sampling2d_3/ShapeShape max_pooling2d_3/MaxPool:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/Shape�
#up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2%
#up_sampling2d_3/strided_slice/stack�
%up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_1�
%up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%up_sampling2d_3/strided_slice/stack_2�
up_sampling2d_3/strided_sliceStridedSliceup_sampling2d_3/Shape:output:0,up_sampling2d_3/strided_slice/stack:output:0.up_sampling2d_3/strided_slice/stack_1:output:0.up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
up_sampling2d_3/strided_slice
up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_3/Const�
up_sampling2d_3/mulMul&up_sampling2d_3/strided_slice:output:0up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_3/mul�
,up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor max_pooling2d_3/MaxPool:output:0up_sampling2d_3/mul:z:0*
T0*/
_output_shapes
:���������  *
half_pixel_centers(2.
,up_sampling2d_3/resize/ResizeNearestNeighbor�
zero_padding2d_5/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
zero_padding2d_5/Pad/paddings�
zero_padding2d_5/PadPad=up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:0&zero_padding2d_5/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2
zero_padding2d_5/Pad�
conv2d_10/Conv2D/ReadVariableOpReadVariableOp(conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_10/Conv2D/ReadVariableOp�
conv2d_10/Conv2DConv2Dzero_padding2d_5/Pad:output:0'conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
conv2d_10/Conv2D�
 conv2d_10/BiasAdd/ReadVariableOpReadVariableOp)conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_10/BiasAdd/ReadVariableOp�
conv2d_10/BiasAddBiasAddconv2d_10/Conv2D:output:0(conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2
conv2d_10/BiasAdd~
conv2d_10/ReluReluconv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
conv2d_10/Relux
IdentityIdentityconv2d_10/Relu:activations:0*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������  :::::W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
�
G__inference_functional_5_layer_call_and_return_conditional_losses_23438

inputs
conv2d_8_23424
conv2d_8_23426
conv2d_10_23432
conv2d_10_23434
identity��!conv2d_10/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall�
 zero_padding2d_3/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������""* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_zero_padding2d_3_layer_call_and_return_conditional_losses_232552"
 zero_padding2d_3/PartitionedCall�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_3/PartitionedCall:output:0conv2d_8_23424conv2d_8_23426*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_233212"
 conv2d_8/StatefulPartitionedCall�
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_232672!
max_pooling2d_3/PartitionedCall�
up_sampling2d_3/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_232862!
up_sampling2d_3/PartitionedCall�
 zero_padding2d_5/PartitionedCallPartitionedCall(up_sampling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_zero_padding2d_5_layer_call_and_return_conditional_losses_232992"
 zero_padding2d_5/PartitionedCall�
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_5/PartitionedCall:output:0conv2d_10_23432conv2d_10_23434*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_233512#
!conv2d_10/StatefulPartitionedCall�
IdentityIdentity*conv2d_10/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������  ::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
K
/__inference_max_pooling2d_3_layer_call_fn_23273

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_232672
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�,
�
 __inference__wrapped_model_23248
input_38
4functional_5_conv2d_8_conv2d_readvariableop_resource9
5functional_5_conv2d_8_biasadd_readvariableop_resource9
5functional_5_conv2d_10_conv2d_readvariableop_resource:
6functional_5_conv2d_10_biasadd_readvariableop_resource
identity��
*functional_5/zero_padding2d_3/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2,
*functional_5/zero_padding2d_3/Pad/paddings�
!functional_5/zero_padding2d_3/PadPadinput_33functional_5/zero_padding2d_3/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2#
!functional_5/zero_padding2d_3/Pad�
+functional_5/conv2d_8/Conv2D/ReadVariableOpReadVariableOp4functional_5_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02-
+functional_5/conv2d_8/Conv2D/ReadVariableOp�
functional_5/conv2d_8/Conv2DConv2D*functional_5/zero_padding2d_3/Pad:output:03functional_5/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
functional_5/conv2d_8/Conv2D�
,functional_5/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp5functional_5_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,functional_5/conv2d_8/BiasAdd/ReadVariableOp�
functional_5/conv2d_8/BiasAddBiasAdd%functional_5/conv2d_8/Conv2D:output:04functional_5/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2
functional_5/conv2d_8/BiasAdd�
functional_5/conv2d_8/ReluRelu&functional_5/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
functional_5/conv2d_8/Relu�
$functional_5/max_pooling2d_3/MaxPoolMaxPool(functional_5/conv2d_8/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
2&
$functional_5/max_pooling2d_3/MaxPool�
"functional_5/up_sampling2d_3/ShapeShape-functional_5/max_pooling2d_3/MaxPool:output:0*
T0*
_output_shapes
:2$
"functional_5/up_sampling2d_3/Shape�
0functional_5/up_sampling2d_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:22
0functional_5/up_sampling2d_3/strided_slice/stack�
2functional_5/up_sampling2d_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2functional_5/up_sampling2d_3/strided_slice/stack_1�
2functional_5/up_sampling2d_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2functional_5/up_sampling2d_3/strided_slice/stack_2�
*functional_5/up_sampling2d_3/strided_sliceStridedSlice+functional_5/up_sampling2d_3/Shape:output:09functional_5/up_sampling2d_3/strided_slice/stack:output:0;functional_5/up_sampling2d_3/strided_slice/stack_1:output:0;functional_5/up_sampling2d_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2,
*functional_5/up_sampling2d_3/strided_slice�
"functional_5/up_sampling2d_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2$
"functional_5/up_sampling2d_3/Const�
 functional_5/up_sampling2d_3/mulMul3functional_5/up_sampling2d_3/strided_slice:output:0+functional_5/up_sampling2d_3/Const:output:0*
T0*
_output_shapes
:2"
 functional_5/up_sampling2d_3/mul�
9functional_5/up_sampling2d_3/resize/ResizeNearestNeighborResizeNearestNeighbor-functional_5/max_pooling2d_3/MaxPool:output:0$functional_5/up_sampling2d_3/mul:z:0*
T0*/
_output_shapes
:���������  *
half_pixel_centers(2;
9functional_5/up_sampling2d_3/resize/ResizeNearestNeighbor�
*functional_5/zero_padding2d_5/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2,
*functional_5/zero_padding2d_5/Pad/paddings�
!functional_5/zero_padding2d_5/PadPadJfunctional_5/up_sampling2d_3/resize/ResizeNearestNeighbor:resized_images:03functional_5/zero_padding2d_5/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2#
!functional_5/zero_padding2d_5/Pad�
,functional_5/conv2d_10/Conv2D/ReadVariableOpReadVariableOp5functional_5_conv2d_10_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02.
,functional_5/conv2d_10/Conv2D/ReadVariableOp�
functional_5/conv2d_10/Conv2DConv2D*functional_5/zero_padding2d_5/Pad:output:04functional_5/conv2d_10/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
functional_5/conv2d_10/Conv2D�
-functional_5/conv2d_10/BiasAdd/ReadVariableOpReadVariableOp6functional_5_conv2d_10_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-functional_5/conv2d_10/BiasAdd/ReadVariableOp�
functional_5/conv2d_10/BiasAddBiasAdd&functional_5/conv2d_10/Conv2D:output:05functional_5/conv2d_10/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2 
functional_5/conv2d_10/BiasAdd�
functional_5/conv2d_10/ReluRelu'functional_5/conv2d_10/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
functional_5/conv2d_10/Relu�
IdentityIdentity)functional_5/conv2d_10/Relu:activations:0*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������  :::::X T
/
_output_shapes
:���������  
!
_user_specified_name	input_3
�
�
G__inference_functional_5_layer_call_and_return_conditional_losses_23407

inputs
conv2d_8_23393
conv2d_8_23395
conv2d_10_23401
conv2d_10_23403
identity��!conv2d_10/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall�
 zero_padding2d_3/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������""* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_zero_padding2d_3_layer_call_and_return_conditional_losses_232552"
 zero_padding2d_3/PartitionedCall�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_3/PartitionedCall:output:0conv2d_8_23393conv2d_8_23395*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_233212"
 conv2d_8/StatefulPartitionedCall�
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_232672!
max_pooling2d_3/PartitionedCall�
up_sampling2d_3/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_232862!
up_sampling2d_3/PartitionedCall�
 zero_padding2d_5/PartitionedCallPartitionedCall(up_sampling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_zero_padding2d_5_layer_call_and_return_conditional_losses_232992"
 zero_padding2d_5/PartitionedCall�
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_5/PartitionedCall:output:0conv2d_10_23401conv2d_10_23403*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_233512#
!conv2d_10/StatefulPartitionedCall�
IdentityIdentity*conv2d_10/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������  ::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�	
�
D__inference_conv2d_10_layer_call_and_return_conditional_losses_23351

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2
Relu�
IdentityIdentityRelu:activations:0*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������:::i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
,__inference_functional_5_layer_call_fn_23547

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_functional_5_layer_call_and_return_conditional_losses_234072
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������  ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�	
�
C__inference_conv2d_8_layer_call_and_return_conditional_losses_23321

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������"":::W S
/
_output_shapes
:���������""
 
_user_specified_nameinputs
�
�
G__inference_functional_5_layer_call_and_return_conditional_losses_23368
input_3
conv2d_8_23332
conv2d_8_23334
conv2d_10_23362
conv2d_10_23364
identity��!conv2d_10/StatefulPartitionedCall� conv2d_8/StatefulPartitionedCall�
 zero_padding2d_3/PartitionedCallPartitionedCallinput_3*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������""* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_zero_padding2d_3_layer_call_and_return_conditional_losses_232552"
 zero_padding2d_3/PartitionedCall�
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_3/PartitionedCall:output:0conv2d_8_23332conv2d_8_23334*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_8_layer_call_and_return_conditional_losses_233212"
 conv2d_8/StatefulPartitionedCall�
max_pooling2d_3/PartitionedCallPartitionedCall)conv2d_8/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_232672!
max_pooling2d_3/PartitionedCall�
up_sampling2d_3/PartitionedCallPartitionedCall(max_pooling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_232862!
up_sampling2d_3/PartitionedCall�
 zero_padding2d_5/PartitionedCallPartitionedCall(up_sampling2d_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_zero_padding2d_5_layer_call_and_return_conditional_losses_232992"
 zero_padding2d_5/PartitionedCall�
!conv2d_10/StatefulPartitionedCallStatefulPartitionedCall)zero_padding2d_5/PartitionedCall:output:0conv2d_10_23362conv2d_10_23364*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_233512#
!conv2d_10/StatefulPartitionedCall�
IdentityIdentity*conv2d_10/StatefulPartitionedCall:output:0"^conv2d_10/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������  ::::2F
!conv2d_10/StatefulPartitionedCall!conv2d_10/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_3
�
L
0__inference_zero_padding2d_3_layer_call_fn_23261

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *T
fORM
K__inference_zero_padding2d_3_layer_call_and_return_conditional_losses_232552
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
K
/__inference_up_sampling2d_3_layer_call_fn_23292

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *S
fNRL
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_232862
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
g
K__inference_zero_padding2d_5_layer_call_and_return_conditional_losses_23299

inputs
identity�
Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2
Pad/paddings�
PadPadinputsPad/paddings:output:0*
T0*J
_output_shapes8
6:4������������������������������������2
Pad�
IdentityIdentityPad:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
~
)__inference_conv2d_10_layer_call_fn_23600

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_conv2d_10_layer_call_and_return_conditional_losses_233512
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_23472
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_232482
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������  ::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:���������  
!
_user_specified_name	input_3
�
f
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_23267

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
2	
MaxPool�
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
,__inference_functional_5_layer_call_fn_23560

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *P
fKRI
G__inference_functional_5_layer_call_and_return_conditional_losses_234382
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������  ::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
input_38
serving_default_input_3:0���������  E
	conv2d_108
StatefulPartitionedCall:0���������  tensorflow/serving/predict:��
�;
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
	trainable_variables

regularization_losses
	variables
	keras_api

signatures
a_default_save_signature
b__call__
*c&call_and_return_all_conditional_losses"�8
_tf_keras_network�8{"class_name": "Functional", "name": "functional_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_3", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_3", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_8", "inbound_nodes": [[["zero_padding2d_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_3", "inbound_nodes": [[["conv2d_8", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_3", "inbound_nodes": [[["max_pooling2d_3", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_5", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_5", "inbound_nodes": [[["up_sampling2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["zero_padding2d_5", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["conv2d_10", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_3", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_3", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_8", "inbound_nodes": [[["zero_padding2d_3", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_3", "inbound_nodes": [[["conv2d_8", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_3", "inbound_nodes": [[["max_pooling2d_3", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_5", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_5", "inbound_nodes": [[["up_sampling2d_3", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_10", "inbound_nodes": [[["zero_padding2d_5", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["conv2d_10", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
�
trainable_variables
regularization_losses
	variables
	keras_api
d__call__
*e&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "ZeroPadding2D", "name": "zero_padding2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "zero_padding2d_3", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
f__call__
*g&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 34, 34, 3]}}
�
trainable_variables
regularization_losses
	variables
	keras_api
h__call__
*i&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_3", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
trainable_variables
regularization_losses
	variables
	keras_api
j__call__
*k&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "UpSampling2D", "name": "up_sampling2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_3", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
 trainable_variables
!regularization_losses
"	variables
#	keras_api
l__call__
*m&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "ZeroPadding2D", "name": "zero_padding2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "zero_padding2d_5", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

$kernel
%bias
&trainable_variables
'regularization_losses
(	variables
)	keras_api
n__call__
*o&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_10", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 34, 34, 3]}}
~
*iter
	+decay
,learning_rate
-momentum
.rho	rms]	rms^	$rms_	%rms`"
	optimizer
<
0
1
$2
%3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
$2
%3"
trackable_list_wrapper
�
/non_trainable_variables
	trainable_variables

0layers
1layer_metrics

regularization_losses
	variables
2layer_regularization_losses
3metrics
b__call__
a_default_save_signature
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
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
�
4non_trainable_variables

5layers
trainable_variables
6layer_metrics
regularization_losses
	variables
7layer_regularization_losses
8metrics
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
):'2conv2d_8/kernel
:2conv2d_8/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
9non_trainable_variables

:layers
trainable_variables
;layer_metrics
regularization_losses
	variables
<layer_regularization_losses
=metrics
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
�
>non_trainable_variables

?layers
trainable_variables
@layer_metrics
regularization_losses
	variables
Alayer_regularization_losses
Bmetrics
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
�
Cnon_trainable_variables

Dlayers
trainable_variables
Elayer_metrics
regularization_losses
	variables
Flayer_regularization_losses
Gmetrics
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
�
Hnon_trainable_variables

Ilayers
 trainable_variables
Jlayer_metrics
!regularization_losses
"	variables
Klayer_regularization_losses
Lmetrics
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
*:(2conv2d_10/kernel
:2conv2d_10/bias
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
�
Mnon_trainable_variables

Nlayers
&trainable_variables
Olayer_metrics
'regularization_losses
(	variables
Player_regularization_losses
Qmetrics
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
.
R0
S1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
	Ttotal
	Ucount
V	variables
W	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
�
	Xtotal
	Ycount
Z
_fn_kwargs
[	variables
\	keras_api"�
_tf_keras_metric�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "categorical_accuracy"}}
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
3:12RMSprop/conv2d_8/kernel/rms
%:#2RMSprop/conv2d_8/bias/rms
4:22RMSprop/conv2d_10/kernel/rms
&:$2RMSprop/conv2d_10/bias/rms
�2�
 __inference__wrapped_model_23248�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *.�+
)�&
input_3���������  
�2�
,__inference_functional_5_layer_call_fn_23547
,__inference_functional_5_layer_call_fn_23418
,__inference_functional_5_layer_call_fn_23449
,__inference_functional_5_layer_call_fn_23560�
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
G__inference_functional_5_layer_call_and_return_conditional_losses_23368
G__inference_functional_5_layer_call_and_return_conditional_losses_23503
G__inference_functional_5_layer_call_and_return_conditional_losses_23534
G__inference_functional_5_layer_call_and_return_conditional_losses_23386�
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
�2�
0__inference_zero_padding2d_3_layer_call_fn_23261�
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
annotations� *@�=
;�84������������������������������������
�2�
K__inference_zero_padding2d_3_layer_call_and_return_conditional_losses_23255�
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
annotations� *@�=
;�84������������������������������������
�2�
(__inference_conv2d_8_layer_call_fn_23580�
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
C__inference_conv2d_8_layer_call_and_return_conditional_losses_23571�
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
�2�
/__inference_max_pooling2d_3_layer_call_fn_23273�
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
annotations� *@�=
;�84������������������������������������
�2�
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_23267�
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
annotations� *@�=
;�84������������������������������������
�2�
/__inference_up_sampling2d_3_layer_call_fn_23292�
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
annotations� *@�=
;�84������������������������������������
�2�
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_23286�
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
annotations� *@�=
;�84������������������������������������
�2�
0__inference_zero_padding2d_5_layer_call_fn_23305�
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
annotations� *@�=
;�84������������������������������������
�2�
K__inference_zero_padding2d_5_layer_call_and_return_conditional_losses_23299�
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
annotations� *@�=
;�84������������������������������������
�2�
)__inference_conv2d_10_layer_call_fn_23600�
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
D__inference_conv2d_10_layer_call_and_return_conditional_losses_23591�
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
2B0
#__inference_signature_wrapper_23472input_3�
 __inference__wrapped_model_23248$%8�5
.�+
)�&
input_3���������  
� "=�:
8
	conv2d_10+�(
	conv2d_10���������  �
D__inference_conv2d_10_layer_call_and_return_conditional_losses_23591�$%I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
)__inference_conv2d_10_layer_call_fn_23600�$%I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
C__inference_conv2d_8_layer_call_and_return_conditional_losses_23571l7�4
-�*
(�%
inputs���������""
� "-�*
#� 
0���������  
� �
(__inference_conv2d_8_layer_call_fn_23580_7�4
-�*
(�%
inputs���������""
� " ����������  �
G__inference_functional_5_layer_call_and_return_conditional_losses_23368�$%@�=
6�3
)�&
input_3���������  
p

 
� "?�<
5�2
0+���������������������������
� �
G__inference_functional_5_layer_call_and_return_conditional_losses_23386�$%@�=
6�3
)�&
input_3���������  
p 

 
� "?�<
5�2
0+���������������������������
� �
G__inference_functional_5_layer_call_and_return_conditional_losses_23503v$%?�<
5�2
(�%
inputs���������  
p

 
� "-�*
#� 
0���������  
� �
G__inference_functional_5_layer_call_and_return_conditional_losses_23534v$%?�<
5�2
(�%
inputs���������  
p 

 
� "-�*
#� 
0���������  
� �
,__inference_functional_5_layer_call_fn_23418|$%@�=
6�3
)�&
input_3���������  
p

 
� "2�/+����������������������������
,__inference_functional_5_layer_call_fn_23449|$%@�=
6�3
)�&
input_3���������  
p 

 
� "2�/+����������������������������
,__inference_functional_5_layer_call_fn_23547{$%?�<
5�2
(�%
inputs���������  
p

 
� "2�/+����������������������������
,__inference_functional_5_layer_call_fn_23560{$%?�<
5�2
(�%
inputs���������  
p 

 
� "2�/+����������������������������
J__inference_max_pooling2d_3_layer_call_and_return_conditional_losses_23267�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
/__inference_max_pooling2d_3_layer_call_fn_23273�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
#__inference_signature_wrapper_23472�$%C�@
� 
9�6
4
input_3)�&
input_3���������  "=�:
8
	conv2d_10+�(
	conv2d_10���������  �
J__inference_up_sampling2d_3_layer_call_and_return_conditional_losses_23286�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
/__inference_up_sampling2d_3_layer_call_fn_23292�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
K__inference_zero_padding2d_3_layer_call_and_return_conditional_losses_23255�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
0__inference_zero_padding2d_3_layer_call_fn_23261�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
K__inference_zero_padding2d_5_layer_call_and_return_conditional_losses_23299�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
0__inference_zero_padding2d_5_layer_call_fn_23305�R�O
H�E
C�@
inputs4������������������������������������
� ";�84������������������������������������