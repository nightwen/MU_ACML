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
 �"serve*2.3.02v2.3.0-0-gb36436b0878߽
�
conv2d_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_49/kernel
}
$conv2d_49/kernel/Read/ReadVariableOpReadVariableOpconv2d_49/kernel*&
_output_shapes
:*
dtype0
t
conv2d_49/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_49/bias
m
"conv2d_49/bias/Read/ReadVariableOpReadVariableOpconv2d_49/bias*
_output_shapes
:*
dtype0
�
conv2d_51/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_51/kernel
}
$conv2d_51/kernel/Read/ReadVariableOpReadVariableOpconv2d_51/kernel*&
_output_shapes
:*
dtype0
t
conv2d_51/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_51/bias
m
"conv2d_51/bias/Read/ReadVariableOpReadVariableOpconv2d_51/bias*
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
RMSprop/conv2d_49/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameRMSprop/conv2d_49/kernel/rms
�
0RMSprop/conv2d_49/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_49/kernel/rms*&
_output_shapes
:*
dtype0
�
RMSprop/conv2d_49/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameRMSprop/conv2d_49/bias/rms
�
.RMSprop/conv2d_49/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_49/bias/rms*
_output_shapes
:*
dtype0
�
RMSprop/conv2d_51/kernel/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_nameRMSprop/conv2d_51/kernel/rms
�
0RMSprop/conv2d_51/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_51/kernel/rms*&
_output_shapes
:*
dtype0
�
RMSprop/conv2d_51/bias/rmsVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameRMSprop/conv2d_51/bias/rms
�
.RMSprop/conv2d_51/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/conv2d_51/bias/rms*
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
�
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
�
regularization_losses
4layer_regularization_losses
5non_trainable_variables
	variables
6metrics

7layers
trainable_variables
8layer_metrics
\Z
VARIABLE_VALUEconv2d_49/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_49/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
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
�
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
�
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
�
 regularization_losses
Hlayer_regularization_losses
Inon_trainable_variables
!	variables
Jmetrics

Klayers
"trainable_variables
Llayer_metrics
\Z
VARIABLE_VALUEconv2d_51/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_51/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

$0
%1

$0
%1
�
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
��
VARIABLE_VALUERMSprop/conv2d_49/kernel/rmsTlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/conv2d_49/bias/rmsRlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/conv2d_51/kernel/rmsTlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUERMSprop/conv2d_51/bias/rmsRlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
�
serving_default_input_16Placeholder*/
_output_shapes
:���������  *
dtype0*$
shape:���������  
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_16conv2d_49/kernelconv2d_49/biasconv2d_51/kernelconv2d_51/bias*
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
#__inference_signature_wrapper_82110
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_49/kernel/Read/ReadVariableOp"conv2d_49/bias/Read/ReadVariableOp$conv2d_51/kernel/Read/ReadVariableOp"conv2d_51/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp0RMSprop/conv2d_49/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_49/bias/rms/Read/ReadVariableOp0RMSprop/conv2d_51/kernel/rms/Read/ReadVariableOp.RMSprop/conv2d_51/bias/rms/Read/ReadVariableOpConst*
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
__inference__traced_save_82312
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_49/kernelconv2d_49/biasconv2d_51/kernelconv2d_51/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rhototalcounttotal_1count_1RMSprop/conv2d_49/kernel/rmsRMSprop/conv2d_49/bias/rmsRMSprop/conv2d_51/kernel/rmsRMSprop/conv2d_51/bias/rms*
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
!__inference__traced_restore_82373��
�
~
)__inference_conv2d_51_layer_call_fn_82238

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
D__inference_conv2d_51_layer_call_and_return_conditional_losses_819892
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
�	
�
D__inference_conv2d_49_layer_call_and_return_conditional_losses_82209

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
H__inference_functional_31_layer_call_and_return_conditional_losses_82045

inputs
conv2d_49_82031
conv2d_49_82033
conv2d_51_82039
conv2d_51_82041
identity��!conv2d_49/StatefulPartitionedCall�!conv2d_51/StatefulPartitionedCall�
!zero_padding2d_39/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8� *U
fPRN
L__inference_zero_padding2d_39_layer_call_and_return_conditional_losses_818932#
!zero_padding2d_39/PartitionedCall�
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_39/PartitionedCall:output:0conv2d_49_82031conv2d_49_82033*
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
GPU 2J 8� *M
fHRF
D__inference_conv2d_49_layer_call_and_return_conditional_losses_819592#
!conv2d_49/StatefulPartitionedCall�
 max_pooling2d_17/PartitionedCallPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *T
fORM
K__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_819052"
 max_pooling2d_17/PartitionedCall�
 up_sampling2d_17/PartitionedCallPartitionedCall)max_pooling2d_17/PartitionedCall:output:0*
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
K__inference_up_sampling2d_17_layer_call_and_return_conditional_losses_819242"
 up_sampling2d_17/PartitionedCall�
!zero_padding2d_41/PartitionedCallPartitionedCall)up_sampling2d_17/PartitionedCall:output:0*
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
GPU 2J 8� *U
fPRN
L__inference_zero_padding2d_41_layer_call_and_return_conditional_losses_819372#
!zero_padding2d_41/PartitionedCall�
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_41/PartitionedCall:output:0conv2d_51_82039conv2d_51_82041*
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
D__inference_conv2d_51_layer_call_and_return_conditional_losses_819892#
!conv2d_51/StatefulPartitionedCall�
IdentityIdentity*conv2d_51/StatefulPartitionedCall:output:0"^conv2d_49/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������  ::::2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall:W S
/
_output_shapes
:���������  
 
_user_specified_nameinputs
�
L
0__inference_max_pooling2d_17_layer_call_fn_81911

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
K__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_819052
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
�
�
#__inference_signature_wrapper_82110
input_16
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_16unknown	unknown_0	unknown_1	unknown_2*
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
 __inference__wrapped_model_818862
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������  ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
input_16
�	
�
D__inference_conv2d_51_layer_call_and_return_conditional_losses_81989

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
�
�
H__inference_functional_31_layer_call_and_return_conditional_losses_82024
input_16
conv2d_49_82010
conv2d_49_82012
conv2d_51_82018
conv2d_51_82020
identity��!conv2d_49/StatefulPartitionedCall�!conv2d_51/StatefulPartitionedCall�
!zero_padding2d_39/PartitionedCallPartitionedCallinput_16*
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
GPU 2J 8� *U
fPRN
L__inference_zero_padding2d_39_layer_call_and_return_conditional_losses_818932#
!zero_padding2d_39/PartitionedCall�
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_39/PartitionedCall:output:0conv2d_49_82010conv2d_49_82012*
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
GPU 2J 8� *M
fHRF
D__inference_conv2d_49_layer_call_and_return_conditional_losses_819592#
!conv2d_49/StatefulPartitionedCall�
 max_pooling2d_17/PartitionedCallPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *T
fORM
K__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_819052"
 max_pooling2d_17/PartitionedCall�
 up_sampling2d_17/PartitionedCallPartitionedCall)max_pooling2d_17/PartitionedCall:output:0*
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
K__inference_up_sampling2d_17_layer_call_and_return_conditional_losses_819242"
 up_sampling2d_17/PartitionedCall�
!zero_padding2d_41/PartitionedCallPartitionedCall)up_sampling2d_17/PartitionedCall:output:0*
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
GPU 2J 8� *U
fPRN
L__inference_zero_padding2d_41_layer_call_and_return_conditional_losses_819372#
!zero_padding2d_41/PartitionedCall�
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_41/PartitionedCall:output:0conv2d_51_82018conv2d_51_82020*
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
D__inference_conv2d_51_layer_call_and_return_conditional_losses_819892#
!conv2d_51/StatefulPartitionedCall�
IdentityIdentity*conv2d_51/StatefulPartitionedCall:output:0"^conv2d_49/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������  ::::2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
input_16
�
g
K__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_81905

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
�
h
L__inference_zero_padding2d_39_layer_call_and_return_conditional_losses_81893

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
�-
�
__inference__traced_save_82312
file_prefix/
+savev2_conv2d_49_kernel_read_readvariableop-
)savev2_conv2d_49_bias_read_readvariableop/
+savev2_conv2d_51_kernel_read_readvariableop-
)savev2_conv2d_51_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop;
7savev2_rmsprop_conv2d_49_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_49_bias_rms_read_readvariableop;
7savev2_rmsprop_conv2d_51_kernel_rms_read_readvariableop9
5savev2_rmsprop_conv2d_51_bias_rms_read_readvariableop
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
value3B1 B+_temp_6e9c02a31c7e4785af613428bd661158/part2	
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_49_kernel_read_readvariableop)savev2_conv2d_49_bias_read_readvariableop+savev2_conv2d_51_kernel_read_readvariableop)savev2_conv2d_51_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop7savev2_rmsprop_conv2d_49_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_49_bias_rms_read_readvariableop7savev2_rmsprop_conv2d_51_kernel_rms_read_readvariableop5savev2_rmsprop_conv2d_51_bias_rms_read_readvariableopsavev2_const"/device:CPU:0*
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
�
�
-__inference_functional_31_layer_call_fn_82185

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
GPU 2J 8� *Q
fLRJ
H__inference_functional_31_layer_call_and_return_conditional_losses_820452
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
�
L
0__inference_up_sampling2d_17_layer_call_fn_81930

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
K__inference_up_sampling2d_17_layer_call_and_return_conditional_losses_819242
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
�
�
-__inference_functional_31_layer_call_fn_82056
input_16
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_16unknown	unknown_0	unknown_1	unknown_2*
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
GPU 2J 8� *Q
fLRJ
H__inference_functional_31_layer_call_and_return_conditional_losses_820452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������  ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
input_16
�
�
H__inference_functional_31_layer_call_and_return_conditional_losses_82006
input_16
conv2d_49_81970
conv2d_49_81972
conv2d_51_82000
conv2d_51_82002
identity��!conv2d_49/StatefulPartitionedCall�!conv2d_51/StatefulPartitionedCall�
!zero_padding2d_39/PartitionedCallPartitionedCallinput_16*
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
GPU 2J 8� *U
fPRN
L__inference_zero_padding2d_39_layer_call_and_return_conditional_losses_818932#
!zero_padding2d_39/PartitionedCall�
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_39/PartitionedCall:output:0conv2d_49_81970conv2d_49_81972*
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
GPU 2J 8� *M
fHRF
D__inference_conv2d_49_layer_call_and_return_conditional_losses_819592#
!conv2d_49/StatefulPartitionedCall�
 max_pooling2d_17/PartitionedCallPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *T
fORM
K__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_819052"
 max_pooling2d_17/PartitionedCall�
 up_sampling2d_17/PartitionedCallPartitionedCall)max_pooling2d_17/PartitionedCall:output:0*
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
K__inference_up_sampling2d_17_layer_call_and_return_conditional_losses_819242"
 up_sampling2d_17/PartitionedCall�
!zero_padding2d_41/PartitionedCallPartitionedCall)up_sampling2d_17/PartitionedCall:output:0*
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
GPU 2J 8� *U
fPRN
L__inference_zero_padding2d_41_layer_call_and_return_conditional_losses_819372#
!zero_padding2d_41/PartitionedCall�
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_41/PartitionedCall:output:0conv2d_51_82000conv2d_51_82002*
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
D__inference_conv2d_51_layer_call_and_return_conditional_losses_819892#
!conv2d_51/StatefulPartitionedCall�
IdentityIdentity*conv2d_51/StatefulPartitionedCall:output:0"^conv2d_49/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������  ::::2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
input_16
�
M
1__inference_zero_padding2d_41_layer_call_fn_81943

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
GPU 2J 8� *U
fPRN
L__inference_zero_padding2d_41_layer_call_and_return_conditional_losses_819372
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
�	
�
D__inference_conv2d_49_layer_call_and_return_conditional_losses_81959

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
~
)__inference_conv2d_49_layer_call_fn_82218

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
GPU 2J 8� *M
fHRF
D__inference_conv2d_49_layer_call_and_return_conditional_losses_819592
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
�
M
1__inference_zero_padding2d_39_layer_call_fn_81899

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
GPU 2J 8� *U
fPRN
L__inference_zero_padding2d_39_layer_call_and_return_conditional_losses_818932
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
�-
�
 __inference__wrapped_model_81886
input_16:
6functional_31_conv2d_49_conv2d_readvariableop_resource;
7functional_31_conv2d_49_biasadd_readvariableop_resource:
6functional_31_conv2d_51_conv2d_readvariableop_resource;
7functional_31_conv2d_51_biasadd_readvariableop_resource
identity��
,functional_31/zero_padding2d_39/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2.
,functional_31/zero_padding2d_39/Pad/paddings�
#functional_31/zero_padding2d_39/PadPadinput_165functional_31/zero_padding2d_39/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2%
#functional_31/zero_padding2d_39/Pad�
-functional_31/conv2d_49/Conv2D/ReadVariableOpReadVariableOp6functional_31_conv2d_49_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-functional_31/conv2d_49/Conv2D/ReadVariableOp�
functional_31/conv2d_49/Conv2DConv2D,functional_31/zero_padding2d_39/Pad:output:05functional_31/conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2 
functional_31/conv2d_49/Conv2D�
.functional_31/conv2d_49/BiasAdd/ReadVariableOpReadVariableOp7functional_31_conv2d_49_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.functional_31/conv2d_49/BiasAdd/ReadVariableOp�
functional_31/conv2d_49/BiasAddBiasAdd'functional_31/conv2d_49/Conv2D:output:06functional_31/conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2!
functional_31/conv2d_49/BiasAdd�
functional_31/conv2d_49/ReluRelu(functional_31/conv2d_49/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
functional_31/conv2d_49/Relu�
&functional_31/max_pooling2d_17/MaxPoolMaxPool*functional_31/conv2d_49/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
2(
&functional_31/max_pooling2d_17/MaxPool�
$functional_31/up_sampling2d_17/ShapeShape/functional_31/max_pooling2d_17/MaxPool:output:0*
T0*
_output_shapes
:2&
$functional_31/up_sampling2d_17/Shape�
2functional_31/up_sampling2d_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:24
2functional_31/up_sampling2d_17/strided_slice/stack�
4functional_31/up_sampling2d_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_31/up_sampling2d_17/strided_slice/stack_1�
4functional_31/up_sampling2d_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4functional_31/up_sampling2d_17/strided_slice/stack_2�
,functional_31/up_sampling2d_17/strided_sliceStridedSlice-functional_31/up_sampling2d_17/Shape:output:0;functional_31/up_sampling2d_17/strided_slice/stack:output:0=functional_31/up_sampling2d_17/strided_slice/stack_1:output:0=functional_31/up_sampling2d_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2.
,functional_31/up_sampling2d_17/strided_slice�
$functional_31/up_sampling2d_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2&
$functional_31/up_sampling2d_17/Const�
"functional_31/up_sampling2d_17/mulMul5functional_31/up_sampling2d_17/strided_slice:output:0-functional_31/up_sampling2d_17/Const:output:0*
T0*
_output_shapes
:2$
"functional_31/up_sampling2d_17/mul�
;functional_31/up_sampling2d_17/resize/ResizeNearestNeighborResizeNearestNeighbor/functional_31/max_pooling2d_17/MaxPool:output:0&functional_31/up_sampling2d_17/mul:z:0*
T0*/
_output_shapes
:���������  *
half_pixel_centers(2=
;functional_31/up_sampling2d_17/resize/ResizeNearestNeighbor�
,functional_31/zero_padding2d_41/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2.
,functional_31/zero_padding2d_41/Pad/paddings�
#functional_31/zero_padding2d_41/PadPadLfunctional_31/up_sampling2d_17/resize/ResizeNearestNeighbor:resized_images:05functional_31/zero_padding2d_41/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2%
#functional_31/zero_padding2d_41/Pad�
-functional_31/conv2d_51/Conv2D/ReadVariableOpReadVariableOp6functional_31_conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02/
-functional_31/conv2d_51/Conv2D/ReadVariableOp�
functional_31/conv2d_51/Conv2DConv2D,functional_31/zero_padding2d_41/Pad:output:05functional_31/conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2 
functional_31/conv2d_51/Conv2D�
.functional_31/conv2d_51/BiasAdd/ReadVariableOpReadVariableOp7functional_31_conv2d_51_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.functional_31/conv2d_51/BiasAdd/ReadVariableOp�
functional_31/conv2d_51/BiasAddBiasAdd'functional_31/conv2d_51/Conv2D:output:06functional_31/conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2!
functional_31/conv2d_51/BiasAdd�
functional_31/conv2d_51/ReluRelu(functional_31/conv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
functional_31/conv2d_51/Relu�
IdentityIdentity*functional_31/conv2d_51/Relu:activations:0*
T0*/
_output_shapes
:���������  2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������  :::::Y U
/
_output_shapes
:���������  
"
_user_specified_name
input_16
�I
�
!__inference__traced_restore_82373
file_prefix%
!assignvariableop_conv2d_49_kernel%
!assignvariableop_1_conv2d_49_bias'
#assignvariableop_2_conv2d_51_kernel%
!assignvariableop_3_conv2d_51_bias#
assignvariableop_4_rmsprop_iter$
 assignvariableop_5_rmsprop_decay,
(assignvariableop_6_rmsprop_learning_rate'
#assignvariableop_7_rmsprop_momentum"
assignvariableop_8_rmsprop_rho
assignvariableop_9_total
assignvariableop_10_count
assignvariableop_11_total_1
assignvariableop_12_count_14
0assignvariableop_13_rmsprop_conv2d_49_kernel_rms2
.assignvariableop_14_rmsprop_conv2d_49_bias_rms4
0assignvariableop_15_rmsprop_conv2d_51_kernel_rms2
.assignvariableop_16_rmsprop_conv2d_51_bias_rms
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
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_49_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_49_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_51_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_51_biasIdentity_3:output:0"/device:CPU:0*
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
AssignVariableOp_13AssignVariableOp0assignvariableop_13_rmsprop_conv2d_49_kernel_rmsIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp.assignvariableop_14_rmsprop_conv2d_49_bias_rmsIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp0assignvariableop_15_rmsprop_conv2d_51_kernel_rmsIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp.assignvariableop_16_rmsprop_conv2d_51_bias_rmsIdentity_16:output:0"/device:CPU:0*
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
�
�
-__inference_functional_31_layer_call_fn_82087
input_16
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_16unknown	unknown_0	unknown_1	unknown_2*
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
GPU 2J 8� *Q
fLRJ
H__inference_functional_31_layer_call_and_return_conditional_losses_820762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������  ::::22
StatefulPartitionedCallStatefulPartitionedCall:Y U
/
_output_shapes
:���������  
"
_user_specified_name
input_16
�
�
-__inference_functional_31_layer_call_fn_82198

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
GPU 2J 8� *Q
fLRJ
H__inference_functional_31_layer_call_and_return_conditional_losses_820762
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
�
h
L__inference_zero_padding2d_41_layer_call_and_return_conditional_losses_81937

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
�%
�
H__inference_functional_31_layer_call_and_return_conditional_losses_82141

inputs,
(conv2d_49_conv2d_readvariableop_resource-
)conv2d_49_biasadd_readvariableop_resource,
(conv2d_51_conv2d_readvariableop_resource-
)conv2d_51_biasadd_readvariableop_resource
identity��
zero_padding2d_39/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2 
zero_padding2d_39/Pad/paddings�
zero_padding2d_39/PadPadinputs'zero_padding2d_39/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2
zero_padding2d_39/Pad�
conv2d_49/Conv2D/ReadVariableOpReadVariableOp(conv2d_49_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_49/Conv2D/ReadVariableOp�
conv2d_49/Conv2DConv2Dzero_padding2d_39/Pad:output:0'conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
conv2d_49/Conv2D�
 conv2d_49/BiasAdd/ReadVariableOpReadVariableOp)conv2d_49_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_49/BiasAdd/ReadVariableOp�
conv2d_49/BiasAddBiasAddconv2d_49/Conv2D:output:0(conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2
conv2d_49/BiasAdd~
conv2d_49/ReluReluconv2d_49/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
conv2d_49/Relu�
max_pooling2d_17/MaxPoolMaxPoolconv2d_49/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
2
max_pooling2d_17/MaxPool�
up_sampling2d_17/ShapeShape!max_pooling2d_17/MaxPool:output:0*
T0*
_output_shapes
:2
up_sampling2d_17/Shape�
$up_sampling2d_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$up_sampling2d_17/strided_slice/stack�
&up_sampling2d_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&up_sampling2d_17/strided_slice/stack_1�
&up_sampling2d_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&up_sampling2d_17/strided_slice/stack_2�
up_sampling2d_17/strided_sliceStridedSliceup_sampling2d_17/Shape:output:0-up_sampling2d_17/strided_slice/stack:output:0/up_sampling2d_17/strided_slice/stack_1:output:0/up_sampling2d_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2 
up_sampling2d_17/strided_slice�
up_sampling2d_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_17/Const�
up_sampling2d_17/mulMul'up_sampling2d_17/strided_slice:output:0up_sampling2d_17/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_17/mul�
-up_sampling2d_17/resize/ResizeNearestNeighborResizeNearestNeighbor!max_pooling2d_17/MaxPool:output:0up_sampling2d_17/mul:z:0*
T0*/
_output_shapes
:���������  *
half_pixel_centers(2/
-up_sampling2d_17/resize/ResizeNearestNeighbor�
zero_padding2d_41/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2 
zero_padding2d_41/Pad/paddings�
zero_padding2d_41/PadPad>up_sampling2d_17/resize/ResizeNearestNeighbor:resized_images:0'zero_padding2d_41/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2
zero_padding2d_41/Pad�
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_51/Conv2D/ReadVariableOp�
conv2d_51/Conv2DConv2Dzero_padding2d_41/Pad:output:0'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
conv2d_51/Conv2D�
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_51/BiasAdd/ReadVariableOp�
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2
conv2d_51/BiasAdd~
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
conv2d_51/Relux
IdentityIdentityconv2d_51/Relu:activations:0*
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
g
K__inference_up_sampling2d_17_layer_call_and_return_conditional_losses_81924

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
�	
�
D__inference_conv2d_51_layer_call_and_return_conditional_losses_82229

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
�%
�
H__inference_functional_31_layer_call_and_return_conditional_losses_82172

inputs,
(conv2d_49_conv2d_readvariableop_resource-
)conv2d_49_biasadd_readvariableop_resource,
(conv2d_51_conv2d_readvariableop_resource-
)conv2d_51_biasadd_readvariableop_resource
identity��
zero_padding2d_39/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2 
zero_padding2d_39/Pad/paddings�
zero_padding2d_39/PadPadinputs'zero_padding2d_39/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2
zero_padding2d_39/Pad�
conv2d_49/Conv2D/ReadVariableOpReadVariableOp(conv2d_49_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_49/Conv2D/ReadVariableOp�
conv2d_49/Conv2DConv2Dzero_padding2d_39/Pad:output:0'conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
conv2d_49/Conv2D�
 conv2d_49/BiasAdd/ReadVariableOpReadVariableOp)conv2d_49_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_49/BiasAdd/ReadVariableOp�
conv2d_49/BiasAddBiasAddconv2d_49/Conv2D:output:0(conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2
conv2d_49/BiasAdd~
conv2d_49/ReluReluconv2d_49/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
conv2d_49/Relu�
max_pooling2d_17/MaxPoolMaxPoolconv2d_49/Relu:activations:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
2
max_pooling2d_17/MaxPool�
up_sampling2d_17/ShapeShape!max_pooling2d_17/MaxPool:output:0*
T0*
_output_shapes
:2
up_sampling2d_17/Shape�
$up_sampling2d_17/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$up_sampling2d_17/strided_slice/stack�
&up_sampling2d_17/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&up_sampling2d_17/strided_slice/stack_1�
&up_sampling2d_17/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&up_sampling2d_17/strided_slice/stack_2�
up_sampling2d_17/strided_sliceStridedSliceup_sampling2d_17/Shape:output:0-up_sampling2d_17/strided_slice/stack:output:0/up_sampling2d_17/strided_slice/stack_1:output:0/up_sampling2d_17/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2 
up_sampling2d_17/strided_slice�
up_sampling2d_17/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_17/Const�
up_sampling2d_17/mulMul'up_sampling2d_17/strided_slice:output:0up_sampling2d_17/Const:output:0*
T0*
_output_shapes
:2
up_sampling2d_17/mul�
-up_sampling2d_17/resize/ResizeNearestNeighborResizeNearestNeighbor!max_pooling2d_17/MaxPool:output:0up_sampling2d_17/mul:z:0*
T0*/
_output_shapes
:���������  *
half_pixel_centers(2/
-up_sampling2d_17/resize/ResizeNearestNeighbor�
zero_padding2d_41/Pad/paddingsConst*
_output_shapes

:*
dtype0*9
value0B."                             2 
zero_padding2d_41/Pad/paddings�
zero_padding2d_41/PadPad>up_sampling2d_17/resize/ResizeNearestNeighbor:resized_images:0'zero_padding2d_41/Pad/paddings:output:0*
T0*/
_output_shapes
:���������""2
zero_padding2d_41/Pad�
conv2d_51/Conv2D/ReadVariableOpReadVariableOp(conv2d_51_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_51/Conv2D/ReadVariableOp�
conv2d_51/Conv2DConv2Dzero_padding2d_41/Pad:output:0'conv2d_51/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  *
paddingVALID*
strides
2
conv2d_51/Conv2D�
 conv2d_51/BiasAdd/ReadVariableOpReadVariableOp)conv2d_51_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_51/BiasAdd/ReadVariableOp�
conv2d_51/BiasAddBiasAddconv2d_51/Conv2D:output:0(conv2d_51/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2
conv2d_51/BiasAdd~
conv2d_51/ReluReluconv2d_51/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2
conv2d_51/Relux
IdentityIdentityconv2d_51/Relu:activations:0*
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
�
H__inference_functional_31_layer_call_and_return_conditional_losses_82076

inputs
conv2d_49_82062
conv2d_49_82064
conv2d_51_82070
conv2d_51_82072
identity��!conv2d_49/StatefulPartitionedCall�!conv2d_51/StatefulPartitionedCall�
!zero_padding2d_39/PartitionedCallPartitionedCallinputs*
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
GPU 2J 8� *U
fPRN
L__inference_zero_padding2d_39_layer_call_and_return_conditional_losses_818932#
!zero_padding2d_39/PartitionedCall�
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_39/PartitionedCall:output:0conv2d_49_82062conv2d_49_82064*
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
GPU 2J 8� *M
fHRF
D__inference_conv2d_49_layer_call_and_return_conditional_losses_819592#
!conv2d_49/StatefulPartitionedCall�
 max_pooling2d_17/PartitionedCallPartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
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
GPU 2J 8� *T
fORM
K__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_819052"
 max_pooling2d_17/PartitionedCall�
 up_sampling2d_17/PartitionedCallPartitionedCall)max_pooling2d_17/PartitionedCall:output:0*
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
K__inference_up_sampling2d_17_layer_call_and_return_conditional_losses_819242"
 up_sampling2d_17/PartitionedCall�
!zero_padding2d_41/PartitionedCallPartitionedCall)up_sampling2d_17/PartitionedCall:output:0*
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
GPU 2J 8� *U
fPRN
L__inference_zero_padding2d_41_layer_call_and_return_conditional_losses_819372#
!zero_padding2d_41/PartitionedCall�
!conv2d_51/StatefulPartitionedCallStatefulPartitionedCall*zero_padding2d_41/PartitionedCall:output:0conv2d_51_82070conv2d_51_82072*
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
D__inference_conv2d_51_layer_call_and_return_conditional_losses_819892#
!conv2d_51/StatefulPartitionedCall�
IdentityIdentity*conv2d_51/StatefulPartitionedCall:output:0"^conv2d_49/StatefulPartitionedCall"^conv2d_51/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������  ::::2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_51/StatefulPartitionedCall!conv2d_51/StatefulPartitionedCall:W S
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
E
input_169
serving_default_input_16:0���������  E
	conv2d_518
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
	regularization_losses

	variables
trainable_variables
	keras_api

signatures
a__call__
*b&call_and_return_all_conditional_losses
c_default_save_signature"�8
_tf_keras_network�8{"class_name": "Functional", "name": "functional_31", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_31", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_16"}, "name": "input_16", "inbound_nodes": []}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_39", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_39", "inbound_nodes": [[["input_16", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_49", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_49", "inbound_nodes": [[["zero_padding2d_39", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_17", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_17", "inbound_nodes": [[["conv2d_49", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_17", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_17", "inbound_nodes": [[["max_pooling2d_17", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_41", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_41", "inbound_nodes": [[["up_sampling2d_17", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_51", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_51", "inbound_nodes": [[["zero_padding2d_41", 0, 0, {}]]]}], "input_layers": [["input_16", 0, 0]], "output_layers": [["conv2d_51", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_31", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_16"}, "name": "input_16", "inbound_nodes": []}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_39", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_39", "inbound_nodes": [[["input_16", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_49", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_49", "inbound_nodes": [[["zero_padding2d_39", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_17", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling2d_17", "inbound_nodes": [[["conv2d_49", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "up_sampling2d_17", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "name": "up_sampling2d_17", "inbound_nodes": [[["max_pooling2d_17", 0, 0, {}]]]}, {"class_name": "ZeroPadding2D", "config": {"name": "zero_padding2d_41", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "name": "zero_padding2d_41", "inbound_nodes": [[["up_sampling2d_17", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_51", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_51", "inbound_nodes": [[["zero_padding2d_41", 0, 0, {}]]]}], "input_layers": [["input_16", 0, 0]], "output_layers": [["conv2d_51", 0, 0]]}}, "training_config": {"loss": "binary_crossentropy", "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "learning_rate": 0.0010000000474974513, "decay": 0.0, "rho": 0.8999999761581421, "momentum": 0.0, "epsilon": 1e-07, "centered": false}}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "input_16", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_16"}}
�
regularization_losses
	variables
trainable_variables
	keras_api
d__call__
*e&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "ZeroPadding2D", "name": "zero_padding2d_39", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "zero_padding2d_39", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
f__call__
*g&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_49", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_49", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 34, 34, 3]}}
�
regularization_losses
	variables
trainable_variables
	keras_api
h__call__
*i&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "MaxPooling2D", "name": "max_pooling2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_17", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
regularization_losses
	variables
trainable_variables
	keras_api
j__call__
*k&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "UpSampling2D", "name": "up_sampling2d_17", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "up_sampling2d_17", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
 regularization_losses
!	variables
"trainable_variables
#	keras_api
l__call__
*m&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "ZeroPadding2D", "name": "zero_padding2d_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "zero_padding2d_41", "trainable": true, "dtype": "float32", "padding": {"class_name": "__tuple__", "items": [{"class_name": "__tuple__", "items": [1, 1]}, {"class_name": "__tuple__", "items": [1, 1]}]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�	

$kernel
%bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
n__call__
*o&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_51", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_51", "trainable": true, "dtype": "float32", "filters": 3, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 34, 34, 3]}}
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
�
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
�
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
*:(2conv2d_49/kernel
:2conv2d_49/bias
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
�
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
�
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
�
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
�
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
*:(2conv2d_51/kernel
:2conv2d_51/bias
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
�
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
4:22RMSprop/conv2d_49/kernel/rms
&:$2RMSprop/conv2d_49/bias/rms
4:22RMSprop/conv2d_51/kernel/rms
&:$2RMSprop/conv2d_51/bias/rms
�2�
-__inference_functional_31_layer_call_fn_82185
-__inference_functional_31_layer_call_fn_82198
-__inference_functional_31_layer_call_fn_82087
-__inference_functional_31_layer_call_fn_82056�
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
H__inference_functional_31_layer_call_and_return_conditional_losses_82024
H__inference_functional_31_layer_call_and_return_conditional_losses_82172
H__inference_functional_31_layer_call_and_return_conditional_losses_82141
H__inference_functional_31_layer_call_and_return_conditional_losses_82006�
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
�2�
 __inference__wrapped_model_81886�
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
annotations� */�,
*�'
input_16���������  
�2�
1__inference_zero_padding2d_39_layer_call_fn_81899�
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
L__inference_zero_padding2d_39_layer_call_and_return_conditional_losses_81893�
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
)__inference_conv2d_49_layer_call_fn_82218�
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
D__inference_conv2d_49_layer_call_and_return_conditional_losses_82209�
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
0__inference_max_pooling2d_17_layer_call_fn_81911�
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
K__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_81905�
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
0__inference_up_sampling2d_17_layer_call_fn_81930�
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
K__inference_up_sampling2d_17_layer_call_and_return_conditional_losses_81924�
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
1__inference_zero_padding2d_41_layer_call_fn_81943�
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
L__inference_zero_padding2d_41_layer_call_and_return_conditional_losses_81937�
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
)__inference_conv2d_51_layer_call_fn_82238�
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
D__inference_conv2d_51_layer_call_and_return_conditional_losses_82229�
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
3B1
#__inference_signature_wrapper_82110input_16�
 __inference__wrapped_model_81886�$%9�6
/�,
*�'
input_16���������  
� "=�:
8
	conv2d_51+�(
	conv2d_51���������  �
D__inference_conv2d_49_layer_call_and_return_conditional_losses_82209l7�4
-�*
(�%
inputs���������""
� "-�*
#� 
0���������  
� �
)__inference_conv2d_49_layer_call_fn_82218_7�4
-�*
(�%
inputs���������""
� " ����������  �
D__inference_conv2d_51_layer_call_and_return_conditional_losses_82229�$%I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+���������������������������
� �
)__inference_conv2d_51_layer_call_fn_82238�$%I�F
?�<
:�7
inputs+���������������������������
� "2�/+����������������������������
H__inference_functional_31_layer_call_and_return_conditional_losses_82006�$%A�>
7�4
*�'
input_16���������  
p

 
� "?�<
5�2
0+���������������������������
� �
H__inference_functional_31_layer_call_and_return_conditional_losses_82024�$%A�>
7�4
*�'
input_16���������  
p 

 
� "?�<
5�2
0+���������������������������
� �
H__inference_functional_31_layer_call_and_return_conditional_losses_82141v$%?�<
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
H__inference_functional_31_layer_call_and_return_conditional_losses_82172v$%?�<
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
-__inference_functional_31_layer_call_fn_82056}$%A�>
7�4
*�'
input_16���������  
p

 
� "2�/+����������������������������
-__inference_functional_31_layer_call_fn_82087}$%A�>
7�4
*�'
input_16���������  
p 

 
� "2�/+����������������������������
-__inference_functional_31_layer_call_fn_82185{$%?�<
5�2
(�%
inputs���������  
p

 
� "2�/+����������������������������
-__inference_functional_31_layer_call_fn_82198{$%?�<
5�2
(�%
inputs���������  
p 

 
� "2�/+����������������������������
K__inference_max_pooling2d_17_layer_call_and_return_conditional_losses_81905�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
0__inference_max_pooling2d_17_layer_call_fn_81911�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
#__inference_signature_wrapper_82110�$%E�B
� 
;�8
6
input_16*�'
input_16���������  "=�:
8
	conv2d_51+�(
	conv2d_51���������  �
K__inference_up_sampling2d_17_layer_call_and_return_conditional_losses_81924�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
0__inference_up_sampling2d_17_layer_call_fn_81930�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
L__inference_zero_padding2d_39_layer_call_and_return_conditional_losses_81893�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_zero_padding2d_39_layer_call_fn_81899�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
L__inference_zero_padding2d_41_layer_call_and_return_conditional_losses_81937�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_zero_padding2d_41_layer_call_fn_81943�R�O
H�E
C�@
inputs4������������������������������������
� ";�84������������������������������������